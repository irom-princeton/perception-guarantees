import os
import random
import argparse
def make_args_parser():
    parser = argparse.ArgumentParser("3D Detection Using Transformers", add_help=False)

    ##### Model #####
    parser.add_argument(
        "--model_name",
        default="3detr",
        type=str,
        help="Name of the model",
        choices=["3detr"],
    )
    ### Encoder
    parser.add_argument(
        "--enc_type", default="vanilla", choices=["masked", "maskedv2", "vanilla"]
    )
    # Below options are only valid for vanilla encoder
    parser.add_argument("--enc_nlayers", default=3, type=int)
    parser.add_argument("--enc_dim", default=256, type=int)
    parser.add_argument("--enc_ffn_dim", default=128, type=int)
    parser.add_argument("--enc_dropout", default=0.1, type=float)
    parser.add_argument("--enc_nhead", default=4, type=int)
    parser.add_argument("--enc_pos_embed", default=None, type=str)
    parser.add_argument("--enc_activation", default="relu", type=str)

    ### Decoder
    parser.add_argument("--dec_nlayers", default=8, type=int)
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_ffn_dim", default=256, type=int)
    parser.add_argument("--dec_dropout", default=0.1, type=float)
    parser.add_argument("--dec_nhead", default=4, type=int)

    ### MLP heads for predicting bounding boxes
    parser.add_argument("--mlp_dropout", default=0.3, type=float)
    parser.add_argument(
        "--nsemcls",
        default=-1,
        type=int,
        help="Number of semantic object classes. Can be inferred from dataset",
    )

    ### Other model params
    parser.add_argument("--preenc_npoints", default=2048, type=int)
    parser.add_argument(
        "--pos_embed", default="fourier", type=str, choices=["fourier", "sine"]
    )
    parser.add_argument("--nqueries", default=128, type=int) # 256
    parser.add_argument("--use_color", default=False, action="store_true")


    ##### Testing #####
    parser.add_argument("--test_only", default=True, action="store_true")
    parser.add_argument("--test_ckpt", default="pretrained/sunrgbd_ep1080.pth", type=str)


    # ##### Number of points #####
    # parser.add_argument('--num_point', type=int, default=40000, help='Point Number [default: 40000]')


    return parser
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import json
import plotly.graph_objects as go

from itertools import product, combinations
from planning.Safe_Planner import *
from nav_sim.env.task_env_exp import TaskEnv
import sys
sys.path.append('utils')
sys.path.append('datasets')
import warnings
warnings.filterwarnings("error")
import IPython as ipy

import torch
from torch.multiprocessing import Pool, Process, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

from nav_sim.test.clustering import cluster, is_box_visible
# from utils.pc_util import preprocess_point_cloud, read_ply, pc_to_axis_aligned_rep, pc_cam_to_3detr, is_inside_camera_fov
# from utils.box_util import box2d_iou
# from utils.make_args import make_args_parser
from omegaconf import OmegaConf

from mmdet3d.apis import LidarDet3DInferencer

import warnings
warnings.filterwarnings("ignore")

# base path
from pathlib import Path
pg_path: Path = Path(__file__).resolve().parent
room_path = pg_path/'nav_sim/sim_data/rooms_multiple/'
taskpath = pg_path/'nav_sim/sim_data/task_multiple.pkl'

# camera + TR3D
num_pc_points = 40000

parser = make_args_parser()
args = parser.parse_args(args=[])

# Model and weights (Scannet version)
model = f"/home/may/mmdetection3d/projects/TR3D/configs/tr3d_1xb16_scannet-3d-18class.py" # TR3D
weights = f"{pg_path}/models/tr3d_1xb16_scannet-3d-18class.pth" # TR3D pretrained weights

# Initialize inferencer
inferencer = LidarDet3DInferencer(model=model, weights=weights)
cp=0

f = open(pg_path/'planning/pre_compute/reachable-2k.pkl', 'rb')
reachable = pickle.load(f)
f = open(pg_path/'planning/pre_compute/Pset-2k.pkl', 'rb')
Pset = pickle.load(f)
dt = 0.1


# Load params from json file
with open(f"{pg_path}/env_params.json", "r") as read_file:
    params = json.load(read_file)

def state_to_planner(state, sp):
    # convert robot state to planner coordinates
    return np.array([[[0,-1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,1,0]]])@np.array(state) + np.array([sp.world.w/2,0,0,0])

def state_to_go1(state, sp):
    x, y, vx, vy = state[0]
    return np.array([y, -x+sp.world.w/2, vy, -vx])

def boxes_to_planner_frame(boxes, sp):
    boxes_new = np.zeros_like(boxes)
    for i in range(len(boxes)):
        #boxes_new[i,:,:] = np.reshape(np.array([[[0,0,0,-1],[1,0,0,0],[0,-1,0,0],[0,0,1,0]]])@np.reshape(boxes[0],(4,1)),(2,2)) + np.array([sp.world.w/2,0])
        boxes_new[i,0,0] = -boxes[i,1,1] + sp.world.w/2
        boxes_new[i,1,0] = -boxes[i,0,1] + sp.world.w/2
        boxes_new[i,:,1] =  boxes[i,:,0]
    return boxes_new

def count_misdetection(pred_boxes, ground_truth, X, piece_bounds):
    if(len(X) > 0):
        is_vis = is_box_visible(X, piece_bounds, visualize=False)
    else: 
        is_vis = [False]*len(ground_truth)
    num_is_vis = 0
    misdetected = 0
    for ii in range(len(ground_truth)):
        detected = False
        num_is_vis += 1 if is_vis[ii] == True else 0
        for jj in range(len(pred_boxes)):
            if is_vis[ii] and ((ground_truth[ii,0,0]>= pred_boxes[jj,0,0]) and (ground_truth[ii,0,1]>= pred_boxes[jj,0,1]) and (ground_truth[ii,1,0]<= pred_boxes[jj,1,0]) and (ground_truth[ii,1,1]<= pred_boxes[jj,1,1])):
                detected = True
                break
        if is_vis[ii] and not detected:
            misdetected+=1
    if num_is_vis == 0:
        return 0
    else:
        return (misdetected/num_is_vis)


            
def plan_env(task):
    # initialize planner
    visualize = False
    task.goal_radius = 1.0
    filename = room_path/str(task.env)/f'cp{cp}'
    grid_data = np.load((room_path/str(task.env)/'occupancy_grid.npz'), allow_pickle=True)
    occupancy_grid = grid_data['arr_0']
    N, M = occupancy_grid.shape
    env = TaskEnv(render=visualize)
    task.init_state = [0.2,-1,0,0]
    task.goal_loc = [7, -2]
    planner_init_state = [5,0.2,0,0]
    sp = Safe_Planner(init_state=planner_init_state, FoV=70*np.pi/180, n_samples=len(Pset)-1,dt=dt,radius = 0.1, sensor_dt=0.2, max_search_iter=2000)
    sp.load_reachable(Pset, reachable)
    env.dt = sp.dt
    env.reset(task)
    t = 0
    observation = env.step([0,0])[0] # initial observation
    steps_taken = 0
    state_traj = []
    gt_obs = [[[obs[0], obs[1], obs[2]],[obs[3], obs[4], obs[5]]] for obs in task.piece_bounds_all]
    ground_truth = boxes_to_planner_frame(np.array(gt_obs), sp)
    done = False
    collided = False
    misdetected = 0
    time_misdetected = 0
    prev_policy = []
    idx_prev = 0
    plan_fail = 0

    while True and not done and not collided:
        state = state_to_planner(env._state, sp)
        boxes = get_box(observation, inferencer, conf_threshold=0,num_boxes= 15, show_viz=False)
        boxes[:,0,:] -= cp
        boxes[:,1,:] += cp
        boxes = boxes_to_planner_frame(boxes, sp)
        ###########################################################################
        mask = (observation[:,2]<2.9)&(observation[:,2]>0.1)&(np.abs(observation[:,1])<3.9)&(observation[:,0]>0.05)# &(observation[:,0]<7.95)
        X = observation[mask]
        X = np.transpose(np.array(X))
        
        misdetected += count_misdetection(boxes, ground_truth, X, task.piece_bounds_all)
        # print("Misdetected: ", misdetected)
        time_misdetected+=1
        ###########################################################################

        st = time.time()
        res = sp.plan(state, boxes)
        t+=(time.time() - st)
        if (steps_taken % 10) == 0 and visualize:
            # sp.show_connection(res[0]) 
            # sp.world.show(true_boxes=ground_truth)
            sp.show(res[0], true_boxes=np.array(ground_truth))
        steps_taken+=1
        if len(res[0]) > 1 and not done and not collided:
            policy_before_trans = np.vstack(res[2])
            policy = (np.array([[0,1],[-1,0]])@policy_before_trans.T).T
            prev_policy = np.copy(policy)
            for step in range(min(int(sp.sensor_dt/sp.dt), len(policy))):
                idx_prev = step
                state = env._state
                state_traj.append(state_to_planner(state, sp))
                for obs in task.piece_bounds_all:
                    if state[0] < obs[3] and state[0] > obs[0]:
                       if state[1] < obs[4] and state[1] > obs[1]: 
                           og_loc = [round(state[0]/0.1)+1 , round((state[1]+4)/0.1)+1]
                        #    if occupancy_grid[og_loc[0], og_loc[1]]:
                           print("Env: ", str(task.env), " Collision")
                           collided = True
                           break
                action = policy[step]
                observation, reward, done, info = env.step(action)
                t += sp.dt
                if done:
                    print("Env: ", str(task.env), " Success!")
                    break
                elif collided:
                    print("Env: ", str(task.env), " Collided")
                    break
        else:
            if (len(prev_policy) > idx_prev+1):
                idx_prev += 1
                action = prev_policy[idx_prev]
                observation, reward, done, info = env.step(action)
                t += sp.dt
            else:
                action = [0,0]
                observation, reward, done, info = env.step(action)
                t += sp.dt
        if t > 140 or plan_fail > 10:
            print("Env: ", str(task.env), " Failed")
            break
    plot_results(filename, state_traj , ground_truth, sp)
    return {"trajectory": np.array(state_traj), "done": done, "collision": collided, "misdetection": (misdetected/time_misdetected)}

def plot_results(filename, state_traj , ground_truth, sp):
    fig, ax = sp.world.show(true_boxes=ground_truth)
    plt.gca().set_aspect('equal', adjustable='box')
    if len(state_traj) >0:
        state_tf = np.squeeze(np.array(state_traj)).T
        print('state tf', state_tf.shape)
        if state_tf.shape == (4,):
            state_tf = state_tf.reshape((4,1))
        ax.plot(state_tf[0, :], state_tf[1, :], c='r', linewidth=1, label='state')
    plt.legend()
    plt.savefig(str(filename) + 'traj_plot.png')
    # plt.show()


# change to TR3D
def get_box(observation_, inferencer, show_viz=False, conf_threshold=0, num_boxes = 15, detect_objects = [2,3,10]):
    mask = (observation_[:,2]<2.9)&(observation_[:,2]>0.1)&(np.abs(observation_[:,1])<3.9)&(observation_[:,0]>0.05)# &(observation[:,0]<7.95)
    X = observation_[mask]
    # X = np.transpose(np.array(X))
    # observation = observation_.copy()
    # observation = observation.T
    observation = np.array(X)

    inputs_all = {'inputs': {'points': observation}, 'pred_score_thr': conf_threshold, 'out_dir': '', 'show': False, 'wait_time': -1, 'no_save_vis': True, 'no_save_pred': False, 'print_result': False}
    inferencer.show_progress = False
    results = inferencer(**inputs_all)
    
    # "cabinet": 0, "bed": 1, "chair": 2, "sofa": 3, "table": 4, "door": 5, "window": 6, "bookshelf": 7, "picture": 8, 
    # "counter": 9, "desk": 10, "curtain": 11, "refrigerator": 12, "showercurtrain": 13, "toilet": 14, "sink": 15, "bathtub": 16, "garbagebin": 17,
    # breakpoint()
    pred_boxes = results["predictions"][0]["bboxes_3d"]

    pred_confidences = results["predictions"][0]["scores_3d"]
    pred_labels = results["predictions"][0]["labels_3d"]

    sort_inds = np.argsort(pred_confidences)[::-1]
    sorted_inds = [i for i in sort_inds[0:num_boxes]]# if (pred_labels[i] in detect_objects)]
    bboxes = [pred_boxes[i] for i in sorted_inds]

    corners = np.zeros((len(bboxes), 2,3))
    for b in range(len(bboxes)):
        bbox = bboxes[b]
        # box corners
        r0 = [bbox[0]-bbox[3]/2, bbox[0]+bbox[3]/2]
        r1 = [bbox[1]-bbox[4]/2, bbox[1]+bbox[4]/2]
        r2 = [bbox[2], bbox[2]+bbox[5]]
        corners[b,0,:] = [r0[0], r1[0], r2[0]]
        corners[b,1,:] = [r0[1], r1[1], r2[1]]

    if show_viz:
        plot_box_pc(observation, corners, np.array([]), np.array([True]*len(corners)))
    
    return corners

# change to TR3D
def get_room_size_box(num_boxes):
    room_size = 8
    boxes = np.zeros((num_boxes, 2,3))
    boxes[:,0,1] = 0*np.ones_like(boxes[:,0,0])
    boxes[:,0,0] = (-room_size/2)*np.ones_like(boxes[:,0,1])
    boxes[:,0,2] = 0*np.ones_like(boxes[:,0,2])
    boxes[:,1,1] = room_size*np.ones_like(boxes[:,1,0])
    boxes[:,1,0] = (room_size/2)*np.ones_like(boxes[:,1,1])
    boxes[:,1,2] = room_size*np.ones_like(boxes[:,1,2])

    return boxes

def plot_box_pc(pc_plot, output_boxes, gt_boxes, is_vis):

    # Visualize
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=pc_plot[:,0], 
        y=pc_plot[:,1], 
        z=pc_plot[:,2],
        mode='markers',
        marker=dict(size=1)
    ))

    for jj, cc in enumerate(output_boxes):
        r0 = [cc[0, 0], cc[1, 0]]
        r1 = [cc[0, 1], cc[1, 1]]
        r2 = [cc[0, 2], cc[1, 2]]

        for s, e in combinations(np.array(list(product(r0, r1, r2))), 2):
            if (np.sum(np.abs(s-e)) == r0[1]-r0[0] or 
                np.sum(np.abs(s-e)) == r1[1]-r1[0] or 
                np.sum(np.abs(s-e)) == r2[1]-r2[0]):
                fig.add_trace(go.Scatter3d(
                    x=[s[0], e[0]], 
                    y=[s[1], e[1]], 
                    z=[s[2], e[2]],
                    mode='lines',
                    line=dict(color='red')
                ))
    for jj, cc in enumerate(gt_boxes):
        r0 = [cc[0, 0], cc[1, 0]]
        r1 = [cc[0, 1], cc[1, 1]]
        r2 = [cc[0, 2], cc[1, 2]]

        for s, e in combinations(np.array(list(product(r0, r1, r2))), 2):
            if (np.sum(np.abs(s-e)) == r0[1]-r0[0] or 
                np.sum(np.abs(s-e)) == r1[1]-r1[0] or 
                np.sum(np.abs(s-e)) == r2[1]-r2[0]):
                fig.add_trace(go.Scatter3d(
                    x=[s[0], e[0]], 
                    y=[s[1], e[1]], 
                    z=[s[2], e[2]],
                    mode='lines',
                    line=dict(color='green')
                ))

    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))
    fig.show()

def initialize_task(task):
    # Initialize task
    task.env= task.base_path.split('/')[-1]
    task.base_path = f'{room_path}/{task.env}'
    task.mesh_parent_folder = f'{pg_path}/nav_sim/sim_data/3D-FUTURE-model-tiny'
    task.goal_radius = 0.5
    task.observation = {}
    task.observation.type = 'rgb'  # 'rgb' or 'lidar'
    task.observation.rgb = {}
    task.observation.depth = {}
    task.observation.lidar = {}
    task.observation.camera_pos = {}
    task.observation.cam_not_inside_obs = {}
    task.observation.is_visible = {}
    task.observation.rgb.x_offset_from_robot_front = 0.05  # no y offset
    task.observation.rgb.z_offset_from_robot_top = 0.05
    task.observation.rgb.tilt = 0  # degrees of tilting down towards the floor
    task.observation.rgb.img_w = 662
    task.observation.rgb.img_h = 376
    task.observation.rgb.aspect = 1.57
    task.observation.rgb.fov = 70  # in PyBullet, this is vertical field of view in degrees
    task.observation.depth.img_w = task.observation.rgb.img_w  # needs to be the same now - assume coming from the same camera
    task.observation.depth.img_h = task.observation.rgb.img_h
    task.observation.lidar.z_offset_from_robot_top = 0.01  # no x/y offset
    task.observation.lidar.horizontal_res = 1  # resolution, in degree,1
    task.observation.lidar.vertical_res = 1  # resolution, in degree , 1
    task.observation.lidar.vertical_fov = 30  # half in one direction, in degree
    task.observation.lidar.max_range = 5 # in meter Anushri changed from 5 to 8
    return task

def multi_run_wrapper(args):
   return plan_env(*args)

if __name__ == '__main__':
    # Load task dataset
    with open(taskpath, 'rb') as f:
        task_dataset = pickle.load(f)
    
    # initialize all tasks
    for task in task_dataset:
        task = initialize_task(task)
    ##################################################################
    # Number of environments
    num_envs = 100

    # Number of parallel threads
    num_parallel = 10
    ##################################################################

    # _, _, _ = render_env(seed=0)

    ##################################################################
    task_id = 0
    batch_size = num_parallel
    save_res = []
    ##################################################################

    collisions = 0
    failed = 0
    for task in task_dataset:
        # save_tasks += [task]
        task_id += 1 
        if task_id%batch_size == 0:
            # if env > 0: # In case code stops running, change starting environment to last batch saved
                batch = math.floor(task_id/batch_size)
                print("Saving batch", str(batch))
                t_start = time.time()
                pool = Pool(num_parallel) # Number of parallel processes
                results = pool.map_async(plan_env, task_dataset[task_id-batch_size:task_id]) # Compute results
                pool.close()
                pool.join()
                # ipy.embed()
                ii = 0
                for result in results.get():
                    # Save data
                    env = task_dataset[task_id-batch_size+ii].env
                    file_batch = room_path/str(env)/f"cp_{cp}.npz"
                    np.savez_compressed(file_batch, data=result)
                    ii+=1
    # task=task_dataset[0]
    # result = plan_env(task)