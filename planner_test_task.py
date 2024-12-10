import os
import random
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import IPython as ipy
from itertools import product, combinations

from planning.Safe_Planner import *
# from nav_sim.env.go1_env import Go1Env
from nav_sim.env.task_env_exp import TaskEnv

from utils.pc_util import preprocess_point_cloud, pc_to_axis_aligned_rep, random_sampling
from utils.box_util import box2d_iou
from utils.make_args import make_args_parser

from nav_sim.asset.util import state_lin_to_bin, state_bin_to_lin
from nav_sim.test.clustering import is_box_visible

from models import build_model
from datasets.sunrgbd import SunrgbdDatasetConfig as dataset_config
import time

import torch
from torch.multiprocessing import Pool, Process, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

from models.model_perception import MLPModelDet

f = open('planning/pre_compute/reachable-2k.pkl', 'rb')
reachable = pickle.load(f)
f = open('planning/pre_compute/Pset-2k.pkl', 'rb')
Pset = pickle.load(f)
dt = 0.1
print("dt=", dt)

# camera + 3DETR
num_pc_points = 40000

np.random.seed(44)
torch.manual_seed(44)

parser = make_args_parser()
args = parser.parse_args(args=[])

# Dataset config: use SUNRGB-D
dataset_config = dataset_config()
# Build model
model, _ = build_model(args, dataset_config)

# Load pre-trained weights
sd = torch.load(args.test_ckpt, map_location=torch.device("cpu")) 
model.load_state_dict(sd["model"]) 

model = model.cuda()
model.eval()

device = torch.device("cuda")

num_in = 32768
num_out = (15, 2,3) # bbox corner representation

# Load params from json file
with open("env_params.json", "r") as read_file:
    params = json.load(read_file)

robot_radius = 0.3
# cp = 0.05 # 85% baseline
# cp=0.75 # 85% PwC
# cp = 1.33 # 95% PwC
# cp = 0.101 # 95% baseline
# cp = 0.83 # 90% PwC
# cp = 0.066 # 90% baseline
# cp = 0.68 # 80% PwC
# cp = 0.025 # 80% baseline
# cp = 0.63 # 75% PwC
# cp = 0.016 # 75% baseline
# cp = 0.6249 # 500 samples
# cp = 0.7086 # 1k samples
# cp = 0.6910 # 1.5k samples
cp = 0.7441 # 2k samples
is_finetune=False
if is_finetune:
    cp=0.65 # 85% PwC
    model_cp = MLPModelDet(num_in, num_out)
    model_cp.to(device)
    model_cp.load_state_dict(torch.load("trained_models/perception_model"))
print("CP: ", cp)

foldername = "../data/perception-guarantees/room_1203_rot/"

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
    filename = foldername + str(task.env) + '/cp' + str(cp)
    grid_data = np.load((foldername + str(task.env) + '/occupancy_grid.npz'), allow_pickle=True)
    occupancy_grid = grid_data['arr_0']
    N, M = occupancy_grid.shape
    env = TaskEnv(render=visualize)
    # init_state = [1,-3,-np.pi/2]
    task.init_state = [0.2,-1,0,0]
    task.goal_loc = [7, -2]
    # task.init_state = [float(v) for v in init_state]
    # task.goal_loc = [float(v) for v in goal_loc]
    planner_init_state = [5,0.2,0,0]
    sp = Safe_Planner(init_state=planner_init_state, FoV=60*np.pi/180, n_samples=len(Pset)-1,dt=dt,radius = 0.1, sensor_dt=0.2, max_search_iter=2000)
    sp.load_reachable(Pset, reachable)
    env.dt = sp.dt
    env.reset(task)
    t = 0
    observation = env.step([0,0])[0] # initial observation
    steps_taken = 0
    state_traj = []
    gt_obs = [[[obs[0], obs[1], obs[2]],[obs[3], obs[4], obs[5]]] for obs in task.piece_bounds_all]
    # print("GT obstacles", gt_obs)
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
        # print(state)
        boxes = get_box(observation, visualize)
        # print(boxes)
        boxes[:,0,:] -= cp
        boxes[:,1,:] += cp
        boxes = boxes_to_planner_frame(boxes, sp)
        ###########################################################################
        X = observation[:, (observation[2, :] >0.1)]
        X = X[:, np.abs(X[1,:]) < 3.9]
        X = X[:, X[0,:]>0.05]
        X = X[:, X[0,:] < 7.95]
        
        X = np.transpose(np.array(X))
        misdetected += count_misdetection(boxes, ground_truth, X, task.piece_bounds_all)
        # print("Misdetected: ", misdetected)
        time_misdetected+=1
        ###########################################################################

        st = time.time()
        # try:
        res = sp.plan(state, boxes)
        #     if len(res[1]) == 0:
        #         plan_fail += 1
        # except:
        #     print("Env: ", str(task.env), " Failed to get plan, Code Error")
        #     continue
            # plot_results(filename, state_traj , ground_truth, sp)
            # return {"trajectory": np.array(state_traj), "done": done, "collision": collided}
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
            if (len(prev_policy) > idx_prev+1): #int(sp.sensor_dt/sp.dt):
                # for kk in range(int(sp.sensor_dt/sp.dt)):
                idx_prev += 1
                action = prev_policy[idx_prev]
                observation, reward, done, info = env.step(action)
                # time.sleep(sp.dt)
                t += sp.dt
            else:
                action = [0,0]
                observation, reward, done, info = env.step(action)
                # time.sleep(sp.dt)
                t += sp.dt
            # for step in range(int(sp.sensor_dt/sp.dt)):
            #     action = [0,0]
            #     observation, reward, done, info = env.step(action)
            #     state = env._state
            #     state_traj.append(state_to_planner(state, sp))
            #     t += sp.dt
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
    plt.savefig(filename + 'traj_plot_2k.png')
    # plt.show()

def get_box(observation_, visualize = False):
    # Filter points with z < 0.01 and abs(y) > 3.5 and x> 0.01 and within a 1m distance of the robot
    # axis transformed, so filter x,y same way
    observation_ = observation_[:, observation_[2, :] < 2.9]

    # ipy.embed()
    observation = np.copy(observation_)
    observation[1,:] = observation_[0,:]
    observation[0,:] = -observation_[1,:]

    if (len(observation[0])>0):
        # Preprocess point cloud (random sampling of points), if there are any LIDAR returns
        points_new = np.transpose(np.array(observation))
        points = np.zeros((1,num_pc_points, 3),dtype='float32')
        points = preprocess_point_cloud(np.array(points_new), num_pc_points)
    else:
        # There are no returns from the LIDAR, object is not visible
        points = np.zeros((1,num_pc_points, 3),dtype='float32')
    
    # Convert from camera frame to world frame
    point_clouds = []
    point_clouds.append(points)
    
    batch_size = 1
    pc = np.array(point_clouds).astype('float32')
    pc = pc.reshape((batch_size, num_pc_points, 3))

    pc_all = torch.from_numpy(pc).to(device)
    pc_min_all = pc_all.min(1).values
    pc_max_all = pc_all.max(1).values
    inputs = {'point_clouds': pc_all, 'point_cloud_dims_min': pc_min_all, 'point_cloud_dims_max': pc_max_all}

    outputs = model(inputs)
    if is_finetune:
        box_features = outputs["box_features"].detach()
        box_features_ = torch.reshape(box_features, (1,1,128,256))
        finetune = model_cp(box_features_)
    
    bbox_pred_points = outputs['outputs']['box_corners'].detach().cpu()
    obj_prob = outputs["outputs"]["objectness_prob"].clone().detach().cpu()
    cls_prob = outputs["outputs"]["sem_cls_prob"].clone().detach().cpu()

    chair_prob = cls_prob[:,:,3]
    sort_box = torch.sort(obj_prob,1,descending=True)

    # Visualize
    if visualize:
        pc_plot = pc[:, pc[0,:,2] > 0.0,:]
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(
            pc_plot[0,:,0], pc_plot[0,:,1],pc_plot[0,:,2]
        )
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('auto')

    num_probs = 0
    num_boxes = 15
    corners = []
    if np.any(np.isnan(np.array(bbox_pred_points))):
            return get_room_size_box(pc_all)
    
    for (sorted_idx,prob) in zip(list(sort_box[1][0,:]), list(sort_box[0][0,:])):
        if (num_probs < num_boxes):
            prob = prob.numpy()
            bbox = bbox_pred_points[range(batch_size), sorted_idx, :, :]
            cc = pc_to_axis_aligned_rep(bbox.numpy())
            flag = False
            if num_probs == 0:
                corners.append(cc)
                num_probs +=1
            else:
                for cc_keep in corners:
                    bb1 = (cc_keep[0,0,0],cc_keep[0,0,1],cc_keep[0,1,0],cc_keep[0,1,1])
                    bb2 = (cc[0,0,0],cc[0,0,1],cc[0,1,0],cc[0,1,1])
                    # Non-maximal supression, check if IoU more than some threshold to keep box
                    if(box2d_iou(bb1,bb2) > 0.1):
                        flag = True
                if not flag:    
                    corners.append(cc)
                    num_probs +=1

            if visualize:
                r0 = [cc[0,0, 0], cc[0,1, 0]]
                r1 = [cc[0,0, 1], cc[0,1, 1]]
                r2 = [cc[0,0, 2], cc[0,1, 2]]

                for s, e in combinations(np.array(list(product(r0, r1, r2))), 2):
                    if (np.sum(np.abs(s-e)) == r0[1]-r0[0] or 
                        np.sum(np.abs(s-e)) == r1[1]-r1[0] or 
                        np.sum(np.abs(s-e)) == r2[1]-r2[0]):
                        if (visualize and not flag):
                            ax.plot3D(*zip(s, e), color=(0.5+0.5*prob, 0.1,0.1))
    
    if is_finetune:
        finetuned_arr = finetune.cpu().detach().numpy()
        finetuned_arr = np.squeeze(finetuned_arr)
        # ipy.embed()
        corners+=finetuned_arr
    
    boxes = np.zeros((len(corners),2,2))
    for i in range(len(corners)):
        # boxes[i,:,:] = corners[i][0,:,0:2]
        boxes[i,:,0] = corners[i][0,:,1]
        boxes[i,0,1] = -corners[i][0,1,0]
        boxes[i,1,1] = -corners[i][0,0,0]

    return boxes

def get_room_size_box( pc_all):
    room_size = 8
    num_chairs =5
    boxes = np.zeros((num_chairs, 2,3))
    # box = torch.tensor(pc_to_axis_aligned_rep(bbox_pred_points.numpy()))
    boxes[:,0,1] = 0*np.ones_like(boxes[:,0,0])
    boxes[:,0,0] = (-room_size/2)*np.ones_like(boxes[:,0,1])
    boxes[:,0,2] = 0*np.ones_like(boxes[:,0,2])
    boxes[:,1,1] = room_size*np.ones_like(boxes[:,1,0])
    boxes[:,1,0] = (room_size/2)*np.ones_like(boxes[:,1,1])
    boxes[:,1,2] = room_size*np.ones_like(boxes[:,1,2])

    pc_min_all = pc_all.min(1).values
    pc_max_all = pc_all.max(1).values
    inputs = {'point_clouds': pc_all, 'point_cloud_dims_min': pc_min_all, 'point_cloud_dims_max': pc_max_all}

    outputs = model(inputs)
    box_features = outputs["box_features"].detach().cpu()
    return boxes

def multi_run_wrapper(args):
   return plan_env(*args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task_dataset', default='/home/zm2074/Projects/data/perception-guarantees/task_1203_rot.pkl',
        nargs='?', help='path to save the task files'
    )
    parser.add_argument(
        '--save_dataset', default='/home/zm2074/Projects/data/perception-guarantees/task_1203_rot.npz',
        nargs='?', help='path to save the task files'
    )
    args = parser.parse_args()

    # Load task dataset
    with open(args.task_dataset, 'rb') as f:
        task_dataset = pickle.load(f)

    # get root repository path
    nav_sim_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


    # Sample random task
    save_tasks = []
    # ipy.embed()
    ii = 0
    for task in task_dataset:

        # task = random.choice(task_dataset)

        # Initialize task
        task.goal_radius = 0.5
        #
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
        task.env= task.base_path.split('/')[-1]
        ii+=1

        # Run environment
        # run_env(task)
        # save_tasks += [task]
        # print(len(save_tasks))

    ##################################################################
    # Number of environments
    num_envs = 100

    # Number of parallel threads
    num_parallel = 10
    ##################################################################

    # _, _, _ = render_env(seed=0)

    ##################################################################
    env = 0
    batch_size = num_parallel
    save_file = args.save_dataset
    save_res = []
    ##################################################################

    collisions = 0
    failed = 0
    for task in task_dataset:
        # save_tasks += [task]
        env += 1 
        if env%batch_size == 0:
            # if env > 0: # In case code stops running, change starting environment to last batch saved
                batch = math.floor(env/batch_size)
                print("Saving batch", str(batch))
                t_start = time.time()
                pool = Pool(num_parallel) # Number of parallel processes
                results = pool.map_async(plan_env, task_dataset[env-batch_size:env]) # Compute results
                pool.close()
                pool.join()
                # ipy.embed()
                ii = 0
                for result in results.get():
                    # Save data
                    file_batch = foldername+ str(env-batch_size+ii) + "/cp_" + str(cp) + "_2k.npz"
                    np.savez_compressed(file_batch, data=result)
                    ii+=1
        # result = plan_env(task)
