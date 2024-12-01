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

from itertools import product, combinations

from nav_sim.env.task_env import TaskEnv
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


# camera + TR3D
num_pc_points = 40000

parser = make_args_parser()
args = parser.parse_args(args=[])

# Model and weights (Scannet version)
model = "tr3d_1xb16_scannet-3d-18class.py" # TR3D
weights = "tr3d_scannet.pth" # TR3D pretrained weights

# Initialize inferencer
inferencer = LidarDet3DInferencer(model=model, weights=weights)

# Load the x,y points to sample
with open('planning/pre_compute/Pset_1.5_7_2K_ramp_unfiltered.pkl', 'rb') as f:
    samples = pickle.load(f)
    # Remove goal
    samples = samples[:-1][:]
# Remove duplicates
sample_proj = [[sample[0], sample[1]] for sample in samples]
s = []
s = [x for x in sample_proj if x not in s and not s.append(x)]
# Transform from planner frame
x = [sample[1] for sample in s]
y = [sample[0]-4 for sample in s]

num_steps = len(x)

# Load params from json file
with open("env_params.json", "r") as read_file:
    params = json.load(read_file)



def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two 2D bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1["x1"] <= bb1["x2"]
    assert bb1["y1"] <= bb1["y2"]
    assert bb2["x1"] <= bb2["x2"]
    assert bb2["y1"] <= bb2["y2"]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def box2d_iou(box1, box2):
    """Compute 2D bounding box IoU.

    Input:
        box1: tuple of (xmin,ymin,xmax,ymax)
        box2: tuple of (xmin,ymin,xmax,ymax)
    Output:
        iou: 2D IoU scalar
    """
    return get_iou(
        {"x1": box1[0], "y1": box1[1], "x2": box1[2], "y2": box1[3]},
        {"x1": box2[0], "y1": box2[1], "x2": box2[2], "y2": box2[3]},
    )





# change to TR3D
def run_env(task):
    visualize = False
    verbose = 0
    init_state = [0,-3.5,0,0]
    goal_loc = [7.5,3.5]
    task.init_state = [float(v) for v in init_state]
    task.goal_loc = [float(v) for v in goal_loc]
    env = TaskEnv(render=visualize)
    env.reset(task)
    camera_pos = []
    cam_not_inside_obs = []
    is_visible = []
    # point_clouds = []
    bbs = []
    outputs = []
    matched_gts =[]
    # Change seed if you want
    # np.random.default_rng(12345)

    # Run
    gt_obs = [[[-obs[4], obs[0], obs[2]],[-obs[1], obs[3], obs[5]]] for obs in task.piece_bounds_all]
    for step in range(num_steps):
        # Execute action
        task, bb, output, matched_gt = run_step(env, task, x, y, step, visualize)
        camera_pos.append(task.observation.camera_pos[step])
        cam_not_inside_obs.append(task.observation.cam_not_inside_obs[step])
        bbs.append(bb)
        outputs.append(output)
        matched_gts.append(matched_gt)

        # Convert from camera frame to world frame
        is_visible.append(task.observation.is_visible[step])

    return {"box_axis_aligned": np.array(bbs), 
            "bbox_labels": np.array(gt_obs), 
            "loss": np.array(is_visible),
            "box_finetune": np.array(outputs),
            "matched_gt": np.array(matched_gts)}
            
# change to TR3D
def run_step(env, task, x, y, step, visualize=False):
    task.init_state = [x[step],y[step],0,0]
    action = [0,0]
    env.reset(task)
    observation, _, _, _ = env.step(action)
    num_chairs = len(task.piece_bounds_all) # Number of chairs in the current environment
    conf_threshold = 0.0 # Confidence threshold for TR3D
    num_boxes = 15

    task.observation.camera_pos[step] = [float(env.lidar_pos[0]), float(env.lidar_pos[1]), float(env.lidar_pos[2])]
    pos = task.observation.camera_pos[step]
    not_inside_xyz = [[(0 if (pos[i]>obs[i]-0.1 and pos[i]<obs[3+i]+0.1) else 1) for i in range(3)] for obs in task.piece_bounds_all]
    gt_obs = [[[-obs[4], obs[0], obs[2]],[-obs[1], obs[3], obs[5]]] for obs in task.piece_bounds_all]
    task.observation.cam_not_inside_obs[step] = all([True if any(obs) == 1 else False for obs in not_inside_xyz])
    is_vis = [False]*num_chairs
    
    # Filter points with z < 0.1, z > 2.9 and abs(y) > 3.9 and x < 0.05, x > 7.95
    observation = observation[:, observation[2, :] < 2.9]

    X = observation[:, (observation[2, :] > 0.1)]
    X = X[:, np.abs(X[1,:]) < 3.9]
    X = X[:, X[0,:] > 0.05]
    X = X[:, X[0,:] < 7.95]
    
    X = np.transpose(np.array(X))
    if(len(X) > 0):
        is_vis = is_box_visible(X, task.piece_bounds_all, visualize)
        for obs_idx, obs in enumerate(task.piece_bounds_all):
            is_vis[obs_idx] = (is_vis[obs_idx] and task.observation.cam_not_inside_obs[step])
    task.observation.is_visible[step] = is_vis

    if (len(observation[0])>0):
        # Get bounding boxes
        output = get_box(observation, inferencer, conf_threshold=conf_threshold)
        bb = match_gt_output_boxes(output, np.array(gt_obs), is_vis)
        matched_gt = match_output_gt_boxes(output, np.array(gt_obs), is_vis)
    else:
        # There are no returns from the LIDAR, object is not visible
        task.observation.is_visible[step] = [False]*num_chairs
        bb = get_room_size_box(num_chairs)
        output = get_room_size_box(num_boxes)
        matched_gt = match_output_gt_boxes(output, np.array(gt_obs), is_vis) 
    return task, bb, output, matched_gt

# change to TR3D
def get_box(observation_, inferencer, show_viz=False, conf_threshold=0, num_boxes = 15, detect_objects = [2,3,10]):
    observation = observation_.copy()
    observation = observation.T

    inputs_all = {'inputs': {'points': observation}, 'pred_score_thr': conf_threshold, 'out_dir': '', 'show': show_viz, 'wait_time': -1, 'no_save_vis': True, 'no_save_pred': False, 'print_result': False}
    inferencer.show_progress = False
    results = inferencer(**inputs_all)
    
    # "cabinet": 0, "bed": 1, "chair": 2, "sofa": 3, "table": 4, "door": 5, "window": 6, "bookshelf": 7, "picture": 8, 
    # "counter": 9, "desk": 10, "curtain": 11, "refrigerator": 12, "showercurtrain": 13, "toilet": 14, "sink": 15, "bathtub": 16, "garbagebin": 17,

    pred_boxes = results["predictions"][0]["bboxes_3d"]

    pred_confidences = results["predictions"][0]["scores_3d"]
    pred_labels = results["predictions"][0]["labels_3d"]

    sort_inds = np.argsort(pred_confidences)[::-1]
    sorted_inds = [i for i in sort_inds[0:num_boxes] if (pred_labels[i] in detect_objects)]
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

# works for TR3D
def match_gt_output_boxes(output_boxes, ground_truth, is_visible):
    max_iou = torch.zeros(ground_truth.shape[0])
    center_diff = 100*torch.ones(ground_truth.shape[0])
    sorted_pred = np.copy(ground_truth)
    for j, val in enumerate(is_visible):
        if val:
            gt = (ground_truth[j,0,0], ground_truth[j,0,1], ground_truth[j,1,0], ground_truth[j,1,1])
            for kk in range(output_boxes.shape[0]):
                pred_ = output_boxes[kk,:,:]
                pred = (pred_[0,0], pred_[0,1], pred_[1,0], pred_[1,1])
                iou = box2d_iou(pred, gt)
                diff = ((((gt[2]+gt[0]-pred[2]-pred[0])**2) + (gt[3]+gt[1]-pred[3]-pred[1])**2)**0.5)/2
                if iou > max_iou[j]:
                    max_iou[j] = iou
                    sorted_pred[j,:,:] = pred_
                    center_diff[j] = diff
                elif iou == 0 and max_iou[j] == 0 and (center_diff[j] > diff):
                    # Centers of the predicted box are closer than before
                    center_diff[j] = diff
                    sorted_pred[j,:,:] = pred_
    return sorted_pred
# works for TR3D
def match_output_gt_boxes(output_boxes, ground_truth, is_visible):
    max_iou = torch.zeros(output_boxes.shape[0])
    center_diff = 100*torch.ones(output_boxes.shape[0])
    sorted_pred = np.zeros_like(output_boxes)
    none_vis = True
    for kk in range(output_boxes.shape[0]):
        pred_ = output_boxes[kk,:,:]
        pred = (pred_[0,0], pred_[0,1], pred_[1,0], pred_[1,1])
        for j, val in enumerate(is_visible):
            if val:
                none_vis = False
                gt = (ground_truth[j,0,0], ground_truth[j,0,1], ground_truth[j,1,0], ground_truth[j,1,1])
                iou = box2d_iou(pred, gt)
                diff = ((((gt[2]+gt[0]-pred[2]-pred[0])**2) + (gt[3]+gt[1]-pred[3]-pred[1])**2)**0.5)/2
                if iou > max_iou[kk]:
                    max_iou[kk] = iou
                    sorted_pred[kk,:,:] = ground_truth[j,:,:]
                    center_diff[kk] = diff
                elif iou == 0 and max_iou[kk] == 0 and (center_diff[kk] > diff):
                    # Centers of the predicted box are closer than before
                    center_diff[kk] = diff
                    sorted_pred[kk,:,:] = ground_truth[j,:,:]
    return sorted_pred

def plot_box_pc(pc, output_boxes, gt_boxes, is_vis):
    # Visualize
    pc_plot = pc[:, pc[0,:,2] > 0.0,:]
    pc_plot = pc_plot[:, pc_plot[0,:,1]<7.95,:]
    pc_plot = pc_plot[:, pc_plot[0,:,1]>0.05,:]
    pc_plot = pc_plot[:, abs(pc_plot[0,:,0]) < 3.9,:]
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(
        pc_plot[0,:,0], pc_plot[0,:,1],pc_plot[0,:,2]
    )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')

    for jj, cc in enumerate(output_boxes):
        r0 = [cc[0, 0], cc[1, 0]]
        r1 = [cc[0, 1], cc[1, 1]]
        r2 = [cc[0, 2], cc[1, 2]]

        for s, e in combinations(np.array(list(product(r0, r1, r2))), 2):
            if (np.sum(np.abs(s-e)) == r0[1]-r0[0] or 
                np.sum(np.abs(s-e)) == r1[1]-r1[0] or 
                np.sum(np.abs(s-e)) == r2[1]-r2[0]):
                ax.plot3D(*zip(s, e), color=(0.5, 0.1,0.1))

    for jj, cc in enumerate(gt_boxes):
        r0 = [cc[0, 0], cc[1, 0]]
        r1 = [cc[0, 1], cc[1, 1]]
        r2 = [cc[0, 2], cc[1, 2]]

        for s, e in combinations(np.array(list(product(r0, r1, r2))), 2):
            if (np.sum(np.abs(s-e)) == r0[1]-r0[0] or 
                np.sum(np.abs(s-e)) == r1[1]-r1[0] or 
                np.sum(np.abs(s-e)) == r2[1]-r2[0]):
                ax.plot3D(*zip(s, e), color=(0.1, 0.5,0.1))
    plt.show()

# change to TR3D
def format_results(results):
    num_envs = len(results)
    num_cam_positions = results[0]["loss"].shape[0]
    num_chairs = results[0]["loss"].shape[1]
    # nqueries = results[0]["box_features"].shape[1]
    # dec_dim = results[0]["box_features"].shape[2]
    num_pred = results[0]["box_finetune"].shape[1]
    print("Number of predicted boxes for finetuning: ", num_pred)
    loss_mask = torch.zeros(num_envs, num_cam_positions, num_chairs)
    model_outputs_all = {
        # "box_features": torch.zeros(num_envs, num_cam_positions, nqueries, dec_dim),
        "box_axis_aligned": torch.zeros(num_envs, num_cam_positions, num_chairs, 2, 3),
    }
    match_outputs_gt = {
        "output": torch.zeros(num_envs, num_cam_positions, num_pred, 2, 3),
        "gt": torch.zeros(num_envs, num_cam_positions, num_pred, 2, 3),
    }
    bboxes_ground_truth_aligned = torch.zeros(num_envs, num_chairs, 2,3)
    for env, result in enumerate(results):
        loss_mask[env,:,:] = torch.from_numpy(results[env]["loss"]).float()
        # model_outputs_all["box_features"][env,:,:,:] = results[env]["box_features"]
        model_outputs_all["box_axis_aligned"][env,:,:,:] = torch.from_numpy(results[env]["box_axis_aligned"])
        bboxes_ground_truth_aligned[env,:,:,:] = torch.from_numpy(results[env]["bbox_labels"])
        match_outputs_gt["output"][env,:,:,:] = torch.from_numpy(results[env]["box_finetune"])
        match_outputs_gt["gt"][env,:,:,:] = torch.from_numpy(results[env]["matched_gt"])
    return model_outputs_all, bboxes_ground_truth_aligned, loss_mask, match_outputs_gt

# change to TR3D

def combine_old_files(filenames, num_files):
    # model_outputs_all = {"box_features": torch.tensor([]), "box_axis_aligned": torch.tensor([])}
    model_outputs_all = {"box_axis_aligned": torch.tensor([])}
    match_outputs_gt = {"output": torch.tensor([]), "gt": torch.tensor([])}
    bboxes_ground_truth_aligned = torch.tensor([])
    loss_mask = torch.tensor([])
    for i in range(num_files):
        features = torch.load(filenames[0]+str(i+1) + ".pt")
        bboxes = torch.load(filenames[1]+str(i+1) + ".pt")
        loss = torch.load(filenames[2]+str(i+1) + ".pt")
        finetune = torch.load(filenames[3]+str(i+1) + ".pt")
        # model_outputs_all["box_features"] = torch.cat((model_outputs_all["box_features"], features["box_features"]))
        model_outputs_all["box_axis_aligned"] = torch.cat((model_outputs_all["box_axis_aligned"], features["box_axis_aligned"]))
        match_outputs_gt["output"] = torch.cat((match_outputs_gt["output"], finetune["output"]))
        match_outputs_gt["gt"] = torch.cat((match_outputs_gt["gt"], finetune["gt"]))
        bboxes_ground_truth_aligned= torch.cat((bboxes_ground_truth_aligned, bboxes))
        loss_mask = torch.cat((loss_mask, loss))
    return model_outputs_all, bboxes_ground_truth_aligned, loss_mask, match_outputs_gt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task_dataset', default='nav_sim/sim_data/task.pkl',
        nargs='?', help='path to task dataset'
    )
    parser.add_argument(
        '--save_dataset', default='nav_sim/sim_data/task.npz',
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
    for task in task_dataset:

        # Initialize task
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

    ##################################################################
    # Number of environments
    num_envs = 10

    # Number of parallel threads
    num_parallel = 2
    ##################################################################

    # _, _, _ = render_env(seed=0)

    ##################################################################
    env = 0
    batch_size = num_parallel
    save_file = args.save_dataset
    save_res = []
    ##################################################################

    for task in task_dataset:
        env += 1 
        if env%batch_size == 0:
            if env>0: # In case code stops running, change starting environment to last batch saved
                batch = math.floor(env/batch_size)
                print("Saving batch", str(batch))
                t_start = time.time()
                pool = Pool(num_parallel) # Number of parallel processes
                results = pool.map_async(run_env, task_dataset[env-batch_size:env]) # Compute results
                pool.close()
                pool.join()
                results = results.get()
                save_res = save_res + results
                # run_env(task)
                t_end = time.time()
                print("Time to generate results: ", t_end - t_start)
                ###########################################################################
                model_outputs_all, bboxes_ground_truth_aligned, loss_mask, match_outputs_gt = format_results(results)
                torch.save(model_outputs_all, args.save_dataset + "data/dataset_intermediate/features"+str(batch) + ".pt")
                torch.save(bboxes_ground_truth_aligned, args.save_dataset + "data/dataset_intermediate/bbox_labels"+str(batch) + ".pt")
                torch.save(loss_mask, args.save_dataset + "data/dataset_intermediate/loss_mask"+str(batch) + ".pt")
                torch.save(match_outputs_gt, args.save_dataset + "data/dataset_intermediate/finetune"+str(batch) + ".pt")
    #################################################################

    # model_outputs_all, bboxes_ground_truth_aligned, loss_mask, match_outputs_gt = format_results(save_res)
    filenames = [args.save_dataset + "data/dataset_intermediate/features", args.save_dataset + "data/dataset_intermediate/bbox_labels", args.save_dataset + "data/dataset_intermediate/loss_mask", args.save_dataset + "data/dataset_intermediate/finetune"]
    model_outputs_all, bboxes_ground_truth_aligned, loss_mask, match_outputs_gt = combine_old_files(filenames, int(len(task_dataset)/num_parallel))
    ###########################################################################
    # # Save processed feature data
    torch.save(model_outputs_all, args.save_dataset + "data/features.pt")
    # # Save ground truth bounding boxes
    torch.save(bboxes_ground_truth_aligned, args.save_dataset + "data/bbox_labels.pt")
    # # Save loss mask
    torch.save(loss_mask, args.save_dataset + "data/loss_mask.pt")
    # # Save all box outputs for finetuning
    torch.save(match_outputs_gt, args.save_dataset + "data/finetune.pt")
    ###########################################################################
