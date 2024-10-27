# %%
import os
import torch
import argparse
import pickle
import json
import numpy as np
import torch
from torch.multiprocessing import Pool, Process, set_start_method
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import time
import gc

# Sim imports
from nav_sim.env.task_env_numcc import TaskEnv

# numcc imports
from numcc.src.engine.engine import prepare_data_udf

import numcc.main_numcc as main_numcc
import numcc.util.misc as misc
from numcc.util.hypersim_dataset import random_crop

from numcc.src.fns import *
from numcc.src.model.nu_mcc import NUMCC
import timm.optim.optim_factory as optim_factory
from numcc.util.misc import NativeScalerWithGradNormCount as NativeScaler

from pathlib import Path

from torch.profiler import profile, record_function, ProfilerActivity

# Add utils to path
import sys
sys.path.append('../utils')

try:
    set_start_method('spawn')
except RuntimeError:
    pass

# memory management
torch.cuda.empty_cache() 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# base path
base_path: Path = Path(__file__).parent.parent.parent

########## Load the x,y points to sample
with open(f'{base_path}/planning/pre_compute/Pset-1.5k.pkl', 'rb') as f:
    state_samples = pickle.load(f)
    # Remove goal
    state_samples = state_samples[:-1][:]
# Remove duplicates
sample_proj = [[sample[0], sample[1]] for sample in state_samples]
s = []
s = [x for x in sample_proj if x not in s and not s.append(x)]
# Transform from planner frame
x = [sample[1] for sample in s]
y = [sample[0]-4 for sample in s]

num_steps = len(x)


# numcc args
numcc_args = main_numcc.get_args_parser().parse_args(args=[])
numcc_args.udf_threshold = 0.23 # calibrate?
numcc_args.resume = base_path/'numcc/pretrained/numcc_hypersim_550c.pth'
numcc_args.use_hypersim = True
numcc_args.run_vis = True
numcc_args.n_groups = 550
numcc_args.blr = 5e-5
numcc_args.save_pc = True
numcc_args.device = torch.device('cuda')

######## LOAD NUMCC MODEL ############
misc.init_distributed_mode(numcc_args)

model = NUMCC(args=numcc_args)
model = model.to(torch.device('cuda'))

# set the model to eval mode
model.eval()

model_without_ddp = model
# %%
# following timm: set wd as 0 for bias and norm layers
param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, numcc_args.weight_decay)
optimizer = torch.optim.AdamW(param_groups, lr=numcc_args.blr, betas=(0.9, 0.95))
loss_scaler = NativeScaler()

misc.load_model(args=numcc_args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)




def dumb_func(task):
    # print("Dumb func")
    return {'delta': 0}

def run_env(args):
    # unpack the arguments
    # task = args['task']
    # visualize = args['visualize'] if 'visualize' in args else False,
    # device: torch.device = args['device'] if device in args else torch.device('cuda')
    task, visualize, device = args

    task = initialize_task(task)
    env = TaskEnv(render=False)
    env.reset(task)

    occupancy_grid_path = os.path.join(task.base_path, 'occupancy_grid.npz')
    with np.load(occupancy_grid_path) as occupancy_grid:
        gt = occupancy_grid['arr_0'] 
    # rotate gt by 180 degrees
    gt = np.rot90(gt, 2)

    pred_grid_env = []
    loss_mask_env = []
    loss_env = []

    for step in range(num_steps):
    # for step in range(3):
        # run step
        task, grid, loss_mask_ = run_step(env, task, x, y, step, device=device)

        if visualize:
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Predicted Grid", "Loss Masked Ground Truth"))
            fig.add_trace(go.Heatmap(z=grid), row=1, col=1)
            fig.add_trace(go.Heatmap(z=gt-loss_mask_), row=1, col=2)
            fig.show()
        
        pred_grid_env.append(grid)
        loss_mask_env.append(loss_mask_)
        loss = expand_pc(grid, gt, loss_mask_)
        loss_env.append(loss)
        

    env.close_pb()
    delta = np.max(np.array(loss_env))

    # clear memory
    del env, task
    torch.cuda.empty_cache()
    gc.collect()

    # return delta
    return {'pred_grid': pred_grid_env, 
            'delta': delta,
            'loss': loss_env}

def initialize_task(task):
    # Initialize task
    task.goal_radius = 0.5
    task.observation = {}
    task.observation.type = 'both'  # 'rgb' or 'lidar'
    task.observation.rgb = {}
    task.observation.depth = {}
    task.observation.lidar = {}
    task.observation.camera_pos = {}
    task.observation.cam_not_inside_obs = {}
    task.observation.is_visible = {}
    task.observation.rgb.x_offset_from_robot_front = 0.05  # no y offset
    task.observation.rgb.z_offset_from_robot_top = 0.8 # 0.05 # elevate
    task.observation.rgb.tilt = 0  # degrees of tilting down towards the floor
    task.observation.rgb.img_w = 662
    task.observation.rgb.img_h = 662 # May changed from 376 for numcc
    task.observation.rgb.aspect = 1.57
    task.observation.rgb.fov = 70  # in PyBullet, this is vertical field of view in degrees
    task.observation.depth.img_w = task.observation.rgb.img_w  # needs to be the same now - assume coming from the same camera
    task.observation.depth.img_h = task.observation.rgb.img_h
    task.observation.lidar.z_offset_from_robot_top = 0.2# 0.01  # no x/y offset
    task.observation.lidar.horizontal_res = 1  # resolution, in degree,1
    task.observation.lidar.vertical_res = 1  # resolution, in degree , 1
    task.observation.lidar.vertical_fov = 30  # half in one direction, in degree
    task.observation.lidar.max_range = 8 # in meter Anushri changed from 5 to 8


    init_state = [0,-3.5,0,0]
    goal_loc = [7.5,3.5]
    task.init_state = [float(v) for v in init_state]
    task.goal_loc = [float(v) for v in goal_loc]
    
    return task

def run_step(env, task, x, y, step,
             device: torch.device = torch.device('cuda')):
    if step%50 == 0:
        print("Step", step)

    action  = [x[step],y[step]]
    pc, _, _, _ = env.step(action) # observation = (pc, rgb)
    task.observation.camera_pos[step] = [float(env.cam_pos[0]), float(env.cam_pos[1]), float(env.cam_pos[2])]

    

    cam_position = torch.tensor(task.observation.camera_pos[step])
    # print("Cam position", cam_position)
    xyz = torch.tensor(pc[0])-cam_position

    # change coordinate system
    forward = xyz[:,:,0]
    left = xyz[:,:,1]
    up = xyz[:,:,2]

    # prep data for inference
    xyz = torch.tensor(torch.stack([-left, -up, forward], -1)).to(torch.float32)
    img = torch.tensor(pc[1]).permute(1,2,0).to(torch.float32) #w,h,3
    img = img / 255.0

    xyz_, img = random_crop(xyz, img, is_train=False)

    ######## LOAD DATA ############
    seen_data = [xyz_, img]

    gt_data = [torch.zeros(seen_data[0].shape), torch.zeros(seen_data[1].shape)]
    seen_data[1] = seen_data[1].permute(2, 0, 1)
    seen_data[0] = seen_data[0].unsqueeze(0)
    seen_data[1] = seen_data[1].unsqueeze(0)

    samples = [
        seen_data,
        gt_data,
    ]

    ######## RUN INFERENCE ############
    # pred_xyz, seen_xyz = run_viz_udf(model, samples, numcc_args.device, numcc_args)
    pred_xyz, seen_xyz = run_viz_udf(model, samples, device, numcc_args)
    cam_position_numcc = np.array([-cam_position[1], -cam_position[2], cam_position[0]])
    # print("Cam position numcc", cam_position_numcc)
    pred_points = pred_xyz + cam_position_numcc # back to sim frame
    seen_xyz = torch.nn.functional.interpolate(
        xyz[None].permute(0, 3, 1, 2), (112, 112),
        mode='bilinear',
    ).permute(0, 2, 3, 1)[0]
    seen_points = seen_xyz.squeeze(0).cpu().numpy().reshape(-1, 3) + cam_position_numcc

    all_points = np.concatenate([pred_points, seen_points], axis = 0)

    # pc_to_occupancy
    pc_for_occ = all_points
    good_points = pc_for_occ[:, 0] != -100

    if good_points.sum() != 0:
        # filter out ceiling and floor
        mask = (pc_for_occ[:, 1] > -3 ) & (pc_for_occ[:, 1] < -0.2)
        pc_for_occ = pc_for_occ[mask]
    # get rid of the middle dimension
    points_2d = pc_for_occ[:, [0, 2]] # right, forward

    # grid
    grid = np.zeros((83,83))
    grid_pitch = 0.10 # from get_room_from_3dfront.py
    min_x = -4
    min_y = 0

    indices = ((points_2d - np.array([min_x, min_y])) / grid_pitch).astype(int)
    if len(indices) > 0:
        indices = indices[(indices[:, 0] >= 0) & (indices[:, 0] < 83) & (indices[:, 1] >= 0) & (indices[:, 1] < 83)]
    grid[indices[:, 0], indices[:, 1]] = 1  # Mark as occupied
    grid = np.rot90(grid)
    loss_mask_ = loss_mask(cam_position)

    return task, grid, loss_mask_

def pc_to_occupancy(points, grid = np.zeros((83,83))):
    # shift to world frame
    good_points = points[:, 0] != -100

    if good_points.sum() != 0:
        # filter out ceiling and floor
        mask = (points[:, 1] > -3 ) & (points[:, 1] < -0.2)
        points = points[mask]
    # get rid of the middle dimension
    points_2d = points[:, [0, 2]] # right, forward

    # grid
    grid_pitch = 0.10 # from get_room_from_3dfront.py
    min_x = -4
    min_y = 0

    indices = ((points_2d - np.array([min_x, min_y])) / grid_pitch).astype(int)
    if len(indices) > 0:
        indices = indices[(indices[:, 0] >= 0) & (indices[:, 0] < 83) & (indices[:, 1] >= 0) & (indices[:, 1] < 83)]
    grid[indices[:, 0], indices[:, 1]] = 1  # Mark as occupied
    grid = np.rot90(grid)

    return grid

def loss_mask(camera_pose, fov=70, grid = np.zeros((83,83))):
    # camera pose in sim frame
    loss_mask = np.ones_like(grid)
    # draw fov from camera_pose
    # convert to grid frame
    camera_pose = np.array([8-camera_pose[0], 4-camera_pose[1]])
    # convert campera pose to index
    camera_pose = (camera_pose/0.1).astype(int)

    for i in range(loss_mask.shape[0]):
        for j in range(loss_mask.shape[1]):
            x = camera_pose[0] - i
            y = np.abs(camera_pose[1] - j)
            angle = np.arctan2(y, x)
            angle = np.rad2deg(angle)
            angle = (angle + 360) % 360
            if (angle >  - fov/2) and (angle < + fov/2):
                loss_mask[i, j] = 0

    # also mask out far enough away
    far = max(0,camera_pose[0] - 50)
    loss_mask[:far, :] = 1
    near = min(83, camera_pose[0] - 10) # 1m
    loss_mask[near:, :] = 1

    return loss_mask

def expand_pc(pred: np.ndarray, 
              gt: np.ndarray,
              loss_mask: np.ndarray):
    loss = 0
    while loss <= 41:
        loss += 1
        # add a pixel around every occupied cell in pred
        pred_pad = pred.copy()

        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                if pred[i, j] == 1:
                    left = max(0, i-loss)
                    right = min(pred.shape[0], i+loss+1)
                    top = max(0, j-loss)
                    bottom = min(pred.shape[1], j+loss+1)
                    pred_pad[left:right, top:bottom] = 1

        # check if pred_pad is a superset of gt
        coverage = pred_pad + loss_mask - gt
        if np.all(coverage >= 0):
            return loss

def run_viz_udf(model, samples, device, args):
    model.eval()
    model = model.to(device)

    t1 = time.time()
    seen_xyz, valid_seen_xyz, query_xyz, unseen_rgb, labels, seen_images, gt_fps_xyz, seen_xyz_hr, valid_seen_xyz_hr = prepare_data_udf(samples, device, is_train=False, is_viz=True, args=args)

    seen_images_no_preprocess = seen_images.clone()


    with torch.no_grad():
        seen_images_hr = None
        
        if args.hr == 1:
            seen_images_hr = preprocess_img(seen_images.clone(), res=args.xyz_size)
            seen_xyz_hr = shrink_points_beyond_threshold(seen_xyz_hr, args.shrink_threshold)

        seen_images = preprocess_img(seen_images)
        query_xyz = shrink_points_beyond_threshold(query_xyz, args.shrink_threshold)
        seen_xyz = shrink_points_beyond_threshold(seen_xyz, args.shrink_threshold)

        if args.distributed:
            latent, up_grid_fea = model.module.encoder(seen_images, seen_xyz, valid_seen_xyz, up_grid_bypass=seen_images_hr)
            fea = model.module.decoderl1(latent)
        else:
            latent, up_grid_fea = model.encoder(seen_images, seen_xyz, valid_seen_xyz, up_grid_bypass=seen_images_hr)
            fea = model.decoderl1(latent)
        centers_xyz = fea['anchors_xyz']
    
        # torch.cuda.empty_cache()
    t2 = time.time()
    # print("Time to encode data", t2-t1)
    # don't forward all at once to avoid oom
    max_n_queries_fwd = args.n_query_udf if not args.hr else int(args.n_query_udf * (args.xyz_size/args.xyz_size_hr)**2)

    # Filter query based on centers xyz # (1, 200, 3)
    offset = 1#0.3
    min_xyz = torch.min(centers_xyz, dim=1)[0][0] - offset
    max_xyz = torch.max(centers_xyz, dim=1)[0][0] + offset

    mask = (torch.rand(1, query_xyz.size()[1]) >= 0).to(args.device)
    mask = mask & (query_xyz[:,:,0] > min_xyz[0]) & (query_xyz[:,:,1] > min_xyz[1]) & (query_xyz[:,:,2] > min_xyz[2])
    mask = mask & (query_xyz[:,:,0] < max_xyz[0]) & (query_xyz[:,:,1] < max_xyz[1]) & (query_xyz[:,:,2] < max_xyz[2])
    query_xyz = query_xyz[mask].unsqueeze(0)

    total_n_passes = int(np.ceil(query_xyz.shape[1] / max_n_queries_fwd))

    pred_points = np.empty((0,3))
    pred_colors = np.empty((0,3))

    if args.distributed:
        for param in model.module.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = False
   
    for p_idx in range(total_n_passes):
        t0 = time.time()
        
        p_start = p_idx     * max_n_queries_fwd
        p_end = (p_idx + 1) * max_n_queries_fwd
        cur_query_xyz = query_xyz[:, p_start:p_end]
        # cur_query_xyz = cur_query_xyz.half()

        # model = model.half()

        with torch.no_grad():
            if args.hr != 1:
                seen_points = seen_xyz
                valid_seen = valid_seen_xyz
            else:
                seen_points = seen_xyz_hr
                valid_seen = valid_seen_xyz_hr

            # valid_seen = valid_seen.half()
            # seen_points = seen_points.half()

            if args.distributed:
                pred = model.module.decoderl2(cur_query_xyz, seen_points, valid_seen, fea, up_grid_fea, custom_centers = None)
                pred = model.module.fc_out(pred)
            else:
                pred = model.decoderl2(cur_query_xyz, seen_points, valid_seen, fea, up_grid_fea, custom_centers = None)
                pred = model.fc_out(pred)

        max_dist = 0.5
        pred_udf = F.relu(pred[:,:,:1]).reshape((-1, 1)) # nQ, 1
        pred_udf = torch.clamp(pred_udf, max=max_dist) 

        # Candidate points
        t = args.udf_threshold
        pos = (pred_udf < t).squeeze(-1) # (nQ, )
        points = cur_query_xyz.squeeze(0) # (nQ, 3)
        points = points[pos].unsqueeze(0) # (1, n, 3)
            
        del pred
        torch.cuda.empty_cache()
        gc.collect()

        # print(pos)

        if torch.sum(pos) > 0:
            points = move_points(model, points, seen_points, valid_seen, fea, up_grid_fea, args, n_iter=args.udf_n_iter)

            # predict final color
            with torch.no_grad():
                if args.distributed:
                    pred = model.module.decoderl2(points, seen_points, valid_seen, fea, up_grid_fea)
                    # pred = model.module.decoderl2(points, seen_points, valid_seen, fea, up_grid_fea).half()
                    pred = model.module.fc_out(pred)
                else:
                    pred = model.decoderl2(points, seen_points, valid_seen, fea, up_grid_fea)
                    # pred = model.decoderl2(points, seen_points, valid_seen, fea, up_grid_fea).half()
                    pred = model.fc_out(pred)

            cur_color_out = pred[:,:,1:].reshape((-1, 3, 256)).max(dim=2)[1] / 255.0
            cur_color_out = cur_color_out.detach().squeeze(0).cpu().numpy()
            if len(cur_color_out.shape) == 1:
                cur_color_out = cur_color_out[None,...]
            pts = points.detach().squeeze(0).cpu().numpy()
            pred_points = np.append(pred_points, pts, axis = 0)
            pred_colors = np.append(pred_colors, cur_color_out, axis = 0)
        
            del pred
            torch.cuda.empty_cache()
            gc.collect()
        
    return pred_points, seen_xyz


if __name__ == '__main__':
    # sim args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task_dataset', default='/home/zm2074/Projects/data/perception-guarantees/task_0803.pkl',
        nargs='?', help='path to save the task files'
    )
    parser.add_argument(
        '--save_dataset', default='/home/zm2074/Projects/data/perception-guarantees/task_numcc/',
        nargs='?', help='path to save the task files'
    )

    parser.add_argument(
        '--num_parallel', default=30,
        nargs='?', help='number of parallel processes'
    )

    parser.add_argument(
        '--visualize', default=False,
        nargs='?', help='visualize'
    )

    args = parser.parse_args()


    # Load sim params from json file
    with open("env_params.json", "r") as read_file:
        params = json.load(read_file)

    # Load task dataset
    with open(args.task_dataset, 'rb') as f:
        task_dataset = pickle.load(f)
   

    ##################################################################
    env = 308
    batch_size = int(args.num_parallel)
    ##################################################################

    for task in task_dataset:
        env += 1 
        # print("Environment", str(env))
        # results = run_env(task, visualize=False)
        # print("Results", results)

        # TODO:
        # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        #     with record_function("model_inference"):
                    ###

        # selected device
        sel_device = [torch.device('cuda')]* batch_size

        # set the device to the GPU for the first two processes
        # doesn't work if both GPU and CPU are used.
        # if batch_size >= 2:
        #     sel_device[:2] = [torch.device('cuda')] * 2
        # else:
        #     sel_device[:1] = [torch.device('cuda')]

        if env%batch_size == 0:
            if env>0: # In case code stops running, change starting environment to last batch saved
                batch = np.floor(env/batch_size)
                print("Saving batch", str(batch))
                t_start = time.time()
                pool = Pool(batch_size) # Number of parallel processes
                # renv_inputs = {
                #     'task': task_dataset[env-batch_size:env],
                #     'visualize':  False, 
                #     'device': sel_device
                # }
                renv_inputs = [
                    (task_dataset[env-batch_size + idx],
                    False, 
                    sel_device[idx]
                    )
                    for idx in range(batch_size)
                ]
                results = pool.map_async(run_env, renv_inputs) # Compute results
                # results = pool.map_async(dumb_func, task_dataset[env-batch_size:env]) # Compute results
                pool.close()
                pool.join()
                results = results.get()
                t_end = time.time()
                print("Time to generate results: ", t_end - t_start)
                for i, result in enumerate(results):
                    print("Environment", str(env), ", Delta", result['delta'])
                ##########################################################################
            pickle.dump(results, open(args.save_dataset + "data/dataset_intermediate/"+str(env)+".pkl", "wb"))
    # ################################################################
    # results = run_env(task_dataset[0], visualize=False)
    # pickle.dump(results, open(args.save_dataset + "data/dataset_intermediate/"+str(env)+".pkl", "wb"))
