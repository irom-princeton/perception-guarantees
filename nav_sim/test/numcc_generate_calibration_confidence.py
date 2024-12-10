# numcc imports
from numcc.src.engine.engine import prepare_data_udf
from numcc.src.engine.engine_viz import generate_html_udf

import numcc.main_numcc as main_numcc
import numcc.util.misc as misc
from numcc.util.hypersim_dataset import random_crop

from numcc.src.fns import *
from numcc.src.model.nu_mcc import NUMCC
import timm.optim.optim_factory as optim_factory
from numcc.util.misc import NativeScalerWithGradNormCount as NativeScaler

# Sim imports
from nav_sim.env.task_env_numcc import TaskEnv

import plotly.express as px
from plotly.subplots import make_subplots
from scipy.ndimage import median_filter, binary_closing
import torch
import numpy as np
import pickle
import os
from pathlib import Path
pg_path = Path(__file__).parent.parent.parent
data_path = pg_path.parent/'data/perception-guarantees'

# Add utils to path
import sys
sys.path.append('../utils')

# memory management
torch.cuda.empty_cache() 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def load_task(task_dataset, task_idx):
    with open(task_dataset, 'rb') as f:
        task_dataset = pickle.load(f)
    task = task_dataset[task_idx]
    task = initialize_task(task)

    # Load the states to calibrate
    with open(pg_path/'planning/pre_compute/Pset-2k.pkl', 'rb') as f:
        state_samples = pickle.load(f)
        # Remove goal
        state_samples = state_samples[:-1][:]
    # Remove duplicates
    sample_proj = [[sample[0], sample[1]] for sample in state_samples]
    s = []
    s = [x for x in sample_proj if x not in s and not s.append(x)]
    # Transform from planner frame
    x = [float(sample[1]) for sample in s]
    y = [float(sample[0]-4) for sample in s]
    task.x = x
    task.y = y
    return task

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
    task.observation.rgb.img_h = 662 #376
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

def run_env(task, model, numcc_args):
    env = TaskEnv(render=False)
    env.reset(task)

    # ground truth occupancy grid
    occupancy_grid_path = os.path.join(task.base_path, 'occupancy_grid.npz')
    with np.load(occupancy_grid_path) as occupancy_grid:
        gt = occupancy_grid['arr_0'] 
    # rotate gt by 180 degrees
    gt = np.rot90(gt, 2)

    num_steps = len(task.x)
    thresholds = np.zeros(num_steps)
    coverages = np.zeros((num_steps, gt.shape[0], gt.shape[1]))
    bad_results = {}

    for step in range(num_steps):
        if (step+1)%50 == 0:
            print(f'Step {step+1}/{num_steps}')
    # step = 13
        x = task.x[step]
        y = task.y[step]
        task, observation = run_step(env, task, x, y, step)

        samples = process_observation(task, observation, cam_position = task.observation.camera_pos[step])
        all_pred_udf, query_xyz, seen_xyz = run_viz_udf(model, samples, numcc_args)
        # fig = visualize(pred_points, seen_xyz, cam_position = task.observation.camera_pos[step])
        t, coverage = find_threshold(torch.cat(all_pred_udf, dim=0), query_xyz, seen_xyz, gt, task.observation.camera_pos[step], task.piece_bounds_all)
        thresholds[step] = t
        coverages[step] = coverage

        if t >= 0.5:
            # plot_coverage(torch.cat(all_pred_udf, dim=0), query_xyz, seen_xyz, gt, task.observation.camera_pos[step], task.piece_bounds_all, t, step)
            step_result = {'all_pred_udf': all_pred_udf, 'query_xyz': query_xyz, 'seen_xyz': seen_xyz, 'gt': gt, 'cam_position': task.observation.camera_pos[step]}
            bad_results[step] = step_result

    env_results = {'thresholds': thresholds, 'coverages': coverages, 'task': task, 'bad_results': bad_results}
    return env_results

def run_step(env, task, x, y, step):
    action  = [x,y]
    observation, _, _, _ = env.step(action) # observation = (pc, rgb)
    task.observation.camera_pos[step] = [float(env.cam_pos[0]), float(env.cam_pos[1]), float(env.cam_pos[2])]

    return task, observation

def process_observation(task, pc, cam_position):
    # shift to camera center
    cam_position = torch.tensor(cam_position)
    xyz = torch.tensor(pc[0])-cam_position

    # change coordinate system
    forward = xyz[:,:,0]
    left = xyz[:,:,1]
    up = xyz[:,:,2]

    observation = (torch.stack([-left, -up, forward], -1), pc[1], 
                   np.array([-cam_position[1], -cam_position[2], cam_position[0]]))

    xyz = torch.tensor(observation[0]).to(torch.float32)
    img = torch.tensor(observation[1]).permute(1,2,0).to(torch.float32) #w,h,3
    img = img / 255.0

    xyz, img = random_crop(xyz, img, is_train=False)


    ######## LOAD DATA ############
    seen_data = [xyz, img]

    gt_data = [torch.zeros(seen_data[0].shape), torch.zeros(seen_data[1].shape)]
    seen_data[1] = seen_data[1].permute(2, 0, 1)
    seen_data[0] = seen_data[0].unsqueeze(0)
    seen_data[1] = seen_data[1].unsqueeze(0)

    samples = [
        seen_data,
        gt_data,
    ]

    return samples

def run_viz_udf(model, samples, args):
    seen_xyz, valid_seen_xyz, query_xyz, unseen_rgb, labels, seen_images, gt_fps_xyz, seen_xyz_hr, valid_seen_xyz_hr = prepare_data_udf(samples, args.device, is_train=False, is_viz=True, args=args)
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
    
    # don't forward all at once to avoid oom
    max_n_queries_fwd = args.n_query_udf if not args.hr else int(args.n_query_udf * (args.xyz_size/args.xyz_size_hr)**2)

    # Filter query based on centers xyz # (1, 200, 3)
    offset = 0.3
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

    all_pred_udf = []

    for p_idx in range(total_n_passes):
        p_start = p_idx     * max_n_queries_fwd
        p_end = (p_idx + 1) * max_n_queries_fwd
        cur_query_xyz = query_xyz[:, p_start:p_end]

        with torch.no_grad():
            if args.hr != 1:
                seen_points = seen_xyz
                valid_seen = valid_seen_xyz
            else:
                seen_points = seen_xyz_hr
                valid_seen = valid_seen_xyz_hr

            if args.distributed:
                pred = model.module.decoderl2(cur_query_xyz, seen_points, valid_seen, fea, up_grid_fea, custom_centers = None)
                pred = model.module.fc_out(pred)
            else:
                pred = model.decoderl2(cur_query_xyz, seen_points, valid_seen, fea, up_grid_fea, custom_centers = None)
                pred = model.fc_out(pred)

        pred_udf = F.relu(pred[:,:,:1]).reshape((-1, 1)) # nQ, 1
        pred_udf = torch.clamp(pred_udf, max=args.max_dist) 
        all_pred_udf.append(pred_udf)

        debug = False
        if debug:
            # Candidate points
            t = 0.6
            pos = (pred_udf < t).squeeze(-1) # (nQ, )
            points = cur_query_xyz.squeeze(0) # (nQ, 3)
            points = points[pos].unsqueeze(0) # (1, n, 3)

            # print(pos)

            if torch.sum(pos) > 0:
                # points = move_points(model, points, seen_points, valid_seen, fea, up_grid_fea, args, n_iter=args.udf_n_iter)

                # predict final color
                with torch.no_grad():
                    if args.distributed:
                        pred = model.module.decoderl2(points, seen_points, valid_seen, fea, up_grid_fea)
                        pred = model.module.fc_out(pred)
                    else:
                        pred = model.decoderl2(points, seen_points, valid_seen, fea, up_grid_fea)
                        pred = model.fc_out(pred)

                cur_color_out = pred[:,:,1:].reshape((-1, 3, 256)).max(dim=2)[1] / 255.0
                cur_color_out = cur_color_out.detach().squeeze(0).cpu().numpy()
                if len(cur_color_out.shape) == 1:
                    cur_color_out = cur_color_out[None,...]
                pts = points.detach().squeeze(0).cpu().numpy()
                pred_points = np.append(pred_points, pts, axis = 0)
                pred_colors = np.append(pred_colors, cur_color_out, axis = 0)
        
    if debug:
        
        img = (seen_images_no_preprocess[0].permute(1, 2, 0) * 255).cpu().numpy().copy().astype(np.uint8)
        with open('351_284_viz.html', 'a') as f:
            generate_html_udf(
                img,
                seen_xyz, seen_images_no_preprocess,
                pred_points,
                pred_colors,
                query_xyz,
                f,
                gt_xyz=None,
                gt_rgb=None,
                mesh_xyz=None,
                centers = centers_xyz,
                fn_pc=None,
                fn_pc_seen = None,
                fn_pc_gt=None
            )
    return all_pred_udf, query_xyz, seen_xyz
def is_chair_visible(seen_xyz, obstacles, cam_position, visualize=False):
    is_vis = [False]*len(obstacles)
    seen_xyz = seen_xyz.squeeze().cpu().numpy()

    right = seen_xyz[:,:,0]
    down = seen_xyz[:,:,1]
    forward = seen_xyz[:,:,2]

    seen_xyz = np.stack([forward, -right, -down],axis=2) + np.array(cam_position)
    for obs_idx, obs in enumerate(obstacles):
        # Check if any visible points are in the ground truth boxes. If more than 100 points are inside the box, it is marked visible
        obs = np.array(obs)
        s=[(seen_xyz[:,:,i]>obs[i]+0.1) & (seen_xyz[:,:,i]<obs[3+i]-0.1) for i in range(3)]
        s=np.array(s)
        is_vis_noise=bool(sum(sum(s[0,:]&s[1,:]&s[2,:]))>100)
        is_vis[obs_idx]  = is_vis_noise
    
    if visualize:
        fig = px.scatter_3d(
            x=seen_xyz[:, :, 0].flatten(), 
            y=seen_xyz[:, :, 1].flatten(), 
            z=seen_xyz[:, :, 2].flatten(),
            opacity=0.8
        )
        fig.update_traces(marker=dict(size=1))

        # add obstacles
        for obs in obstacles:
            x = [obs[0], obs[3], obs[3], obs[0], obs[0]]
            y = [obs[1], obs[1], obs[4], obs[4], obs[1]]
            z = [obs[2], obs[2], obs[5], obs[5], obs[2]]
            fig.add_scatter3d(x=x, y=y, z=z, mode='lines')

        fig.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ))
        fig.show()
    return is_vis

def loss_mask(cam_position: np.ndarray, piece_bounds_all: list, is_viz: list, gt: np.ndarray=np.zeros((83,83)), fov: int=60, far: int=48, near: int=15) -> np.ndarray:
    mask_grid = np.zeros_like(gt).astype(float) # 1 means counted, 0 means not counted
    # camera pose in sim frame, convert to grid frame
    camera_pose = np.array([8-cam_position[0], 4-cam_position[1]])
    camera_pose = (camera_pose*83/8).astype(int)

    # first check if camera is inside obstacles
    if gt[camera_pose[0], camera_pose[1]] == 1:
        return mask_grid
    
    # field of view
    for i in range(mask_grid.shape[0]):
        for j in range(mask_grid.shape[1]):
            x = camera_pose[0] - i
            y = np.abs(camera_pose[1] - j)
            distance = np.sqrt(x**2 + y**2)
            angle = np.arctan2(y, x)
            angle = np.rad2deg(angle)
            angle = (angle + 360) % 360
            if (angle >  - fov/2) and (angle < + fov/2) and near<distance<far:
                mask_grid[i, j] = 1

    # count occlusion
    for piece_i, piece_bound in enumerate(piece_bounds_all):
        if is_viz[piece_i]==False: # mask out invisble pieces
            mins = (np.floor(np.array([8-piece_bound[3], 4-piece_bound[4]])*82/8)).astype(int) 
            maxs = (np.ceil(np.array([8-piece_bound[0], 4-piece_bound[1]])*83/8)).astype(int) 
            piece_mesh = np.meshgrid(np.arange(mins[0], min(82, maxs[0]+1)), np.arange(mins[1], min(82, maxs[1]+1)))
            piece_mesh = np.array(piece_mesh).reshape(2, -1).T
            piece_mesh = piece_mesh[gt[piece_mesh[:,0], piece_mesh[:,1]]==1]
            mask_grid[piece_mesh[:,0], piece_mesh[:,1]] = 0
            
    true_grid = np.logical_and(mask_grid, gt).astype(int)

    return true_grid

def plot_coverage(pred_udf, cur_query_xyz, seen_xyz, gt, cam_position, t, piece_bounds_all, step=0.0):
    pos = (pred_udf < t).squeeze(-1) # (nQ, )
    points = cur_query_xyz.squeeze(0) # (nQ, 3)
    points = points[pos].unsqueeze(0) # (1, n, 3)

    pred_points = np.empty((0,3))
    if torch.sum(pos) > 0:
        pts = points.detach().squeeze(0).cpu().numpy()
        pred_points = np.append(pred_points, pts, axis = 0)
    
    pred_grid = visualize(pred_points, seen_xyz, cam_position)
    is_viz = is_chair_visible(seen_xyz, piece_bounds_all, cam_position)
    true_grid = loss_mask(cam_position, piece_bounds_all, is_viz, fov=60, gt=gt)
    coverage = pred_grid - true_grid


    fig = make_subplots(rows=1, cols=3, subplot_titles=(f'Threshold {np.round(t,2)}', 'Predicted Grid', 'True Grid'))

    fig.add_trace(px.imshow(coverage).data[0], row=1, col=1)
    fig.add_trace(px.imshow(pred_grid).data[0], row=1, col=2)
    fig.add_trace(px.imshow(true_grid).data[0], row=1, col=3)

    # fig.show()
    fig.write_image(f'coverage_{np.round(t,2)}_step{step}.png')

    return 

def find_coverage(pred_udf, cur_query_xyz, seen_xyz, gt, cam_position, t, piece_bounds_all):
    pos = (pred_udf < t).squeeze(-1) # (nQ, )
    points = cur_query_xyz.squeeze(0) # (nQ, 3)
    points = points[pos].unsqueeze(0) # (1, n, 3)

    pred_points = np.empty((0,3))
    if torch.sum(pos) > 0:
        pts = points.detach().squeeze(0).cpu().numpy()
        pred_points = np.append(pred_points, pts, axis = 0)
    
    pred_grid = visualize(pred_points, seen_xyz, cam_position)
    is_viz = is_chair_visible(seen_xyz, piece_bounds_all, cam_position)
    true_grid = loss_mask(cam_position, piece_bounds_all=piece_bounds_all, is_viz=is_viz, fov=60, gt=gt)
    coverage = pred_grid - true_grid

    # fig = px.imshow(coverage, title=f'Threshold {np.round(t,2)}')
    # fig.show()

    # fig.write_image(f'coverage_{np.round(t,2)}.png')

    return coverage

def find_threshold(pred_udf, cur_query_xyz, seen_xyz, gt, cam_position, piece_bounds_all):
    def loss(coverage):
        coverage = find_coverage(pred_udf, cur_query_xyz, seen_xyz, gt, cam_position, t, piece_bounds_all)
        if np.all(coverage >= 0):
            return 0
        return 1
    # find minimum t such that loss(t) = 0
    t = 0.23 # initial guess
    coverage = find_coverage(pred_udf, cur_query_xyz, seen_xyz, gt, cam_position, t, piece_bounds_all)

    while loss(coverage) != 0 and t < 1:
        t += 0.002
        coverage = find_coverage(pred_udf, cur_query_xyz, seen_xyz, gt, cam_position, t, piece_bounds_all)

        # print(t)
    return t, coverage



def visualize(pred_points, seen_xyz, cam_position, ceiling = -2, floor = -0.5):
    cam_position_numcc = np.array([-cam_position[1], -cam_position[2], cam_position[0]])
    pred_points = pred_points + cam_position_numcc # back to sim frame
    seen_points = seen_xyz.squeeze(0).cpu().numpy().reshape(-1, 3) + cam_position_numcc
    all_points = np.concatenate([pred_points, seen_points], axis = 0)

    # pc_to_occupancy
    pc_for_occ = all_points
    good_points = pc_for_occ[:, 0] != -100

    if good_points.sum() != 0:
        # filter out ceiling and floor
        mask = (pc_for_occ[:, 1] > ceiling ) & (pc_for_occ[:, 1] < floor)
        pc_for_occ = pc_for_occ[mask]
    # get rid of the middle dimension
    points_2d = pc_for_occ[:, [0, 2]] # right, forward

    # grid
    grid = np.zeros((83,83))
    grid_pitch = 8/82 # from get_room_from_3dfront.py
    min_x = -4
    min_y = 0

    indices = ((points_2d - np.array([min_x, min_y])) / grid_pitch).astype(int)
    if len(indices) > 0:
        indices = indices[(indices[:, 0] >= 0) & (indices[:, 0] < 83) & (indices[:, 1] >= 0) & (indices[:, 1] < 83)]    
    grid[indices[:, 0], indices[:, 1]] = 1  # Mark as occupied
    grid = np.rot90(grid)
    
    bc1 = binary_closing(grid, np.ones((4,1))).astype(int)
    bc2 = binary_closing(grid, np.ones((1,4))).astype(int)
    bc = np.logical_or(bc1, bc2).astype(int)
    mf = median_filter(bc, size=2)
    grid = np.logical_or(grid, mf).astype(int)

    # fig = px.imshow(grid)
    # fig.show()

    return grid


def main():

    # numcc args
    numcc_args = main_numcc.get_args_parser().parse_args(args=[])
    numcc_args.udf_threshold = 0.56 #0.23 # calibrate?
    numcc_args.max_dist = 1 #0.5
    numcc_args.resume = str(pg_path/'numcc/pretrained/numcc_hypersim_550c.pth')
    numcc_args.use_hypersim = True
    numcc_args.run_vis = True
    numcc_args.n_groups = 550
    numcc_args.blr = 5e-5
    numcc_args.save_pc = False

    ######## LOAD MODEL ############
    misc.init_distributed_mode(numcc_args)

    model = NUMCC(args=numcc_args)
    model = model.to(numcc_args.device)
    model_without_ddp = model

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, numcc_args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=numcc_args.blr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(args=numcc_args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    model.eval()

    ######## RUN ENVIRONMENTS ############

    ## load task
    task_dataset = pg_path.parent/'data/perception-guarantees/task_1210_rot.pkl'

    # for task_idx in range(162,400):
    task_idx = 351
    print(f'Running task {task_idx}')
    task = load_task(task_dataset, task_idx)

    env_results = run_env(task, model, numcc_args)
    
        # with open(data_path /'task_numcc'/f'task_0803_{task_idx}.pkl', 'wb') as f:
        #     pickle.dump(env_results, f)
    

    return 


if __name__ == '__main__':
    main()

    