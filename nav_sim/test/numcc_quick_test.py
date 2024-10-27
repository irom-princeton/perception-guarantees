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
import torch
import numpy as np
import pickle
import os
from pathlib import Path
pg_path = Path(__file__).parent.parent.parent

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

def run_env(task):
    env = TaskEnv(render=False)
    env.reset(task)

    # ground truth occupancy grid
    occupancy_grid_path = os.path.join(task.base_path, 'occupancy_grid.npz')
    with np.load(occupancy_grid_path) as occupancy_grid:
        gt = occupancy_grid['arr_0'] 
    # rotate gt by 180 degrees
    gt = np.rot90(gt, 2)

    return env, gt

def run_step(env, task, x, y, step):
    action  = [x[step],y[step]]
    observation, _, _, _ = env.step(action) # observation = (pc, rgb)
    task.observation.camera_pos[step] = [float(env.cam_pos[0]), float(env.cam_pos[1]), float(env.cam_pos[2])]

    return task, observation

def process_observation(task, pc):
    # shift to camera center
    cam_position = torch.tensor(task.observation.camera_pos[0])
    xyz = torch.tensor(pc[0])-cam_position

     # change coordinate system
    forward = xyz[:,:,0]
    left = xyz[:,:,1]
    up = xyz[:,:,2]

    observation = (torch.stack([-left, -up, forward], -1), pc[1], 
                   np.array([-cam_position[1], -cam_position[2], cam_position[0]]))

    xyz = torch.tensor(observation[0]).to(torch.float32)
    # img = torch.tensor(observation[1]).to(torch.float32) #3,w,h
    img = torch.tensor(observation[1]).permute(1,2,0).to(torch.float32) #w,h,3
    img = img / 255.0

    xyz, img = random_crop(xyz, img, is_train=False)
    # print(xyz.shape)
    # fig = px.scatter_3d(x=xyz[:,:,0].reshape(112*112), y=xyz[:,:,1].reshape(112*112), z=xyz[:,:,2].reshape(112*112))
    # fig.update_traces(marker_size=0.5)
    # fig.show()


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

def run_viz_udf(samples, args):

    ######## LOAD MODEL ############
    misc.init_distributed_mode(args)

    model = NUMCC(args=args)
    model = model.to(args.device)
    model_without_ddp = model

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.blr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)


    model.eval()

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

        # max_dist = 0.5
        max_dist = args.max_dist
        pred_udf = F.relu(pred[:,:,:1]).reshape((-1, 1)) # nQ, 1
        pred_udf = torch.clamp(pred_udf, max=max_dist) 

        # Candidate points
        t = args.udf_threshold
        pos = (pred_udf < t).squeeze(-1) # (nQ, )
        points = cur_query_xyz.squeeze(0) # (nQ, 3)
        points = points[pos].unsqueeze(0) # (1, n, 3)

        if torch.sum(pos) > 0:
            points = move_points(model, points, seen_points, valid_seen, fea, up_grid_fea, args, n_iter=args.udf_n_iter)

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
        
    rank = misc.get_rank()
    out_folder = os.path.join('experiments/', f'{args.exp_name}', 'viz')
    Path(out_folder).mkdir(parents= True, exist_ok=True)
    prefix = os.path.join(out_folder, n_exp)
    img = (seen_images_no_preprocess[0].permute(1, 2, 0) * 255).cpu().numpy().copy().astype(np.uint8)

    # gt_xyz = samples[1][0].to(device).reshape(-1, 3)
    # gt_rgb = samples[1][1].to(device).reshape(-1, 3)
    # mesh_xyz = samples[2].to(device).reshape(-1, 3) if args.use_hypersim else None
    gt_xyz = None
    gt_rgb = None
    mesh_xyz = None

    fn_pc = None
    fn_pc_seen = None
    fn_pc_gt = None
    if args.save_pc:
        out_folder_ply = os.path.join('experiments/', f'{args.exp_name}', 'ply')
        Path(out_folder_ply).mkdir(parents= True, exist_ok=True)
        prefix_pc = os.path.join(out_folder_ply, n_exp)
        fn_pc = prefix_pc + '.ply'

        # seen
        out_folder_ply = os.path.join('experiments/', f'{args.exp_name}', 'ply_seen')
        Path(out_folder_ply).mkdir(parents= True, exist_ok=True)
        prefix_pc = os.path.join(out_folder_ply, n_exp)
        fn_pc_seen = prefix_pc +'_seen' +'.ply'

        # gt
        out_folder_ply = os.path.join('experiments/', f'{args.exp_name}', 'ply_gt')
        Path(out_folder_ply).mkdir(parents= True, exist_ok=True)
        prefix_pc = os.path.join(out_folder_ply, n_exp)
        fn_pc_gt = prefix_pc +'_gt' +'.ply'

    with open(prefix + '.html', 'a') as f:
        generate_html_udf(
            img,
            seen_xyz, seen_images_no_preprocess,
            pred_points,
            pred_colors,
            query_xyz,
            f,
            gt_xyz=gt_xyz,
            gt_rgb=gt_rgb,
            mesh_xyz=mesh_xyz,
            centers = centers_xyz,
            fn_pc=fn_pc,
            fn_pc_seen = fn_pc_seen,
            fn_pc_gt=fn_pc_gt
        )

    return pred_points, pred_colors, seen_xyz, prefix


def visualize(pred_points, pred_colors, seen_xyz, prefix, cam_position, ceiling = 0.3, floor = 0.5):
    pred_points = pred_points + cam_position
    seen_points = seen_xyz.squeeze(0).cpu().numpy().reshape(-1, 3) + cam_position
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
    grid_pitch = 0.10 # from get_room_from_3dfront.py
    min_x = -4
    min_y = 0

    indices = ((points_2d - np.array([min_x, min_y])) / grid_pitch).astype(int)
    if len(indices) > 0:
        indices = indices[(indices[:, 0] >= 0) & (indices[:, 0] < 83) & (indices[:, 1] >= 0) & (indices[:, 1] < 83)]
    grid[indices[:, 0], indices[:, 1]] = 1  # Mark as occupied
    grid = np.rot90(grid)

    fig = px.imshow(grid)
    fig.show()

    return fig

if __name__ == '__main__':

    ## name experiment
    n_exp = 'test_clean_code'

    ## load task
    task_dataset = pg_path.parent/'data/perception-guarantees/task_0803.pkl'
    task_idx = 100

    task = load_task(task_dataset, task_idx)

    ## load env
    env, gt = run_env(task)

    ## run step
    x = [0.1]
    y = [0.1]

    task, pc = run_step(env, task, x, y, 0)
    cam_position = np.array(task.observation.camera_pos[0])

    ## process observation
    samples = process_observation(task, pc)

    ## run numcc inference
    # numcc args
    use_hypersim = True
    run_vis = True
    weights = '/home/zm2074/Projects/perception-guarantees/numcc/pretrained/numcc_hypersim_550c.pth'
    n_groups = 550
    blr = 5e-5
    thres = 0.23

    numcc_args = main_numcc.get_args_parser().parse_args(args=[])
    numcc_args.udf_threshold = thres #0.23 # calibrate?
    numcc_args.max_dist = 1 #0.5
    numcc_args.resume = weights
    numcc_args.use_hypersim = use_hypersim
    numcc_args.run_vis = run_vis
    numcc_args.n_groups = n_groups
    numcc_args.blr = blr
    numcc_args.save_pc = False

    pred_points, pred_colors, seen_xyz, prefix = run_viz_udf(samples, numcc_args)


    ## visualization