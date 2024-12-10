from pathlib import Path
import pickle
import gc
import matplotlib.pyplot as plt
import time
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from scipy.ndimage import median_filter

from planning.Occ_Planner import Safe_Planner
from nav_sim.env.task_env_numcc_exp import TaskEnv
from nav_sim.test.numcc_generate_calibration_confidence import loss_mask, is_chair_visible

# numcc imports
from numcc.src.engine.engine import prepare_data_udf

import numcc.main_numcc as main_numcc
import numcc.util.misc as misc
from numcc.util.hypersim_dataset import random_crop

from numcc.src.fns import *
from numcc.src.model.nu_mcc import NUMCC
import timm.optim.optim_factory as optim_factory
from numcc.util.misc import NativeScalerWithGradNormCount as NativeScaler


# base path
base_path: Path = Path(__file__).parent
foldername = f'{base_path.parent}/data/perception-guarantees/room_1203_rot/'
taskpath = f'{base_path.parent}/data/perception-guarantees/task_1203_rot.pkl'

# load pre-sampled points
Pset = pickle.load(open(f'{base_path}/planning/pre_compute/Pset-2k.pkl', 'rb'))
reachable = pickle.load(open(f'{base_path}/planning/pre_compute/reachable-2k.pkl', 'rb'))

# numcc args
numcc_args = main_numcc.get_args_parser().parse_args(args=[])
numcc_args.udf_threshold = 0.23 # 0.49
numcc_args.resume = f'{base_path}/numcc/pretrained/numcc_hypersim_550c.pth'
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
model_without_ddp = model

# following timm: set wd as 0 for bias and norm layers
param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, numcc_args.weight_decay)
optimizer = torch.optim.AdamW(param_groups, lr=numcc_args.blr, betas=(0.9, 0.95))
loss_scaler = NativeScaler()

misc.load_model(args=numcc_args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

model.eval()

cp = 0

def state_to_planner(state, sp):
    # convert robot state to planner coordinates
    return (np.array([[[0,-1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,1,0]]])@np.array(state) + np.array([8/2,0,0,0])).squeeze()

def state_to_go1(state, sp):
    x, y, vx, vy = state[0]
    return np.array([y, -x+8/2, vy, -vx])

def plan_env(task):
    print("Env: ", str(task.env))
    visualize = False
    filename = foldername + str(task.env) + '/numcc_' + str(numcc_args.udf_threshold)
    gt_data = np.load((foldername + str(task.env) + '/occupancy_grid.npz'), allow_pickle=True)
    gt_grid = gt_data['arr_0']
    # make gt_grid 83*83 if it's not
    if gt_grid.shape != (83, 83):
        gt = np.zeros((83, 83))
        gt[:min(83,gt_grid.shape[0]), :min(83,gt_grid.shape[1])] = gt_grid[:min(83,gt_grid.shape[0]), :min(83,gt_grid.shape[1])]
        gt_grid = gt
    gt_grid = np.rot90(gt_grid, 2)

    env = TaskEnv(render=False)

    planner_init_state = [5,0.5,0,0]
    sp = Safe_Planner(init_state=planner_init_state, 
                      n_samples=len(Pset),
                      Pset=Pset,
                      reachable=reachable,
                      sensor_dt=0.2,)
    planner_goal = state_to_planner(np.array([task.goal_loc[0], task.goal_loc[1],0.5,0]), sp)
    
    env.dt = sp.dt
    env.reset(task)
    t = 0
    steps_taken = 0
    state_traj = []
    done = False
    collided = False
    misdetected = 0
    time_misdetected = 0
    prev_policy = []
    idx_prev = 0
    plan_fail = 0

    observation = env.step([0,0])[0]
    cam_position = torch.tensor([float(env.cam_pos[0]), float(env.cam_pos[1]), float(env.cam_pos[2])])

    while True and not done and not collided:
        state = state_to_planner(env._state, sp)
        # print('state: ', env._state)

        # DETECTION
        grid = get_map(observation, cam_position)

        # PLANNING
        res = sp.plan(state, planner_goal, grid)
        steps_taken+=1

        # breakpoint()
        misdetected += count_misdetected(gt_grid, sp.world.map_design)
        time_misdetected += 1

        if visualize and steps_taken % 10 == 0:
            # plot grid and state with plotly
            fig = go.Figure()
            fig.add_trace(go.Heatmap(z=sp.world.map_design-gt_grid))
            fig.add_trace(go.Scatter(x=[sp.world.state_to_pixel(state)[1]], y=[sp.world.state_to_pixel(state)[0]], mode='markers', marker=dict(size=10, color='red')))
            # plot plan in green
            if len(res['idx_solution']) > 1:
                x_waypoints = np.vstack(res['x_waypoints'])
                for i in range(len(x_waypoints)-1):
                    x1, y1 = sp.world.state_to_pixel(x_waypoints[i])
                    x2, y2 = sp.world.state_to_pixel(x_waypoints[i+1])
                    fig.add_trace(go.Scatter(x=[y1, y2], y=[x1, x2], mode='lines', line=dict(color='green', width=2)))
            fig.show()

        # plt.clf()
        # fig = plt.imshow(sp.world.map_design+gt_grid*5, cmap='coolwarm')
        # plt.scatter(sp.world.state_to_pixel(state)[1], sp.world.state_to_pixel(state)[0], color='red')
        # plt.savefig(f'{steps_taken}_map.png')

        if len(res['idx_solution']) > 1 and not done and not collided:
            policy_before_trans = np.vstack(res['u_waypoints'])
            policy = (np.array([[0,1],[-1,0]])@policy_before_trans.T).T
            prev_policy = np.copy(policy)
            for step in range(min(int(sp.sensor_dt/sp.dt), len(policy))):
                idx_prev = step
                state = env._state
                state_traj.append(state_to_planner(state, sp))
                # for obs in task.piece_bounds_all:
                #     if state[0] < obs[3] and state[0] > obs[0]:
                #        if state[1] < obs[4] and state[1] > obs[1]: 
                # og_loc = [round(state[0]/0.1)+1 , round((state[1]+4)/0.1)+1]
                og_loc = sp.world.state_to_pixel(state_traj[-1])
                if gt_grid[og_loc[0], og_loc[1]]:
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
            plan_fail += 1
            if (len(prev_policy) > idx_prev+1): #int(sp.sensor_dt/sp.dt):
                # for kk in range(int(sp.sensor_dt/sp.dt)):
                idx_prev += 1
                action = prev_policy[idx_prev]
                state = env._state
                state_traj.append(state_to_planner(state, sp))
                observation, reward, done, info = env.step(action)
                # time.sleep(sp.dt)
                t += sp.dt
            else:
                action = [0,0] # ICS was considered so shouldn't be a problem
                state = env._state
                state_traj.append(state_to_planner(state, sp))
                observation, reward, done, info = env.step(action)
                # time.sleep(sp.dt)
                t += sp.dt
        # print(f"Time: {t}")
        if t > 40: # or plan_fail > 10:
            print("Env: ", str(task.env), " Failed")
            break
    plot_results(filename, state_traj , gt_grid, sp)
    # create_gif([f'{step+1}_map.png' for step in range(steps_taken-1)], 'output.gif')
    print("misdetected: ", misdetected)
    return {"trajectory": np.array(state_traj), "done": done, "collision": collided, "misdetection": (misdetected/time_misdetected)}

def create_gif(image_paths, output_gif_path, duration=500):
    """Creates a GIF from a list of image paths."""

    images = [Image.open(image_path) for image_path in image_paths]
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=duration,
        loop=0  # 0 means infinite loop
    )


def plot_results(filename, state_traj , ground_truth, sp):
    plt.clf()
    plt.imshow(ground_truth*5 + sp.world.map_design, cmap='coolwarm')
    if len(state_traj) >0:
        for state in state_traj:
            x,y = sp.world.state_to_pixel(state)[1], sp.world.state_to_pixel(state)[0]
            plt.scatter(x, y, color='red', s=1)
    plt.savefig(filename + 'traj_plot_2k_ssdt0.2_bl.png')

def initialize_task(task):
    task.goal_radius = 1.5
    task.init_state = [0.5,-1,0,0]
    task.goal_loc = [7, -2]
    task.observation = {}
    task.observation.type = 'both'
    task.observation.rgb = {}
    task.observation.depth = {}
    task.observation.lidar = {}
    task.observation.camera_pos = {}
    task.observation.cam_not_inside_obs = {}
    task.observation.is_visible = {}
    task.observation.rgb.x_offset_from_robot_front = 0.05  # no y offset
    task.observation.rgb.z_offset_from_robot_top = 0.8
    task.observation.rgb.tilt = 0  # degrees of tilting down towards the floor
    task.observation.rgb.img_w = 662
    task.observation.rgb.img_h = 662
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
    return task


def get_map(pc, cam_position):
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
    pred_xyz, seen_xyz = run_viz_udf(model, samples, numcc_args.device, numcc_args)
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
        mask = (pc_for_occ[:, 1] > -2 ) & (pc_for_occ[:, 1] < -0.5)
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
    grid = median_filter(grid, size=2)

    return grid

def run_viz_udf(model, samples, device, args):
    model.eval()
    model = model.to(device)
    seen_xyz, valid_seen_xyz, query_xyz, unseen_rgb, labels, seen_images, gt_fps_xyz, seen_xyz_hr, valid_seen_xyz_hr = prepare_data_udf(samples, device, is_train=False, is_viz=True, args=args)

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

        max_dist = 1 # 0.5
        pred_udf = F.relu(pred[:,:,:1]).reshape((-1, 1)) # nQ, 1
        pred_udf = torch.clamp(pred_udf, max=max_dist) 

        # Candidate points
        t = args.udf_threshold
        pos = (pred_udf < t).squeeze(-1) # (nQ, )
        points = cur_query_xyz.squeeze(0) # (nQ, 3)
        points = points[pos].unsqueeze(0) # (1, n, 3)
            
        if torch.sum(pos) > 0:
            # points = move_points(model, points, seen_points, valid_seen, fea, up_grid_fea, args, n_iter=args.udf_n_iter)
            pts = points.detach().squeeze(0).cpu().numpy()
            pred_points = np.append(pred_points, pts, axis = 0)
        
        
    return pred_points, seen_xyz

def expand_pc(grid: np.ndarray, 
              cp: int) -> np.ndarray:
    """
    Expand the grid by cp pixels
    """
    grid_pad = grid.copy()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:
                left = max(0, i-cp)
                right = min(grid.shape[0], i+cp+1)
                top = max(0, j-cp)
                bottom = min(grid.shape[1], j+cp+1)
                grid_pad[left:right, top:bottom] = 1
    return grid_pad

def count_misdetected(gt, pred):
    x,y = np.where(pred == 0.5)
    predicted_free = (x[(x != 82) & (x != 81) & (y != 82) & (y != 81)], 
                      y[(x != 82) & (x != 81) & (y != 82) & (y != 81)])

    return np.sum(gt[predicted_free] == 1) > 0


if __name__ == "__main__":
    with open(taskpath, 'rb') as f:
        task_dataset = pickle.load(f)

    # get root repository path
    nav_sim_path = base_path/"nav_sim"

    num_tasks = len(task_dataset)
    num_envs = 100

    collisions = 0
    fails = 0
    for i in range(num_tasks-num_envs, num_tasks):
        task = task_dataset[i]
        task = initialize_task(task)
        result = plan_env(task)

        file_batch = foldername+ str(task.env) + "/cp_" + str(cp) + "_numcc_ssdt0.2_bl.npz"
        np.savez_compressed(file_batch, data=result)