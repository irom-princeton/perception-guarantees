import pickle
import numpy as np
from scipy.ndimage import zoom

def initialize_task(task_config): #TODO: support overwriting defaults from config
    """
    Initialize a task by setting its status to 'initialized'.
    
    Args:
        task_dataset: The pickled task to be initialized.
    """
    task_dataset = pickle.load(open(task_config.path, "rb"))

    for task in task_dataset:

        # task = random.choice(task_dataset)

        # Initialize task
        task.goal_radius = task_config.goal_radius  # in meters
        task.observation = {}
        task.observation.type = task_config.observation.type  # 'rgb' or 'lidar' or 'rgbd'
        task.observation.rgb = {}
        task.observation.depth = {}
        task.observation.lidar = {}
        task.observation.camera_pos = {}
        task.observation.cam_not_inside_obs = {}
        task.observation.is_visible = {}
        task.observation.rgb.x_offset_from_robot_front = task_config.observation.rgb.x_offset_from_robot_front  # no y offset
        task.observation.rgb.z_offset_from_robot_top = task_config.observation.rgb.z_offset_from_robot_top
        task.observation.rgb.tilt = 0  # degrees of tilting down towards the floor
        task.observation.rgb.img_w = task_config.observation.rgb.img_w  # width of the image in pixels
        task.observation.rgb.img_h = task_config.observation.rgb.img_h  # height of the image in pixels
        task.observation.rgb.aspect = task_config.observation.rgb.aspect  # aspect ratio of the image
        task.observation.rgb.fov = task_config.observation.rgb.fov  # in PyBullet, this is vertical field of view in degrees
        task.observation.depth.img_w = task.observation.rgb.img_w  # needs to be the same now - assume coming from the same camera
        task.observation.depth.img_h = task.observation.rgb.img_h
        task.observation.lidar.z_offset_from_robot_top = 0.01  # no x/y offset
        task.observation.lidar.horizontal_res = 1  # resolution, in degree,1
        task.observation.lidar.vertical_res = 1  # resolution, in degree , 1
        task.observation.lidar.vertical_fov = 30  # half in one direction, in degree
        task.observation.lidar.max_range = 5 # in meter Anushri changed from 5 to 8
        task.env= task.base_path.split('/')[-1]
        task.init_state = task_config.init_state
        task.goal_loc = [7, -2]

        

    return task_dataset

def load_and_interpolate_gt(task, map_size):
    # Load the occupancy grid
    grid_data = np.load(task.base_path + '/occupancy_grid.npz', allow_pickle=True)
    gt_grid = grid_data['arr_0']

    # Resize gt_grid to match planner's map_size using bilinear interpolation
    if gt_grid.shape != map_size:
        zoom_factors = (
            map_size[0] / gt_grid.shape[0],
            map_size[1] / gt_grid.shape[1],
        )
        gt_grid = zoom(gt_grid, zoom=zoom_factors, order=1)  # order=1: bilinear interpolation

    gt_grid = np.rot90(gt_grid, 2)
    
    return gt_grid