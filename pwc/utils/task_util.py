import pickle
import numpy as np

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
        task.goal_radius = 1
        task.observation = {}
        task.observation.type = task_config.observation.type  # 'rgb' or 'lidar' or 'rgbd'
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
        task.init_state = [0.2,-1,0,0]
        task.goal_loc = [7, -2]

        # grid_data = np.load((task_config.room_folder + str(task.env) + '/occupancy_grid.npz'), allow_pickle=True)
        # occupancy_grid = grid_data['arr_0']
        # task.occupancy_grid = occupancy_grid

    return task_dataset