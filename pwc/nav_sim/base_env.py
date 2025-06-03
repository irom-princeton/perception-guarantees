"""Navigation simulation in PyBullet

The robot action include forward velocity and z angular velocity.

Please contact the author(s) of this library if you have any questions.
Authors: Allen Z. Ren (allen.ren@princeton.edu)
"""

import numpy as np
import pickle

class BaseEnv():

    def __init__(
        self,
        render=False,
        config=None,
    ):
        """
        Args:
            render (bool): whether to render the environment with PyBullet for GUI visulization
        """
        super(BaseEnv, self).__init__()

        # Room layout
        self.room_dim = config.room.room_dim
        self.wall_thickness = config.room.wall_thickness
        self.wall_height = config.room.wall_height
        self.ground_rgba = config.room.ground_rgba
        self.back_wall_rgba = self.left_wall_rgba = self.right_wall_rgba = self.front_wall_rgba = config.room.wall_rgba

        # Robot dimensions TODO: get Go1 dimensions
        self.robot_half_dim = config.robot.half_dim  # (x, y, z) half dimensions
        self.robot_com_height = self.robot_half_dim[2]
        self.lidar_height = config.robot.lidar_height  # height of LiDAR above robot top
        self.camera_thickness = config.robot.camera_thickness  # thickness of camera box

        # Dynamics model
        self.load_dynamics_model(config.dynamics.params_path)
        self.dt = config.dynamics.dt  # 10 Hz for now #Anushri changed from 0.1 to 2

        # Mode
        self.mode = config.mode

    def load_dynamics_model(self, params_path):
        """
        Load dynamics model parameters from a file.
        
        Args:
            params_path (str): Path to the parameters file.
        """
        with open(params_path, 'rb') as f:
            [k1, k2, A, B, R, BRB] = pickle.load(f)
        self.k1 = k1
        self.k2 = k2
        self.k3 = A[2,3]
        self.k4 = A[3,2]
        self.k5 = B[3,1]
        self.k6 = B[2,0]
        return
    
    def reset(self, task=None):
        """
        Reset the environment - initialize PyBullet if first time, reset task, reset obstacles, reset robot
        
        Args:
            task (dict, optional): Task to be reset.
        """
        if task is not None:
            self.reset_task(task)
            
        self._state = task.init_state
        self.move_camera(self._state)

        return None

    def reset_task(self, task):
        """
        Reset task by loading some info into class variables.
        """
        self.task = task
        self._goal_loc = np.array(task.goal_loc)
        self._goal_radius = task.goal_radius
        self._init_dist_to_goal = np.linalg.norm(
            np.array(task.init_state[:2]) - self._goal_loc
        )
        self.observation_type = task.observation.type
        self.rgb_cfg = task.observation.rgb
        self.depth_cfg = task.observation.depth
        self.lidar_cfg = task.observation.lidar

    def step(self, action):
        """
        Step the environment. Terminate episode if robot at goal.
        
        Args:
            action (np.ndarray): Action to be applied.
        
        Returns:
            np.ndarray: Observation.
            float: Reward.
            bool: Whether the episode is done.
            dict: Additional information.
        """
        self._state, _ = self.move_robot(action, self._state)
        return None, None, None, None
    
    def reset_obstacles(self, task):
        """
        Load furniture meshes at specified poses.

        Args:
            task (dict): Task dict.
        """

        pass

    def reset_robot(self, state):
        """
        Reset the robot with (x, y, yaw) state input and fixed height at COM.
        
        Args:
            state (np.ndarray): State to be reset.
        """
        self._state = state

    def move_robot(self, action, state):
        """
        Move the robot with Go1 dynamics. Right-hand coordinates.

        Args:
            action (np.ndarray): to be applied.

        Returns:
            state: after action applied
        """
        x, y, vx, vy = state
        ux, uy = action
        x_new = x + vx* self.dt
        y_new = y + vy * self.dt
        vx_new = vx-self.k1*self.dt*vx+self.k5*ux*self.dt -self.k4*vy*self.dt
        vy_new = vy-self.k2*self.dt*vy+self.k6*uy*self.dt -self.k3*vx*self.dt
        if self.mode == 'experiment':
            state = np.array([x_new, y_new, vx_new, vy_new])
        elif self.mode == 'calibration':
            state = np.array([ux, uy, 0, 0]) # for calibration

        return state, action

    def move_camera(self, state):
        """
        Move camera and LiDAR to follow the robot. Update camera/LiDAR visualization if render.
                
        Args:
            state (np.ndarray): State of the robot.
        """
        x, y, vx, vy = state
        yaw = 0 # face +y direction
        robot_top_height = self.robot_half_dim[2] * 2

        # camera
        # these are from task
        rgb_height = robot_top_height + self.rgb_cfg.z_offset_from_robot_top

        self.cam_pos = np.array([x, y, rgb_height])


