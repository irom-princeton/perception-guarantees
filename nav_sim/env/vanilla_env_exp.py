"""Navigation simulation in PyBullet

The robot action include forward velocity and z angular velocity.

Please contact the author(s) of this library if you have any questions.
Authors: Allen Z. Ren (allen.ren@princeton.edu)
"""

import numpy as np
import torch
import pybullet as pb
from pybullet_utils import bullet_client as bc
import pybullet_data

from nav_sim.util.misc import rgba2rgb
# from robot_descriptions.loaders.pybullet import load_robot_description

k1=3.968; k2=2.517; k3=0.1353; k4=-0.5197; k5 = 4.651; k6 = 2.335
# print("Controller gains, k1", k1, " k2 ", k2, " k3 ", k3, " k4 ", k4, " k5 ", k5, " k6 ", k6)

class VanillaEnv():

    def __init__(
        self,
        render=False,
    ):
        """
        Args:
            render (bool): whether to render the environment with PyBullet for GUI visulization
        """
        super(VanillaEnv, self).__init__()

        # PyBullet ID
        self._physics_client_id = -1

        # GUI, observation model
        self.render = render

        # Room layout
        self.room_dim = 8
        self.wall_thickness = 0.1
        self.wall_height = 4
        self.ground_rgba = [175/255, 175/255, 175/255, 1.0]
        self.back_wall_rgba = [199 / 255, 182 / 255, 191 / 255, 1.0]
        self.left_wall_rgba = [199 / 255, 182 / 255, 191 / 255, 1.0]
        self.right_wall_rgba = [199 / 255, 182 / 255, 191 / 255, 1.0]
        self.front_wall_rgba = [199 / 255, 182 / 255, 191 / 255, 1.0]

        # Robot dimensions TODO: get Go1 dimensions
        self.robot_half_dim = [0.323, 0.14, 0.2]
        self.robot_com_height = self.robot_half_dim[2]+0.3
        self.lidar_height = 0.15
        self.camera_thickness = 0.04

        # Dynamics model
        self.xdot_range = [0.5, 1.0] 
        self.dt = 2  # 10 Hz for now #Anushri changed from 0.1 to 2

    def reset(self, task=None):
        """
        Reset the environment - initialize PyBullet if first time, reset task, reset obstacles, reset robot
        
        Args:
            task (dict, optional): Task to be reset.
        """
        # Start PyBullet session if first time - and set obstacle
        if self._physics_client_id < 0:
            self.init_pb()

        # Reset task
        if task is not None:
            self.reset_task(task)
            self.reset_obstacles(task)

        # Reset robot state
        self._state = task.init_state

        # Reset robot
        self.reset_robot(self._state)

        # Reset camera
        self.move_camera(self._state)

        # Reset timer
        self.step_elapsed = 0

        # Return observation at current state
        return self._get_obs(self._state)

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
        # TODO: Set up LiDAR noise model

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
        # Move car in simulation
        self._state, _ = self.move_robot(action, self._state)

        # Move camera and LiDAR
        self.move_camera(self._state)

        # Setp sim to update
        self._p.stepSimulation()

        # Reward and done signal- small penalty fors wandering around
        dist_to_goal_center = np.linalg.norm(self._state[:2] - self._goal_loc)
        ratio_dist_to_goal_center = dist_to_goal_center / self._init_dist_to_goal
        reward = 1 - ratio_dist_to_goal_center

        # Done signal
        if dist_to_goal_center < self._goal_radius:
            done = True
        else:
            done = False

        # Return
        info = {
            'state': self._state,
        }
        return self._get_obs(self._state), reward, done, info

    def _get_obs(self, state=None):
        """
        Get RGB, depth, or LiDAR observation at robot's current state or given a state. State argument could be helpful when we collect perception data without the robot.

        Args:
            state (np.ndarray): (x, y, yaw)

        Returns:
            np.ndarray: RGB or depth image, of the shape (C, H, W)
        """
        if state is not None:
            self.move_camera(state)
        if self.observation_type == 'rgb':
            rgb_cfg = self.rgb_cfg
            depth_cfg = self.depth_cfg

            # Get view matrix
            init_camera_vector = (1, 0, 0)  # x-axis
            init_up_vector = (0, 0, 1)  # z-axis
            camera_vector = self.cam_rot_matrix.dot(init_camera_vector)
            up_vector = self.cam_rot_matrix.dot(init_up_vector)
            view_matrix = self._p.computeViewMatrix(
                self.cam_pos, self.cam_pos + 0.1*camera_vector, up_vector
            )

            # Get Image
            far = 7.5 # 1000.0
            near = 1 #0.01
            projection_matrix = self._p.computeProjectionMatrixFOV(
                fov=rgb_cfg.fov, aspect=rgb_cfg.aspect, nearVal=near,
                farVal=far
            )
            _, _, rgb_img, depth, _ = self._p.getCameraImage(
                rgb_cfg.img_w, rgb_cfg.img_h, view_matrix, projection_matrix,
                flags=self._p.ER_NO_SEGMENTATION_MASK, shadow=1,
                lightDirection=[1, 1, 1]
            )
            depth = np.reshape(depth, (1, depth_cfg.img_h, depth_cfg.img_w))
            depth = far * near / (far - (far-near) * depth)

            # # Convert RGB to CHW and uint8
            # # rgb = rgba2rgb(rgb_img).transpose(2, 0, 1)
            # pc = self._get_point_cloud(depth, rgb_cfg.img_w, rgb_cfg.img_h, view_matrix, projection_matrix)
            # # print("Depth map, ", pc.shape, pc)
            # # print("Height: ", rgb_cfg.img_h, " Width: ", rgb_cfg.img_w)
            # return pc.T #rgb
            return self._get_point_cloud(0, self.rgb_cfg.img_w, self.rgb_cfg.img_h, view_matrix, projection_matrix)

        elif self.observation_type == 'lidar':
            return self._get_lidar()

    def _get_point_cloud(self, depth, width, height, view_matrix, proj_matrix):
        # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer

        # get a depth image
        # "infinite" depths will have a value close to 1
        image_arr = pb.getCameraImage(width=width, height=height, viewMatrix=view_matrix, projectionMatrix=proj_matrix, 
                                      flags=self._p.ER_NO_SEGMENTATION_MASK, shadow=1,
                                      lightDirection=[1, 1, 1])
        depth = np.array(image_arr[3])
        rbg_img = np.array(image_arr[2])
        rgb = rgba2rgb(rbg_img).transpose(2, 0, 1)

        # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
        proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        # create a grid with pixel coordinates and depth values
        # y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
        y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
        r, g, b = rgb[0].reshape(-1), rgb[1].reshape(-1), rgb[2].reshape(-1)
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h, r, g, b], axis=1)
        # filter out "infinite" depths
        # pixels = pixels[z < 0.99999]
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels[:,0:4].T).T
        points /= points[:, 3: 4]
        points = points[:, :3]
        # print("Points shape: ", points.shape)
        points = np.hstack((points, pixels[:, 4:]))
        return points

    def _get_lidar(self):
        """
        Simulate LiDAR measurement at robot's current state.

        Returns:
            np.ndarray: LiDAR measurement, of the shape (3, N) 
            Comment from Anushri: Changed this measurement from np to float so that we can easily use it in omegaConf
        """
        lidar_cfg = self.lidar_cfg

        # Calculate ray direction
        yaw_all = np.arange(
            -np.pi,
            np.pi + 1e-5,
            lidar_cfg.horizontal_res * np.pi / 180,
        )
        tilt_all = np.arange(
            -lidar_cfg.vertical_fov * np.pi / 180 / 2,
            lidar_cfg.vertical_fov * np.pi / 180 / 2,
            lidar_cfg.vertical_res * np.pi / 180,
        )
        ray_to = []
        for yaw in yaw_all:
            for tilt in tilt_all:
                rot_matrix = self._p.getMatrixFromQuaternion(
                    self._p.getQuaternionFromEuler([0, -tilt, yaw])
                )
                vector = np.array(rot_matrix).reshape(3, 3).dot([1, 0, 0])
                ray_to += [
                    self.lidar_pos
                    + vector / np.linalg.norm(vector) * lidar_cfg.max_range
                ]
        num_rays = len(ray_to)

        # Get rays
        ray_from = [self.lidar_pos] * num_rays
        ray_output = self._p.rayTestBatch(
            ray_from,
            ray_to,
        )  # 1st hit for each ray

        # # Visualize rays
        # out_num = 0
        # for ind in range(num_rays):
        # self._p.addUserDebugLine(ray_from[ind], ray_to[ind], lineWidth=1)

        # Get point cloud
        point_cloud = []
        for ind, out in enumerate(ray_output):
            if out[0] != -1:
                # print(out[2] * lidar_cfg.max_range)
                point = (
                    self.lidar_pos + lidar_cfg.max_range * out[2] *
                    (np.array(ray_to[ind]) - self.lidar_pos)
                    / np.linalg.norm(np.array(ray_to[ind]) - self.lidar_pos)
                )
                point_cloud += [point]
                # self._p.addUserDebugLine(
                #     self.lidar_pos, point, lineColorRGB=[0, 0, 0.5], lineWidth=2
                # )
        point_cloud = np.array(point_cloud).T
        return point_cloud

    def init_pb(self):
        """
        Initialize the PyBullet client and build wall and floors.
        
        The room has back wall at [0-wall_thickness/2, 0], with left and right walls at two sides, and front wall at [room_dim+wall_thickness/2, 0].
        """
        if self.render:
            self._p = bc.BulletClient(
                connection_mode=pb.GUI, options='--width=2400 --height=1600'
            )
        else:
            self._p = bc.BulletClient()
        self._physics_client_id = self._p._client
        p = self._p
        p.resetSimulation()
        p.setTimeStep(self.dt)
        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, -9.8)
        if self.render:
            p.resetDebugVisualizerCamera(3.80, 225, -60, [3.5, 0.5, 0])
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        

        # Build ground and walls
        ground_collision_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[
                self.room_dim / 2 + self.wall_thickness / 2,
                self.room_dim / 2 + self.wall_thickness / 2,
                self.wall_thickness / 2
            ]
        )
        ground_visual_id = p.createVisualShape(
            p.GEOM_BOX, rgbaColor=self.ground_rgba, halfExtents=[
                self.room_dim / 2 + self.wall_thickness / 2,
                self.room_dim / 2 + self.wall_thickness / 2,
                self.wall_thickness / 2
            ]
        )
        self.ground_id = p.createMultiBody(
            baseMass=0,  # FIXED
            baseCollisionShapeIndex=ground_collision_id,
            baseVisualShapeIndex=ground_visual_id,
            basePosition=[self.room_dim / 2, 0, -self.wall_thickness / 2]
        )
        wall_collision_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[
                self.wall_thickness / 2, self.room_dim / 2,
                self.wall_height / 2
            ]
        )
        # wall_back_visual_id = p.createVisualShape(
        #     p.GEOM_BOX, rgbaColor=self.back_wall_rgba, halfExtents=[
        #         self.wall_thickness / 2, self.room_dim / 2,
        #         self.wall_height / 2
        #     ]
        # )
        # self.wall_back_id = p.createMultiBody(
        #     baseMass=0, baseCollisionShapeIndex=wall_collision_id,
        #     baseVisualShapeIndex=wall_back_visual_id,
        #     basePosition=[-self.wall_thickness / 2, 0, self.wall_height / 2]
        # )
        wall_left_visual_id = p.createVisualShape(
            p.GEOM_BOX, rgbaColor=self.left_wall_rgba, halfExtents=[
                self.wall_thickness / 2, self.room_dim / 2,
                self.wall_height / 2
            ]
        )
        self.wall_left_id = p.createMultiBody(  # positive in y
            baseMass=0,
            baseCollisionShapeIndex=wall_collision_id,
            baseVisualShapeIndex=wall_left_visual_id,
            basePosition=[
                self.room_dim / 2,
                self.room_dim/2 + self.wall_thickness/2,
                self.wall_height / 2
            ],
            baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]))
        wall_right_visual_id = p.createVisualShape(
            p.GEOM_BOX, rgbaColor=self.right_wall_rgba, halfExtents=[
                self.wall_thickness / 2, self.room_dim / 2,
                self.wall_height / 2
            ]
        )
        self.wall_right_id = p.createMultiBody(  # negative in y
            baseMass=0,
            baseCollisionShapeIndex=wall_collision_id,
            baseVisualShapeIndex=wall_right_visual_id,
            basePosition=[
                self.room_dim / 2,
                -self.room_dim/2 - self.wall_thickness/2,
                self.wall_height / 2
            ],
            baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]))
        wall_front_visual_id = p.createVisualShape(
            p.GEOM_BOX, rgbaColor=self.front_wall_rgba, halfExtents=[
                self.wall_thickness / 2, self.room_dim / 2,
                self.wall_height / 2
            ]
        )
        self.wall_front_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=wall_collision_id,
            baseVisualShapeIndex=wall_front_visual_id, basePosition=[
                self.room_dim + self.wall_thickness / 2, 0,
                self.wall_height / 2
            ]
        )
        # self.wall_top_id = p.createMultiBody(
        #     # for blocking view - same as ground
        #     baseMass=0,
        #     baseCollisionShapeIndex=ground_collision_id,
        #     baseVisualShapeIndex=ground_visual_id,
        #     basePosition=[
        #         self.room_dim / 2, 0,
        #         self.wall_height + self.wall_thickness / 2
        #     ]
        # )

        # Build camera and LiDAR visualization if render
        # if self.render:
        #     camera_visual_id = p.createVisualShape(
        #         p.GEOM_BOX, rgbaColor=[0, 0.8, 0, 1.0],
        #         halfExtents=[0.03, 0.20, 0.05]
        #     )
        #     self.camera_id = p.createMultiBody(
        #         baseMass=0, baseCollisionShapeIndex=-1,
        #         baseVisualShapeIndex=camera_visual_id, basePosition=[0, 0, 0]
        #     )
        #     lidar_visual_id = p.createVisualShape(
        #         p.GEOM_CYLINDER, rgbaColor=[0, 0.8, 0, 1.0], radius=0.08,
        #         length=self.lidar_height
        #     )
        #     self.lidar_id = p.createMultiBody(
        #         baseMass=0, baseCollisionShapeIndex=-1,
        #         baseVisualShapeIndex=lidar_visual_id, basePosition=[0, 0, 0]
        #     )

        # Initialize obstacle id list - excluding walls and floors
        self._obs_id_all = []

    def close_pb(self):
        """
        Close the PyBullet client.
        """
        if "_goal_id" in vars(self).keys():
            self._p.removeBody(self._goal_id)
            del self._goal_id
        if "robot_id" in vars(self).keys():
            self._p.removeBody(self.robot_id)
            del self.robot_id
        for obs_id in self._obs_id_all:
            self._p.removeBody(obs_id)
        self._obs_id_all = []
        self._p.disconnect()
        self._physics_client_id = -1

    def reset_obstacles(self, task):
        """
        Load furniture meshes at specified poses.

        # TODO: skip if no need to switch furniture

        Args:
            task (dict): Task dict.
        """

        # Remove existing ones - excluding walls
        for obs_id in self._obs_id_all:
            self._p.removeBody(obs_id)
        self._obs_id_all = []

        # Load furniture
        furniture_dict = task.furniture
        for key, value in furniture_dict.items():

            print(value.path)
            obj_collision_id = self._p.createCollisionShape(
                self._p.GEOM_MESH,
                fileName=value.path,
                flags=self._p.GEOM_FORCE_CONCAVE_TRIMESH  # concave
            )
            obj_visual_id = self._p.createVisualShape(
                self._p.GEOM_MESH, fileName=value.path
            )
            obj_id = self._p.createMultiBody(
                baseMass=0,  # static, allow concave
                baseCollisionShapeIndex=obj_collision_id,
                baseVisualShapeIndex=obj_visual_id,
                basePosition=value.position,
                baseOrientation=self._p.getQuaternionFromEuler([
                    np.pi / 2, 0, value.yaw
                ])
            )
            self._obs_id_all += [obj_id]

        # Render goal location if render
        if self.render:
            if "_goal_id" in vars(self).keys():
                self._p.removeBody(self._goal_id)
            goal_visual_id = self._p.createVisualShape(
                self._p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05],
                rgbaColor=[0.7, 0, 0, 1]
            )
            self._goal_id = self._p.createMultiBody(
                baseMass=0, baseVisualShapeIndex=goal_visual_id,
                basePosition=[self._goal_loc[0], self._goal_loc[1], 0.05],
                baseOrientation=self._p.getQuaternionFromEuler([0, 0, 0])
            )

    def reset_robot(self, state):
        """
        Reset the robot with (x, y, yaw) state input and fixed height at COM.
        
        Args:
            state (np.ndarray): State to be reset.
        """
        yaw = 0 #np.pi/2
        if "robot_id" in vars(self).keys():
            self._p.resetBasePositionAndOrientation(
                self.robot_id,
                posObj=np.append(state[:2], self.robot_com_height),
                ornObj=self._p.getQuaternionFromEuler([0, 0, state[2]])
            )
        else:
            # self.robot_id = load_robot_description('go1_description')
            robot_visual_id = self._p.createVisualShape(
                self._p.GEOM_BOX, halfExtents=self.robot_half_dim,
                rgbaColor=[0.5, 0.5, 0.5, 1]
            )
            self.robot_id = self._p.createMultiBody(
                baseMass=0,  # static
                baseVisualShapeIndex=robot_visual_id,
                basePosition=np.append(state[:2], self.robot_com_height),
                baseOrientation=self._p.getQuaternionFromEuler([
                    0, 0, state[2]
                ])
            )

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
        vx_new = vx-k1*self.dt*vx+k5*ux*self.dt -k4*vy*self.dt
        vy_new = vy-k2*self.dt*vy+k6*uy*self.dt -k3*vx*self.dt
        # print(vx_new, vy_new)
        state = np.array([x_new, y_new, vx_new, vy_new])
        # state = np.array([ux,uy,0,0])
        yaw = 0#  np.pi/2

        # Update visual
        self._p.resetBasePositionAndOrientation(
            self.robot_id, posObj=np.append(state[:2], self.robot_com_height),
            ornObj=self._p.getQuaternionFromEuler([0, 0, yaw])
        )
        return state, action

    def move_camera(self, state):
        """
        Move camera and LiDAR to follow the robot. Update camera/LiDAR visualization if render.
        
        TODO: add noise to camera/LiDAR pose
        
        Args:
            state (np.ndarray): State of the robot.
        """
        x, y, vx, vy = state
        yaw = 0 #np.pi/2 # face +y direction
        robot_top_height = self.robot_half_dim[2] * 2

        # camera
        rgb_height = robot_top_height + self.rgb_cfg.z_offset_from_robot_top
        rgb_x_body_center = self.robot_half_dim[
            0
        ] + self.rgb_cfg.x_offset_from_robot_front + self.camera_thickness / 2
        rgb_x_body_front = self.robot_half_dim[
            0] + self.rgb_cfg.x_offset_from_robot_front + self.camera_thickness
        camera_roll_noise = 0
        camera_yaw_noise = 0
        # if self.camera_tilt_range is not None:
        #     camera_tilt = np.random.uniform(
        #         self.camera_tilt_range[0], self.camera_tilt_range[1], 1
        #     )[0]
        # else:
        #     camera_tilt = self.camera_tilt
        #     if self.camera_tilt_noise_std > 0:  # in deg
        #         camera_tilt += np.random.normal(
        #             0, self.camera_tilt_noise_std, 1
        #         )[0]
        # if self.camera_roll_noise_std > 0:  # in deg
        #     camera_roll_noise = np.random.normal(
        #         0, self.camera_roll_noise_std, 1
        #     )[0]
        # if self.camera_yaw_noise_std > 0:  # in deg
        #     camera_yaw_noise = np.random.normal(
        #         0, self.camera_yaw_noise_std, 1
        #     )[0]
        rot_matrix = [
            camera_roll_noise / 180 * np.pi, self.rgb_cfg.tilt / 180 * np.pi,
            yaw + camera_yaw_noise / 180 * np.pi
        ]
        rot_matrix = self._p.getMatrixFromQuaternion(
            self._p.getQuaternionFromEuler(rot_matrix)
        )
        self.cam_rot_matrix = np.array(rot_matrix).reshape(3, 3)
        self.cam_pos_center = np.array([x, y, rgb_height]
                                      ) + self.cam_rot_matrix.dot(
                                          (rgb_x_body_center, 0, 0)
                                      )  # for visualization
        self.cam_pos = np.array([x, y, rgb_height]) + self.cam_rot_matrix.dot(
            (rgb_x_body_front, 0, 0)
        )

        # LiDAR
        self.lidar_pos = np.array([
            x, y, robot_top_height + self.lidar_cfg.z_offset_from_robot_top
            + +self.lidar_height / 2
        ])

        # Update visuals
        # if self.render:
        #     self._p.resetBasePositionAndOrientation(
        #         self.camera_id, posObj=self.cam_pos_center,
        #         ornObj=self._p.getQuaternionFromEuler([0, 0, yaw])
        #     )
        #     self._p.resetBasePositionAndOrientation(
        #         self.lidar_id, posObj=self.lidar_pos,
        #         ornObj=self._p.getQuaternionFromEuler([0, 0, 0])
        #     )

    def seed(self, seed=0):
        """
        Set seed for reproducibility
        
        Args:
            seed (int): seed value
        """
        self.seed_val = seed
        self.rng = np.random.default_rng(seed=self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(
            self.seed_val
        )  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True