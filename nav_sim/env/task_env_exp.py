"""Navigation simulation in PyBullet

Use the task dataset instead of meshes in the asset folder.

"""

import os
import numpy as np
import pybullet as pb
from pybullet_utils import bullet_client as bc

from nav_sim.env.vanilla_env_exp import VanillaEnv


class TaskEnv(VanillaEnv):

    def __init__(
        self,
        render=False,
    ):
        """
        Args:
            render (bool): whether to render the environment with PyBullet for GUI visulization
        """
        super(TaskEnv, self).__init__(render=render)

    def init_pb(self):
        """
        Initialize the PyBullet client.
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

        # Build camera and LiDAR visualization if render
        if self.render:
            camera_visual_id = p.createVisualShape(
                p.GEOM_BOX, rgbaColor=[0, 0.8, 0, 1.0],
                halfExtents=[0.03, 0.20, 0.05]
            )
            self.camera_id = p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=camera_visual_id, basePosition=[0, 0, 0]
            )
            lidar_visual_id = p.createVisualShape(
                p.GEOM_CYLINDER, rgbaColor=[0, 0.8, 0, 1.0], radius=0.08,
                length=self.lidar_height
            )
            self.lidar_id = p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=lidar_visual_id, basePosition=[0, 0, 0]
            )

        # Initialize obstacle id list - excluding walls and floors
        self._obs_id_all = []

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

        # Load wall and ceiling
        room_obj_all = [
            os.path.join(task.base_path, 'floor.obj'),
            os.path.join(task.base_path, 'wall.obj'),
        ]
        for obj_path in room_obj_all:
            obj_collision_id = self._p.createCollisionShape(
                self._p.GEOM_MESH,
                fileName=obj_path,
                flags=self._p.GEOM_FORCE_CONCAVE_TRIMESH  # concave
            )
            obj_visual_id = self._p.createVisualShape(
                self._p.GEOM_MESH, fileName=obj_path
            )
            obj_id = self._p.createMultiBody(
                baseMass=0,  # static, allow concave
                baseCollisionShapeIndex=obj_collision_id,
                baseVisualShapeIndex=obj_visual_id,
                baseOrientation=self._p.getQuaternionFromEuler([0, 0, 0])
            )
            self._obs_id_all += [obj_id]

        # Load furniture
        for piece_id, piece_pos in zip(task.piece_id_all, task.piece_pos_all):
            piece_path = os.path.join(
                task.mesh_parent_folder, piece_id, task.mesh_name
            )
            obj_collision_id = self._p.createCollisionShape(
                self._p.GEOM_MESH,
                fileName=piece_path,
                flags=self._p.GEOM_FORCE_CONCAVE_TRIMESH  # concave
            )
            obj_visual_id = self._p.createVisualShape(
                self._p.GEOM_MESH,
                fileName=piece_path,
            )
            obj_id = self._p.createMultiBody(
                baseMass=0,  # static, allow concave
                baseCollisionShapeIndex=obj_collision_id,
                baseVisualShapeIndex=obj_visual_id,
                basePosition=[piece_pos[0], piece_pos[1], 0],
                baseOrientation=self._p.getQuaternionFromEuler([
                    np.pi / 2, 0, 0
                ])
            )
            self._obs_id_all += [obj_id]

        # Render goal location if render
        if self.render:
            if "_goal_id" in vars(self).keys():
                self._p.removeBody(self._goal_id)
            goal_visual_id = self._p.createVisualShape(
                self._p.GEOM_BOX,
                halfExtents=[0.05, 0.05, 0.05],
                rgbaColor=[0.7, 0, 0, 1],
            )
            self._goal_id = self._p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=goal_visual_id,
                basePosition=[self._goal_loc[0], self._goal_loc[1], 0.05],
                baseOrientation=self._p.getQuaternionFromEuler([0, 0, 0]),
            )

    def reset_robot(self, state):
        """
        Reset the robot with (x, y, yaw) state input and fixed height at COM.
        
        Args:
            state (np.ndarray): State to be reset.
        """
        yaw = np.pi/2 # face +y direction
        if "robot_id" in vars(self).keys():
            self._p.resetBasePositionAndOrientation(
                self.robot_id,
                posObj=np.append(state[:2], self.robot_com_height),
                ornObj=self._p.getQuaternionFromEuler([0, 0, yaw])
            )
        else:
            robot_visual_id = self._p.createVisualShape(
                self._p.GEOM_BOX, halfExtents=self.robot_half_dim,
                rgbaColor=[0.5, 0.5, 0.5, 1]
            )
            self.robot_id = self._p.createMultiBody(
                baseMass=0,  # static
                baseVisualShapeIndex=robot_visual_id,
                basePosition=np.append(yaw, self.robot_com_height),
                baseOrientation=self._p.getQuaternionFromEuler([
                    0, 0, yaw
                ])
            )
