import os
import random
import argparse
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

from nav_sim.test.clustering import cluster, is_box_visible
from utils.pc_util import preprocess_point_cloud, read_ply, pc_to_axis_aligned_rep, pc_cam_to_3detr, is_inside_camera_fov
from utils.box_util import box2d_iou
from utils.make_args import make_args_parser
from omegaconf import OmegaConf

from mmdet3d.apis import LidarDet3DInferencer

import warnings
warnings.filterwarnings("ignore")


# Model and weights (Scannet version)
model = "projects/TR3D/configs/tr3d_1xb16_scannet-3d-18class.py" # TR3D
weights = "tr3d_1xb16_scannet-3d-18class.pth" # TR3D pretrained weights

# Initialize inferencer
inferencer = LidarDet3DInferencer(model=model, weights=weights)


def run_env(task):
    env = TaskEnv(render=True)
    env.reset(task)

    # Press any key to start
    print("\n=========================================")
    input("Press any key to start")
    print("=========================================\n")

    # Run
    for step in range(100):

        # Execute action
        action = [0.1, 0]
        observation, reward, done, info = env.step(action)

        # summarize the step in one line
        print(
            '\nStep: {}, Action: {}, Reward: {}, Done: {}, Info: {}\n'.format(
                step, action, reward, done, info
            )
        )

        # Show RGB image or LiDAR scan
        if task.observation.type == 'rgb':
            plt.imshow(observation.transpose(1, 2, 0))
            plt.show()
        elif task.observation.type == 'lidar':
            # Filter points with z < 0.01 and abs(y) > 3.5 and x> 0.01
            observation = observation[:, observation[2, :] > 0.1]
            observation = observation[:, np.abs(observation[1, :]) < 3.5]
            observation = observation[:, observation[0, :] > 0.01]
            results = get_box(observation, inferencer, show_viz=True)
            # scores = results['predictions'][0]['scores_3d']
            # bboxes = results['predictions'][0]['bboxes_3d']
            # print(max(scores))
            print('Scan - number of points: ', observation.shape[1])
            plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(
                observation[0, :], observation[1, :], observation[2, :]
            )
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()
        elif task.observation.type == 'both':
            # Filter points with z < 0.01 and abs(y) > 3.5 and x> 0.01
            observation = observation[:, observation[2, :] > 0.1]
            observation = observation[:, observation[2, :] < 2.9]
            observation = observation[:, np.abs(observation[1, :]) < 3.5]
            observation = observation[:, observation[0, :] > 0.01]

            # np.save('observation.npy', observation)

            bboxes = get_box(observation, inferencer, 15)
            print(bboxes)

            plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(
                observation[0, :], observation[1, :], observation[2, :]
            )

            for bbox in bboxes:
                # box corners
                r0 = [bbox[0]-bbox[3]/2, bbox[0]+bbox[3]/2]
                r1 = [bbox[1]-bbox[4]/2, bbox[1]+bbox[4]/2]
                r2 = [bbox[2], bbox[2]+bbox[5]]
                for s, e in combinations(np.array(list(product(r0, r1, r2))), 2):
                    if (np.sum(np.abs(s-e)) == np.abs(r0[1]-r0[0]) or 
                        np.sum(np.abs(s-e)) == np.abs(r1[1]-r1[0]) or 
                        np.sum(np.abs(s-e)) == np.abs(r2[1]-r2[0])):
                        ax.plot3D(*zip(s, e), color=(0.5, 0.1, 0.1))

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_zlim((0,4))
            plt.show()

def get_box(observation_, inferencer, num_boxes, show_viz=False):
    observation = observation_.copy()
    observation = observation.T
    # Convert depth from Sim frame to ScanNet frame
    # Sim frame:     +X(Front), +Y(Left),  +Z(Up)
    # ScanNet frame: +X(Right), +Y(Front), +Z(Up)
    # observation[:,1] = observation_.T[:,0]
    # observation[:,0] = -observation_.T[:,1]
    # print(observation_.T[0])
    # print(observation[0])

    conf_threshold = 0.05
    inputs_all = {'inputs': {'points': observation}, 'pred_score_thr': conf_threshold, 'out_dir': '', 'show': show_viz, 'wait_time': -1, 'no_save_vis': True, 'no_save_pred': False, 'print_result': False}
    inferencer.show_progress = False
    results = inferencer(**inputs_all)
    
    detect_objects = range(16) # "cabinet": 0, "bed": 1, "chair": 2, "sofa": 3, "table": 4, "door": 5, "window": 6, "bookshelf": 7, "picture": 8, 
    # "counter": 9, "desk": 10, "curtain": 11, "refrigerator": 12, "showercurtrain": 13, "toilet": 14, "sink": 15, "bathtub": 16, "garbagebin": 17,

    pred_boxes = results["predictions"][0]["bboxes_3d"]
    num_predictions = len(pred_boxes) # Number of predicted boxes

    pred_confidences = results["predictions"][0]["scores_3d"]
    pred_labels = results["predictions"][0]["labels_3d"]

    # filter_inds = [i for i in range(num_predictions) if ((pred_confidences[i] > conf_threshold) and (pred_labels[i] in detect_objects))]
    # return [pred_boxes[i] for i in filter_inds]

    sort_inds = np.argsort(pred_confidences)[::-1]
    sorted_inds = [i for i in sort_inds[0:num_boxes] if (pred_labels[i] in detect_objects)]
    return [pred_boxes[i] for i in sorted_inds]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task_dataset', default='/home/zm2074/Projects/data/perception-guarantees/task.pkl',
        nargs='?', help='path to save the task files'
    )
    parser.add_argument(
        '--save_dataset', default='/home/zm2074/Projects/data/perception-guarantees/task.npz',
        nargs='?', help='path to save the task files'
    )
    args = parser.parse_args()

    # Load task dataset
    with open(args.task_dataset, 'rb') as f:
        task_dataset = pickle.load(f)

    # Sample random task
    task = random.choice(task_dataset)

    # get root repository path
    nav_sim_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Initialize task
    task.goal_radius = 0.5
    task.init_state = [0.5,0.2,0,0]
    task.observation = {}
    task.observation.type = 'both'  # 'rgb' or 'lidar' or 'both
    task.observation.rgb = {}
    task.observation.depth = {}
    task.observation.lidar = {}
    task.observation.rgb.x_offset_from_robot_front = 0.01  # no y offset
    task.observation.rgb.z_offset_from_robot_top = 0
    task.observation.rgb.tilt = 5  # degrees of tilting down towards the floor
    task.observation.rgb.img_w = 256
    task.observation.rgb.img_h = 256
    task.observation.rgb.aspect = 1
    task.observation.rgb.fov = 70  # in PyBullet, this is vertical field of view in degrees
    task.observation.depth.img_w = task.observation.rgb.img_w  # needs to be the same now - assume coming from the same camera
    task.observation.depth.img_h = task.observation.rgb.img_h
    task.observation.lidar.z_offset_from_robot_top = 0.01  # no x/y offset
    task.observation.lidar.horizontal_res = 1  # resolution, in degree
    task.observation.lidar.vertical_res = 1  # resolution, in degree
    task.observation.lidar.vertical_fov = 30  # half in one direction, in degree
    task.observation.lidar.max_range = 5  # in meter

    # Run environment
    run_env(task)