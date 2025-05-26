"""
Generate the navigation simulation and the calibration dataset with the task dataset.

Please contact the author(s) of this library if you have any questions.
Authors: Anushri Dixit (anushri.dixit@princeton.edu), Allen Z. Ren (allen.ren@princeton.edu)
"""

import os
import random
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
# from multiprocessing import Pool
import math
import json

from itertools import product, combinations
import torch
from torch.multiprocessing import Pool, Process, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

from clustering import is_box_visible
from nav_sim.env.task_env import TaskEnv
import sys
sys.path.append('../utils')
sys.path.append('../datasets')
from utils.pc_util import preprocess_point_cloud, pc_cam_to_3detr, random_sampling
import warnings
warnings.filterwarnings("error")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils.make_args import make_args_parser

# camera + 3DETR
num_pc_points = 40000

parser = make_args_parser()
args = parser.parse_args(args=[])

def combine_old_files(filenames, num_files):
    model_outputs_all = {"box_features": torch.tensor([]), "box_axis_aligned": torch.tensor([])}
    match_outputs_gt = {"output": torch.tensor([]), "gt": torch.tensor([])}
    bboxes_ground_truth_aligned = torch.tensor([])
    loss_mask = torch.tensor([])
    for i in range(num_files):
        features = torch.load(filenames[0]+str(i+1) + ".pt")
        bboxes = torch.load(filenames[1]+str(i+1) + ".pt")
        loss = torch.load(filenames[2]+str(i+1) + ".pt")
        finetune = torch.load(filenames[3]+str(i+1) + ".pt")
        model_outputs_all["box_features"] = torch.cat((model_outputs_all["box_features"], features["box_features"]))
        model_outputs_all["box_axis_aligned"] = torch.cat((model_outputs_all["box_axis_aligned"], features["box_axis_aligned"]))
        match_outputs_gt["output"] = torch.cat((match_outputs_gt["output"], finetune["output"]))
        match_outputs_gt["gt"] = torch.cat((match_outputs_gt["gt"], finetune["gt"]))
        bboxes_ground_truth_aligned= torch.cat((bboxes_ground_truth_aligned, bboxes))
        loss_mask = torch.cat((loss_mask, loss))
    return model_outputs_all, bboxes_ground_truth_aligned, loss_mask, match_outputs_gt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task_dataset', default='/home/zm2074/Projects/data/perception-guarantees/task_0803.pkl',
        nargs='?', help='path to save the task files'
    )
    parser.add_argument(
        '--save_dataset', default='/home/zm2074/Projects/data/perception-guarantees/calibrate_2k/',
        nargs='?', help='path to save the task files'
    )
    args = parser.parse_args()

    # Load task dataset
    with open(args.task_dataset, 'rb') as f:
        task_dataset = pickle.load(f)

    # get root repository path
    nav_sim_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


    # Sample random task
    save_tasks = []
    for task in task_dataset:

        # Initialize task
        task.goal_radius = 0.5
        #
        task.observation = {}
        task.observation.type = 'rgb'  # 'rgb' or 'lidar'
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

        task.mesh_parent_folder = '/home/zm2074/Projects/data/perception-guarantees/3D-FUTURE-model-tiny'
    ##################################################################
    # Number of environments
    num_envs = 400

    # Number of parallel threads
    num_parallel = 10
    ##################################################################

    # model_outputs_all, bboxes_ground_truth_aligned, loss_mask, match_outputs_gt = format_results(save_res)
    filenames = [args.save_dataset + "data/dataset_intermediate/features", args.save_dataset + "data/dataset_intermediate/bbox_labels", args.save_dataset + "data/dataset_intermediate/loss_mask", args.save_dataset + "data/dataset_intermediate/finetune"]
    model_outputs_all, bboxes_ground_truth_aligned, loss_mask, match_outputs_gt = combine_old_files(filenames, int(len(task_dataset)/num_parallel))
    ###########################################################################
    # # Save processed feature data
    torch.save(model_outputs_all, args.save_dataset + "data/features.pt")
    # # Save ground truth bounding boxes
    torch.save(bboxes_ground_truth_aligned, args.save_dataset + "data/bbox_labels.pt")
    # # Save loss mask
    torch.save(loss_mask, args.save_dataset + "data/loss_mask.pt")
    # # Save all box outputs for finetuning
    torch.save(match_outputs_gt, args.save_dataset + "data/finetune.pt")
    ###########################################################################
