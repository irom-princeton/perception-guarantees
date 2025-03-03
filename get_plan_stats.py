import os
import random
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import IPython as ipy
import time
import torch
import shutil

num_envs = 100

# cp = 0.7441 # 2k samples
# cp = 0.6249 # 500 samples
cp = 0.7086 # 1k samples
# cp = 0.6910 # 1.5k samples
# cp = 0.75
from pathlib import Path
# base path
base_path: Path = Path(__file__).parent
foldername = f'{base_path.parent}/data/perception-guarantees/room_1203_rot/'
taskpath = f'{base_path.parent}/data/perception-guarantees/task_1203_rot.pkl'

# filename = 'numcc_1.26_0213.npz'
# filename_if = 'numcc_0.49_0120.npz'
# filename = 'cp_0_numcc_ssdt0.5.npz'
filename = 'cp_1.1_4k_0218.npz'
# filename = 'numcc_0.24_ssdt0.2_fix_misdetect.npz'
# pic_name = "numcc_0.49traj_plot_2k_ssdt0.2_fd.png"
# pic_name = 'cp0.7441traj_plot_2k.png'
# filename = 'cp_confidence.npz'
# pic_name = 'cpconfidencetraj_plot_confidence.png'
dst =  "../data/perception-guarantees/results/"
goal_loc_planner_frame = [6,7]
traj = {}
done= []
coll = []
misdetect = []
dist_from_goal = 0
envs = []

p = []
q = []

# Load the x,y points to sample
with open('planning/pre_compute/Pset-4k.pkl', 'rb') as f:
    samples = pickle.load(f)
    # Remove goal
    samples = samples[:-1][:]
# Remove duplicates
sample_proj = [[sample[0], sample[1]] for sample in samples]
s = []
s = [x for x in sample_proj if x not in s and not s.append(x)]
# Transform from planner frame
for sample in s:
    x = sample[1]
    y = sample[0]-4
    p.append([x,y])
pd = np.ones(len(p))/len(p)
qd = np.zeros_like(pd)

def state_to_pixel(state: np.ndarray) -> np.ndarray:
    """
    Convert planner state to pixel location

    Args:
        state (np.ndarray): State to convert

    Returns:
        np.ndarray: Pixel location
    """
    forward = state[1]
    right = state[0]
    room_size = 8

    x = int(83-np.floor(forward/room_size*83))
    y = int(np.floor(right/room_size*83))
    pix_loc = np.array([x, y])

with open(taskpath, 'rb') as f:
    task_dataset = pickle.load(f)

# get root repository path
nav_sim_path = base_path/"nav_sim"

num_tasks = len(task_dataset)

collisions = 0
fails = 0
for i in range(num_envs):
    task = task_dataset[i]
    env= task.base_path.split('/')[-1]

    # if filename_if exists, use it
    # if os.path.exists(foldername + str(env) + "/" + filename_if):
    #     file_env = foldername + str(env) + "/" + filename_if
    # else:
    file_env = foldername + str(env) + "/" + filename
    # pic_src = foldername + str(env) + "/" + pic_name
    # pic_dst = dst + str(i) +pic_name
    # shutil.copyfile(pic_src, pic_dst)

    # check if file exists
    if not os.path.exists(file_env):
        print("File does not exist: ", str(env))
        continue

    data_ = np.load(file_env, allow_pickle=True)
    traj_info = data_["data"].item()
    # ipy.embed()
    # print(data[tra])
    # print("Environment ", i)
    traj[env] = traj_info['trajectory']
    envs.append(env)
    done.append(int(traj_info['done']))
    coll.append(int(traj_info['collision']==False))
    misdetect.append(traj_info['misdetection'])
breakpoint() # additional indexing for bbox experiments!
traj_length = 0
for i in range(len(envs)):
    env = envs[i]
    if len(traj[env]) == 0:
        dist_from_goal += np.linalg.norm(np.array(goal_loc_planner_frame)-np.array([5,0.2]))
    else:
        if done[i] == 1:
            traj_length+= np.sum(np.linalg.norm(np.array(traj[env][:-1,0,0:2]) - np.array(traj[env][1:,0, 0:2]), axis=1))
        if done[i] == 0:
            dist_from_goal += np.linalg.norm(np.array(traj[env][-1,0,0:2]-np.array(goal_loc_planner_frame)))-1
        for j in range(len(traj[env][:,0])):
            idx = np.argmin(np.linalg.norm(traj[env][j,0,0:2] - p, axis=1))
            q.append(p[idx])
            qd[idx] += 1

# qd = qd/len(q)
# ipy.embed()

# kl = 0
# for i in range(len(p)):
#     if qd[i] > 0:
#         kl+= qd[i]*np.log(qd[i]/pd[i])
# breakpoint()

print("Average trajectory length: ", traj_length/np.sum(done))
print("Successful task completion: ", np.mean(done))
print("Safety rate: ", np.mean(coll))
print("Misdetection rate: ", len(np.where(np.array(misdetect)>0)[0])/num_envs)
print("Failed in environments: ", np.array(envs)[np.where(np.array(done)<1)[0]])
print("Collisions in environments: ", np.array(envs)[np.where(np.array(coll)<1)[0]])
print("Misdetections in environments: ", np.array(envs)[np.where(np.array(misdetect)>0)[0]])
print("Average distance from goal if failed: ", dist_from_goal/(np.sum(1-np.array(done))) )
# print("KL-divergence", kl)

