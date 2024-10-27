import numpy as np
import torch
from torch.utils.data import DataLoader
import IPython as ipy
import argparse 
from pc_dataset import PointCloudDataset
from loss_fn import *
from copy import deepcopy
from matplotlib import pyplot as plt
import scipy.special
import scipy.optimize as opt
import pickle
import os
import functools

import sys
sys.path.append('../utils')

from pathlib import Path
base_path = Path(__file__).parent
data_path = base_path.parent / 'data'

# from Ani
def diff_conditional_success(epsilon_hat, desired_success_prob, N, delta):
	v = np.floor((N+1)*epsilon_hat)
	a = N+1-v
	b = v
	return scipy.special.betaincinv(a, b, delta) - desired_success_prob

def main_numcc():
	folder_path = data_path/'perception-guarantees/task_numcc/data/dataset_intermediate/'
	losses = []

	file_names = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
	for filename in file_names:
		with open(f'{folder_path}/{filename}', 'rb') as f:
			data = pickle.load(f)
			delta = data[0]['delta']
			losses.append(delta)
	
	desired_epsilon = 0.15
	desired_success_prob = 1-desired_epsilon
	delta = 0.01
	N = len(losses)
	epsilon_hat = opt.bisect(diff_conditional_success, desired_epsilon/10, 0.5, args=(desired_success_prob, N, delta))

	q_level = np.ceil((N+1)*(1-epsilon_hat))/N
	qhat = np.quantile(losses, q_level, method = 'higher')
	print(f'qhat: {qhat}')
	return qhat
	


def main_box(raw_args=None):


	###################################################################
	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--verbose", type=int, default=1, help="print more (default: 1)")

	args = parser.parse_args(raw_args)
	verbose = args.verbose
	###################################################################

	###################################################################
	# Initialize dataset and dataloader
	dataset = PointCloudDataset("/media/zm2074/Data Drive/data/perception-guarantees/calibrate_1.5k/data/features.pt", "/media/zm2074/Data Drive/data/perception-guarantees/calibrate_1.5k/data/bbox_labels.pt", "/media/zm2074/Data Drive/data/perception-guarantees/calibrate_1.5k/data/loss_mask.pt")
	dataloader_cp = DataLoader(dataset, batch_size=len(dataset))
	###################################################################

	###################################################################
	# Device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	torch.cuda.set_device(0)
	# device = torch.device('cpu')
	###################################################################

	#################################################################
	# Without finetuning
	for i, data in enumerate(dataloader_cp, 0):
		print(i)
		inputs, targets, loss_mask = data
		boxes_3detr = targets["bboxes_3detr"].to(device)
		boxes_gt = targets["bboxes_gt"].to(device)
		loss_mask = loss_mask.to(device)


		corners_pred = boxes_3detr
		corners_gt = boxes_gt
		tol = 0.887

		B, K = corners_gt.shape[0], corners_gt.shape[1]

		# 2D projection
		corners_gt = corners_gt[:,:,:,:,0:2]
		corners_pred = corners_pred[:,:,:,:,0:2]

		# Ensure that corners of predicted bboxes satisfy basic constraints
		corners1_pred = torch.min(corners_pred[:, :, :, 0, :][:,:,None,:], corners_pred[:, :, :, 1, :][:,:,None,:])
		corners2_pred = torch.max(corners_pred[:, :, :, 0, :][:,:,None,:], corners_pred[:, :, :, 1, :][:,:,None,:])

		# Compute the mean position of the ground truth and predicted bounding boxes
		pred_center = torch.div(corners_pred[:, :, 0, :][:,:,None,:] + corners_pred[:, :, 1, :][:,:,None,:],2)
		gt_center = torch.div(corners_gt[:, :, 0, :][:,:,None,:] + corners_gt[:, :, 1, :][:,:,None,:],2)

		# Calculate the scaling between predicted and ground truth boxes
		corners1_diff = (corners1_pred - corners_gt[:,:,:,0,:][:,:,None,:])
		corners2_diff = (corners_gt[:,:,:,1,:][:,:,None,:] - corners2_pred)
		corners1_diff = torch.squeeze(corners1_diff,2)
		corners2_diff = torch.squeeze(corners2_diff,2)
		corners1_diff_mask = torch.mul(loss_mask,corners1_diff.amax(dim=3))
		corners2_diff_mask = torch.mul(loss_mask, corners2_diff.amax(dim=3))
		corners1_diff_mask[loss_mask == 0] = -np.inf
		corners2_diff_mask[loss_mask == 0] = -np.inf
		# ipy.embed()
		corners1_diff_mask = corners1_diff_mask.amax(dim=2)
		corners2_diff_mask = corners2_diff_mask.amax(dim=2)
		delta_all = torch.maximum(corners1_diff_mask, corners2_diff_mask)

		delta = delta_all.amax(dim=1)
		delta, indices = torch.sort(delta, dim=0, descending=False)
		idx = math.ceil((B+1)*(tol))-1
		
		idx = math.ceil((B+1)*(tol))-1



		ipy.embed()
		scaling_cp = scale_prediction(boxes_3detr, boxes_gt, loss_mask, 0.887) #for coverage of 0.85 w.p. 0.99 
		average_cp = scale_prediction_average(boxes_3detr, boxes_gt, loss_mask, 0.887)
		print('CP quantile prediction', scaling_cp)
		print('CP quantile prediction (for baseline CP-avg.)', average_cp)
	#################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
	main_numcc() 

