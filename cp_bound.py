import numpy as np
import torch
from torch.utils.data import DataLoader
import IPython as ipy
import argparse 
from pc_dataset import PointCloudDataset
from loss_fn import *
from copy import deepcopy
from matplotlib import pyplot as plt

from pathlib import Path
pg_path: Path = Path(__file__).resolve().parent

import sys
sys.path.append('../utils')

def main(raw_args=None):


	###################################################################
	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--verbose", type=int, default=1, help="print more (default: 1)")

	args = parser.parse_args(raw_args)
	verbose = args.verbose
	###################################################################

	###################################################################
	# Initialize dataset and dataloader
	dataset = PointCloudDataset(pg_path/"nav_sim/output/data/features.pt", 
							 	pg_path/"nav_sim/output/data/bbox_labels.pt", 
								pg_path/"nav_sim/output/data/loss_mask.pt")
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
		targets, loss_mask = data
		boxes_3detr = targets["bboxes_3detr"].to(device)
		boxes_gt = targets["bboxes_gt"].to(device)
		loss_mask = loss_mask.to(device)
		breakpoint()
		scaling_cp = scale_prediction(boxes_3detr[:400], boxes_gt[:400], loss_mask[:400], 0.887) #for coverage of 0.85 w.p. 0.99 
		average_cp = scale_prediction_average(boxes_3detr, boxes_gt, loss_mask, 0.887)
		print('CP quantile prediction', scaling_cp)
		print('CP quantile prediction (for baseline CP-avg.)', average_cp)
	#################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
	main() 

