import time
from mmdet3d.apis import LidarDet3DInferencer
from plyfile import PlyData
import pandas as pd
import numpy as np
import torch
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes

#########################################################################
# Input .ply file
ply_file = "Pointcloud_ZED.ply" 
#########################################################################

#########################################################################
# Model and weights (Scannet version)
model = "projects/TR3D/configs/tr3d_1xb16_scannet-3d-18class.py" # TR3D
weights = "tr3d_1xb16_scannet-3d-18class.pth" # TR3D pretrained weights

# Initialize inferencer
inferencer = LidarDet3DInferencer(model=model, weights=weights)
#########################################################################

#########################################################################
# Load points from .ply file
plydata = PlyData.read(ply_file)
data = plydata.elements[0].data # read data
data_pd = pd.DataFrame(data) # convert to DataFrame
num_subsample = 100000 # Number of points to subsample
data_np = np.zeros((num_subsample,6), dtype=float) # initialize array to store data: xyzrgb
property_names = data[0].dtype.names # read names of properties

# Convert depth from Zed frame to ScanNet frame
# Zed frame:     +X(Right), +Y(Up),    +Z(Backward)
# ScanNet frame: +X(Right), +Y(Front), +Z(Up)
rand_inds = np.random.choice(range(data_pd.shape[0]),size=(num_subsample),replace=False)
data_np[:,0] = data_pd['x'][rand_inds]
data_np[:,1] = -1*data_pd['z'][rand_inds]
data_np[:,2] = data_pd['y'][rand_inds]

# Colors
data_np[:,3] = data_pd['red'][rand_inds]
data_np[:,4] = data_pd['green'][rand_inds]
data_np[:,5] = data_pd['blue'][rand_inds]

data_np = data_np.astype(np.float32)
#########################################################################

#########################################################################    
# Initialize inputs to inferencer

inputs_all = {'inputs': {'points': data_np}, 'pred_score_thr': 0.3, 'out_dir': '', 'show': False, 'wait_time': -1, 'no_save_vis': True, 'no_save_pred': True, 'print_result': False}

# Don't show progress bar during inference
inferencer.show_progress = False
#########################################################################

#########################################################################
# Perform inference
t_start = time.time()
results = inferencer(**inputs_all)
t_end = time.time()
print("Inference time: ", t_end - t_start, " seconds")
#########################################################################


#########################################################################
# Perform visualization

# Initialize visualizer
visualizer = Det3DLocalVisualizer()

# Set point cloud in visualizer
visualizer.set_points(data_np, mode="xyzrgb")

# Filter bounding boxes by semantic category and confidence
conf_threshold = 0.3
detect_objects = [2] # "cabinet": 0, "bed": 1, "chair": 2, "sofa": 3, "table": 4, "door": 5, "window": 6, "bookshelf": 7, "picture": 8, 
# "counter": 9, "desk": 10, "curtain": 11, "refrigerator": 12, "showercurtrain": 13, "toilet": 14, "sink": 15, "bathtub": 16, "garbagebin": 17,

pred_boxes = results["predictions"][0]["bboxes_3d"]
num_predictions = len(pred_boxes) # Number of predicted boxes

pred_confidences = results["predictions"][0]["scores_3d"]
pred_labels = results["predictions"][0]["labels_3d"]

filter_inds = [i for i in range(num_predictions) if ((pred_confidences[i] > conf_threshold) and (pred_labels[i] in detect_objects))]

filtered_boxes = LiDARInstance3DBoxes(torch.tensor([pred_boxes[i] for i in filter_inds]))

# Draw 3D bboxes
visualizer.draw_bboxes_3d(filtered_boxes, bbox_color=len(filter_inds)*[[0,255,0]])
visualizer.show()
#########################################################################

