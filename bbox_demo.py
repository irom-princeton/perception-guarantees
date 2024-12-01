import time
from mmdet3d.apis import LidarDet3DInferencer
from plyfile import PlyData
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from itertools import combinations, product
# third_party/mmdetection3d/mmdet3d/visualization/local_visualizer.py
from mmdet3d.visualization import Det3DLocalVisualizer
# third_party/mmdetection3d/mmdet3d/apis/inferencers/lidar_det3d_inferencer.py
from mmdet3d.structures import LiDARInstance3DBoxes

#########################################################################
# Input .ply file
ply_file = "Pointcloud_ZED.ply" 
#########################################################################

#########################################################################
# Model and weights (Scannet version)
model = "tr3d_1xb16_scannet-3d-18class.py" # TR3D
weights = "tr3d_scannet.pth" # TR3D pretrained weights

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
detect_objects = set([2]) # "cabinet": 0, "bed": 1, "chair": 2, "sofa": 3, "table": 4, "door": 5, "window": 6, "bookshelf": 7, "picture": 8, 
# "counter": 9, "desk": 10, "curtain": 11, "refrigerator": 12, "showercurtrain": 13, "toilet": 14, "sink": 15, "bathtub": 16, "garbagebin": 17,

pred_boxes = results["predictions"][0]["bboxes_3d"]
num_predictions = len(pred_boxes) # Number of predicted boxes

pred_confidences = results["predictions"][0]["scores_3d"]
pred_labels = results["predictions"][0]["labels_3d"]

# filter_inds = [i for i in range(num_predictions) if ((pred_confidences[i] > conf_threshold) and (pred_labels[i] in detect_objects))]

# filtered_boxes = LiDARInstance3DBoxes(torch.tensor([pred_boxes[i] for i in filter_inds]))

#visualizer.draw_bboxes_3d(filtered_boxes, bbox_color=len(filter_inds)*[[0,255,0]])
#visualizer.show()

#################################################################################
# Draw and save boxes

num_boxes = 15
sort_inds_by_confidence = np.argsort(pred_confidences)[::-1]
sort_inds_in_detect_objects = [i for i in sort_inds_by_confidence if (pred_labels[i] in detect_objects)]
top_num_boxes_indices = sort_inds_in_detect_objects[0:num_boxes]
bboxes = [pred_boxes[i] for i in top_num_boxes_indices]


corners = np.zeros((len(bboxes), 2,3))
for b in range(len(bboxes)):
    bbox = bboxes[b]
    # box corners
    r0 = [bbox[0]-bbox[3]/2, bbox[0]+bbox[3]/2]
    r1 = [bbox[1]-bbox[4]/2, bbox[1]+bbox[4]/2]
    r2 = [bbox[2], bbox[2]+bbox[5]]
    corners[b,0,:] = [r0[0], r1[0], r2[0]]
    corners[b,1,:] = [r0[1], r1[1], r2[1]]

#############################################################################

xyz_data = data_pd[['x', 'y', 'z']].to_numpy()
pc = xyz_data[:num_subsample].reshape(1, -1, 3)  
pc_plot = pc
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(
    pc_plot[0, :, 0], pc_plot[0, :, 1], pc_plot[0, :, 2], s=1, alpha=0.5, label="Point Cloud"
)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


ax.set_xlim(np.min(pc[0, :, 0]), np.max(pc[0, :, 0]))
ax.set_ylim(np.min(pc[0, :, 1]), np.max(pc[0, :, 1]))
ax.set_zlim(np.min(pc[0, :, 2]), np.max(pc[0, :, 2]))

#Draw bounding boxes
for corner in corners:
    # Extract corners of the box
    x_min, y_min, z_min = corner[0]
    x_max, y_max, z_max = corner[1]

    # Define corner points
    points = np.array([
        [x_min, y_min, z_min], [x_min, y_min, z_max],
        [x_min, y_max, z_min], [x_min, y_max, z_max],
        [x_max, y_min, z_min], [x_max, y_min, z_max],
        [x_max, y_max, z_min], [x_max, y_max, z_max]
    ])

    # Draw edges between points
    for start, end in combinations(points, 2):
        # Only connect points that share one coordinate
        if np.sum(np.abs(np.array(start) - np.array(end)) <= 1e-6) == 2:
            ax.plot3D(*zip(start, end), color='red')

output_file = "output/point_cloud_boxes.png"  # Replace with your desired file path
plt.legend()
plt.savefig(output_file)
plt.close(fig)

print(f"Plot saved to {output_file}")