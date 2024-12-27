import torch
from torch.utils.data import Dataset
import IPython as ipy

class PointCloudDataset(Dataset):
    def __init__(self, features_file, bbox_labels_file, loss_mask_file):

        # Point cloud features from 3DETR
        self.features = torch.load(features_file)
        # self.feature_dims = self.features["box_features"].shape[2:]

        # Loss mask array
        self.loss_mask = torch.load(loss_mask_file)

        # Prediction of axis-aligned box from 3DETR
        self.bbox_3detr = self.features["box_axis_aligned"]

        # Repeat grounth truth labels for each robot viewpoint location
        num_locations = self.features["box_axis_aligned"].shape[1]
        self.bbox_labels = torch.load(bbox_labels_file)
        sb = self.bbox_labels.shape
        self.bbox_labels = self.bbox_labels.view((sb[0], 1, sb[1], sb[2], sb[3]))
        self.bbox_labels = self.bbox_labels.expand(-1, num_locations, -1, -1, -1)


    def __len__(self):
        return self.bbox_labels.shape[0]
    
    def __numobjects__(self):
        return self.bbox_labels.shape[2]

    def __getitem__(self, idx):
        '''
        idx: environment index.
        Returns features and labels.
            features: batch of box features returned by 3DETR on the raw point clouds
            labels: batch of labels; this contains the batch of ground truth bbox vertices
                    and also contains the predictions made by 3DETR (which will be used to compute loss).
        '''

        # Get features
        # features = self.features["box_features"][idx, :, :, :]

        # Get loss mask
        loss_mask = self.loss_mask[idx, :]

        # Get label
        labels = {'bboxes_gt': self.bbox_labels[idx, :, :, :, :],
                'bboxes_3detr': self.bbox_3detr[idx, :, :, :, :]}
        
        
        return labels, loss_mask
