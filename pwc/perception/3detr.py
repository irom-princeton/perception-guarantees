import numpy as np
import torch

from pwc.perception.models import build_model
from pwc.perception.datasets.sunrgbd import SunrgbdDatasetConfig

from pwc.utils.pc_util import preprocess_point_cloud, pc_to_axis_aligned_rep, pc_cam_to_3detr
from pwc.utils.box_util import box2d_iou
from pwc.utils.clustering import is_box_visible
from pwc.utils.make_args import make_args_parser

from pwc.perception.perception_model import PerceptionModel


class Perception3DETR(PerceptionModel):
    def __init__(self,
                 config,):
        self.model_name = "3DETR"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.visualize = config.visualize
        self.num_chairs = config.num_chairs
        self.num_boxes = config.num_boxes
        self.num_pc_points = config.num_pc_points
        self.room_thresholds = config.room_thresholds
        
        self.load_model()

    def load_model(self):
        # Dataset config: use SUNRGB-D
        dataset_config = SunrgbdDatasetConfig()

        # Parse default arguments
        parser = make_args_parser()
        args = parser.parse_args(args=[])

        # Build model
        model, _ = build_model(args, dataset_config)

        # Load pre-trained weights
        sd = torch.load(args.test_ckpt, map_location=torch.device("cpu")) 
        model.load_state_dict(sd["model"]) 

        model = model.cuda()
        model.eval()

        device = torch.device("cuda")
        # device = torch.device("cpu")
        model.to(device)
        self.model = model
        return
    
    def run_step(self,
                 observation: torch.Tensor,
                 piece_bounds_all: list,
                 pos: list):
        
        # filter invisible chairs
        not_inside_xyz = [[(0 if (pos[i]>obs[i]-0.1 and pos[i]<obs[3+i]+0.1) else 1) for i in range(3)] for obs in piece_bounds_all]
        gt_obs = [[[-obs[4], obs[0], obs[2]],[-obs[1], obs[3], obs[5]]] for obs in piece_bounds_all]
        cam_not_inside_obs = all([True if any(obs) == 1 else False for obs in not_inside_xyz])
        is_vis = [False]*self.num_chairs

        # Filter points with z < 0.01 and abs(y) > 3.9 and x> 0.05
        observation = observation[:, observation[2, :] < self.room_thresholds.zmax]
        X = observation[:, (observation[2, :] > self.room_thresholds.zmin) ]
        X = X[:, np.abs(X[1,:]) < self.room_thresholds.ymax]
        X = X[:, X[0,:] > self.room_thresholds.xmin]
        X = X[:, X[0,:] < self.room_thresholds.xmax]
        
        X = np.transpose(np.array(X))
        if(len(X) > 0):
            is_vis = is_box_visible(X, piece_bounds_all, visualize=self.visualize)
            for obs_idx, obs in enumerate(piece_bounds_all):
                is_vis[obs_idx] = (is_vis[obs_idx] and cam_not_inside_obs)

        # predict bounding boxes
        pc_all = self.process_input(observation)

        if (len(observation[0])>0):
            output, box_features = self.predict(pc_all)
            bb = self.match_gt_output_boxes(output, np.array(gt_obs), is_vis)
        else:
            bb, _ = self.get_room_size_box(pc_all, num_preds=self.num_chairs)
            output, box_features = self.get_room_size_box(pc_all, num_preds=self.num_boxes)

        matched_gt = self.match_output_gt_boxes(output, np.array(gt_obs), is_vis) 
        
        results = {
            'cam_not_inside_obs': cam_not_inside_obs,
            'is_vis': is_vis,
            'bb': bb,
            'box_features': box_features,
            'output': output,
            'matched_gt': matched_gt,
            }
        
        return results
    

    def process_input(self, 
                      observation: torch.Tensor,):
        if (len(observation[0])>0):
            # Preprocess point cloud (random sampling of points), if there are any LIDAR returns
            points_ = np.zeros_like(np.transpose(np.array(observation)).shape)
            points_ = pc_cam_to_3detr(np.transpose(np.array(observation)))
            points = np.zeros((1,self.num_pc_points, 3),dtype='float32')
            points = preprocess_point_cloud(np.array(points_), self.num_pc_points)

            pc = np.array(points).astype('float32')
            pc = pc.reshape((1, self.num_pc_points, 3))
            pc_all = torch.from_numpy(pc).to(self.device)
        else:
            # There are no returns from the camera, object is not visible
            points = np.zeros((1,self.num_pc_points, 3),dtype='float32')
            pc_all = torch.from_numpy(points).to(self.device)
        return pc_all
        

    def predict(self, 
                pc_all: torch.Tensor):
        
        pc_min_all = pc_all.min(1).values
        pc_max_all = pc_all.max(1).values
        inputs = {'point_clouds': pc_all, 'point_cloud_dims_min': pc_min_all, 'point_cloud_dims_max': pc_max_all}

        outputs = self.model(inputs)
        bbox_pred_points = outputs['outputs']['box_corners'].detach().cpu()
        cls_prob = outputs["outputs"]["sem_cls_prob"].clone().detach().cpu()
        box_features = outputs["box_features"].detach().cpu()

        chair_prob = cls_prob[:,:,3]
        obj_prob = outputs["outputs"]["objectness_prob"].clone().detach().cpu()
        sort_box = torch.sort(obj_prob,1,descending=True)

        num_probs = 0
        corners = np.zeros((self.num_boxes, 2,3))
        if np.any(np.isnan(np.array(bbox_pred_points))):
            return self.get_room_size_box(pc_all, self.num_boxes)
        
        for (sorted_idx,prob) in zip(list(sort_box[1][0,:]), list(sort_box[0][0,:])):
            if (num_probs < self.num_boxes):
                prob = prob.numpy()
                bbox = bbox_pred_points[0, sorted_idx, :, :]
                cc = pc_to_axis_aligned_rep(bbox.numpy())
                flag = False
                if num_probs == 0:
                    corners[num_probs,:,:] = cc
                    num_probs +=1
                else:
                    for cc_keep in corners:
                        bb1 = (cc_keep[0,0],cc_keep[0,1],cc_keep[1,0],cc_keep[1,1])
                        bb2 = (cc[0,0],cc[0,1],cc[1,0],cc[1,1])
                        # Non-maximal supression, check if IoU more than some threshold to keep box
                        if(box2d_iou(bb1,bb2) > 0.1):
                            flag = True
                    if not flag:    
                        corners[num_probs,:,:] = cc
                        num_probs +=1
        return corners, box_features
    
    def get_room_size_box(self, 
                          pc_all: torch.Tensor,
                          num_preds: int = 5):
        room_size = 8
        boxes = np.zeros((num_preds, 2,3))
        boxes[:,0,1] = 0*np.ones_like(boxes[:,0,0])
        boxes[:,0,0] = (-room_size/2)*np.ones_like(boxes[:,0,1])
        boxes[:,0,2] = 0*np.ones_like(boxes[:,0,2])
        boxes[:,1,1] = room_size*np.ones_like(boxes[:,1,0])
        boxes[:,1,0] = (room_size/2)*np.ones_like(boxes[:,1,1])
        boxes[:,1,2] = room_size*np.ones_like(boxes[:,1,2])

        pc_min_all = pc_all.min(1).values
        pc_max_all = pc_all.max(1).values
        inputs = {'point_clouds': pc_all, 'point_cloud_dims_min': pc_min_all, 'point_cloud_dims_max': pc_max_all}

        outputs = self.model(inputs)
        box_features = torch.zeros_like(outputs["box_features"]).detach().cpu()
        return boxes, box_features
    
    def match_gt_output_boxes(self,
                              output_boxes, 
                              ground_truth, 
                              is_visible):
        max_iou = torch.zeros(ground_truth.shape[0])
        center_diff = 100*torch.ones(ground_truth.shape[0])
        sorted_pred = np.copy(ground_truth)
        for j, val in enumerate(is_visible):
            if val:
                gt = (ground_truth[j,0,0], ground_truth[j,0,1], ground_truth[j,1,0], ground_truth[j,1,1])
                for kk in range(output_boxes.shape[0]):
                    pred_ = output_boxes[kk,:,:]
                    pred = (pred_[0,0], pred_[0,1], pred_[1,0], pred_[1,1])
                    iou = box2d_iou(pred, gt)
                    diff = ((((gt[2]+gt[0]-pred[2]-pred[0])**2) + (gt[3]+gt[1]-pred[3]-pred[1])**2)**0.5)/2
                    if iou > max_iou[j]:
                        max_iou[j] = iou
                        sorted_pred[j,:,:] = pred_
                        center_diff[j] = diff
                    elif iou == 0 and max_iou[j] == 0 and (center_diff[j] > diff):
                        # Centers of the predicted box are closer than before
                        center_diff[j] = diff
                        sorted_pred[j,:,:] = pred_
        return sorted_pred

    def match_output_gt_boxes(self, 
                              output_boxes, 
                              ground_truth, 
                              is_visible):
        max_iou = torch.zeros(output_boxes.shape[0])
        center_diff = 100*torch.ones(output_boxes.shape[0])
        sorted_pred = np.zeros_like(output_boxes)
        none_vis = True
        for kk in range(output_boxes.shape[0]):
            pred_ = output_boxes[kk,:,:]
            pred = (pred_[0,0], pred_[0,1], pred_[1,0], pred_[1,1])
            for j, val in enumerate(is_visible):
                if val:
                    none_vis = False
                    gt = (ground_truth[j,0,0], ground_truth[j,0,1], ground_truth[j,1,0], ground_truth[j,1,1])
                    iou = box2d_iou(pred, gt)
                    diff = ((((gt[2]+gt[0]-pred[2]-pred[0])**2) + (gt[3]+gt[1]-pred[3]-pred[1])**2)**0.5)/2
                    if iou > max_iou[kk]:
                        max_iou[kk] = iou
                        sorted_pred[kk,:,:] = ground_truth[j,:,:]
                        center_diff[kk] = diff
                    elif iou == 0 and max_iou[kk] == 0 and (center_diff[kk] > diff):
                        # Centers of the predicted box are closer than before
                        center_diff[kk] = diff
                        sorted_pred[kk,:,:] = ground_truth[j,:,:]
        return sorted_pred