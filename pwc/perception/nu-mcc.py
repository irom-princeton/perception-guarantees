import torch
from scipy.ndimage import median_filter, binary_closing

# numcc imports
from pwc.perception.numcc.src.engine.engine import prepare_data_udf

import pwc.perception.numcc.main_numcc as main_numcc
import pwc.perception.numcc.util.misc as misc
from pwc.perception.numcc.util.hypersim_dataset import random_crop

from pwc.perception.numcc.src.fns import *
from pwc.perception.numcc.src.model.nu_mcc import NUMCC
import timm.optim.optim_factory as optim_factory
from pwc.perception.numcc.util.misc import NativeScalerWithGradNormCount as NativeScaler
from pwc.perception.numcc.src.engine.engine_viz import generate_html_udf

from pwc.perception.perception_model import PerceptionModel

class PerceptionNUMCC(PerceptionModel):
    def __init__(self,
                 config,):
        self.model_name = "NU-MCC"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ckpt_path = config.ckpt_path
        self.udf_threshold = config.udf_threshold
        self.visualizez_pc = config.visualize_pc
        
        self.load_model()

    def load_model(self):
        # numcc args
        numcc_args = main_numcc.get_args_parser().parse_args(args=[])
        numcc_args.udf_threshold = self.udf_threshold
        numcc_args.resume = self.ckpt_path
        numcc_args.use_hypersim = True
        numcc_args.run_vis = True
        numcc_args.n_groups = 550
        numcc_args.blr = 5e-5
        numcc_args.save_pc = True
        numcc_args.device = torch.device('cuda')

        self.numcc_args = numcc_args

        ######## LOAD NUMCC MODEL ############
        misc.init_distributed_mode(numcc_args)

        model = NUMCC(args=numcc_args)
        model = model.to(self.device)
        model_without_ddp = model

        # following timm: set wd as 0 for bias and norm layers
        param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, numcc_args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=numcc_args.blr, betas=(0.9, 0.95))
        loss_scaler = NativeScaler()

        misc.load_model(args=numcc_args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

        model.eval()
        
        self.model = model

    def get_map(self, pc, cam_position):
        xyz = torch.tensor(pc[0])-cam_position
        # change coordinate system
        forward = xyz[:,:,0]
        left = xyz[:,:,1]
        up = xyz[:,:,2]

        # prep data for inference
        xyz = torch.tensor(torch.stack([-left, -up, forward], -1)).to(torch.float32)
        img = torch.tensor(pc[1]).permute(1,2,0).to(torch.float32) #w,h,3
        img = img / 255.0

        xyz_, img = random_crop(xyz, img, is_train=False)

        ######## LOAD DATA ############
        seen_data = [xyz_, img]

        gt_data = [torch.zeros(seen_data[0].shape), torch.zeros(seen_data[1].shape)]
        seen_data[1] = seen_data[1].permute(2, 0, 1)
        seen_data[0] = seen_data[0].unsqueeze(0)
        seen_data[1] = seen_data[1].unsqueeze(0)

        samples = [
            seen_data,
            gt_data,
        ]

        ######## RUN INFERENCE ############
        pred_xyz, seen_xyz = self.run_viz_udf(samples)
        cam_position_numcc = np.array([-cam_position[1], -cam_position[2], cam_position[0]])
        # print("Cam position numcc", cam_position_numcc)
        pred_points = pred_xyz + cam_position_numcc # back to sim frame
        seen_xyz = torch.nn.functional.interpolate(
            xyz[None].permute(0, 3, 1, 2), (112, 112),
            mode='bilinear',
        ).permute(0, 2, 3, 1)[0]
        seen_points = seen_xyz.squeeze(0).cpu().numpy().reshape(-1, 3) + cam_position_numcc

        all_points = np.concatenate([pred_points, seen_points], axis = 0)

        # pc_to_occupancy
        pc_for_occ = all_points
        good_points = pc_for_occ[:, 0] != -100

        if good_points.sum() != 0:
            # filter out ceiling and floor
            mask = (pc_for_occ[:, 1] > -2 ) & (pc_for_occ[:, 1] < -0.8)
            pc_for_occ = pc_for_occ[mask]
        # get rid of the middle dimension
        points_2d = pc_for_occ[:, [0, 2]] # right, forward

        # grid
        grid = np.zeros((83,83))
        grid_pitch = 8/82 # from get_room_from_3dfront.py
        min_x = -4
        min_y = 0

        indices = ((points_2d - np.array([min_x, min_y])) / grid_pitch).astype(int)
        if len(indices) > 0:
            indices = indices[(indices[:, 0] >= 0) & (indices[:, 0] < 83) & (indices[:, 1] >= 0) & (indices[:, 1] < 83)]
        grid[indices[:, 0], indices[:, 1]] = 1  # Mark as occupied
        grid = np.rot90(grid)

        bc1 = binary_closing(grid, np.ones((4,1))).astype(int)
        bc2 = binary_closing(grid, np.ones((1,4))).astype(int)
        bc = np.logical_or(bc1, bc2).astype(int)
        mf = median_filter(bc, size=2)
        grid = np.logical_or(grid, mf).astype(float)

        return grid


    def run_viz_udf(self, samples):

        seen_xyz, valid_seen_xyz, query_xyz, unseen_rgb, labels, seen_images, gt_fps_xyz, seen_xyz_hr, valid_seen_xyz_hr = prepare_data_udf(samples, self.device, is_train=False, is_viz=True, args=self.numcc_args)
        seen_images_no_preprocess = seen_images.clone()
        with torch.no_grad():
            seen_images_hr = None
            
            if self.numcc_args.hr == 1:
                seen_images_hr = preprocess_img(seen_images.clone(), res=self.numcc_args.xyz_size)
                seen_xyz_hr = shrink_points_beyond_threshold(seen_xyz_hr, self.numcc_args.shrink_threshold)

            seen_images = preprocess_img(seen_images)
            query_xyz = shrink_points_beyond_threshold(query_xyz, self.numcc_args.shrink_threshold)
            seen_xyz = shrink_points_beyond_threshold(seen_xyz, self.numcc_args.shrink_threshold)

            if self.numcc_args.distributed:
                latent, up_grid_fea = self.model.module.encoder(seen_images, seen_xyz, valid_seen_xyz, up_grid_bypass=seen_images_hr)
                fea = self.model.module.decoderl1(latent)
            else:
                latent, up_grid_fea = self.model.encoder(seen_images, seen_xyz, valid_seen_xyz, up_grid_bypass=seen_images_hr)
                fea = self.model.decoderl1(latent)
            centers_xyz = fea['anchors_xyz']
        
        # don't forward all at once to avoid oom
        max_n_queries_fwd = self.numcc_args.n_query_udf if not self.numcc_args.hr else int(self.numcc_args.n_query_udf * (self.numcc_args.xyz_size/self.numcc_args.xyz_size_hr)**2)

        # Filter query based on centers xyz # (1, 200, 3)
        offset = 0.3
        min_xyz = torch.min(centers_xyz, dim=1)[0][0] - offset
        max_xyz = torch.max(centers_xyz, dim=1)[0][0] + offset

        mask = (torch.rand(1, query_xyz.size()[1]) >= 0).to(self.device)
        mask = mask & (query_xyz[:,:,0] > min_xyz[0]) & (query_xyz[:,:,1] > min_xyz[1]) & (query_xyz[:,:,2] > min_xyz[2])
        mask = mask & (query_xyz[:,:,0] < max_xyz[0]) & (query_xyz[:,:,1] < max_xyz[1]) & (query_xyz[:,:,2] < max_xyz[2])
        query_xyz = query_xyz[mask].unsqueeze(0)

        total_n_passes = int(np.ceil(query_xyz.shape[1] / max_n_queries_fwd))

        if self.numcc_args.distributed:
            for param in self.model.module.parameters():
                param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = False
    
        all_pred_udf = []
        for p_idx in range(total_n_passes):        
            p_start = p_idx     * max_n_queries_fwd
            p_end = (p_idx + 1) * max_n_queries_fwd
            cur_query_xyz = query_xyz[:, p_start:p_end]

            with torch.no_grad():
                if self.numcc_args.hr != 1:
                    seen_points = seen_xyz
                    valid_seen = valid_seen_xyz
                else:
                    seen_points = seen_xyz_hr
                    valid_seen = valid_seen_xyz_hr

                if self.numcc_args.distributed:
                    pred = self.model.module.decoderl2(cur_query_xyz, seen_points, valid_seen, fea, up_grid_fea, custom_centers = None)
                    pred = self.model.module.fc_out(pred)
                else:
                    pred = self.model.decoderl2(cur_query_xyz, seen_points, valid_seen, fea, up_grid_fea, custom_centers = None)
                    pred = self.model.fc_out(pred)

            max_dist = 1 # 0.5
            pred_udf = F.relu(pred[:,:,:1]).reshape((-1, 1)) # nQ, 1
            pred_udf = torch.clamp(pred_udf, max=max_dist) 

            all_pred_udf.append(pred_udf)

        # nonlinearly transform all udfs
        all_pred_udf = torch.exp(torch.cat(all_pred_udf, dim=0).squeeze())

        # Candidate points
        t = self.udf_threshold
        pos = (all_pred_udf < t).squeeze(-1) # (nQ, )
        points = query_xyz.squeeze(0) # (nQ, 3)
        points = points[pos].unsqueeze(0) # (1, n, 3)
        
        pred_points = np.empty((0,3))
        if torch.sum(pos) > 0:
            # points = move_points(model, points, seen_points, valid_seen, fea, up_grid_fea, args, n_iter=args.udf_n_iter)
            pts = points.detach().squeeze(0).cpu().numpy()
            pred_points = np.append(pred_points, pts, axis = 0)
        
        img = (seen_images_no_preprocess[0].permute(1, 2, 0) * 255).cpu().numpy().copy().astype(np.uint8)
        if self.visualizez_pc:
            with open('nonlinear_scale.html', 'a') as f:
                generate_html_udf(
                    img,
                    seen_xyz, seen_images,
                    pred_points,
                    np.zeros_like(pred_points), # dummy
                    query_xyz,
                    f,
                    gt_xyz=None,
                    gt_rgb=None,
                    mesh_xyz=None,
                    centers = centers_xyz,
                    fn_pc=None,
                    fn_pc_seen = None,
                    fn_pc_gt=None
                )
            
        return pred_points, seen_xyz
    
    def count_misdetected(self, gt, pred):
        # ignore walls
        mask = np.zeros_like(gt) #TODO: hardcoded...
        mask[0:2, :] = 1
        mask[:, 0:2] = 1
        mask[-2:, :] = 1
        mask[:, -2:] = 1
        pix_loc = np.array([78,51]) # starting point
        # ignore initial position
        for i in range(-15, 16):
            for j in range(-15, 16):
                if (np.linalg.norm(np.array([i,j])) < 15 and 
                    pix_loc[0]+i >= 0 and pix_loc[0]+i < pred.shape[0] and 
                    pix_loc[1]+j >= 0 and pix_loc[1]+j < pred.shape[1]):
                    mask[pix_loc[0]+i, pix_loc[1]+j] = 1
        # mask out pred
        predicted_free = np.where((pred == 0.5) & (mask == 0))
        return np.sum(gt[predicted_free] == 1) > 0


        