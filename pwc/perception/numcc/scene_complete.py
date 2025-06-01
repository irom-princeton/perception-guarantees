import os
import torch
from pathlib import Path

from src.engine.engine import prepare_data_udf
from src.engine.engine_viz import generate_html_udf

import main_numcc
import util.misc as misc
from src.engine.engine_viz import run_viz_udf
from util.hypersim_utils import read_h5py, read_img
from util.hypersim_dataset import random_crop, get_camera_pos_file_name_from_frame_name, get_camera_orientation_file_name_from_frame_name, read_scale_from_frame_name

from src.fns import *
from src.model.nu_mcc import NUMCC
import timm.optim.optim_factory as optim_factory
from util.misc import NativeScalerWithGradNormCount as NativeScaler

torch.cuda.empty_cache() 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

use_hypersim = True
run_vis = True
weights = '/home/zm2074/Projects/perception-guarantees/models/numcc/pretrained/numcc_hypersim_550c.pth'
n_groups = 550
blr = 5e-5

args = main_numcc.get_args_parser().parse_args(args=[])
args.resume = weights
args.use_hypersim = use_hypersim
args.run_vis = run_vis
args.n_groups = n_groups
args.blr = blr

def run_viz_udf(model, samples, device, args):
    model.eval()

    seen_xyz, valid_seen_xyz, query_xyz, unseen_rgb, labels, seen_images, gt_fps_xyz, seen_xyz_hr, valid_seen_xyz_hr = prepare_data_udf(samples, device, is_train=False, is_viz=True, args=args)

    seen_images_no_preprocess = seen_images.clone()


    with torch.no_grad():
        seen_images_hr = None
        
        if args.hr == 1:
            seen_images_hr = preprocess_img(seen_images.clone(), res=args.xyz_size)
            seen_xyz_hr = shrink_points_beyond_threshold(seen_xyz_hr, args.shrink_threshold)

        seen_images = preprocess_img(seen_images)
        query_xyz = shrink_points_beyond_threshold(query_xyz, args.shrink_threshold)
        seen_xyz = shrink_points_beyond_threshold(seen_xyz, args.shrink_threshold)

        if args.distributed:
            latent, up_grid_fea = model.module.encoder(seen_images, seen_xyz, valid_seen_xyz, up_grid_bypass=seen_images_hr)
            fea = model.module.decoderl1(latent)
        else:
            latent, up_grid_fea = model.encoder(seen_images, seen_xyz, valid_seen_xyz, up_grid_bypass=seen_images_hr)
            fea = model.decoderl1(latent)
        centers_xyz = fea['anchors_xyz']
    
    # don't forward all at once to avoid oom
    max_n_queries_fwd = args.n_query_udf if not args.hr else int(args.n_query_udf * (args.xyz_size/args.xyz_size_hr)**2)

    # Filter query based on centers xyz # (1, 200, 3)
    offset = 0.3
    min_xyz = torch.min(centers_xyz, dim=1)[0][0] - offset
    max_xyz = torch.max(centers_xyz, dim=1)[0][0] + offset

    mask = (torch.rand(1, query_xyz.size()[1]) >= 0).to(args.device)
    mask = mask & (query_xyz[:,:,0] > min_xyz[0]) & (query_xyz[:,:,1] > min_xyz[1]) & (query_xyz[:,:,2] > min_xyz[2])
    mask = mask & (query_xyz[:,:,0] < max_xyz[0]) & (query_xyz[:,:,1] < max_xyz[1]) & (query_xyz[:,:,2] < max_xyz[2])
    query_xyz = query_xyz[mask].unsqueeze(0)

    total_n_passes = int(np.ceil(query_xyz.shape[1] / max_n_queries_fwd))

    pred_points = np.empty((0,3))
    pred_colors = np.empty((0,3))

    if args.distributed:
        for param in model.module.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = False


    for p_idx in range(total_n_passes):
        p_start = p_idx     * max_n_queries_fwd
        p_end = (p_idx + 1) * max_n_queries_fwd
        cur_query_xyz = query_xyz[:, p_start:p_end]

        with torch.no_grad():
            if args.hr != 1:
                seen_points = seen_xyz
                valid_seen = valid_seen_xyz
            else:
                seen_points = seen_xyz_hr
                valid_seen = valid_seen_xyz_hr

            if args.distributed:
                pred = model.module.decoderl2(cur_query_xyz, seen_points, valid_seen, fea, up_grid_fea, custom_centers = None)
                pred = model.module.fc_out(pred)
            else:
                pred = model.decoderl2(cur_query_xyz, seen_points, valid_seen, fea, up_grid_fea, custom_centers = None)
                pred = model.fc_out(pred)

        max_dist = 0.5
        pred_udf = F.relu(pred[:,:,:1]).reshape((-1, 1)) # nQ, 1
        pred_udf = torch.clamp(pred_udf, max=max_dist) 

        

        # Candidate points
        t = args.udf_threshold
        pos = (pred_udf < t).squeeze(-1) # (nQ, )
        points = cur_query_xyz.squeeze(0) # (nQ, 3)
        points = points[pos].unsqueeze(0) # (1, n, 3)

        # print(pos)

        if torch.sum(pos) > 0:
            points = move_points(model, points, seen_points, valid_seen, fea, up_grid_fea, args, n_iter=args.udf_n_iter)

            # predict final color
            with torch.no_grad():
                if args.distributed:
                    pred = model.module.decoderl2(points, seen_points, valid_seen, fea, up_grid_fea)
                    pred = model.module.fc_out(pred)
                else:
                    pred = model.decoderl2(points, seen_points, valid_seen, fea, up_grid_fea)
                    pred = model.fc_out(pred)

            cur_color_out = pred[:,:,1:].reshape((-1, 3, 256)).max(dim=2)[1] / 255.0
            cur_color_out = cur_color_out.detach().squeeze(0).cpu().numpy()
            if len(cur_color_out.shape) == 1:
                cur_color_out = cur_color_out[None,...]
            pts = points.detach().squeeze(0).cpu().numpy()
            pred_points = np.append(pred_points, pts, axis = 0)
            pred_colors = np.append(pred_colors, cur_color_out, axis = 0)
        
    rank = misc.get_rank()
    out_folder = os.path.join('/home/zm2074/Projects/perception-guarantees/models/numcc/experiments/', f'{args.exp_name}', 'viz')
    Path(out_folder).mkdir(parents= True, exist_ok=True)
    prefix = os.path.join(out_folder, '1')
    img = (seen_images_no_preprocess[0].permute(1, 2, 0) * 255).cpu().numpy().copy().astype(np.uint8)

    # gt_xyz = samples[1][0].to(device).reshape(-1, 3)
    # gt_rgb = samples[1][1].to(device).reshape(-1, 3)
    #mesh_xyz = samples[2].to(device).reshape(-1, 3) if args.use_hypersim else None
    gt_xyz = None
    gt_rgb = None
    mesh_xyz = None

    fn_pc = None
    fn_pc_seen = None
    fn_pc_gt = None
    if args.save_pc:
        out_folder_ply = os.path.join('experiments/', f'{args.exp_name}', 'ply')
        Path(out_folder_ply).mkdir(parents= True, exist_ok=True)
        prefix_pc = os.path.join(out_folder_ply, '0')
        fn_pc = prefix_pc + '.ply'

        # seen
        out_folder_ply = os.path.join('experiments/', f'{args.exp_name}', 'ply_seen')
        Path(out_folder_ply).mkdir(parents= True, exist_ok=True)
        prefix_pc = os.path.join(out_folder_ply, '0')
        fn_pc_seen = prefix_pc +'_seen' +'.ply'

        # gt
        out_folder_ply = os.path.join('experiments/', f'{args.exp_name}', 'ply_gt')
        Path(out_folder_ply).mkdir(parents= True, exist_ok=True)
        prefix_pc = os.path.join(out_folder_ply, '0')
        fn_pc_gt = prefix_pc +'_gt' +'.ply'

    with open(prefix + '.html', 'a') as f:
        generate_html_udf(
            img,
            seen_xyz, seen_images_no_preprocess,
            pred_points,
            pred_colors,
            query_xyz,
            f,
            gt_xyz=gt_xyz,
            gt_rgb=gt_rgb,
            mesh_xyz=mesh_xyz,
            centers = centers_xyz,
            fn_pc=fn_pc,
            fn_pc_seen = fn_pc_seen,
            fn_pc_gt=fn_pc_gt
        )



def main(args):

    ######## LOAD MODEL ############
    misc.init_distributed_mode(args)

    model = NUMCC(args=args)
    model = model.to(args.device)
    model_without_ddp = model

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.blr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    ######## LOAD DATA ############
    
    frame_xyz_path = '/home/zm2074/Projects/perception-guarantees/models/numcc/ml-hypersim/ai_002_001/images/scene_cam_00_geometry_hdf5/frame.0000.position.hdf5'
    frame_path = '/home/zm2074/Projects/perception-guarantees/models/numcc/ml-hypersim/ai_002_001/images/scene_cam_00_final_preview/frame.0000.tonemap.jpg'
    xyz = read_h5py(frame_xyz_path)
    img = read_img(frame_path)
    
    xyz, img = random_crop(xyz, img, is_train=False)

    seen_data = [xyz, img]
    seen_frame = frame_path
    # get camera positions
    camera_positions = read_h5py(get_camera_pos_file_name_from_frame_name(seen_frame))
    camera_position = camera_positions[int(seen_frame.split('.')[-3])]

    # get camera orientations
    cam_orientations = read_h5py(get_camera_orientation_file_name_from_frame_name(seen_frame))
    cam_orientation = cam_orientations[int(seen_frame.split('.')[-3])]
    cam_orientation = cam_orientation * (-1.0)

    # rotate to camera direction
    seen_data[0] = torch.matmul(seen_data[0], cam_orientation)

    # shift to camera center
    camera_position = torch.matmul(camera_position, cam_orientation)
    seen_data[0] -= camera_position
    # to meter
    asset_to_meter_scale = read_scale_from_frame_name(seen_frame)
    seen_data[0] = seen_data[0] * asset_to_meter_scale

    seen_data[0][..., 0] *= -1
    seen_data[1] = seen_data[1].permute(2, 0, 1)
    gt_data = [torch.zeros(seen_data[0].shape), torch.zeros(seen_data[1].shape)]
    seen_data[0] = seen_data[0].unsqueeze(0)
    seen_data[1] = seen_data[1].unsqueeze(0)

    # print(seen_data[0].shape, gt_data[0].shape)
    
    
    samples = [
        seen_data,
        gt_data,
    ]

    run_viz_udf(model, samples, args.device, args)







if __name__ == '__main__':
    main(args)