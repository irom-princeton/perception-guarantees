import os
import sys
import torch
import IPython as ipy
import time
import math

# Point cloud IO
import numpy as np
from plyfile import PlyData, PlyElement

# Mesh IO
import trimesh

import matplotlib.pyplot as pyplot


# ----------------------------------------
# Point Cloud Sampling
# ----------------------------------------


def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """Input is NxC, output is num_samplexC"""
    if replace is None:
        replace = pc.shape[0] < num_sample
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]


def preprocess_point_cloud(point_cloud, num_points_to_sample):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:,0:3] # do not use color for now
    # floor_height = np.percentile(point_cloud[:,2],0.99)
    # height = point_cloud[:,2] - floor_height
    # point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, num_points_to_sample)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,3)
    return pc



# ----------------------------------------
# Simple Point manipulations
# ----------------------------------------
def shift_scale_points(pred_xyz, src_range, dst_range=None):
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    if dst_range is None:
        dst_range = [
            torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device),
            torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),
        ]

    if pred_xyz.ndim == 4:
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]

    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape

    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
    prop_xyz = (
        ((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff
    ) + dst_range[0][:, None, :]
    return prop_xyz


def scale_points(pred_xyz, mult_factor):
    if pred_xyz.ndim == 4:
        mult_factor = mult_factor[:, None]
    scaled_xyz = pred_xyz * mult_factor[:, None, :]
    return scaled_xyz


def rotate_point_cloud(points, rotation_matrix=None):
    """Input: (n,3), Output: (n,3)"""
    # Rotate in-place around Z axis.
    if rotation_matrix is None:
        rotation_angle = np.random.uniform() * 2 * np.pi
        sinval, cosval = np.sin(rotation_angle), np.cos(rotation_angle)
        rotation_matrix = np.array(
            [[cosval, sinval, 0], [-sinval, cosval, 0], [0, 0, 1]]
        )
    ctr = points.mean(axis=0)
    rotated_data = np.dot(points - ctr, rotation_matrix) + ctr
    return rotated_data, rotation_matrix


def rotate_pc_along_y(pc, rot_angle):
    """Input ps is NxC points with first 3 channels as XYZ
    z is facing forward, x is left ward, y is downward
    """
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def roty_batch(t):
    """Rotation about the y-axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = np.zeros(tuple(list(input_shape) + [3, 3]))
    c = np.cos(t)
    s = np.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 2] = s
    output[..., 1, 1] = 1
    output[..., 2, 0] = -s
    output[..., 2, 2] = c
    return output


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def point_cloud_to_bbox(points):
    """Extract the axis aligned box from a pcl or batch of pcls
    Args:
        points: Nx3 points or BxNx3
        output is 6 dim: xyz pos of center and 3 lengths
    """
    which_dim = len(points.shape) - 2  # first dim if a single cloud and second if batch
    mn, mx = points.min(which_dim), points.max(which_dim)
    lengths = mx - mn
    cntr = 0.5 * (mn + mx)
    return np.concatenate([cntr, lengths], axis=which_dim)


def write_bbox(scene_bbox, out_filename):
    """Export scene bbox to meshes
    Args:
        scene_bbox: (N x 6 numpy array): xyz pos of center and 3 lengths
        out_filename: (string) filename

    Note:
        To visualize the boxes in MeshLab.
        1. Select the objects (the boxes)
        2. Filters -> Polygon and Quad Mesh -> Turn into Quad-Dominant Mesh
        3. Select Wireframe view.
    """

    def convert_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file
    # trimesh.io.export.export_mesh(mesh_list, out_filename, file_type="ply")
    mesh_list.export(out_filename, file_type="ply")

    return


def write_oriented_bbox(scene_bbox, out_filename, colors=None):
    """Export oriented (around Z axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename: (string) filename
    """

    def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3, 3))
        rotmat[2, 2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    if colors is not None:
        if colors.shape[0] != len(scene_bbox):
            colors = [colors for _ in range(len(scene_bbox))]
            colors = np.array(colors).astype(np.uint8)
        assert colors.shape[0] == len(scene_bbox)
        assert colors.shape[1] == 4

    scene = trimesh.scene.Scene()
    for idx, box in enumerate(scene_bbox):
        box_tr = convert_oriented_box_to_trimesh_fmt(box)
        if colors is not None:
            box_tr.visual.main_color[:] = colors[idx]
            box_tr.visual.vertex_colors[:] = colors[idx]
            for facet in box_tr.facets:
                box_tr.visual.face_colors[facet] = colors[idx]
        scene.add_geometry(box_tr)

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file
    # trimesh.io.export.export_mesh(mesh_list, out_filename, file_type="ply")
    mesh_list.export(out_filename, file_type="ply")


    return


#################################################################################
# Newly added functions

def pc_to_axis_aligned_rep(points):
    """
    Args:
        points: N x 3 points or B x N x 3
    Output:
        (2,3) array: (x1,y1,z1), (x2,y2,z2) corresponding to opposing corners of axis-aligned bounding box
    """

    which_dim = len(points.shape) - 2  # first dim if a single cloud and second if batch

    # Vertex corresponding to min (x,y,z) and max (x,y,z)
    vert_min = points.min(which_dim)
    vert_max = points.max(which_dim)

    if len(points.shape) == 3: # Batch
        B = points.shape[0] # Batch size
        vert_min = vert_min.reshape((B, 1, 3))
        vert_max = vert_max.reshape((B, 1, 3))

        # Return (B,2,3) array
        verts = np.concatenate((vert_min, vert_max), 1)
    else: # No batch
        vert_min = vert_min.reshape((1,3))
        vert_max = vert_max.reshape((1,3))
        verts = np.concatenate((vert_min, vert_max), 0)

    return verts


def write_oriented_bbox_ply_from_outputs(outputs, out_filename, prob_threshold=0.5):
    '''
    Write ply file corresponding to bounding boxes using outputs of 3DETR model.
    Args:
        outputs: outputs from 3DETR model.
        out_filename: name of ply file to write.
        prob_threshold: probability above which we consider something an object.
    '''

    # Parse outputs
    centers = outputs["outputs"]["center_unnormalized"] 
    centers = centers.cpu().detach().numpy()
    lengths = outputs["outputs"]["size_unnormalized"]
    lengths = lengths.cpu().detach().numpy()

    inds = outputs["outputs"]["objectness_prob"] > prob_threshold
    inds = inds.cpu()
    inds = inds[0, :]

    centers = centers[:,inds,:]
    lengths = lengths[:,inds,:]

    angles = outputs["outputs"]["angle_continuous"]
    angles = angles[:,inds]
    angles = angles.cpu().detach().numpy()

    # ipy.embed()
    scene_bbox = np.concatenate((centers, lengths), 2)
    scene_bbox = scene_bbox[0,:,:]

    scene_bbox = np.concatenate((scene_bbox, angles.T), 1)

    write_oriented_bbox(scene_bbox, out_filename)

    # Number of objects (detected above prob_threshold)
    num_objects = inds.sum().item()

    return num_objects



def write_oriented_bbox_camera_coord(scene_bbox, out_filename):
    """Export oriented (around Y axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Y axis.
            Z forward, X rightward, Y downward. heading angle of positive X is 0,
            heading angle of negative Z is 90 degrees.
        out_filename: (string) filename
    """

    def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3, 3))
        rotmat[1, 1] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0, :] = np.array([cosval, 0, sinval])
        rotmat[2, :] = np.array([-sinval, 0, cosval])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file
    # trimesh.io.export.export_mesh(mesh_list, out_filename, file_type="ply")
    mesh_list.export(out_filename, file_type="ply")

    return


def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

def read_ply_realsense(filename):
    """ read XYZ point cloud from filename PLY file from RealSense """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data

    # Change to correct coordinate system:
    # Real-Sense: +X (right), +Y (down),    +Z (forward) [But export to ply is different; correction below]
    # 3DETR:    +X (right),   +Y (forward), +Z (up) 
    pc_array = np.array([[x, -z, y, r, g, b] for x,y,z,r,g,b in pc])

    write_ply_rgb(pc_array[:,0:3], pc_array[:,3:6], "input_pc_realsense_transformed.obj")

    pc_array = pc_array[:,0:3] 


    return pc_array

def pc_cam_to_gibson(points_in_cam_frame, cam_pos):
    '''
    Convert point cloud from camera coordinate system to Gibson world coordinate system.
    ASSUMPTION: Camera is facing in the +Y world direction.  
    Camera: +X (right), +Y (up),    +Z (backward)
    Gibson world:    +X (right),   +Y (forward), +Z (up) 
    Args:
        points_in_cam_frame: Nx3
        cam_pos: (3,)
    '''
    
    # Convert directions and add cam_pos
    
    # points_new = [(points_in_cam_frame[i,0], -points_in_cam_frame[i,2], points_in_cam_frame[i,1]) for i in range(points_in_cam_frame.shape[0])]

    points_new = np.zeros_like(points_in_cam_frame)
    points_new[:,0] = points_in_cam_frame[:,0] + cam_pos[0]
    points_new[:,1] = -points_in_cam_frame[:,2] + cam_pos[1]
    points_new[:,2] = points_in_cam_frame[:,1] + cam_pos[2]

    return points_new

def pc_cam_to_3detr(points_in_cam_frame):
    '''
    Convert point cloud from camera coordinate system to 3DETR world coordinate system.
    ASSUMPTION: Camera is facing in the +X world direction.  
    Camera: +X (forward), +Y (left),    +Z (upward)
    Gibson world:    +X (right),   +Y (forward), +Z (up) 
    Args:
        points_in_cam_frame: Nx3
        cam_pos: (3,)
    '''
    
    # Convert directions and add cam_pos
    
    # points_new = [(points_in_cam_frame[i,1], -points_in_cam_frame[i,0], points_in_cam_frame[i,2]) for i in range(points_in_cam_frame.shape[0])]

    points_new = np.zeros_like(points_in_cam_frame)
    points_new[:,0] = -points_in_cam_frame[:,1]
    points_new[:,1] = points_in_cam_frame[:,0]
    points_new[:,2] = points_in_cam_frame[:,2]

    return points_new


def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

def write_ply_rgb(points, colors, out_filename, num_classes=None):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as OBJ file """
    colors = colors.astype(int)
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i,:]
        fout.write('v %f %f %f %d %d %d\n' % (points[i,0], points[i,1], points[i,2],c[0],c[1],c[2]))
    fout.close()

def is_inside_camera_fov(state, obstacles, fov):
    """" find whether the obstacle is within FOV of camera
         input: state (x,y,orientation)
                obstacles (list of bounding boxes of all obstacles)
                fov (horizontal fov in degrees, it is the same as the vertical fov if aspect = 1)
    """
    fov = math.radians(fov)
    # Find the range of the camera FOV in min and max angles
    rng = np.mod([2*np.pi - (fov/2) + state[2], 2*np.pi + (fov/2) + state[2]],2*np.pi)
    # Find the angle between the camera and the obstacle center 
    angles = np.mod([2*np.pi + np.arctan2(((obs[1]+obs[4])/2)-state[1], ((obs[0]+obs[3])/2)-state[0])for obs in obstacles], 2*np.pi)
    # Check if the obstacle is within the FOV of the camera
    is_vis = [1 if ((ang >= rng[0] and ang <= rng[1]) or 
                    (rng[0] > rng[1] and ang > np.pi and ang-2*np.pi >= rng[0]-2*np.pi and ang-2*np.pi <= rng[1]) or 
                    (rng[0] > rng[1] and ang >= rng[0]-2*np.pi and ang <= rng[1]))  else 0 for ang in angles]
    # print(is_vis, rng, angles)
    # print("Obstacle", obstacles)
    # print("Cam: ", state)
    return is_vis

# def is_inside_camera_fov(state, pc, fov):
#     """" find whether the obstacle is within FOV of camera
#          input: state (x,y,orientation)
#                 obstacles (list of bounding boxes of all obstacles)
#                 fov (horizontal fov in degrees, it is the same as the vertical fov if aspect = 1)
#     """
#     fov = math.radians(fov)
#     # Find the range of the camera FOV in min and max angles
#     rng = np.mod([2*np.pi - (fov/2) + state[2], 2*np.pi + (fov/2) + state[2]],2*np.pi)
#     # Find the angle between the camera and the obstacle center 
#     angles = np.mod([2*np.pi + np.arctan2(((obs[1]+obs[4])/2)-state[1], ((obs[0]+obs[3])/2)-state[0])for obs in obstacles], 2*np.pi)
#     # Check if the obstacle is within the FOV of the camera
#     is_vis = [1 if ((ang >= rng[0] and ang <= rng[1]) or 
#                     (rng[0] > rng[1] and ang > np.pi and ang-2*np.pi >= rng[0]-2*np.pi and ang-2*np.pi <= rng[1]) or 
#                     (rng[0] > rng[1] and ang >= rng[0]-2*np.pi and ang <= rng[1]))  else 0 for ang in angles]
#     # print(is_vis, rng, angles)
#     # print("Obstacle", obstacles)
#     # print("Cam: ", state)
#     return is_vis





