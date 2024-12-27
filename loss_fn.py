'''
Implements loss function for box prediction.
'''

import torch
import numpy as np
import IPython as ipy
import math


def box_loss_diff(
    corners_pred: torch.Tensor,
    corners_gt: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    loss_mask: torch.Tensor
):
    """
    Box loss for optimization.
    Input:
        corners_pred: torch Tensor (B, K, 2, 3). Predicted.
        corners_gt: torch Tensor (B, K, 2, 3). Ground truth.
        Assumes that all boxes are axis-aligned.
        w1, w2, w3: weights on the three loss terms.
    Returns:
        B x K x 1  matrix of losses.
    """
    assert len(corners_gt.shape) == 4
    assert len(corners_pred.shape) == 4
    assert corners_gt.shape[2] == 2
    assert corners_gt.shape[3] == 3
    assert corners_gt.shape[0] == corners_pred.shape[0]
    assert corners_gt.shape[1] == corners_pred.shape[1]
    assert corners_gt.shape[2] == corners_pred.shape[2]
    assert corners_gt.shape[3] == corners_pred.shape[3]

    B, K = corners_gt.shape[0], corners_gt.shape[1]

    # Ensure that corners of predicted bboxes satisfy basic constraints
    corners1_pred = torch.min(corners_pred[:, :, 0, :][:,:,None,:], corners_pred[:, :, 1, :][:,:,None,:])
    corners2_pred = torch.max(corners_pred[:, :, 0, :][:,:,None,:], corners_pred[:, :, 1, :][:,:,None,:])


    # Calculate volume of ground truth and predicted boxes
    vol_gt = torch.prod(corners_gt[:, :, 1, :][:,:,None,:] - corners_gt[:, :, 0, :][:,:,None,:], 3)
    vol_pred = torch.prod(corners2_pred - corners1_pred, 3)

    # Calculate intersection between predicted and ground truth boxes
    corners1_int = torch.max(corners1_pred, corners_gt[:,:,0,:][:,:,None,:])
    corners2_int = torch.min(corners2_pred, corners_gt[:,:,1,:][:,:,None,:])
    corners_int_diff = (corners2_int - corners1_int).clamp(min=0)
    vol_int = torch.prod(corners_int_diff, 3)

    # Find smallest box that encloses predicted and ground truth boxes
    EPS = 1e-6
    corners1_enclosing = torch.min(corners1_pred, corners_gt[:,:,0,:][:,:,None,:])
    corners2_enclosing = torch.max(corners2_pred, corners_gt[:,:,1,:][:,:,None,:])
    corners_enclosing_diff = (corners2_enclosing - corners1_enclosing)
    vol_enclosing = torch.prod(corners_enclosing_diff, 3).clamp(min=EPS)

    # Compute volume of GT\PRED and PRED\GT
    vol_gt_minus_pred = vol_gt - vol_int
    vol_pred_minus_gt = vol_pred - vol_int

    # Compute volume of union
    vol_union = (vol_pred + vol_gt - vol_int).clamp(min=EPS)

    # Now compute all the terms in the loss
    l1 = vol_gt_minus_pred / vol_gt
    l2 = vol_pred_minus_gt / vol_pred
    l3 = (vol_enclosing - vol_union)/vol_enclosing

    losses = w1*l1 + w2*l2 + w3*l3

    # Mask loss in locations where object was not visible
    losses = torch.mul(loss_mask, losses.view((B, K)))

    # Take max across locations
    losses = losses.amax(dim=1)

    # Take mean across environments
    mean_loss = losses.mean()

    return mean_loss

box_loss_diff_jit = torch.jit.script(box_loss_diff)


def box_loss_true(
    corners_pred: torch.Tensor,
    corners_gt: torch.Tensor,
    loss_mask: torch.Tensor,
    tol
):
    """
    Box loss corresponding to enclosure of ground truth boxes.
    Input:
        corners_pred: torch Tensor (B, K, 2, 3). Predicted.
        corners_gt: torch Tensor (B, K, 2, 3). Ground truth.
        Assumes that all boxes are axis-aligned.
        loss_mask: mask on loss (based on whether object is visible).
        tol: tolerance on checking enclosure.
    Returns:
        Mean loss across environments. The loss for each env is 0 if all predicted boxes for that env encapsulate the
        ground truth box, and 0 otherwise.
    """
    assert len(corners_gt.shape) == 4
    assert len(corners_pred.shape) == 4
    assert corners_gt.shape[2] == 2
    assert corners_gt.shape[3] == 3
    assert corners_gt.shape[0] == corners_pred.shape[0]
    assert corners_gt.shape[1] == corners_pred.shape[1]
    assert corners_gt.shape[2] == corners_pred.shape[2]
    assert corners_gt.shape[3] == corners_pred.shape[3]

    B, K = corners_gt.shape[0], corners_gt.shape[1]

    # Ensure that corners of predicted bboxes satisfy basic constraints
    corners1_pred = torch.min(corners_pred[:, :, 0, :][:,:,None,:], corners_pred[:, :, 1, :][:,:,None,:])
    corners2_pred = torch.max(corners_pred[:, :, 0, :][:,:,None,:], corners_pred[:, :, 1, :][:,:,None,:])

    # Calculate volume of ground truth boxes
    vol_gt = torch.prod(corners_gt[:, :, 1, :][:,:,None,:] - corners_gt[:, :, 0, :][:,:,None,:], 3)

    # Calculate intersection between predicted and ground truth boxes
    corners1_int = torch.max(corners1_pred, corners_gt[:,:,0,:][:,:,None,:])
    corners2_int = torch.min(corners2_pred, corners_gt[:,:,1,:][:,:,None,:])
    corners_int_diff = (corners2_int - corners1_int).clamp(min=0)
    vol_int = torch.prod(corners_int_diff, 3)

    # Compute volume of GT\PRED
    vol_gt_minus_pred = vol_gt - vol_int

    # Check enclosure and take max loss across locations in each environment
    not_enclosed = (vol_gt_minus_pred/vol_gt > tol).float()

    # Mask loss (0 for locations where object was not visible)
    not_enclosed = torch.mul(loss_mask, not_enclosed.view((not_enclosed.shape[0], not_enclosed.shape[1])))

    # Take max loss across locations in each environment
    losses = not_enclosed.amax(dim=1)

    # Take mean across environments in the batch
    mean_loss = losses.mean()

    return mean_loss, not_enclosed

def scale_prediction(
    corners_pred: torch.Tensor,
    corners_gt: torch.Tensor,
    loss_mask: torch.Tensor,
    tol
):
    """
    Provides an estimate of how much to scale the predicted bounding box using conformal prediction
    Input:
        corners_pred: torch Tensor (B, K, num_pred, 2, 3). Predicted.
        corners_gt: torch Tensor (B, K, num_chairs, 2, 3). Ground truth.
        Assumes that all boxes are axis-aligned.
        loss_mask: mask on loss (based on whether object is visible).
        tol: tolerance on checking enclosure.
    Returns:
        The scaling factor (how much to increase or decrease the l, w, h of the BB prediction)
    """
    assert len(corners_gt.shape) == 5
    assert len(corners_pred.shape) == 5
    assert corners_gt.shape[3] == 2
    assert corners_gt.shape[4] == 3
    assert corners_gt.shape[0] == corners_pred.shape[0]
    assert corners_gt.shape[1] == corners_pred.shape[1]
    assert corners_gt.shape[2] == corners_pred.shape[2]
    assert corners_gt.shape[3] == corners_pred.shape[3]
    assert corners_gt.shape[4] == corners_pred.shape[4]

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
    breakpoint()
    corners1_diff_mask = corners1_diff_mask.amax(dim=2)
    corners2_diff_mask = corners2_diff_mask.amax(dim=2)
    delta_all = torch.maximum(corners1_diff_mask, corners2_diff_mask)

    delta = delta_all.amax(dim=1)
    delta, indices = torch.sort(delta, dim=0, descending=False)
    idx = math.ceil((B+1)*(tol))-1
    return delta[idx]


def scale_prediction_average(
    corners_pred: torch.Tensor,
    corners_gt: torch.Tensor,
    loss_mask: torch.Tensor,
    tol
):
    """
    Provides an estimate of how much to scale the predicted bounding box using conformal prediction
    Input:
        corners_pred: torch Tensor (B, K, num_pred, 2, 3). Predicted.
        corners_gt: torch Tensor (B, K, num_chairs, 2, 3). Ground truth.
        Assumes that all boxes are axis-aligned.
        loss_mask: mask on loss (based on whether object is visible).
        tol: tolerance on checking enclosure.
    Returns:
        The scaling factor (how much to increase or decrease the l, w, h of the BB prediction)
    """
    assert len(corners_gt.shape) == 5
    assert len(corners_pred.shape) == 5
    assert corners_gt.shape[3] == 2
    assert corners_gt.shape[4] == 3
    assert corners_gt.shape[0] == corners_pred.shape[0]
    assert corners_gt.shape[1] == corners_pred.shape[1]
    assert corners_gt.shape[2] == corners_pred.shape[2]
    assert corners_gt.shape[3] == corners_pred.shape[3]
    assert corners_gt.shape[4] == corners_pred.shape[4]

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
    corners1_diff_mask = torch.mul(loss_mask,corners1_diff.mean(dim=3))
    corners2_diff_mask = torch.mul(loss_mask, corners2_diff.mean(dim=3))
    corners1_diff_mask[loss_mask == 0] = 0
    corners2_diff_mask[loss_mask == 0] = 0
    # ipy.embed()
    n = torch.sum(loss_mask,2)
    n = torch.sum(n,1)
    n[n==0] = 1
    corners1_diff_mask = corners1_diff_mask.sum(dim=2)
    corners2_diff_mask = corners2_diff_mask.sum(dim=2)
    corners1_diff_mask = corners1_diff_mask.sum(dim=1)/n
    corners2_diff_mask = corners2_diff_mask.sum(dim=1)/n
    delta = torch.maximum(corners1_diff_mask, corners2_diff_mask)

    # delta = delta_all.amax(dim=1)
    delta, indices = torch.sort(delta, dim=0, descending=False)
    idx = math.ceil((B+1)*(tol))-1
    
    idx = math.ceil((B+1)*(tol))-1
    return delta[idx]

# if __name__=='__main__':
#
#     bboxes = np.load("test_bboxes.npz")
#     bbox1 = torch.tensor(bboxes["bbox1"])
#     bbox2 = torch.tensor(bboxes["bbox2"])
#
#     # ipy.embed()
#     # box_loss_tensor(bbox1, bbox2, 1, 1, 1)
#
#     bboxes1 = torch.cat((bbox1, bbox2), 0)
#     bboxes2 = torch.cat((bbox2, bbox1), 0)
#     box_loss_tensor(bboxes1, bboxes2, 1, 1, 1)

