#%%
import numpy as np
import torch
from torch.utils.data import DataLoader
import scipy.optimize as opt

from pwc.calibration.base_calibration import Calibration
from pwc.utils.pc_dataset import PointCloudDataset
from pwc.utils.loss_fn import diff_conditional_success, scale_prediction, scale_prediction_average

#%%

class PwC(Calibration):
    """
    Perceive with Confidence (PwC) calibration approach
    and CP-avg. Baseline approach.
    Works with bounding box predictor
    """

    def __init__(self, name: str = "PwC"):
        """
        Initialize the PwC calibration method.

        Args:
            name (str): Name of the calibration method.
        """
        super().__init__(name)

    def calibrate(self,
                  calibration_dataset_base_path: str = "/media/zm2074/Data Drive/data/perception-guarantees/PwC_calibration/calibrate_4k_rot/data/",
                  epsilon: float = 0.15,
                  delta: float = 0.01,):
        """
        Calibrate the predictions based on the targets.

        Args:
            calibration_dataset_base_path (str): Path to the folder for calibration dataset.
            epsilon (float): desired epsilon = 1-coverage.
            delta (float): desired delta.
        """
        # Initialize dataset and dataloader
        dataset = PointCloudDataset(calibration_dataset_base_path+'features.pt',
                                   calibration_dataset_base_path+'bbox_labels.pt',
                                   calibration_dataset_base_path+'loss_mask.pt')
        dataloader_cp = DataLoader(dataset, batch_size=len(dataset))
        
        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(0)

        #################################################################
        # Without finetuning
        for i, data in enumerate(dataloader_cp, 0):
            inputs, targets, loss_mask = data
            boxes_3detr = targets["bboxes_3detr"].to(device)
            boxes_gt = targets["bboxes_gt"].to(device)
            loss_mask = loss_mask.to(device)

            N = boxes_gt.shape[0]

            # Compute epsilon_hat
            epsilon_hat = opt.bisect(diff_conditional_success, epsilon/10, 0.5, args=(1-epsilon, N, delta))
            q_level = np.ceil((N+1)*(1-epsilon_hat))/N

            scaling_cp = scale_prediction(boxes_3detr, boxes_gt, loss_mask, q_level)
            average_cp = scale_prediction_average(boxes_3detr, boxes_gt, loss_mask, q_level)
            print('CP quantile prediction', scaling_cp)
            print('CP quantile prediction (for baseline CP-avg.)', average_cp)
        #################################################################
        return scaling_cp, average_cp

#%%
if __name__ == "__main__":
    pwc = PwC()
    pwc.calibrate(calibration_dataset_base_path="/media/zm2074/Data Drive/data/perception-guarantees/PwC_calibration/calibrate_2k/data/",
                  epsilon=0.0,
                  delta=0.01)
# %%
