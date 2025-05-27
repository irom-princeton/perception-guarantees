import numpy as np
import pickle
import scipy.optimize as opt
import plotly.graph_objects as go

from pwc.calibration.base_calibration import Calibration
from pwc.utils.loss_fn import diff_conditional_success

class PwCNuMCC(Calibration):
    """
    Perceive with Confidence (PwC) calibration approach for 
    and CP-avg. baseline.
    Works with occupancy predictor.
    """

    def __init__(self, name: str = "PwC-NU-MCC"):
        """
        Initialize the PwCNuMCC calibration method.

        Args:
            name (str): Name of the calibration method.
        """
        super().__init__(name)

    def calibrate(self, 
                  calibration_dataset_base_path: str = "/media/zm2074/Data Drive/data/perception-guarantees/numcc_calibration_data/data_confidence_4k_0130/",
                  epsilon: float = 0.15, 
                  delta: float = 0.01,
                  visualize: bool = True):
        """
        Calibrate the predictions based on the targets.

        Args:
            calibration_dataset_base_path (str): Path to the folder for calibration dataset.
            epsilon (float): desired epsilon = 1-coverage.
            delta (float): desired delta.
        """
        losses = []
        losses_avg = []

        for task_idx in range(300):
            data = pickle.load(open(f'{calibration_dataset_base_path}task_1210_{task_idx}.pkl', 'rb'))
            losses.append(np.max(data['thresholds']))
            losses_avg.append(np.mean(data['thresholds']))

        if visualize:
            # plot histogram of losses
            fig = go.Figure(data=[go.Histogram(x=losses)])
            fig.show()

        N = len(losses)
        epsilon_hat = opt.bisect(diff_conditional_success, epsilon/10, 0.5, args=(1-epsilon, N, delta))

        q_level = np.ceil((N+1)*(1-epsilon_hat))/N
        qhat = np.quantile(losses, q_level, method = 'higher')
        qhat_avg = np.quantile(losses_avg, q_level, method = 'higher')
        print(f'CP quantile prediction: {qhat}')
        print(f'CP quantile prediction (for baseline CP-avg.): {qhat_avg}')

        return qhat, qhat_avg