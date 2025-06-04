import os
import wandb
from copy import deepcopy
import torch
from torch.utils.data import DataLoader, random_split
from omegaconf import OmegaConf

from pwc.calibration.base_calibration import Calibration

from pwc.perception.models.model_perception import MLPModel
from pwc.utils.loss_fn import box_loss_diff, box_loss_true
from pwc.utils.pc_dataset import PointCloudDataset
from pwc.utils.pac_util import PAC_Bayes_regularizer

class PACBayesBox(Calibration):
    """
    PAC-Bayes calibration approach.
    Trains a bounding box predictor on top of 3DETR output features.
    """

    def __init__(self, name: str = "PACBayes"):
        """
        Initialize the PAC-Bayes calibration method.

        Args:
            name (str): Name of the calibration method.
        """
        super().__init__(name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def calibrate(self, 
                  calibration_dataset_base_path: str = "/media/zm2074/Data Drive/data/perception-guarantees/PAC_Bayes_calibration/calibrate_4k_rot/data/",
                  config: OmegaConf = None):
        """
        Calibrate the predictions based on the targets.

        Args:
            calibration_dataset_base_path (str): Path to the folder for calibration dataset.
        """

        self.config = config
        self.w1 = torch.tensor(self.config.w1).to(self.device)
        self.w2 = torch.tensor(self.config.w2).to(self.device)
        self.w3 = torch.tensor(self.config.w3).to(self.device)

        if self.config.use_wandb:
            wandb.init(
                project="pac-bayes-calibration",
                name=self.name,
                config=self.config,
            )

        # --------Initialize dataset and dataloader--------
        dataset = PointCloudDataset(calibration_dataset_base_path+'features.pt',
                                calibration_dataset_base_path+'bbox_labels.pt',
                                calibration_dataset_base_path+'loss_mask.pt')
        
        prior_data, post_data = random_split(dataset, [len(dataset) - self.config.N, self.config.N])

        params = {'batch_size': self.config.batch_size,
                    'shuffle': False}

        loaders = {
            'prior': DataLoader(prior_data, **params),
            'post': DataLoader(post_data, **params)
        }

        # --------Model shape setup--------
        num_in = dataset.feature_dims[0]*dataset.feature_dims[1]
        num_out = (self.config.num_objects,2,3) # 5 boxess * bbox corner representation

        #--------Prior Model--------
        prior = MLPModel(num_in, num_out)
        prior.init_logvar(-10)
        prior.to(self.device)

        print("Trainig prior...")
        self.train(model=prior,
                   dataloader=loaders['prior'],
                   loss_fn=self.loss_prior,
                   training_config={
                       'num_epochs': self.config.prior_num_epochs,
                       'lr': self.config.prior_lr,})
        
        prior.init_logvar(-5)
        self.prior = prior
        
        # --------Posterior model--------

        posterior = MLPModel(num_in, num_out)
        posterior.load_state_dict(deepcopy(prior.state_dict()))
        posterior.to(self.device)

        self.train(model=posterior,
                   dataloader=loaders['post'],
                   loss_fn=self.loss_posterior,
                   training_config={
                       'num_epochs': self.config.posterior_num_epochs,
                       'lr': self.config.posterior_lr,})

        
        # Save model
        save_dir = calibration_dataset_base_path + "trained_models/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(posterior.state_dict(), "trained_models/perception_model")
        if self.config.verbose:
            print('Saved trained model.')
        ###################################################################


    def train(self,
              model: MLPModel,
              dataloader: DataLoader,
              loss_fn: callable,
              training_config: dict,):
        """
        Train a given model with given configs.
        """
        
        optimizer = torch.optim.Adam(model.parameters(), lr = training_config['lr'])

        for epoch in range(training_config['num_epochs']):

            #------Initialize running losses for this epoch------
            current_loss = 0.0
            current_loss_true = 0.0
            num_batches = 0

            #------Iterate over the DataLoader for training data------
            for i, data in enumerate(dataloader, 0):
                inputs, targets, loss_mask = data
                inputs = inputs.to(self.device)
                boxes_3detr = targets["bboxes_3detr"].to(self.device)
                boxes_gt = targets["bboxes_gt"].to(self.device)
                loss_mask = loss_mask.to(self.device)

                # forward pass
                model.init_xi()
                outputs = model(inputs)
                loss, loss_true = loss_fn(model, outputs, boxes_3detr, boxes_gt, loss_mask) #prior, N, delta, device stored in self

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update current loss for this epoch (summing across batches)
                current_loss += loss.item()
                current_loss_true += loss_true.item()
                num_batches += 1
            
            #------Optional: log to wandb, print------
            if self.config.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "loss/train": current_loss / num_batches,
                    "loss/true": current_loss_true / num_batches,
                })
            print_interval = 1
            if self.config.verbose and (epoch % print_interval == 0):
                print("epoch: ", epoch, "; loss: ", '{:02.6f}'.format(current_loss/num_batches),
                    "; loss true: ", '{:02.6f}'.format(current_loss_true / num_batches), end='\r')


    def loss_prior(self,
                   model: MLPModel,
                   outputs: torch.Tensor,
                   boxes_3detr: torch.Tensor,
                   boxes_gt: torch.Tensor,
                   loss_mask: torch.Tensor):
        """
        Compute the loss for the prior model.
        """
        loss = box_loss_diff(outputs + boxes_3detr, boxes_gt, self.w1, self.w2, self.w3, loss_mask)
        loss_true, not_enclosed = box_loss_true(outputs + boxes_3detr, boxes_gt, loss_mask, 0.01)

        return loss, loss_true
    
    def loss_posterior(self,
                       model: MLPModel,
                       outputs: torch.Tensor,
                       boxes_3detr: torch.Tensor,
                       boxes_gt: torch.Tensor,
                       loss_mask: torch.Tensor):
        """
        Compute the loss for the posterior model.
        """

        loss = box_loss_diff(outputs + boxes_3detr, boxes_gt, self.w1, self.w2, self.w3, loss_mask)
        loss_true, not_enclosed = box_loss_true(outputs + boxes_3detr, boxes_gt, loss_mask, 0.01)

        reg = PAC_Bayes_regularizer(model, self.prior, self.N, self.delta, self.device)
        loss += torch.sqrt(reg / 2)

        return loss, loss_true

#%%
if __name__ == "__main__":
    calibrator = PACBayesBox()

    config = OmegaConf.create({
        "batch_size": 100,
        "learning_rate": 1e-4,
        "prior_num_epochs": 100,
        "prior_lr": 0.01,
        "posterior_num_epochs": 1000,
        "posterior_lr": 1e-4,
        "w1": 1.0,
        "w2": 0.1,
        "w3": 1.0,
        "N": 350,  # Number of samples for PAC-Bayes
        "num_objects": 5,  # Number of objects in the dataset
        "delta": 0.01,  # Confidence level for PAC-Bayes
        "use_wandb": False,
        "verbose": True
    })

    calibrator.calibrate(
        calibration_dataset_base_path="/media/zm2074/Data Drive/data/perception-guarantees/PwC_calibration/calibrate_2k/data/",
        config=config
    )
