import os
import wandb
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from omegaconf import OmegaConf
from pathlib import Path

from pwc.calibration.base_calibration import Calibration

from pwc.perception.models.model_inflation import InflationModel
from pwc.utils.loss_fn import box_loss_diff, box_loss_true
from pwc.utils.pc_dataset import PointCloudDataset
from pwc.utils.pac_util import PAC_Bayes_regularizer

class PACBayesScalar(Calibration):
    """
    PAC-Bayes calibration approach.
    Trains a bounding box predictor on top of 3DETR output features.
    """

    def __init__(self, 
                 name: str = "PACBayes-scalar",
                 config: OmegaConf = None):
        """
        Initialize the PAC-Bayes calibration method.

        Args:
            name (str): Name of the calibration method.
        """
        super().__init__(name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.config = config
        if 'runtime_config' in config:
            self.runtime_config = config.runtime_config
            #--------Load model--------#
            state_dict = torch.load(self.runtime_config.trained_model_path, map_location=self.device)
            self.model = InflationModel(weight_size=config.weight_size)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

    def calibrate(self, 
                  config: OmegaConf = None):
        """
        Calibrate the predictions based on the targets.

        Args:
            calibration_dataset_base_path (str): Path to the folder for calibration dataset.
        """

        if config is not None:
            self.training_config = config
        else:
            self.training_config = self.config.training_config
        self.w1 = torch.tensor(self.training_config.w1).to(self.device)
        self.w2 = torch.tensor(self.training_config.w2).to(self.device)
        self.w3 = torch.tensor(self.training_config.w3).to(self.device)

        calibration_dataset_base_path = self.training_config.dataset_base_path

        if self.training_config.use_wandb:
            wandb.init(
                project="PwC-PAC-0613",
                name=self.training_config.save_name,
                config={**self.training_config},
            )

        # --------Initialize dataset and dataloader--------
        dataset = PointCloudDataset(calibration_dataset_base_path+'features.pt',
                                calibration_dataset_base_path+'bbox_labels.pt',
                                calibration_dataset_base_path+'loss_mask.pt')
        
        subset = Subset(dataset, range(self.training_config.N_total))  # Use only the first N_total samples
        dataset = subset.dataset
        
        prior_data, post_data = random_split(dataset, [len(dataset) - self.training_config.N, self.training_config.N])

        params = {'batch_size': self.training_config.batch_size,
                    'shuffle': False}

        loaders = {
            'prior': DataLoader(prior_data, **params),
            'post': DataLoader(post_data, **params)
        }

        #--------Create save directory--------
        save_dir = Path(__file__).parents[1] / "perception" / "models" / "trained_models"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        #--------Prior Model--------
        prior = InflationModel(self.config.weight_size)
        prior.init_logvar(-4)
        # prior.init_mu(0.5) # initializing mu
        prior.init_mu(0.05) # initializing mu
        prior.to(self.device)

        print("Training prior...")
        self.train(model=prior,
                   dataloader=loaders['prior'],
                   loss_fn=self.loss_prior,
                   training_config={
                       'num_epochs': self.training_config.prior_num_epochs,
                       'lr': self.training_config.prior_lr,})
        
        self.prior = prior
        print(f"Prior inflation: {prior.layer.mu}")
        torch.save(prior.state_dict(), f"{save_dir}/{self.training_config.save_name}_prior.pth")
        if self.training_config.verbose:
            print(f'Saved trained prior model to {save_dir}/{self.training_config.save_name}_prior.pth.')
        
        # --------Posterior model--------

        posterior = InflationModel(self.config.weight_size)
        posterior.load_state_dict(deepcopy(prior.state_dict()))
        posterior.to(self.device)
        print("Training posterior...")
        self.train(model=posterior,
                   dataloader=loaders['post'],
                   loss_fn=self.loss_posterior,
                   training_config={
                       'num_epochs': self.training_config.posterior_num_epochs,
                       'lr': self.training_config.posterior_lr,})

        print(f"Posterior inflation: {posterior.layer.mu}")
        
        # Save model

        torch.save(posterior.state_dict(), f"{save_dir}/{self.training_config.save_name}_posterior.pth")
        if self.training_config.verbose:
            print(f'Saved trained posterior model to {save_dir}/{self.training_config.save_name}_posterior.pth.')
        


    def train(self,
              model: InflationModel,
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
                outputs = model(boxes_gt.shape[:3]) # B x K x N

                # inflation_factor = outputs*torch.ones_like(boxes_3detr)
                # inflation_factor[:,:,:,0,:] *= -1.0
                inflation_factor = outputs[..., None, None] * torch.ones_like(boxes_3detr)
                inflation_factor[:,:,:,0,:] *= -1.0

                loss, loss_true = loss_fn(model, inflation_factor, boxes_3detr, boxes_gt, loss_mask) #prior, N, delta, device stored in self

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update current loss for this epoch (summing across batches)
                current_loss += loss.item()
                current_loss_true += loss_true.item()
                num_batches += 1
            
            #------Optional: log to wandb, print------
            if self.training_config.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "loss/train": current_loss / num_batches,
                    "loss/true": current_loss_true / num_batches,
                    "mu": model.layer.mu.item(),
                    "logvar": model.layer.logvar.item(),
                })
            print_interval = 1
            if self.training_config.verbose and (epoch % print_interval == 0):
                print(f"epoch: {epoch}; loss: {current_loss/num_batches:02.6f}; loss true: {current_loss_true/num_batches:02.6f}; mu: {model.layer.mu.item():02.6f}", end='\r')


    def loss_prior(self,
                   model: None,
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
                       model: InflationModel,
                       outputs: torch.Tensor,
                       boxes_3detr: torch.Tensor,
                       boxes_gt: torch.Tensor,
                       loss_mask: torch.Tensor):
        """
        Compute the loss for the posterior model.
        """

        loss = box_loss_diff(outputs + boxes_3detr, boxes_gt, self.w1, self.w2, self.w3, loss_mask)
        loss_true, not_enclosed = box_loss_true(outputs + boxes_3detr, boxes_gt, loss_mask, 0.01)
        
        # breakpoint()
        # TODO: To be deleted...
        # hj = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
        # hj = torch.autograd.grad(reg, model.parameters(), retain_graph=True)
        
        reg = PAC_Bayes_regularizer(model, self.prior, self.training_config.N, self.training_config.delta, self.device)
        # loss += torch.sqrt(reg / 2)
        # breakpoint()
        loss += torch.sqrt(reg / self.training_config.N)
        
        return loss, loss_true
    
    def evaluate(self):
        """
        Evaluate the bound on the trained model.
        """
        self.training_config = self.config.training_config
        self.w1 = torch.tensor(self.training_config.w1).to(self.device)
        self.w2 = torch.tensor(self.training_config.w2).to(self.device)
        self.w3 = torch.tensor(self.training_config.w3).to(self.device)

        calibration_dataset_base_path = self.training_config.dataset_base_path

        # --------Initialize dataset and dataloader--------
        print("Loading calibration dataset...")
        dataset = PointCloudDataset(calibration_dataset_base_path+'features.pt',
                                calibration_dataset_base_path+'bbox_labels.pt',
                                calibration_dataset_base_path+'loss_mask.pt')
        
        subset = Subset(dataset, range(self.training_config.N_total))  # Use only the first N_total samples
        loader = DataLoader(subset, batch_size=self.training_config.N_total, shuffle=False)

        # --------Load prior and posterior models--------
        save_dir = Path(__file__).parents[1] / "perception" / "models" / "trained_models"
        
        print(f"Loading prior...")
        prior = InflationModel(self.config.weight_size)
        prior.load_state_dict(torch.load(f"{save_dir}/{self.training_config.save_name}_prior.pth", map_location=self.device))
        prior.to(self.device)
        prior.eval()

        print(f"Loading posterior...")
        posterior = InflationModel(self.config.weight_size)
        posterior.load_state_dict(torch.load(f"{save_dir}/{self.training_config.save_name}_posterior.pth", map_location=self.device))
        posterior.to(self.device)
        posterior.eval()

        #--------Evaluate the bound--------
        bound = self.compute_bound(
            dataloader=loader,
            posterior=posterior,
            prior=prior,
            # loss_fn=self.loss_coverage,
            loss_fn=self.loss_prior,
        )
        print(f"PAC-Bayes bound: {bound.item():.4f}")


    def compute_bound(self,
                      dataloader: DataLoader,
                      posterior: InflationModel,
                      prior: InflationModel,
                      loss_fn: callable,):
        """
        Compute the PAC-Bayes bound for the posterior model.
        """
        # PAC-Bayes regularizer
        reg = PAC_Bayes_regularizer(posterior, prior, self.training_config.N, self.training_config.delta, self.device)
        
        # Training loss
        for i, data in enumerate(dataloader, 0):
            inputs, targets, loss_mask = data
            inputs = inputs.to(self.device)
            boxes_3detr = targets["bboxes_3detr"].to(self.device)
            boxes_gt = targets["bboxes_gt"].to(self.device)
            loss_mask = loss_mask.to(self.device)

            # forward pass
            outputs = posterior(boxes_gt.shape[:3]) # B x K x N
            inflation_factor = outputs[..., None, None] * torch.ones_like(boxes_3detr)
            inflation_factor[:,:,:,0,:] *= -1.0

            # not_enclosed = loss_fn(prior, inflation_factor, boxes_3detr, boxes_gt, loss_mask) #prior, N, delta, device stored in self
            # take max over states and objects
            # not_enclosed = not_enclosed.amax(dim=1).amax(dim=1)  # B
            train_loss, _ = loss_fn(posterior, inflation_factor, boxes_3detr, boxes_gt, loss_mask) #prior, N, delta, device stored in self

        # train_loss = not_enclosed.sum().item() / len(not_enclosed)  # average over batch
            
        # Compute the PAC-Bayes bound
        # bound = train_loss + torch.sqrt(reg / 2)
        regularizer = torch.sqrt(reg / self.training_config.N)
        bound = train_loss + regularizer
        print(f"Train loss: {train_loss:.4f}, Regularizer: {regularizer.item():.4f}, Bound: {bound.item():.4f}")
        return bound
    
    def loss_coverage(self,
                      model: None,
                      outputs: torch.Tensor,
                      boxes_3detr: torch.Tensor,
                      boxes_gt: torch.Tensor,
                      loss_mask: torch.Tensor):
        """
        Compute the coverage loss.
        """
        mean_loss, not_enclosed = box_loss_true(outputs + boxes_3detr, boxes_gt, loss_mask, 0.01)
        return not_enclosed
    
    def calibrate_runtime(self,
                          corners: torch.Tensor,
                          _ = None):

        boxes_3detr = torch.Tensor(np.array(corners)).squeeze()
        boxes_3detr = boxes_3detr.to(self.device)

        self.model.init_xi()
        outputs = self.model(boxes_3detr.shape[:1]) # B x K x N # sample inflation factor

        inflation_factor = outputs[..., None, None] * torch.ones_like(boxes_3detr)
        inflation_factor[:,0,:] *= -1.0
        
        #--------Combine with 3DETR boxes--------
        inflated_corners = inflation_factor + boxes_3detr
        inflated_corners = inflated_corners.cpu().detach().numpy().squeeze()

        boxes = np.zeros((len(corners),2,2))
        for i in range(len(corners)):
            # boxes[i,:,:] = corners[i][0,:,0:2]
            boxes[i,:,0] = corners[i,:,1]
            boxes[i,0,1] = -corners[i,1,0]
            boxes[i,1,1] = -corners[i,0,0]
    
        return boxes
    
    def calibrate_runtime_pwc(self,
                          boxes_3detr: torch.Tensor,
                          config: OmegaConf = None):
        if config is not None:
            self.runtime_config = config

        boxes_3detr = boxes_3detr.to(self.device)

        self.model.init_xi()
        outputs = self.model(boxes_3detr.shape[:1]) # B x K x N # sample inflation factor
        outputs = torch.ones_like(outputs) * 0.75

        inflation_factor = outputs[..., None, None] * torch.ones_like(boxes_3detr)
        inflation_factor[:,0,:] *= -1.0
        
        #--------Combine with 3DETR boxes--------
        boxes = inflation_factor + boxes_3detr
        boxes = boxes.cpu().detach().numpy().squeeze()
    
        return boxes

        

#%%
if __name__ == "__main__":

    config = OmegaConf.create({
        "training_config": {
        "dataset_base_path": "/media/zm2074/Data Drive/data/perception-guarantees/PwC_calibration/calibrate_2k/data/",
        "batch_size": 50,
        "learning_rate": 1e-4,
        "prior_num_epochs": 50,
        "prior_lr": 0.01,
        "posterior_num_epochs": 100,
        "posterior_lr": 1e-3, # TODO:  # 1e-4,
        "w1": 1.0,
        "w2": 0.1,
        "w3": 1.0,
        "N_total": 400,  # Total number of samples in the dataset
        "N": 350,  # Number of samples for PAC-Bayes
        "num_objects": 5,  # Number of objects in the dataset
        "delta": 0.01,  # Confidence level for PAC-Bayes
        "use_wandb": False, # TODO:  # True,
        "verbose": True,
        "save_name": "PAC_inflation_model_avg-maskgt",
        },
        "weight_size": 1
    })

    calibrator = PACBayesScalar(config=config)

    calibrator.calibrate()
