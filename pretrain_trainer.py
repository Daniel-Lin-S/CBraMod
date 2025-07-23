import numpy as np
import torch
from ptflops import get_model_complexity_info
from torch.nn import MSELoss
from torchinfo import summary
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
from argparse import Namespace

from utils.util import generate_mask


class Trainer(object):
    """
    Train a model using the provided data loader and parameters.
    Supports masking for pre-training tasks and can use multiple GPUs.

    Attributes
    ----------
    params : Namespace
        Parameters used for training, including device, epochs, batch size, etc.
    data_loader : DataLoader
        DataLoader for the training dataset.
    model : torch.nn.Module
        The model to be trained, should be a PyTorch model.
    criterion : torch.nn.Module
        Loss function used for training, set to MSELoss.
    optimizer : torch.optim.Optimizer
        Optimizer used for training, set to AdamW.
    optimizer_scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler for the optimizer.
    device : torch.device
        Device on which the model and data will be processed
        (CPU or CUDA).
    data_length : int
        Length of the data loader, used for scheduling and logging.
    model_dir : str
        Directory where the model checkpoints will be saved.
    """
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer
    optimizer_scheduler: torch.optim.lr_scheduler._LRScheduler
    device: torch.device
    data_length: int

    def __init__(
            self, params: Namespace,
            data_loader: DataLoader,
            model: torch.nn.Module
        ) -> None:
        """
        Parameters
        ----------
        params : Namespace
            Parameters for training,
            - cuda: int, device id for CUDA
            - parallel: bool, whether to use DataParallel
            - epochs: int, number of training epochs
            - batch_size: int, batch size for training
            - lr: float, learning rate for the optimizer
            - weight_decay: float, weight decay for the optimizer
            - lr_scheduler: str, type of learning rate scheduler
            - need_mask: bool, whether to apply masking
            - mask_ratio: float, ratio of masked patches
            - clip_value: float, gradient clipping value
            - model_dir: str, directory to save the model checkpoints
        data_loader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        model : torch.nn.Module
            The model to be trained, should be a PyTorch model.
        """
        self.params = params
        self.device = torch.device(
            f"cuda:{self.params.cuda}" if torch.cuda.is_available() else "cpu"
        )
        self.data_loader = data_loader
        self.model = model.to(self.device)
        self.criterion = MSELoss(reduction='mean').to(self.device)

        if self.params.parallel and self.device != "cpu":
            device_ids = list(range(torch.cuda.device_count()))

            if len(device_ids) > 1:
                self.model = torch.nn.DataParallel(
                    self.model, device_ids=device_ids
                )
            else:
                warnings.warn(
                    "Only one GPU detected, DataParallel will not be used."
                )
        elif self.device == "cpu":
            warnings.warn(
                "Training on CPU, DataParallel will not be used."
                "Please set --parallel to False or use CUDA GPU."
            )

        self.data_length = len(self.data_loader)

        summary(self.model, input_size=(1, 19, 30, 200))

        macs, params = get_model_complexity_info(
            self.model, (19, 30, 200), as_strings=True,
            print_per_layer_stat=True, verbose=True
        )
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.params.lr,
            weight_decay=self.params.weight_decay
        )

        self._configure_scheduler(self.params.lr_scheduler)

    def train(self):
        """
        Train the model using the provided
        data loader and parameters.
        """
        best_loss = float('inf')

        for epoch in range(self.params.epochs):
            losses = []
            for x in tqdm(self.data_loader, mininterval=10):
                self.optimizer.zero_grad()
                x = x.to(self.device)/100
                if self.params.need_mask:
                    bz, ch_num, patch_num, _ = x.shape
                    mask = generate_mask(
                        bz, ch_num, patch_num,
                        mask_ratio=self.params.mask_ratio,
                        device=self.device,
                    )

                    y = self.model(x, mask=mask)

                    # compute loss on the masked part
                    masked_x = x[mask == 1]
                    masked_y = y[mask == 1]
                    loss = self.criterion(masked_y, masked_x)

                    # non_masked_x = x[mask == 0]
                    # non_masked_y = y[mask == 0]
                    # non_masked_loss = self.criterion(non_masked_y, non_masked_x)
                    # loss = 0.8 * masked_loss + 0.2 * non_masked_loss
                else:
                    y = self.model(x)
                    loss = self.criterion(y, x)

                loss.backward()
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.params.clip_value
                    )

                self.optimizer.step()
                self.optimizer_scheduler.step()
                losses.append(loss.data.cpu().numpy())

            mean_loss = np.mean(losses)
            learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']

            print(
                f'Epoch {epoch+1}: '
                f'Training Loss: {mean_loss:.6f}, '
                f'Learning Rate: {learning_rate:.6f}')

            if  mean_loss < best_loss:
                model_path = rf'{self.params.model_dir}/epoch{epoch+1}_loss{mean_loss}.pth'
                torch.save(self.model.state_dict(), model_path)
                print("Epoch loss decresaed, model save in ", model_path)
                best_loss = mean_loss

    def _configure_scheduler(self, scheduler: str) -> None:
        """
        Configure the learning rate scheduler based on the specified type.

        Parameters
        ----------
        scheduler : str
            The type of learning rate scheduler to use. Supported types are:
            'CosineAnnealingLR', 'ExponentialLR', 'StepLR', 'MultiStepLR', 'CyclicLR'.
        """
        if scheduler == 'CosineAnnealingLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=40*self.data_length,
                eta_min=1e-5
            )
        elif scheduler=='ExponentialLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.999999999
            )
        elif scheduler=='StepLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5 * self.data_length, gamma=0.5
            )
        elif scheduler=='MultiStepLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[
                    10*self.data_length,
                    20*self.data_length,
                    30*self.data_length
                ], gamma=0.1
            )
        elif scheduler=='CyclicLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=1e-6,
                max_lr=0.001, step_size_up=self.data_length*5,
                step_size_down=self.data_length*2,
                mode='exp_range', gamma=0.9, cycle_momentum=False
            )
        else:
            raise ValueError(
                f"Unsupported lr_scheduler: {scheduler}"
            )
