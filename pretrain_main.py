import argparse
from torch.utils.data import DataLoader
import os
import yaml

from datasets.pretraining_dataset import PretrainingDataset
from models.cbramod import CBraMod
from pretrain_trainer import Trainer
from utils.util import setup_seed


def main():
    parser = argparse.ArgumentParser(
        description='pre-train CBraMod on a large dataset.')
    # -------- General Parameters --------
    parser.add_argument(
        '--seed', type=int, default=42,
        help='random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--cuda', type=int, default=0,
        help='cuda device id (default: 0)'
    )
    parser.add_argument(
        '--parallel', type=bool, default=False,
        help='whether to use DataParallel to compute on multiple GPUs (default: False)'
    )
    # -------- Training Parameters --------
    parser.add_argument(
        '--epochs', type=int, default=40,
        help='number of epochs for training (default: 5)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='batch size for training (default: 128)'
    )
    parser.add_argument(
        '--need_mask', type=bool, default=True,
        help='Whether to use masked input for training (default: True)'
        'If False, the model simply reconstructs the input.'
    )
    parser.add_argument(
        '--mask_ratio', type=float, default=0.5,
        help='Mask ratio for the masked input (default: 0.5)'
    )
    # -------- Optimizer Parameters --------
    parser.add_argument(
        '--lr', type=float, default=5e-4,
        help='learning rate of the optimiser (default: 5e-4)'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=5e-2,
        help='weight decay of the optimiser (default: 5e-2)'
    )
    parser.add_argument(
        '--clip_value', type=float, default=1.,
        help='gradient clipping value for the optimiser (default: 1.0)'
    )
    parser.add_argument(
        '--lr_scheduler', type=str, default='CosineAnnealingLR',
        choices=['CosineAnnealingLR', 'ExponentialLR', 'StepLR', 'MultiStepLR', 'CyclicLR'],
        help='The learning rate scheduler to use (default: CosineAnnealingLR)'
    )
    # -------- I/O settings --------
    parser.add_argument(
        '--model_configs', type=str, default='configs/default.yaml',
        help='Path to the configuration file for the CBraMod model.'
        'Must have key "CBraMod" with model parameters.'
        'If you want to use a different configuration than the public '
        'version, please create a new configuration file. '
    )
    parser.add_argument(
        '--dataset_dir', type=str, default='data/lmdb/pretrain',
        help='Directory where the pretraining dataset is stored '
        '(default: data/lmdb/pretrain)'
    )
    parser.add_argument(
        '--model_dir', type=str, default='checkpoints',
        help='Directory to save the model checkpoints '
        '(default: checkpoints)'
    )

    params = parser.parse_args()
    print("Parameters: ", params)

    if not os.path.exists(params.dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory '{params.dataset_dir}' does not exist."
        )
    if not os.path.exists(params.model_configs):
        raise FileNotFoundError(
            f"Model configuration file '{params.model_configs}' does not exist."
        )

    if not os.path.exists(params.model_dir):
        os.makedirs(params.model_dir)

    setup_seed(params.seed)

    pretrained_dataset = PretrainingDataset(dataset_dir=params.dataset_dir)

    print('Number of samples in the pretraining dataset: ',
          len(pretrained_dataset))

    data_loader = DataLoader(
        pretrained_dataset,
        batch_size=params.batch_size,
        num_workers=8,
        shuffle=True,
    )

    # load configuration
    with open(params.model_configs, 'r') as f:
        model_configs = yaml.safe_load(f)

    model = CBraMod(**model_configs['CBraMod'])

    trainer = Trainer(params, data_loader, model)
    trainer.train()

    print('Training completed.')
    pretrained_dataset.db.close()

if __name__ == '__main__':
    main()
