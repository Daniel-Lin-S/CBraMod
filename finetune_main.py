import argparse
import random

import numpy as np
import torch

from datasets import (
    custom_dataset, seedv_dataset, physio_dataset, shu_dataset, isruc_dataset,
    chb_dataset, speech_dataset, mumtaz_dataset, seedvig_dataset,
    stress_dataset, tuev_dataset, tuab_dataset, bciciv2a_dataset
)

from finetune_trainer import Trainer

from models import (
    model_for_faced, model_for_seedv, model_for_physio, model_for_shu,
    model_for_isruc, model_for_chb, model_for_speech, model_for_mumtaz,
    model_for_seedvig, model_for_stress, model_for_tuev, model_for_tuab,
    model_for_bciciv2a
)


def main():
    parser = argparse.ArgumentParser(description='Big model downstream')
    parser.add_argument('--seed', type=int, default=3407,
                        help='random seed (default: 3407)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda index (default: 0)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs for fine-tuning(default: 50)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for fine-tuning (default: 64)')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help='The optimizer used to train networks. Options: [AdamW, SGD]')
    parser.add_argument('--weight_decay', type=float, default=5e-2,
                        help='weight decay for optimizer (default: 5e-2)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate for optimizer (default: 1e-3)')
    parser.add_argument('--clip_value', type=float, default=1,
                        help='If non-zero, gradient clipping will be applied with this value'
                        ' (default: 1). Used to stabalise training')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--classifier', type=str, default='avgpooling_patch_reps',
                        help= 'The type of classifier. Default is avgpooling_patch_reps.'
                        'Options: [all_patch_reps, avgpooling_patch_reps]'
                        'avgpooling_patch_reps: flatten the channel and feature dimensions '
                        ' by taking the average, leaving a single point for each time segment. '
                        'The build a linear classifier (linear probing paradigm)'
                        'all_patch_reps: Apply MLP to all features.')

    """############ Downstream dataset settings ############"""
    parser.add_argument('--downstream_dataset', type=str, default='FACED',
                        help='[FACED, SEED-V, PhysioNet-MI, SHU-MI, ISRUC, CHB-MIT, BCIC2020-3, Mumtaz2016, SEED-VIG, MentalArithmetic, TUEV, TUAB, BCIC-IV-2a]')
    parser.add_argument('--datasets_dir', type=str,
                        default='/data/lmdb/Faced',
                        help='Path to the datasets directory')
    parser.add_argument('--num_of_classes', type=int, default=9, help='number of classes')
    parser.add_argument('--model_dir', type=str, default='/data/models_weights/',
                        help='The directory to save the model weights')
    """############ Downstream dataset settings ############"""

    parser.add_argument('--num_workers', type=int, default=16,
                        help='num_workers for data loader (default: 16)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Amount of smoothing when computing the cross entropy loss (default: 0.1)'
                        'Must be a float in range [0.0, 1.0]')
    parser.add_argument('--multi_lr', type=bool, default=True,
                        help='If true, different learning rates for different modules')
    parser.add_argument('--frozen', type=bool,
                        default=False,
                        help='If true, freeze the weights of pretrained backbone'
                        ' during fine-tuning')
    parser.add_argument('--use_pretrained_weights', type=bool,
                        default=True, help='if True, load pretrained weights of the backbone'
                        '. Otherwise, train the backbone from scratch')
    parser.add_argument('--foundation_dir', type=str,
                        default='pretrained_weights/pretrained_weights.pth',
                        help='The path to the pretrained weights of the foundation model')

    params = parser.parse_args()
    print(params)

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    print('The downstream dataset is {}'.format(params.downstream_dataset))

    # Load dataset and model
    if params.downstream_dataset == 'FACED':
        load_dataset = custom_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_faced.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'SEED-V':
        load_dataset = seedv_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_seedv.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'PhysioNet-MI':
        load_dataset = physio_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_physio.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'SHU-MI':
        load_dataset = shu_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_shu.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'ISRUC':
        load_dataset = isruc_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_isruc.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'CHB-MIT':
        load_dataset = chb_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_chb.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'BCIC2020-3':
        load_dataset = speech_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_speech.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'Mumtaz2016':
        load_dataset = mumtaz_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_mumtaz.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'SEED-VIG':
        load_dataset = seedvig_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_seedvig.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_regression()
    elif params.downstream_dataset == 'MentalArithmetic':
        load_dataset = stress_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_stress.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'TUEV':
        load_dataset = tuev_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_tuev.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'TUAB':
        load_dataset = tuab_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_tuab.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'BCIC-IV-2a':
        load_dataset = bciciv2a_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_bciciv2a.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    print('Done!!!!!')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
