import torch
from models.cbramod import CBraMod
import argparse
import time

from datasets import faced_dataset

parser = argparse.ArgumentParser(description='Quick example for using CBraMod with FACED dataset')
parser.add_argument('--datasets_dir', type=str, default='data/lmdb/FACED',
                    help='Directory containing the FACED dataset')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for loading data')
parser.add_argument('--foundation_dir', type=str,
                    default='pretrained_weights/pretrained_weights.pth',
                    help='Path to the pre-trained weights of the foundation model')

params = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CBraMod().to(device)
model.load_state_dict(
    torch.load(params.foundation_dir,
               map_location=device,
               weights_only=True))

dataset_loader = faced_dataset.LoadDataset(params)
data_loaders = dataset_loader.get_data_loader()

# extract an eeg sample
# shape: (batch_size, num_of_channels, time_segments, points_per_patch)
# 200 time points corresponds to 1 second (all re-sampled to 200Hz)
eeg_sample, _ = next(iter(data_loaders['train']))

print('shape of eeg data:', eeg_sample.shape)

start_time = time.time()
features = model(eeg_sample)
end_time = time.time()
print('Time taken to extract features:', end_time - start_time, 'seconds')
print('shape of embedded features:', features.shape)

