import torch
import torch.nn as nn
from models.cbramod import CBraMod
from einops.layers.torch import Rearrange


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CBraMod().to(device)
print('Loading pre-trained model ...')
model.load_state_dict(
    torch.load('pretrained_weights/pretrained_weights.pth', map_location=device))
print('Finished loading')
model.proj_out = nn.Identity()
classifier = nn.Sequential(
  Rearrange('b c s p -> b (c s p)'),
  nn.Linear(22*4*200, 4*200),
  nn.ELU(),
  nn.Dropout(0.1),
  nn.Linear(4 * 200, 200),
  nn.ELU(),
  nn.Dropout(0.1),
  nn.Linear(200, 4),
).to(device)

# shape: (batch_size, num_of_channels, time_segments, points_per_patch)
# 200 time points corresponds to 1 second (all re-sampled to 200Hz)
mock_eeg = torch.randn((8, 22, 4, 200)).to(device)
print('shape of mock_eeg:', mock_eeg.shape)

features = model(mock_eeg)
print('shape of embedded features:', features.shape)

# logits.shape = (batch_size, num_of_classes)
logits = classifier(features)

print('shape of classifier output:', logits.shape)