"""
Visualise EEG data of subjects from the FACED dataset using the CBraMod foundation model
with t-SNE dimensionality reduction.
Subject's labels (age or gender) can be used to mark the subjects in the visualisation.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy import signal
import torch
from models.cbramod import CBraMod
import argparse

from utils.util import to_tensor


parser = argparse.ArgumentParser(
    description="Visualize EEG data using CBraMod foundation model")
parser.add_argument(
    '--data_dir', default='data/FACED',
    type=str,
    help="Directory containing subject .pkl files")
parser.add_argument(
    '--info_file', default='Recording_info.csv',
    type=str,
    help='File with information of subjects')
parser.add_argument(
    '--aggregate', default='all',
    type=str,
    help='Aggregation method for features of each subject. '
    'Options: [all, temporal, spatial]. \n'
    'all: mean over all channels, clips and segments; '
    'temporal: mean over all clips and segments; '
    'spatial: mean over all clips and channels')
parser.add_argument(
    '--label', default='Age',
    type=str,
    help='Labels used for marking the subjects. Options: [Age, Gender]')
parser.add_argument(
    '--figure_dir', default='figure',
    type=str,
    help='Directory to save the figure')


params = parser.parse_args()

print('Visualising {}'.format(params.label), ' ...')
print('Aggregation Method: {}'.format(params.aggregate))

# Load subject labels
labels_df = pd.read_csv(params.info_file)

# Initialize storage for features and labels
all_features = []
all_labels = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CBraMod().to(device)
model.load_state_dict(
    torch.load('pretrained_weights/pretrained_weights.pth',
               map_location=device))

# Process each subject's data
with torch.no_grad():
    model.eval()
    for subject_file in os.listdir(params.data_dir):
        if subject_file.endswith(".pkl"):
            subject_id = os.path.splitext(subject_file)[0]

            # Load the subject's data
            with open(os.path.join(params.data_dir, subject_file), "rb") as f:
                data = pickle.load(f)  # Shape: (28, 32, 7500)

            eeg = signal.resample(data, 6000, axis=2)
            # segment into 30 1-second patches
            eeg_ = eeg.reshape(28, 32, 30, 200)
            subj_data = to_tensor(eeg_).float().to(device)

            features = model(subj_data)

            # Aggregate features
            if params.aggregate == 'all':
                # Take mean value for channels, clips and patches
                subject_features = features.mean(dim=(0, 1, 2)).cpu().numpy()
            elif params.aggregate == 'temporal':
                subject_features = features.mean(dim=(0, 2)).cpu().numpy()
                subject_features = subject_features.reshape(-1)
            elif params.aggregate == 'spatial':
                subject_features = features.mean(dim=(0, 1)).cpu().numpy()
                subject_features = subject_features.reshape(-1)
            else:
                raise ValueError(
                    "Invalid aggregation method: {}, must be one of "
                    "[all, temporal, spatial]".format(params.aggregate))

            all_features.append(subject_features)
            
            # Get subject's label (e.g., age or gender)
            subject_label = labels_df[
                labels_df["sub"] == subject_id][params.label].values[0]
            all_labels.append(subject_label)

# Convert to arrays
all_features = np.array(all_features) # Shape: (num_subjects, feature_dim)

if params.label == 'Gender':
    label_mapping = {'M': 0, 'F': 1}
    all_labels = [label_mapping[label] for label in all_labels]

all_labels = np.array(all_labels)

# Use t-SNE to reduce the dimension
reducer = TSNE(n_components=2, perplexity=30, random_state=42)
reduced_features = reducer.fit_transform(all_features)

# Visualisation
plt.rcParams.update({'font.size': 15})

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    reduced_features[:, 0], reduced_features[:, 1],
    c=all_labels, cmap="viridis", s=50)
if params.label == 'Gender':
    colorbar = plt.colorbar(scatter, ticks=[0, 1])
    colorbar.set_ticklabels(['M', 'F'])
    colorbar.set_label(params.label)
else:
    plt.colorbar(scatter, label=params.label)
plt.title("tSNE Plot of Foundation Model Representations", fontsize=20)
plt.xlabel("tSNE Dimension 1", fontsize=17)
plt.ylabel("tSNE Dimension 2", fontsize=17)

figure_path = os.path.join(
    params.figure_dir,
    "subject_{}_agg[{}].png".format(params.label, params.aggregate)
)
plt.savefig(figure_path, dpi=500)
plt.close()
print('Figure saved to {}'.format(figure_path))
