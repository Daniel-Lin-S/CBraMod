"""
Visually check the discriminative power of the foundation model representations
of EEG data from the FACED dataset.

The following aspects are checked:
- Visualisation of features using t-SNE or UMAP
- PCA analysis of features
- Clustering analysis of features using HDBSCAN
Labels like emotions and subject id are used to mark the points in the visualisation.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import torch
import argparse

from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA

from models.cbramod import CBraMod
from utils.util import to_tensor
from utils.data_util import faced_labels
from utils.visualisations import scatter_with_labels


parser = argparse.ArgumentParser(
    description="Visualize EEG data using CBraMod foundation model")
parser.add_argument(
    '--data_dir', default='data/FACED',
    type=str,
    help="Directory containing subject .pkl files")
parser.add_argument(
    '--info_file', default='Recording_info.csv',
    type=str,
    help='Path to the csv file with information of subjects')
parser.add_argument(
    '--figure_dir', default='figure',
    type=str,
    help='Directory to save the figure')
parser.add_argument(
    '--visualisation', default='tSNE',
    type=str,
    choices=['tSNE', 'UMAP'],
    help='Type of visualisation to use')


params = parser.parse_args()

# Load subject labels
labels_df = pd.read_csv(params.info_file)
# remove spaces before and after column names
labels_df.columns = labels_df.columns.str.strip()

# Initialize storage for features and labels
all_features = []
all_clip_labels = []
all_sub_labels = []  # subject labels

clip_labels, emotion_names = faced_labels()

# load pre-trained model
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
            for clip_idx in range(28):
                clip_features = features[clip_idx].mean(dim=(0, 1)).cpu().numpy()
                all_features.append(clip_features)
                all_clip_labels.append(clip_labels[clip_idx])
                all_sub_labels.append(int(subject_id[3:]))

# Convert to arrays
all_features = np.array(all_features)  # Shape: (num_subjects * num_clips, feature_dim)
all_clip_labels = np.array(all_clip_labels)
all_sub_labels = np.array(all_sub_labels)

# Choose visualisation method
if params.visualisation == 'UMAP':
    reducer = UMAP(n_components=2, random_state=42)
elif params.visualisation == 'tSNE':
    reducer = TSNE(n_components=2, perplexity=30, random_state=42)
else:
    raise ValueError(
        "Unsupported visualisation method: {}".format(
            params.visualisation))

reduced_features = reducer.fit_transform(all_features)

# ----- Visualisation -----
plt.rcParams.update({'font.size': 15})

# map clip labels 0-8 to emotion names
emotion_labels = np.array(
    [emotion_names[label] for label in all_clip_labels])

scatter_with_labels(
    reduced_features[:, 0], reduced_features[:, 1],
    emotion_labels,
    os.path.join(params.figure_dir, 'features_emotions_{}.png'.format(
        params.visualisation.lower()
    )),
    label_name='Emotion',
    xlabel='Dimension 1',
    ylabel='Dimension 2',
    title='{} Plot of Foundation Model Representations'.format(
        params.visualisation
    ),
)

# Select first 20 subjects for easier visualisation
mask = all_sub_labels < 20
filtered_labels = all_sub_labels[mask]
filtered_features = reduced_features[mask]

scatter_with_labels(
    filtered_features[:, 0], filtered_features[:, 1],
    filtered_labels,
    os.path.join(params.figure_dir, 'features_subject_{}.png'.format(
        params.visualisation.lower()
    )),
    label_name='Subject',
    xlabel='Dimension 1',
    ylabel='Dimension 2',
    title='{} Plot of Foundation Model Representations'.format(
            params.visualisation
        ) + '\n'+ 'for first 20 subjects',
    cmap_name='tab20'
)

# ----- PCA analysis -----
pca = PCA(n_components=10)
pca_features = pca.fit_transform(all_features)
print('Shape of PCA features:', pca_features.shape)

# plot explained variance
pca_var_path = os.path.join(
    params.figure_dir, 'features_pca_variance.png')
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title('PCA Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.savefig(pca_var_path, dpi=500)
plt.close()
print('Figure saved to {}'.format(pca_var_path))

# plot the first two PCA components with emotion labels
scatter_with_labels(
    pca_features[:, 0], pca_features[:, 1],
    emotion_labels,
    os.path.join(params.figure_dir, 'features_pca_emotions.png'),
    label_name='Emotion',
    xlabel='PCA Component 1',
    ylabel='PCA Component 2',
    title='PCA Componenets of Foundation Model Representations'
)

# ----- Clustering analysis -----
pca_threshold = 0.99
ndims = np.argmax(
    np.cumsum(pca.explained_variance_ratio_) >= pca_threshold) + 1
main_features = pca_features[:, :ndims]
print('Shape of main features:', main_features.shape)

clusterer = HDBSCAN(min_cluster_size=30, min_samples=5)
cluster_labels = clusterer.fit_predict(main_features)

num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print('Found {} clusters'.format(num_clusters))
noise_points = np.sum(cluster_labels == -1)
print('Number of noise points: {} out of {}'.format(
    noise_points, len(cluster_labels)))

print('Cluster sizes: {}'.format(
    pd.Series(cluster_labels).value_counts().to_dict()))

if num_clusters > 1:
    # Compute metrics
    silhouette_avg = silhouette_score(main_features, cluster_labels)
    print(f'Silhouette Score: {silhouette_avg:.4f}')
    # Calinski-Harabasz Index
    ch_index = calinski_harabasz_score(main_features, cluster_labels)
    print(f'Calinski-Harabasz Index: {ch_index:.4f}')
else:
    print('Warning: Not enough clusters to compute metrics')

# plot clusters on the first two dimensions
if ndims > 1:
    scatter_with_labels(
        main_features[:, 0], main_features[:, 1],
        cluster_labels,
        os.path.join(params.figure_dir, 'features_pca_clusters.png'),
        label_name='Cluster',
        xlabel='PCA Component 1',
        ylabel='PCA Component 2',
        title='Clusters of Foundation Model Representations'
    )

else:
    print("Not enough dimensions to plot clusters")
