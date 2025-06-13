"""
Preprocess the FACED dataset and store it in LMDB format.

The original dataset can be downloaded from 
https://www.synapse.org/Synapse:syn50614194/files/
Only the "Processed_data" folder is required.
Number of subjects: 123.
Each pkl file corresponds to a subject's EEG data.

Shape of each file:
(28, 32, 7500)
28: number of video clips
32: number of electrodes
7500: number of time points (30 seconds at 250 Hz)

Basic information
-----------------
Number of channels (electrodes): 32
Number of subjects: 123
Number of classes (emotions): 9
Sampling rate: 250 Hz
Trial duration: 30 seconds

Labelling of emotions:
0: Anger
1: Disgust
2: Fear
3: Sadness
4: Neutral
5: Amusement
6: Inspiration
7: Joy
8: Tenderness

Steps of EEG pre-processing:
1. Adjust the sampling rate to 200 Hz (6000 time points)
2. Split each trial into three 10-second segments.
3. Segment each 10-second sample into 30 patches of 1 second length each (reshaping)
4. Store the processed data in LMDB format, with each segment as a sample.
"""

from scipy import signal
import os
import lmdb
import pickle
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Preprocess the FACED dataset for CBraMod')
parser.add_argument(
    '--lmdb_path', type=str,
    default='data/lmdb/FACED',
    help='The path to store the processed LMDB dataset')
parser.add_argument(
    '--root_dir', type=str,
    default='data/FACED/Processed_data',
    help='The path to the original FACED dataset')
parser.add_argument(
    '--verbose', type=bool,
    default=True,
    help='Whether to print the processing information')

# Labels for emotions for the 28 video clips (3 clips for each emotion)
group1 = np.tile(np.repeat(np.arange(4), 3), 1)
group2 = np.repeat(4, 4)
group3 = np.tile(np.repeat(np.arange(5, 9), 3), 1)

clip_labels = np.concatenate((group1, group2, group3))

params = parser.parse_args()

files = [
    file for file in os.listdir(params.root_dir)
    if file.endswith('.pkl')]
files = sorted(files)

# 80 patients for training, 20 for validation, and 23 for testing
files_dict = {
    'train' : files[:80],
    'val' : files[80:100],
    'test' : files[100:],
}

dataset = {
    'train' : list(),
    'val' : list(),
    'test' : list(),
}

db = lmdb.open(params.lmdb_path, map_size=6612500172)

for files_key in files_dict.keys():
    for file in files_dict[files_key]:
        f = open(os.path.join(params.root_dir, file), 'rb')
        array = pickle.load(f)  # (28, 32, 7500)

        # adjust the sampling rate to 200Hz
        eeg = signal.resample(array, 6000, axis=2)
        # segment into 30 1-second patches
        eeg_ = eeg.reshape(28, 32, 30, 200)

        # split each trial into three 10-second segments
        # i - index of video clip, j - index of segment
        if eeg_.shape[0] != len(clip_labels):
            raise ValueError(
                "Number of video clips in {} does not match "
                "the expected number: {}".format(
                    len(file, clip_labels)))

        for i, (samples, label) in enumerate(zip(eeg_, clip_labels)):
            for j in range(3):
                sample = samples[:, 10*j:10*(j+1), :]
                sample_key = f'{file}-{i}-{j}'
                data_dict = {
                    'sample': sample, 'label': label
                }

                # write into LMDB database
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()
                dataset[files_key].append(sample_key)

        if params.verbose:
            print('Finished loading file ', file)


# Store the keys of the dataset in LMDB
txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()

if params.verbose:
    print(
        'Preprocessing completed and data stored in LMDB format in ',
        params.lmdb_path)
