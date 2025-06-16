"""
Preprocess the BCIC-2020 Imagined Speech Dataset and store it in LMDB format.

The original dataset can be downloaded from
https://osf.io/pq7vb/files/osfstorage
Each mat file corresponds to one subject.

Raw data shape:
(n_time_points, n_channels, n_trials)
training set: 300 trials
validation set: 50 trials
test set: 50 trials
trial duration: 3.1 seconds


Basic information
-----------------
Number of channels (electrodes): 64
Number of subjects: 15
Number of classes (words): 5
Sampling rate: 256Hz


Labels of words:
0: Hello
1: Help me
2: Stop
3: Thank you
4: Yes

Steps of pre-processing
1. Discard the first 0.1 second
2. Resample the data to 200Hz
3. Segment the data into 3 segments of 200 points each (1 second)
"""

import h5py
from scipy.io import loadmat
from scipy import signal
import os
import lmdb
import pickle
import numpy as np
import pandas as pd
import argparse

from data_utils import store_eeg_data


parser = argparse.ArgumentParser(
    description='Preprocess BCIC-2020 Imagined Speech Dataset')
parser.add_argument(
    '--data_dir', type=str, default='data/datasets/Imagined speech',
    help='Directory containing the dataset (with training, validation, and test sets).')
parser.add_argument(
    '--lmdb_dir', type=str, default='data/lmdb/Speech',
    help='Directory to store the processed LMDB dataset.')
parser.add_argument(
    '--verbose', type=int, default=1,
    help='0 - no output, 1 - print progress, 2 - print detailed progress.')


params = parser.parse_args()


train_dir = os.path.join(params.data_dir, 'Training set')
val_dir = os.path.join(params.data_dir, 'Validation set')
test_dir = os.path.join(params.data_dir, 'Test set')

files_dict = {
    'train' : sorted([file for file in os.listdir(train_dir)]),
    'val' : sorted([file for file in os.listdir(val_dir)]),
    'test' : sorted([
            file for file in os.listdir(test_dir)
            if file.endswith('.mat')
        ]),
}

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

db = lmdb.open(params.lmdb_dir, map_size=3000000000)


# Load training data
for file in files_dict['train']:
    data = loadmat(os.path.join(train_dir, file))  # dict
    eeg = data['epo_train'][0][0][4].transpose(
        2, 1, 0)    # (n_trials, n_channels, n_time_points)
    # truncate the first 0.1 second
    eeg = eeg[:, :, -768:]
    # resample to 200Hz
    eeg = signal.resample(eeg, num=600, axis=2).reshape(
        300, 64, 3, 200)   # (n_trials, n_channels, n_segments, points_per_segment)

    labels = data['epo_train'][0][0][5].transpose(1, 0)   # (n_trials, n_classes)
    labels = np.argmax(labels, axis=1)   # (n_trials,)

    # load data into LMDB
    prefix = 'train-{}'.format(file[:-4])
    store_verbose = params.verbose >= 2
    store_eeg_data(
        dataset, db, prefix, 'train',
        eeg, labels, verbose=store_verbose)

    if params.verbose >= 1:
        print('Finished loading {}'.format(file))


if params.verbose >= 1:
    print('Finished loading training data')

for file in files_dict['val']:
    data = loadmat(os.path.join(val_dir, file))
    eeg = data['epo_validation'][0][0][4].transpose(2, 1, 0)
    labels = data['epo_validation'][0][0][5].transpose(1, 0)
    eeg = eeg[:, :, -768:]
    labels = np.argmax(labels, axis=1)
    eeg = signal.resample(eeg, 600, axis=2).reshape(50, 64, 3, 200)

    prefix = 'val-{}'.format(file[:-4])
    store_verbose = params.verbose >= 2
    store_eeg_data(
        dataset, db, prefix,
        'val', eeg, labels, verbose=store_verbose)

    if params.verbose >= 1:
        print('Finished loading {}'.format(file))

if params.verbose >= 1:
    print('Finished loading validation data')

# Load test labels
test_answer_path = os.path.join(
    params.data_dir, 'Test set/Track3_Answer Sheet_Test.xlsx')
df = pd.read_excel(test_answer_path)
df_ = df.head(53)
test_labels = df_.values   # shape 
test_labels = test_labels[2:, 1:][:, 1:30:2].transpose(
    1, 0)  # shape (n_subjects, n_trials)

# Load test data
for j, file in enumerate(files_dict['test']):
    data = h5py.File(os.path.join(test_dir, file))
    eeg = data['epo_test']['x'][:]
    eeg = eeg[:, :, -768:]
    eeg = signal.resample(eeg, 600, axis=2).reshape(50, 64, 3, 200)

    labels = test_labels[j]
    labels = labels - 1  # convert to 0-indexed labels

    prefix = 'test-{}'.format(file[:-4])
    store_verbose = params.verbose >= 2
    store_eeg_data(
        dataset, db, prefix,
        'test', eeg, labels, verbose=store_verbose)

    if params.verbose >= 1:
        print('Finished loading {}'.format(file))

if params.verbose >= 1:
    print('Finished loading test data')


# Store the keys of the dataset in LMDB
txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()
