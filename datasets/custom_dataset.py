from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from utils.util import to_tensor
import lmdb
import pickle
from argparse import Namespace
from typing import List, Tuple


class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            mode: str='train',
    ):
        """
        Custom dataset for loading data from a LMDB database.
        Parameters
        ----------
        data_dir: str
            Path to the LMDB database directory.
        mode: str
            Mode of the dataset, can be 'train', 'val', or 'test'.
        """
        super(CustomDataset, self).__init__()
        self.db = lmdb.open(
            data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]

    def __len__(self) -> int:
        return len((self.keys))

    def __getitem__(
            self, idx: int
        ) -> Tuple[np.ndarray, int]:
        """
        Parameter
        ----------
        idx: int
            Index of the data sample to retrieve.

        Returns
        -------
        data : np.ndarray
            The data sample as a NumPy array
            of shape (channels, time_segments, num_points).
        label : int
            The corresponding label for the data sample.
        """
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        label = pair['label']
        return data / 100, label

    def collate(
            self,
            batch: List[tuple]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate function to combine a batch of data samples.

        Parameters
        ----------
        batch: list
            A list of tuples, where each tuple contains
            a data sample (array) and its corresponding label (scalar).
        
        Returns
        -------
        tuple
            A tuple containing:
            - x_data: Tensor of input data samples.
            with shape (batch_size, channels, time_segments, num_points).
            - y_label: Tensor of corresponding labels.
            with shape (batch_size,).
        """
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label).long()


class LoadDataset(object):
    def __init__(self, params: Namespace):
        """
        LoadDataset class for managing dataset loading.

        Parameters
        ----------
        params: Namespace
            Parameters containing dataset directory and batch size.
            - datasets_dir: str. Path to the datasets directory.
            - batch_size: int. Size of the batches to be loaded.
        """
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self) -> DataLoader:
        train_set = CustomDataset(self.datasets_dir, mode='train')
        val_set = CustomDataset(self.datasets_dir, mode='val')
        test_set = CustomDataset(self.datasets_dir, mode='test')
        print(
            'Dataset sizes: train {} val {} test {}'.format(
                len(train_set), len(val_set), len(test_set)
            ))
        total_size = len(train_set) + len(val_set) + len(test_set)
        print('Total number of samples in the dataset:', total_size)

        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
        }
        return data_loader
