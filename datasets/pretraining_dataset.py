import pickle
from typing import List
from lmdb import open, Environment
from torch.utils.data import Dataset

from utils.util import to_tensor


class PretrainingDataset(Dataset):
    db: Environment
    keys: List[str]

    def __init__(
            self,
            dataset_dir: str
    ):
        """
        Parameters
        ----------
        dataset_dir : str
            Path to the LMDB dataset directory.
            The dataset should be stored in LMDB format with keys and values
            serialized using pickle.
        """
        super(PretrainingDataset, self).__init__()
        self.db = open(
            dataset_dir, readonly=True,
            lock=False, readahead=True, meminit=False
        )
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))
        # self.keys = self.keys[:100000]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        with self.db.begin(write=False) as txn:
            patch = pickle.loads(txn.get(key.encode()))

        patch = to_tensor(patch)

        return patch
