import lmdb
import numpy as np
import pickle
from typing import List


def store_eeg_data(
    dataset: dict,
    db: lmdb.Environment,
    file_prefix: str,
    dataset_key: str,
    eeg: np.ndarray,
    labels: np.ndarray,
    verbose: bool = False
    ) -> None:
    """
    Store a group of EEG data into LMDB database.
    dataset and db will be modified in-place.

    Parameters
    ----------
    dataset : dict
        Dictionary to store the keys of the samples.
    db : lmdb.Environment
        LMDB database environment.
    file_prefix : str
        Prefix for the sample keys,
        e.g. the subject's id, type of data (train / test / val).
    dataset_key : str
        Key in the dataset dictionary to store the sample keys,
        e.g. 'train', 'val', or 'test'.
    eeg : np.ndarray
        EEG data of shape (n_trials, n_channels,
        n_segments, points_per_segment).
    labels : np.ndarray
        Labels of the samples, shape (n_trials,).
    verbose : bool, optional
        If True, print the sample keys as they are stored.
        Default is False.
    """
    for i, (sample, label) in enumerate(zip(eeg, labels)):
        sample_key = '{}-{}'.format(
            file_prefix, i
        )
        data_dict = {
            'sample': sample, 'label': label,
        }
        txn = db.begin(write=True)
        txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
        txn.commit()
        if verbose:
            print(f'Stored sample: {sample_key}')

        dataset[dataset_key].append(sample_key)


def faced_labels() -> tuple[np.ndarray, List[str]]:
    """
    Generate (emotion) labels for video clips and
    corresponding emotion names for the FACED dataset.
    """

    group1 = np.tile(np.repeat(np.arange(4), 3), 1)
    group2 = np.repeat(4, 4)
    group3 = np.tile(np.repeat(np.arange(5, 9), 3), 1)

    clip_labels = np.concatenate((group1, group2, group3))

    emotion_names = [
        "Anger", "Disgust", "Fear", "Sadness", "Neutral",
        "Amusement", "Inspiration", "Joy", "Tenderness"
    ]

    return clip_labels, emotion_names
