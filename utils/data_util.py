import numpy as np
from typing import List


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
