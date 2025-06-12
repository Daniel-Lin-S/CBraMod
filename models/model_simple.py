import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from argparse import Namespace


class Model(nn.Module):
    """
    A simple model used to classify EEG data using
    a linear classifier or simple MLP.
    """
    def __init__(self, param: Namespace):
        """
        Initializes the Model with a CBraMod backbone and a classifier.

        Parameters
        ----------
        param: Namespace
            Parameters containing model configuration:
            - classifier: str. Type of classifier to use
            ('avgpooling_patch_reps' or 'all_patch_reps').
            - dropout: float. Dropout rate for the classifier.
            - num_of_classes: int. Number of output classes.
            - cuda: int. CUDA device index for loading pre-trained weights.
            - n_electrodes: int. Number of channels in the data.
            - time_segments: int. Number of time segments in the data.
            - ndim: int. Number of dimensions of feature in each time segment.
        """
        super(Model, self).__init__()

        agg_dim = param.n_electrodes * param.time_segments * param.ndim
        temporal_dim = param.time_segments * param.ndim

        if param.classifier == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(param.ndim, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(agg_dim, temporal_dim),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(temporal_dim, param.ndim),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, num_of_channels, time_segments, points_per_patch)
            where time_segments is the number of segments and
            points_per_patch is the number of points in each segment.

        Returns
        -------
        out: torch.Tensor
            Output tensor of shape (batch_size, num_of_classes)
            containing the logit probabilities for each class.
        """
        out = self.classifier(x)
        return out
