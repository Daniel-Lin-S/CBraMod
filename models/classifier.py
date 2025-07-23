import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from argparse import Namespace
import yaml

from .cbramod import CBraMod


class Classifier(nn.Module):
    """
    Classifier model based on the CBraMod architecture.
    Used for downstream classification tasks.
    """
    def __init__(self, param : Namespace):
        """
        Parameters
        ----------
        param: Namespace
            Parameters containing model configuration:
            - use_pretrained_weights: bool. Whether to use pre-trained weights.
            - foundation_dir: str. Path to the pre-trained weights.
            - classifier: str. Type of classifier to use 
              ('avgpooling_patch_reps' or 'all_patch_reps').
            - dropout: float. Dropout rate for the classifier.
            - num_of_classes: int. Number of output classes.
            - cuda: int. CUDA device index for loading pre-trained weights.
            - n_electrodes: int. Number of channels in the data.
            - time_segments: int. Number of time segments in the data.
            - ndim: int. Number of dimensions of feature in each time segment.
            should be 200 if using the published version of CBramod
        """
        super(Classifier, self).__init__()
        with open(param.foundation_configs, 'r') as f:
            foundation_configs = yaml.safe_load(f)

        self.backbone = CBraMod(**foundation_configs['CBraMod'])

        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            self.backbone.load_state_dict(
                torch.load(param.foundation_dir, map_location=map_location))
        self.backbone.proj_out = nn.Identity()
        if param.classifier == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(param.ndim, param.num_of_classes)
            )
        elif param.classifier == 'all_patch_reps':
            agg_dim = param.n_electrodes * param.time_segments * param.ndim
            temporal_dim = param.time_segments * param.ndim
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(agg_dim, temporal_dim),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(temporal_dim, param.ndim),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(param.ndim, param.num_of_classes),
            )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
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
        feats = self.backbone(x)
        out = self.classifier(feats)
        return out
