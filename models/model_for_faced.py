import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from argparse import Namespace

from .cbramod import CBraMod


class Model(nn.Module):
    def __init__(self, param: Namespace):
        """
        Initializes the Model with a CBraMod backbone and a classifier.

        Parameters
        ----------
        param: Namespace
            Parameters containing model configuration:
            - use_pretrained_weights: bool. Whether to use pre-trained weights.
            - foundation_dir: str. Path to the pre-trained weights.
            - classifier: str. Type of classifier to use ('avgpooling_patch_reps' or 'all_patch_reps').
            - dropout: float. Dropout rate for the classifier.
            - num_of_classes: int. Number of output classes.
            - cuda: int. CUDA device index for loading pre-trained weights.
        """
        super(Model, self).__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )

        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))
        self.backbone.proj_out = nn.Identity()

        if param.classifier == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(32 * 10 * 200, 10 * 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(10 * 200, 200),
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
        feats = self.backbone(x)
        out = self.classifier(feats)
        return out



