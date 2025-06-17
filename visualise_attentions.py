"""
It is possible to capture and visualise the attention weights
to study what the model is focusing on during the forward pass.

This file contains functions to capture and visualise
attention weights from the CBraMod model's TransformerEncoder.
"""

from models.cbramod import CBraMod
import torch
import matplotlib.pyplot as plt
from typing import Optional


def load_cbramod_model(
        foundation_path : str,
        device: Optional[torch.device]=None
    ) -> CBraMod:
    """
    Initialise and load the CBraMod model from the specified directory.
    All parameters are set to their default values (
    consistent with public pre-trained weights)

    Parameters
    ----------
    foundation_path : str
        Path to the pth file containing the
        pre-trained CBraMod model weights.
    device : Optional[torch.device]
        Device to which the model will be moved (CPU or GPU).
        If None, it defaults to 'cuda:0' if available, otherwise 'cpu'.
    """
    if device is None:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    model = CBraMod(
        in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
        seq_len=30, n_layer=12, nhead=8).to(device)
    model.load_state_dict(
        torch.load(foundation_path, map_location=device))
    
    return model


def capture_attention_weights(
        model: CBraMod, input_tensor: torch.Tensor,
        layer: int=0
    ) -> dict:
    """
    Capture spatial and temporal attention weights from the TransformerEncoder
    in the CBraMod model during the forward pass.

    Parameters
    ----------
    model : CBraMod
        The CBraMod model containing the TransformerEncoder.
    input_tensor : torch.Tensor
        Input tensor of shape (batch_size, num_of_channels, time_segments, in_dim).
    layer: int
        The index of the layer from which to capture attention weights.
        Default is 0, which captures from the first layer.

    Returns
    -------
    dict
        A dictionary containing spatial and temporal attention weights:
        {
            'spatial_attention': numpy array of shape
                (batch_size, n_channels, n_channels),
            'temporal_attention': numpy array of shape
                (batch_size, n_segments, n_segments)
        }
    """
    model.eval()

    model_dtype = next(model.parameters()).dtype
    input_tensor = input_tensor.to(dtype=model_dtype)

    spatial_attention_weights = []
    temporal_attention_weights = []

    def hook_spatial(module, input, output):
        x = module.norm1(input[0])
        bz, ch_num, patch_num, patch_size = x.shape
        xs = x[:, :, :, :patch_size // 2]
        xs = xs.transpose(1, 2).contiguous().view(
            bz * patch_num, ch_num, patch_size // 2)
        # shape (batch_size, num_channels, num_channels)
        attn_weights = module.self_attn_s(
            xs, xs, xs, need_weights=True)[1]
        attn_weights = attn_weights.view(bz, patch_num, ch_num, ch_num).mean(dim=1)
        spatial_attention_weights.append(attn_weights)

    def hook_temporal(module, input, output):
        x = module.norm1(input[0])
        bz, ch_num, patch_num, patch_size = x.shape
        xt = x[:, :, :, patch_size // 2:]
        xt = xt.contiguous().view(
            bz * ch_num, patch_num, patch_size // 2)
        # shape (batch_size, num_segments, num_segments)
        attn_weights = module.self_attn_t(
            xt, xt, xt, need_weights=True)[1]
        attn_weights = attn_weights.view(
            bz, ch_num, patch_num, patch_num).mean(dim=1)
        temporal_attention_weights.append(attn_weights)

    if layer < 0 or layer >= len(model.encoder.layers):
        raise ValueError(
                "Invalid layer index: {layer}. "
                "Must be between 0 and {}.".format(
                    len(model.encoder.layers) - 1
            ))

    # Register hooks for spatial and temporal attention
    model.encoder.layers[layer].register_forward_hook(hook_spatial)
    model.encoder.layers[layer].register_forward_hook(hook_temporal)

    # Perform the forward pass
    model(input_tensor)

    spatial_attention = torch.cat(
        spatial_attention_weights, dim=0).mean(dim=0)
    temporal_attention = torch.cat(
        temporal_attention_weights, dim=0).mean(dim=0)

    return {
        'spatial_attention': spatial_attention.detach().cpu().numpy(),
        'temporal_attention': temporal_attention.detach().cpu().numpy()
    }


def plot_attention_weights(
        attention_weights: dict,
        title: str="Attention Weights",
        figure_path: str="attention_weights.png",
    ):
    """
    Visualise the spatial and temporal attention weights.

    Parameters
    ----------
    attention_weights : dict
        Dictionary containing spatial and temporal attention weights.
        Expected keys are 'spatial_attention' and 'temporal_attention'.
    title : str
        Title for the attention weights plot.
        Default is "Attention Weights".
    figure_path : str
        Path to save the attention weights plot.
        Default is "attention_weights.png".
    """
    spatial_attention = attention_weights['spatial_attention']
    temporal_attention = attention_weights['temporal_attention']

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=20)

    # Spatial Attention Plot
    axes[0].imshow(spatial_attention, aspect='auto', cmap='viridis')
    axes[0].set_title("Spatial Attention", fontsize=18)
    axes[0].set_xlabel("Channel", fontsize=16)
    axes[0].set_ylabel("Channel", fontsize=16)
    cbar = fig.colorbar(axes[0].images[0], ax=axes[0], orientation='vertical')
    cbar.set_label("Attention Weight")

    # Temporal Attention Plot
    axes[1].imshow(temporal_attention, aspect='auto', cmap='viridis')
    axes[1].set_title("Temporal Attention", fontsize=18)
    axes[1].set_xlabel("Segment", fontsize=16)
    axes[1].set_ylabel("Segment", fontsize=16)
    cbar = fig.colorbar(axes[1].images[0], ax=axes[1], orientation='vertical')
    cbar.set_label("Attention Weight")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the super title
    plt.savefig(figure_path, dpi=500)
    plt.close()
