import copy
from typing import Optional, Any, Union, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the Transformer Encoder.
    This layer consists of two multi-head self-attention mechanisms
    and a feedforward neural network, with layer normalisation and dropout.
    Self-attention is applied on temporal and spatial dimensions separately.

    Attributes
    -----------
    self_attn_s : nn.MultiheadAttention
        Multi-head self-attention mechanism for spatial attention.
    self_attn_t : nn.MultiheadAttention
        Multi-head self-attention mechanism for temporal attention.
    linear1 : nn.Linear
        Linear layer for the self-attention block
    linear2 : nn.Linear
        Linear layer for the feedforward (fully-connected) block.
    norm1 : nn.LayerNorm
        Layer normalisation applied before self-attention block
    norm2 : nn.LayerNorm
        Layer normalisation applied before feedforward block
    dropout1 : nn.Dropout
        Dropout layer applied after self-attention block
    dropout2 : nn.Dropout
        Dropout layer applied after feedforward block
    activation : Callable[[Tensor], Tensor]
        Activation function used in the feedforward network.
    activation_relu_or_gelu : int
        An integer indicating the type of activation function used:
        1 for ReLU, 2 for GELU, and 0 for other functions.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                dropout: float = 0.1,
                activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                layer_norm_eps: float = 1e-5, batch_first: bool = False,
                bias: bool = True, device: Optional[torch.device]=None,
                dtype: Optional[torch.dtype]=None) -> None:
        """
        Parameters
        ----------
        d_model : int
            Dimension of the input and output features.
        nhead : int
            Total number of attention heads.
        dim_feedforward : int, optional
            Dimension of the hidden layero of feedforward network
            (default is 2048).
        dropout : float, optional
            Dropout probability (default is 0.1).
        activation : Union[str, Callable[[Tensor], Tensor]], optional
            Activation function to use in the feedforward network.
            It can be a string ('relu', 'gelu') or a callable function.
            Default is F.relu.
        layer_norm_eps : float, optional
            Value added on the standard deviation in layer normalisation
            to avoid division by zero.
            (default is 1e-5).
        batch_first : bool, optional
            Used for Multi-headed attention.
            If True, the input and output tensors are of shape
            (batch_size, seq_len, d_model).
            Otherwise, the shapes are
            (seq_len, batch_size, d_model).
        bias : bool, optional
            Whether to add a bias term in the linear layers.
            Default is True.
        device : Optional[torch.device], optional
            The device on which the module will be allocated.
            If None, the default device is used.
        dtype : Optional[torch.dtype], optional
            The data type of the module's parameters.
            If None, the default data type is used.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        # spatial self-attention
        self.self_attn_s = nn.MultiheadAttention(d_model//2, nhead // 2, dropout=dropout,
                                                 bias=bias, batch_first=batch_first,
                                                 **factory_kwargs)
        # temporal self-attention
        self.self_attn_t = nn.MultiheadAttention(d_model//2, nhead // 2, dropout=dropout,
                                                 bias=bias, batch_first=batch_first,
                                                 **factory_kwargs)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        src : Tensor
            Input tensor of shape (batch_size, num_of_channels, patch_num, d_model).
            This corresponds to the EEG signal.
        src_mask : Optional[Tensor]
            Attention mask tensor of shape (patch_num, patch_num)
            or (batch_size, patch_num, patch_num).
            If None, no attention mask is applied.
        src_key_padding_mask : Optional[Tensor]
            Key padding mask tensor of shape (batch_size, patch_num).
            (if masked, that token is ignored in the attention).
            If None, no key padding mask is applied.
        """

        x = src
        # residual self-attention
        x = x + self._sa_block(
            self.norm1(x), src_mask, src_key_padding_mask)
        # residual feedforward (Fully-connected) network
        x = x + self._ff_block(self.norm2(x))
        return x

    # self-attention block
    def _sa_block(
            self, x: Tensor,
            attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor]) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, num_of_channels, patch_num, d_model).
        attn_mask : Optional[Tensor]
            Attention mask tensor of shape (patch_num, patch_num)
            or (batch_size, patch_num, patch_num).
            If None, no attention mask is applied.
        key_padding_mask : Optional[Tensor]
            Key padding mask tensor of shape (batch_size, patch_num).
            If None, no key padding mask is applied.
        
        Return
        -------
        Tensor
            Output tensor of the same shape as x.
        """
        bz, ch_num, patch_num, patch_size = x.shape
        xs = x[:, :, :, :patch_size // 2]
        xt = x[:, :, :, patch_size // 2:]
        xs = xs.transpose(1, 2).contiguous().view(
            bz*patch_num, ch_num, patch_size // 2)
        xt = xt.contiguous().view(
            bz*ch_num, patch_num, patch_size // 2)
        # attention applied across channels
        xs = self.self_attn_s(xs, xs, xs,
                             attn_mask=attn_mask,
                             key_padding_mask=key_padding_mask,
                             need_weights=False)[0]
        xs = xs.contiguous().view(bz, patch_num, ch_num, patch_size//2).transpose(1, 2)
        # attention applied across patches (time segments)
        xt = self.self_attn_t(xt, xt, xt,
                              attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask,
                              need_weights=False)[0]
        xt = xt.contiguous().view(bz, ch_num, patch_num, patch_size//2)
        x = torch.concat((xs, xt), dim=3)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor of shape
            (batch_size, num_of_channels, patch_num, d_model).
        
        Return
        -------
        Tensor
            Output tensor of shape (batch_size, num_of_channels, patch_num, d_model).
            the feedforward (fully-connected) network.
        """
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoder(nn.Module):
    """
    A class used for assembling stack of TransformerEncoderLayer.
    """
    def __init__(
            self, encoder_layer: TransformerEncoderLayer,
            num_layers: int,
            norm: Optional[nn.Module]=None) -> None:
        """
        Parameters
        ----------
        encoder_layer : nn.Module
            An instance of TransformerEncoderLayer to be cloned.
        num_layers : int
            The number of sub-encoder-layers in the encoder.
        norm : Optional[nn.Module]
            An optional module for final normalisation of the output.
        """
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        src : Tensor
            Input tensor of shape (batch_size, num_of_channels, patch_num, d_model).
        mask : Optional[Tensor]
            Attention mask tensor of shape (patch_num, patch_num)
            or (batch_size, patch_num, patch_num).
            If None, no attention mask is applied.
        src_key_padding_mask : Optional[Tensor]
            Currently a place-holder
            Key padding mask tensor of shape (batch_size, patch_num).
            If None, no key padding mask is applied.

        Returns
        -------
        Tensor
            Output tensor of the same shape as src.
        """

        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError(
            f"activation should be relu/gelu, not {activation}")

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


def _generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
        dtype: torch.dtype = torch.get_default_dtype(),
) -> Tensor:
    r"""Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


if __name__ == '__main__':
    encoder_layer = TransformerEncoderLayer(
        d_model=256, nhead=4, dim_feedforward=1024, batch_first=True,
        activation=F.gelu
    )
    encoder = TransformerEncoder(encoder_layer, num_layers=2)
    encoder = encoder.cuda()

    a = torch.randn((4, 19, 30, 256)).cuda()
    b = encoder(a)
    print(a.shape, b.shape)
