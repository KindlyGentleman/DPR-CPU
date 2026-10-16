"""Dense 3x3 inference prototypes, retaining padding, channel and batch semantics.

Winograd uses the F(2x2,3x3) input/filter/output transforms. Unfold/GEMM is
deliberately separate. Algebraic equivalence does not establish FP32 parity or
full-model performance; those require measured gates.
"""

import copy
import torch
from torch import nn
from torch.nn import functional as F


def _check_layer(layer):
    if (layer.kernel_size != (3, 3) or layer.stride != (1, 1)
            or layer.dilation != (1, 1) or layer.groups != 1):
        raise ValueError('Prototype requires dense 3x3, groups=1, stride=1, dilation=1')
    if layer.padding_mode != 'zeros' or layer.padding not in ((0, 0), (1, 1)):
        raise ValueError('Only symmetric zero padding 0 or 1 supported; external reflect padding remains external')
    if layer.weight.dtype not in (torch.float32, torch.float64):
        raise ValueError('FP32 or FP64 mathematical validation only')


class FrozenConv(nn.Module):
    def __init__(self, layer):
        super().__init__()
        _check_layer(layer)
        self.register_buffer('weight', layer.weight.detach().clone())
        self.register_buffer('bias', None if layer.bias is None else layer.bias.detach().clone())
        self.padding = layer.padding[0]
        self.in_channels = layer.in_channels
        self.out_channels = layer.out_channels

    def _shape(self, x):
        if (x.ndim != 4 or x.shape[1] != self.in_channels or x.dtype != self.weight.dtype
                or x.device != self.weight.device):
            raise ValueError('NCHW input must match convolution channels, dtype and device')
        h, w = x.shape[-2]+2*self.padding-2, x.shape[-1]+2*self.padding-2
        if h < 1 or w < 1:
            raise ValueError('Input too small for 3x3 convolution')
        return h, w


class WinogradF2Conv(FrozenConv):
    transform_algorithm = 'F(2x2,3x3)'

    def __init__(self, layer):
        super().__init__(layer)
        bt = self.weight.new_tensor([[1, 0, -1, 0], [0, 1, 1, 0],
                                     [0, -1, 1, 0], [0, 1, 0, -1]])
        g = self.weight.new_tensor([[1, 0, 0], [.5, .5, .5], [.5, -.5, .5], [0, 0, 1]])
        at = self.weight.new_tensor([[1, 1, 1, 0], [0, 1, -1, -1]])
        self.register_buffer('bt', bt)
        self.register_buffer('at', at)
        # Frozen filter preparation, outside warm timing; retain its setup cost.
        transformed = g @ self.weight @ g.t()
        self.register_buffer('transformed_weight', transformed.permute(2, 3, 0, 1).contiguous().reshape(
            16, self.out_channels, self.in_channels))

    def forward(self, x):
        h, w = self._shape(x)
        p = self.padding
        padded = F.pad(x, (p, p+w % 2, p, p+h % 2))
        patches = F.unfold(padded, kernel_size=4, stride=2)
        n, channels, tiles = x.shape[0], x.shape[1], patches.shape[-1]
        d = patches.reshape(n, channels, 4, 4, tiles).permute(0, 1, 4, 2, 3)
        v = self.bt @ d @ self.bt.t()
        v = v.permute(3, 4, 1, 0, 2).contiguous().reshape(16, channels, n*tiles)
        m = torch.bmm(self.transformed_weight, v)
        m = m.reshape(4, 4, self.out_channels, n, tiles).permute(3, 2, 4, 0, 1)
        y = self.at @ m @ self.at.t()
        packed = y.permute(0, 1, 3, 4, 2).contiguous().reshape(n, self.out_channels*4, tiles)
        result = F.fold(packed, (h+h % 2, w+w % 2), kernel_size=2, stride=2)[..., :h, :w]
        if self.bias is not None:
            result = result+self.bias.view(1, -1, 1, 1)
        return result


class TiledConv(FrozenConv):
    def __init__(self, layer, tile=16, materialize_tiles=False):
        super().__init__(layer)
        if type(tile) is not int or tile < 1:
            raise ValueError('Positive integer tile required')
        self.tile = tile
        self.materialize_tiles = materialize_tiles

    def forward(self, x):
        h, w = self._shape(x)
        if self.padding:
            x = F.pad(x, (self.padding,)*4)
        result = x.new_empty(x.shape[0], self.out_channels, h, w)
        for top in range(0, h, self.tile):
            for left in range(0, w, self.tile):
                bottom, right = min(h, top+self.tile), min(w, left+self.tile)
                patch = x[..., top:bottom+2, left:right+2]
                if self.materialize_tiles:
                    patch = patch.contiguous()
                result[..., top:bottom, left:right] = F.conv2d(patch, self.weight, self.bias)
        return result


class UnfoldConv(FrozenConv):
    def forward(self, x):
        h, w = self._shape(x)
        columns = F.unfold(x, kernel_size=3, padding=self.padding)
        result = torch.matmul(self.weight.reshape(self.out_channels, -1), columns)
        result = result.reshape(x.shape[0], self.out_channels, h, w)
        if self.bias is not None:
            result = result+self.bias.view(1, -1, 1, 1)
        return result


class ReflectPaddedPrototype(nn.Module):
    """Retain the original Conv2d's internal reflect-padding boundary."""
    def __init__(self, core, padding):
        super().__init__()
        self.core, self.padding = core, padding

    def forward(self, x):
        return self.core(F.pad(x, self.padding, mode='reflect'))


def eligible_hot_convolution(layer):
    return (type(layer) is nn.Conv2d and layer.in_channels == layer.out_channels == 256
            and layer.kernel_size == (3, 3) and layer.stride == layer.dilation == (1, 1)
            and layer.groups == 1 and ((layer.padding_mode == 'zeros' and layer.padding == (0, 0))
                                      or (layer.padding_mode == 'reflect' and layer.padding == (1, 1))))


def prototype_conv(layer, kind, tile=16):
    if layer.padding_mode == 'reflect' and layer.padding == (1, 1):
        # Copy only metadata; frozen cores clone tensors and never mutate the source.
        unpadded = copy.copy(layer)
        unpadded.padding, unpadded.padding_mode = (0, 0), 'zeros'
        unpadded._reversed_padding_repeated_twice = (0, 0, 0, 0)
        core = prototype_conv(unpadded, kind, tile)
        return ReflectPaddedPrototype(core, layer._reversed_padding_repeated_twice).eval()
    if kind == 'winograd':
        return WinogradF2Conv(layer).eval()
    if kind == 'tiled':
        return TiledConv(layer, tile).eval()
    if kind == 'tiled_small':
        # Captured batch-one FP32 256-channel tile: 16 KiB input + 4 KiB output.
        # Weight packing/streaming is separate; no measured L1-residency claim.
        return TiledConv(layer, tile=2, materialize_tiles=True).eval()
    if kind == 'unfold':
        return UnfoldConv(layer).eval()
    raise ValueError(f'Unknown prototype {kind}')


def replace_hot_convolutions(model, kind, tile=16):
    """Replace only original dense 256x256x3x3 ordinary layers in a fresh model."""
    replaced = []
    for name, layer in list(model.named_modules()):
        if eligible_hot_convolution(layer):
            parent_name, _, child_name = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, prototype_conv(layer, kind, tile))
            replaced.append(name)
    return replaced
