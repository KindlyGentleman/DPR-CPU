"""Local convolution layouts; explicit boundaries preserve detector/FFT layouts."""

import torch
from torch import nn
from torch.nn import functional as F


class PrepackedConv(nn.Module):
    """FP32 oneDNN weights, with reflect padding outside the packed convolution."""

    def __init__(self, conv):
        super().__init__()
        from torch.utils.mkldnn import to_mkldnn

        self.padding_mode = conv.padding_mode
        self.explicit_padding = conv._reversed_padding_repeated_twice
        if self.padding_mode != "zeros":
            conv.padding = (0, 0)
        self.packed = to_mkldnn(conv.eval(), dtype=torch.float32)

    def forward(self, x):
        if self.padding_mode != "zeros":
            x = F.pad(x, self.explicit_padding, mode=self.padding_mode)
        return self.packed(x.to_mkldnn()).to_dense()


class EncoderLayout(nn.Module):
    """Only WPC encoder uses NHWC; final encoder boundary restores NCHW."""

    def __init__(self, layer, first=False, last=False):
        super().__init__()
        self.layer = layer.to(memory_format=torch.channels_last)
        self.first, self.last = first, last

    def forward(self, x):
        if self.first:
            a, m = x
            x = (a.contiguous(memory_format=torch.channels_last), m)
        out = self.layer(x)
        if self.last:
            if isinstance(out, tuple):
                return (out[0].contiguous(), out[1])
            return out.contiguous()
        return out
