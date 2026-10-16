"""Experimental pointwise kernels; no BatchNorm folding or precision changes."""

import torch
from torch import nn
from torch.nn import functional as F


class SpectralGEMM(nn.Module):
    def __init__(self, conv, kind="linear"):
        super().__init__()
        self.weight = nn.Parameter(conv.weight.detach().clone(), requires_grad=False)
        self.bias = (
            None
            if conv.bias is None
            else nn.Parameter(conv.bias.detach().clone(), requires_grad=False)
        )
        self.groups, self.kind = conv.groups, kind

    def forward(self, x):
        b, c, h, w = x.shape
        g = self.groups
        weight = self.weight[:, :, 0, 0]
        if self.kind == "linear":
            a = x.permute(0, 2, 3, 1)
            if g == 1:
                out = F.linear(a, weight, self.bias)
            else:
                out = torch.cat(
                    [
                        F.linear(
                            a[..., i * c // g : (i + 1) * c // g],
                            weight[
                                i
                                * weight.shape[0]
                                // g : (i + 1)
                                * weight.shape[0]
                                // g
                            ],
                            (
                                None
                                if self.bias is None
                                else self.bias[
                                    i
                                    * weight.shape[0]
                                    // g : (i + 1)
                                    * weight.shape[0]
                                    // g
                                ]
                            ),
                        )
                        for i in range(g)
                    ],
                    -1,
                )
            return out.permute(0, 3, 1, 2)
        a = x.reshape(b, g, c // g, h * w)
        mat = weight.reshape(g, -1, c // g)
        if self.kind == "bmm":
            out = torch.bmm(
                mat.unsqueeze(0).expand(b, -1, -1, -1).reshape(b * g, -1, c // g),
                a.reshape(b * g, c // g, h * w),
            ).reshape(b, -1, h, w)
        else:
            out = torch.matmul(mat.unsqueeze(0), a).reshape(b, -1, h, w)
        if self.bias is not None:
            out = out + self.bias[None, :, None, None]
        return out


class PackedFourierUnit(nn.Module):
    """NHWC interleaved GEMM and complex strided inverse view.

    The inverse view uses even non-last strides, last stride one. Group slices
    preserve [real(c0),imag(c0),real(c1),imag(c1),...] exactly.
    """

    def __init__(self, unit):
        super().__init__()
        self.conv_layer, self.bn, self.relu = unit.conv_layer, unit.bn, unit.relu
        self.groups = unit.groups

    def forward(self, x):
        b, c, h, w = x.shape
        z = torch.fft.rfft2(x, norm="ortho")
        a = (
            torch.view_as_real(z)
            .permute(0, 2, 3, 1, 4)
            .reshape(b, h, w // 2 + 1, c * 2)
        )
        weight = self.conv_layer.weight[:, :, 0, 0]
        g = self.groups
        if g == 1:
            y = F.linear(a, weight, self.conv_layer.bias)
        else:
            oc = weight.shape[0]
            y = torch.cat(
                [
                    F.linear(
                        a[..., i * c * 2 // g : (i + 1) * c * 2 // g],
                        weight[i * oc // g : (i + 1) * oc // g],
                        (
                            None
                            if self.conv_layer.bias is None
                            else self.conv_layer.bias[i * oc // g : (i + 1) * oc // g]
                        ),
                    )
                    for i in range(g)
                ],
                -1,
            )
        y = self.relu(self.bn(y.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
        z = torch.view_as_complex(
            y.reshape(b, h, w // 2 + 1, -1, 2).permute(0, 3, 1, 2, 4)
        )
        return torch.fft.irfft2(z, s=(h, w), norm="ortho")


class BufferFourierUnit(nn.Module):
    """Instance-local scratch packing; outputs never alias reusable buffers."""

    def __init__(self, unit):
        super().__init__()
        self.conv_layer, self.bn, self.relu = unit.conv_layer, unit.bn, unit.relu
        self.groups = unit.groups
        self._packed, self._inverse = None, None

    def forward(self, x):
        if torch.is_grad_enabled():
            raise RuntimeError("Static buffer backend requires inference_mode")
        b, c, h, w = x.shape
        z = torch.fft.rfft2(x, norm="ortho")
        shp = (b, c, 2, h, w // 2 + 1)
        if self._packed is None or tuple(self._packed.shape) != shp:
            self._packed = torch.empty(shp, dtype=x.dtype, device=x.device)
        self._packed.copy_(torch.view_as_real(z).permute(0, 1, 4, 2, 3))
        y = self.relu(
            self.bn(self.conv_layer(self._packed.view(b, c * 2, h, w // 2 + 1)))
        )
        shp = (b, y.shape[1] // 2, h, w // 2 + 1, 2)
        if self._inverse is None or tuple(self._inverse.shape) != shp:
            self._inverse = torch.empty(shp, dtype=x.dtype, device=x.device)
        self._inverse.copy_(y.view(b, -1, 2, h, w // 2 + 1).permute(0, 1, 3, 4, 2))
        return torch.fft.irfft2(
            torch.view_as_complex(self._inverse), s=(h, w), norm="ortho"
        )
