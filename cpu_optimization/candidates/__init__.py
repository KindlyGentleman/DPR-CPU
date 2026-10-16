"""Isolated factory: mutate only a freshly loaded experimental model."""

import torch
from .spectral import SpectralGEMM, PackedFourierUnit, BufferFourierUnit

NAMES = {
    "reference",
    "spectral_linear",
    "spectral_bmm",
    "spectral_matmul",
    "packed_gemm",
    "static_buffers",
    "compile_default",
    "compile_reduce",
    "compile_autotune",
    "compile_selective",
    "local_channels_last",
    "encoder_channels_last",
    "mkldnn_prepack",
    "jit_trace",
    "prepack_bmm",
    "phase2_winograd",
    "phase2_tiled",
    "phase2_tiled_small",
    "phase2_unfold",
}


class LocalChannelsLast(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer.to(memory_format=torch.channels_last)

    def forward(self, x):
        return self.layer(x.contiguous(memory_format=torch.channels_last))


def apply_candidate(model, name):
    from deeppr.ffc import FourierUnit

    if name not in NAMES:
        raise ValueError(f"Unknown candidate {name}")
    if name == "reference":
        return model
    if name.startswith('phase2_'):
        from ..phase2_extension.prototypes import replace_hot_convolutions

        replaced = replace_hot_convolutions(model, name.removeprefix('phase2_'))
        if not replaced:
            raise ValueError('No eligible hot convolutions; refusing an identity prototype trial')
        model._phase2_replaced_layers = replaced
        return model.eval()
    if name == "prepack_bmm":
        return apply_candidate(apply_candidate(model, "mkldnn_prepack"), "spectral_bmm")
    if name == "jit_trace":
        from .compiler import TracedNetwork

        return TracedNetwork(model).eval()
    if name.startswith("compile_"):
        from torch import _dynamo

        _dynamo.config.suppress_errors = False
        if name == "compile_selective":
            # FFTs remain eager; independent spectral pointwise kernels compile.
            for module in model.modules():
                if isinstance(module, FourierUnit):
                    module.conv_layer = torch.compile(
                        module.conv_layer, fullgraph=False, dynamic=False
                    )
            return model
        mode = {
            "compile_default": "default",
            "compile_reduce": "reduce-overhead",
            "compile_autotune": "max-autotune",
        }[name]
        return torch.compile(model, fullgraph=False, dynamic=False, mode=mode)
    if name == "encoder_channels_last":
        from .layout import EncoderLayout

        for i in range(4):
            model.model[i] = EncoderLayout(model.model[i], first=i == 0, last=i == 3)
        return model.eval()
    if name == "mkldnn_prepack":
        from .layout import PrepackedConv

        # Only spatial 3x3 ordinary convolutions; WPC and spectral 1x1 untouched.
        for parent in list(model.modules()):
            for key, module in list(parent.named_children()):
                if type(module) is torch.nn.Conv2d and module.kernel_size == (3, 3):
                    setattr(parent, key, PrepackedConv(module))
        return model.eval()
    for parent in list(model.modules()):
        for key, module in list(parent.named_children()):
            if isinstance(module, FourierUnit):
                if name.startswith("spectral_"):
                    module.conv_layer = SpectralGEMM(
                        module.conv_layer, name.split("_")[1]
                    )
                elif name == "packed_gemm":
                    setattr(parent, key, PackedFourierUnit(module))
                elif name == "static_buffers":
                    setattr(parent, key, BufferFourierUnit(module))
                elif name == "local_channels_last":
                    module.conv_layer = LocalChannelsLast(module.conv_layer)
    return model.eval()
