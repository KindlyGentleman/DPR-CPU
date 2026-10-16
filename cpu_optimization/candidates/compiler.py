"""Compiler paths that do not require a new native extension at installation."""

import torch


class TracedNetwork(torch.nn.Module):
    """Static-shape TorchScript trace; masks remain runtime tensor inputs.

    No optimize_for_inference/Conv-BN folding is requested. First-use tracing is
    counted as startup and is never included in warmed comparison clocks.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
        self.traced = None

    def forward(self, x, m, false_scale=False):
        if false_scale:
            raise ValueError("Traced backend supports false_scale=False only")
        if self.traced is None:
            self.model(x, m, false_scale=False)
            self.traced = torch.jit.trace(
                self.model, (x, m), check_trace=True, strict=False
            )
        return self.traced(x, m)
