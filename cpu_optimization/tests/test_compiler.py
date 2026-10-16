import pytest
import torch
from cpu_optimization.candidates.compiler import TracedNetwork


def test_traced_network_preserves_dynamic_mask_values():
    class Model(torch.nn.Module):
        def forward(self, x, m, false_scale=False):
            return torch.relu(x * m + 1)

    eager = Model().eval()
    traced = TracedNetwork(eager).eval()
    a = torch.randn(1, 1, 8, 8)
    first = torch.ones_like(a)
    second = first.clone()
    second[:, :, 3:6, 3:6] = 0
    with torch.inference_mode():
        assert torch.equal(traced(a, first), eager(a, first))
        assert torch.equal(traced(a, second), eager(a, second))
    assert traced.traced is not None
    with pytest.raises(ValueError, match="false_scale"):
        traced(a, first, false_scale=True)
