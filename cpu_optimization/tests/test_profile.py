import copy
import types
import pytest
import torch
from cpu_optimization.benchmarks.profile import wpc_forward


@pytest.mark.parametrize("kind", ["full", "beamstop", "gaps"])
def test_instrumented_wpc_is_bitwise_original(kind):
    from deeppr.weightedpartialconv2d import WeightedPartialConv2d

    torch.set_num_threads(4)
    a = WeightedPartialConv2d(
        1, 4, 3, padding=1, padding_mode="reflect", return_mask=True
    ).eval()
    b = copy.deepcopy(a)
    b.forward = types.MethodType(wpc_forward, b)
    x = torch.rand(1, 1, 32, 32)
    mask = torch.ones_like(x)
    if kind == "beamstop":
        mask[:, :, 12:20, 12:20] = 0
    if kind == "gaps":
        mask[:, :, :, ::5] = 0
    with torch.inference_mode():
        y, ym = a(x, mask)
        z, zm = b(x, mask)
    assert torch.equal(y, z) and torch.equal(ym, zm)
