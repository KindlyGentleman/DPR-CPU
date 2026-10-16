import numpy as np
import pytest
import torch
from cpu_optimization.fft.plans import FFTWPlan


@pytest.mark.parametrize("norm", ["ortho", "backward"])
@pytest.mark.parametrize("shape", [(1, 2, 8, 8), (1, 3, 16, 16)])
def test_reused_fftw_plans_preserve_scale_and_inputs(norm, shape):
    pytest.importorskip("pyfftw")
    rng = np.random.default_rng(2)
    a = rng.normal(size=shape).astype("float32")
    saved = a.copy()
    f = FFTWPlan(shape, "forward", norm, 1)
    inv = FFTWPlan(shape, "inverse", norm, 1)
    reference = torch.fft.rfft2(torch.from_numpy(a), norm=norm)
    out = f.bridge(torch.from_numpy(a))
    torch.testing.assert_close(out, reference, rtol=1e-4, atol=1e-5)
    held = out.clone()
    reconstructed = inv.bridge(out)
    torch.testing.assert_close(reconstructed, torch.from_numpy(a), rtol=1e-4, atol=1e-5)
    assert np.array_equal(a, saved) and torch.equal(out, held)
    f.bridge(torch.zeros_like(torch.from_numpy(a)))
    assert torch.equal(out, held)
