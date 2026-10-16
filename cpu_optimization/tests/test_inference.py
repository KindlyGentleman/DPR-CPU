import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from cpu_optimization.api import DPRCPU, load_reference
from cpu_optimization.benchmarks.common import load_cases


@pytest.fixture(scope="module")
def reference():
    return load_reference(Path("pretrained/param_dpr1.pt"))


def test_checkpoint_loading_and_consecutive_sample_isolation(reference):
    state = torch.load(
        "pretrained/param_dpr1.pt", map_location="cpu", weights_only=True
    )["model_state_dict"]
    assert set(state) == set(reference.state_dict())
    assert all(torch.equal(state[k], v) for k, v in reference.state_dict().items())
    model = DPRCPU(reference, "reference", Path("pretrained/param_dpr1.pt"), False)
    a, b = load_cases()[:2]
    first = model.predict(a["intensity"], a["valid"])
    held = first.copy()
    second = model.predict(b["intensity"], b["valid"])
    np.testing.assert_allclose(first, a["archived_reference"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(second, b["archived_reference"], rtol=1e-4, atol=1e-5)
    assert first.dtype == np.float32 and first.shape == (64, 64)
    assert np.array_equal(first, held)
    assert np.array_equal(model.predict(a["intensity"], a["valid"]), first)


@pytest.mark.parametrize(
    "error",
    [
        ImportError("optional backend missing"),
        RuntimeError("native extension load failed"),
    ],
)
def test_optional_backend_failure_is_explicit(monkeypatch, reference, error):
    import cpu_optimization.api as api
    import cpu_optimization.candidates as candidates

    monkeypatch.setattr(api, "load_reference", lambda _: reference)

    def fail(*_):
        raise error

    monkeypatch.setattr(candidates, "apply_candidate", fail)
    with pytest.raises(type(error)):
        DPRCPU.from_pretrained(backend="packed_gemm")
    with pytest.warns(RuntimeWarning, match="falling back"):
        model = DPRCPU.from_pretrained(backend="packed_gemm", allow_fallback=True)
    assert model.backend == "reference"


def test_static_scratch_allocations_are_bounded():
    from deeppr.ffc import FourierUnit
    from cpu_optimization.candidates.spectral import BufferFourierUnit

    mod = BufferFourierUnit(FourierUnit(8, 8).eval())
    with torch.inference_mode():
        mod(torch.randn(1, 8, 16, 16))
        ptr = (mod._packed.data_ptr(), mod._inverse.data_ptr())
        for _ in range(32):
            mod(torch.randn(1, 8, 16, 16))
            assert ptr == (mod._packed.data_ptr(), mod._inverse.data_ptr())
    assert not list(mod.buffers()) or all(not b.requires_grad for b in mod.buffers())


@pytest.mark.parametrize("mask_kind", ["full", "beamstop", "gaps"])
def test_wpc_arbitrary_masks_and_reflect_are_untouched(mask_kind):
    from deeppr.weightedpartialconv2d import WeightedPartialConv2d
    from cpu_optimization.candidates import apply_candidate

    mod = WeightedPartialConv2d(
        1, 4, 3, padding=1, padding_mode="reflect", return_mask=True
    ).eval()
    original = copy.deepcopy(mod)
    x = torch.rand(1, 1, 32, 32)
    m = torch.ones_like(x)
    if mask_kind == "beamstop":
        m[:, :, 12:20, 12:20] = 0
    if mask_kind == "gaps":
        m[:, :, :, ::5] = 0
    # Candidates that only replace FourierUnits cannot modify WPC weights/behavior.
    wrapped = apply_candidate(torch.nn.Sequential(mod), "static_buffers")[0]
    with torch.inference_mode():
        a, am = original(x, m)
        b, bm = wrapped(x, m)
    assert torch.equal(a, b) and torch.equal(am, bm) and bm.dtype == torch.bool


def test_lfu_patch_order_remains_original():
    from deeppr.ffc import SpectralTransform

    class Capture(torch.nn.Module):
        def forward(self, x):
            self.saved = x.clone()
            return torch.zeros_like(x)

    layer = SpectralTransform(8, 16).eval()
    capture = Capture()
    layer.lfu = capture
    x = torch.arange(1 * 8 * 8 * 8, dtype=torch.float32).reshape(1, 8, 8, 8)
    with torch.inference_mode():
        mid = layer.conv1(x)
        layer(x)
    # Original order is horizontal-half outer, vertical-half inner, then channels.
    expected = torch.cat(
        [
            mid[:, :2, :4, :4],
            mid[:, :2, 4:, :4],
            mid[:, :2, :4, 4:],
            mid[:, :2, 4:, 4:],
        ],
        1,
    )
    assert torch.equal(capture.saved, expected)
