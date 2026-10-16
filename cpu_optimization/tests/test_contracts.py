import copy
import numpy as np
import pytest
import torch
from pathlib import Path

torch.set_num_threads(4)

from cpu_optimization.api import prepare_inputs, verify_checkpoint, DPRCPU
from cpu_optimization.candidates.spectral import (
    SpectralGEMM,
    PackedFourierUnit,
    BufferFourierUnit,
)
from cpu_optimization.statistics import paired_summary, decision
from cpu_optimization.quality import image_metrics, compare_outputs


def test_input_semantics_signed_and_no_alias():
    x = np.ones((512, 512), np.float32)
    x[0, 0] = -0.4
    m = np.ones_like(x, bool)
    m[0, 1] = False
    a, b = prepare_inputs(x, m)
    assert a.shape == b.shape == (1, 1, 512, 512)
    assert a[0, 0, 0, 0] == np.float32(-0.4)
    assert b[0, 0, 0, 1] == 0 and b.sum() == 512 * 512 - 1
    a.zero_()
    assert x[0, 1] == 1


@pytest.mark.parametrize(
    "fault", ["shape", "mask", "nan", "overflow", "complex", "zero"]
)
def test_invalid_input(fault):
    x = np.ones((512, 512), np.float64)
    m = np.ones_like(x)
    if fault == "shape":
        x = x[:4]
    if fault == "mask":
        m[0, 0] = 0.5
    if fault == "nan":
        x[0, 0] = np.nan
    if fault == "overflow":
        x[0, 0] = 1e100
    if fault == "complex":
        x = x.astype(complex)
    if fault == "zero":
        m[:] = 0
    with pytest.raises(ValueError):
        prepare_inputs(x, m)


def test_checkpoint_hash_rejects_before_deserialize(tmp_path):
    f = tmp_path / "bad.pt"
    f.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_checkpoint(f)
    assert verify_checkpoint(Path("pretrained/param_dpr1.pt")).startswith("823572aa")


def test_checkpoint_hash_supports_python310_api(monkeypatch):
    import hashlib

    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    assert verify_checkpoint(Path("pretrained/param_dpr1.pt")).startswith("823572aa")


@pytest.mark.parametrize("kind", ["linear", "bmm", "matmul"])
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_spectral_gemm_group_order(kind, groups):
    torch.manual_seed(23)
    conv = torch.nn.Conv2d(8, 12, 1, groups=groups, bias=True).eval()
    x = torch.randn(2, 8, 8, 5)
    replacement = SpectralGEMM(conv, kind)
    with torch.inference_mode():
        torch.testing.assert_close(replacement(x), conv(x), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            replacement(x[:, :, ::2]), conv(x[:, :, ::2]), rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize(
    "shape,groups",
    [((1, 4, 8, 8), 1), ((1, 8, 16, 16), 2), ((2, 8, 12, 10), 4), ((1, 16, 32, 32), 1)],
)
@pytest.mark.parametrize("wrapper", [PackedFourierUnit, BufferFourierUnit])
def test_fourier_order_normalization_and_state(shape, groups, wrapper):
    from deeppr.ffc import FourierUnit

    torch.manual_seed(34)
    unit = FourierUnit(shape[1], shape[1], groups).eval()
    candidate = wrapper(copy.deepcopy(unit)).eval()
    a, b = torch.randn(shape), torch.randn(shape)
    with torch.inference_mode():
        out = candidate(a)
        saved = out.clone()
        torch.testing.assert_close(out, unit(a), rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(candidate(b), unit(b), rtol=1e-4, atol=1e-5)
        assert torch.equal(out, saved)
        torch.testing.assert_close(candidate(a), saved, rtol=0, atol=0)


def test_block_bootstrap_and_incomplete_cannot_promote():
    b = np.ones(36)
    c = b * 0.8
    s = paired_summary(b, c, seed=4)
    assert s["speedup"] == pytest.approx(1.25)
    assert s["ci_low"] == pytest.approx(1.25)
    assert decision(s, True, True, True) == "accepted"
    assert decision(s, False, True, True) == "rejected"
    assert decision(s, True, False, True) == "inconclusive"
    assert decision(s, True, True, False) == "inconclusive"
    with pytest.raises(ValueError):
        paired_summary([1], [0])


def test_quality_distinguishes_parity_from_reconstruction():
    truth = np.zeros((64, 64))
    truth[20:30, 20:30] = 0.1
    poor = np.ones_like(truth) * 0.05
    assert image_metrics(truth, poor)["failure"] == 1
    assert compare_outputs(poor, poor)["allclose"]
    shifted = np.roll(truth, 3, 0)
    assert image_metrics(truth, shifted)["aligned_nrmse"] < 1e-12


def test_relative_tolerance_is_anchored_to_reference():
    # This boundary passes if the candidate incorrectly sets its own tolerance.
    reference = np.array([10000.0])
    candidate = np.array([10001.00008])
    assert not compare_outputs(reference, candidate)["allclose"]


def test_api_auto_reference_and_explicit_unknown():
    with pytest.raises(ValueError, match="backend"):
        DPRCPU.from_pretrained(backend="not_real", threads=4)
    # No model allocation required to inspect the locked auto policy.
    assert DPRCPU.auto_backend() == "reference"


def test_api_disables_ambient_cpu_autocast():
    class Probe(torch.nn.Module):
        def forward(self, x, m, false_scale=False):
            assert not torch.is_autocast_enabled(
                "cpu"
            ), "DPR must remain FP32 under ambient autocast"
            assert torch.is_inference_mode_enabled()
            return x[:, :, 224:288, 224:288]

    api = DPRCPU(Probe(), "reference", None, False)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = api.predict(np.ones((512, 512), np.float32), np.ones((512, 512), bool))
    assert output.dtype == np.float32
