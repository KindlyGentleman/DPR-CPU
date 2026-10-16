import json
import zipfile
import pytest
from cpu_optimization.benchmarks import common, threading


def test_sampled_rss_cannot_satisfy_peak_memory_gate():
    peaks, gate = threading.memory_evidence([[{"rss": 100}], [{"rss": 90}]])
    assert peaks == {"reference_worker_bytes": None, "candidate_worker_bytes": None}
    assert gate is None
    peaks, gate = threading.memory_evidence(
        [[{"rss": 100, "peak_wset": 150}], [{"rss": 90, "peak_wset": 180}]]
    )
    assert peaks["candidate_worker_bytes"] == 180
    assert gate is False


def test_source_archive_survives_parent_directory_named_results(tmp_path, monkeypatch):
    root = tmp_path / "results" / "checkout"
    home = root / "cpu_optimization"
    (root / "deeppr").mkdir(parents=True)
    (home / "results" / "diagnostic").mkdir(parents=True)
    (root / "deeppr" / "kernel.py").write_text("preserve = True\n")
    (home / "results" / "diagnostic" / "graph.py").write_text("exclude = True\n")
    monkeypatch.setattr(common, "ROOT", root)
    monkeypatch.setattr(common, "HOME", home)
    monkeypatch.setattr(common, "environment", lambda: {})
    monkeypatch.setattr(common, "command", lambda *args: "test")
    run = common.new_run("nested", "reference", tmp_path / "evidence")
    with zipfile.ZipFile(run / "source.zip") as archive:
        assert "deeppr/kernel.py" in archive.namelist()
        assert not any("graph.py" in name for name in archive.namelist())


def test_error_before_run_still_reaches_stderr(monkeypatch, capsys):
    monkeypatch.setattr(common, "_ACTIVE_RUN", None)

    def fail():
        raise RuntimeError("early diagnostic retained")

    with pytest.raises(SystemExit):
        common.cli(fail)
    assert "early diagnostic retained" in capsys.readouterr().err


def test_prepack_bmm_combination_contains_both_isolated_changes():
    import copy
    import torch
    from deeppr.ffc import FourierUnit
    from cpu_optimization.candidates import NAMES, apply_candidate
    from cpu_optimization.candidates.layout import PrepackedConv
    from cpu_optimization.candidates.spectral import SpectralGEMM

    assert "prepack_bmm" in NAMES
    original = torch.nn.Sequential(
        torch.nn.Conv2d(8, 8, 3, padding=1, padding_mode="reflect"), FourierUnit(8, 8)
    ).eval()
    candidate = apply_candidate(copy.deepcopy(original), "prepack_bmm")
    assert isinstance(candidate[0], PrepackedConv)
    assert isinstance(candidate[1].conv_layer, SpectralGEMM)
    assert type(original[0]) is torch.nn.Conv2d
    x = torch.randn(1, 8, 16, 16)
    with torch.inference_mode():
        torch.testing.assert_close(candidate(x), original(x), rtol=1e-4, atol=1e-5)


def test_explicit_affinity_respects_available_cpu_set(monkeypatch):
    import sys
    from types import SimpleNamespace
    from cpu_optimization import api

    calls = []

    class Process:
        def cpu_affinity(self, cpus=None):
            if cpus is None:
                return [0, 1, 2, 3]
            calls.append(cpus)

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=Process))
    api.configure_affinity([2, 0])
    assert calls == [[0, 2]]
    with pytest.raises(ValueError):
        api.configure_affinity([4])
    with pytest.raises(ValueError):
        api.configure_affinity([])


def test_default_affinity_needs_no_optional_dependency(monkeypatch):
    import sys
    from cpu_optimization import api

    monkeypatch.setitem(sys.modules, "psutil", None)
    api.configure_affinity(None)
    with pytest.raises(ImportError, match="psutil"):
        api.configure_affinity([0])
