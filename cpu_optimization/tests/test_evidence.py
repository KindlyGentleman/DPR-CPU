from pathlib import Path
import numpy as np
import pytest
from cpu_optimization.benchmarks.common import new_run, write_json, load_cases, topology


def test_immutable_trial_and_real_dev_inputs(tmp_path):
    run = new_run("test", "reference", tmp_path)
    assert (run / "run.json").exists()
    with pytest.raises(FileExistsError):
        new_run("test", "reference", tmp_path)
    assert load_cases()[0]["intensity"].shape == (512, 512)
    assert all(c["split"] == "development" for c in load_cases())


def test_topology_affinities_are_explicit():
    info = topology()
    assert info["logical_count"] >= 1
    if info["cores"]:
        cpus = [c for core in info["cores"] for c in core["logical_cpus"]]
        assert len(cpus) == len(set(cpus))


def test_process_failure_is_terminal_and_keeps_partial_data(tmp_path):
    import json
    from cpu_optimization.benchmarks.common import record_failed_trial

    run = new_run("failed", "reference", tmp_path)
    (run / "partial.csv").write_text("keep me")
    record_failed_trial(run, "process timeout")
    assert json.loads((run / "run.json").read_text())["status"] == "blocked"
    assert json.loads((run / "summary.json").read_text())["status"] == "blocked"
    before = (run / "summary.json").read_bytes()
    record_failed_trial(run, "different failure")
    assert before == (run / "summary.json").read_bytes()
    assert (run / "partial.csv").read_text() == "keep me"
