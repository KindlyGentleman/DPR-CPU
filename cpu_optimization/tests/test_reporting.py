import json
from pathlib import Path
import pytest

from cpu_optimization.benchmarks import summarize


def test_characterization_summary_gets_durable_identity(tmp_path, monkeypatch):
    run = tmp_path / "results" / "profile"
    run.mkdir(parents=True)
    (run / "run.json").write_text(
        json.dumps({"candidate": "reference_profile", "track": "E"})
    )
    (run / "summary.json").write_text(
        json.dumps({"status": "inconclusive", "reason": "profile"})
    )
    monkeypatch.setattr(summarize, "HOME", tmp_path)
    trials = summarize.collect()
    assert trials[0]["summary"]["candidate"] == "reference_profile"
    assert trials[0]["summary"]["track"] == "E"


def test_pool_switch_annotation_requires_paired_timing():
    characterization = {"summary": {"candidate": "reference"}}
    paired_screen = {"summary": {"candidate": "reference", "speedup": 1.12}}
    fixed_workers = {
        "summary": {
            "candidate": "reference",
            "speedup": 1.12,
            "methodology": "persistent_worker_processes",
        }
    }
    assert not summarize.exploratory(characterization)
    assert summarize.exploratory(paired_screen)
    assert not summarize.exploratory(fixed_workers)
