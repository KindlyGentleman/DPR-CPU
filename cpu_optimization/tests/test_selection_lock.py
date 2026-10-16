import json
import pytest
from cpu_optimization.benchmarks import common


def test_published_replay_requires_matching_immutable_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "HOME", tmp_path)
    with pytest.raises(ValueError, match="selection"):
        common.require_selection("reference", 6, [0, 1])
    (tmp_path / "config").mkdir()
    (tmp_path / "config/selection.json").write_text(
        json.dumps(
            {"selected": {"backend": "reference", "threads": 6, "affinity": [0, 1]}}
        )
    )
    common.require_selection("reference", 6, [0, 1])
    with pytest.raises(ValueError, match="locked"):
        common.require_selection("prepack_bmm", 6, [0, 1])
    with pytest.raises(ValueError, match="locked"):
        common.require_selection("reference", 4, [0, 1])
