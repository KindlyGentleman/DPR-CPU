"""Scientific data and CPU wrapper contracts; historical timing is never rerun."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'cpu_study'


def wrapper():
    spec = importlib.util.spec_from_file_location('reproduce', DATA / 'reproduce.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('problem', ['shape', 'mask', 'nan', 'empty'])
def test_invalid_detector_input_is_rejected(problem):
    intensity = np.ones((512, 512), dtype=np.float32)
    valid = np.ones((512, 512), dtype=bool)
    if problem == 'shape':
        intensity = intensity[:64, :64]
    elif problem == 'mask':
        valid = np.full((512, 512), 2)
    elif problem == 'nan':
        intensity[0, 0] = np.nan
    else:
        valid[:] = False
    with pytest.raises(ValueError):
        wrapper().validate_inputs(intensity, valid)


def test_signed_measurements_are_preserved_with_valid_mask():
    intensity = np.ones((512, 512), dtype=np.float32)
    intensity[0, 0] = -0.4
    valid = np.ones((512, 512), dtype=bool)
    x, m = wrapper().validate_inputs(intensity, valid)
    assert x.shape == (1, 1, 512, 512)
    assert x[0, 0, 0, 0] == np.float32(-0.4)
    assert m.dtype == np.float32 and m.sum() == 512 * 512


def test_checkpoint_mismatch_is_rejected_before_loading():
    checkpoint = ROOT / 'LICENSE.txt'  # authentic file, definitely not the checkpoint
    with pytest.raises(ValueError, match='SHA-256'):
        wrapper().verify_checkpoint(checkpoint)


def test_reproduction_cannot_overwrite_archived_evidence():
    with pytest.raises(ValueError):
        wrapper().validate_output_path(DATA / 'examples.npz')


def test_reconstruction_drift_stops_reproduction():
    reference = np.full((64, 64), .01, dtype=np.float32)
    prediction = reference.copy()
    prediction[0, 0] = .0102
    with pytest.raises(RuntimeError, match='parity'):
        wrapper().verify_reference(prediction, reference)


def test_evidence_hashes_and_base_denominator():
    meta = json.loads((DATA / 'provenance.json').read_text())
    # A clean clone must not depend on local outputs from fresh demonstrations.
    assert set(meta['included_data_sha256']) == {
        'results.csv', 'final_metrics.csv', 'paired_timings.csv', 'examples.npz'}
    for name, digest in meta['included_data_sha256'].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest
    import csv
    with (DATA / 'final_metrics.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 570
    assert len({r['sample_id'] for r in rows}) == 570
    assert len({r['base_id'] for r in rows}) == 15
    assert sum(int(r['failure']) for r in rows) == 251
    assert all(int(r['failure']) == (float(r['aligned_nrmse']) > .5) for r in rows)


def test_paired_intervals_replay_from_recorded_calls():
    import csv
    with (DATA / 'results.csv').open(newline='') as f:
        summaries = [r for r in csv.DictReader(f) if r['section'] == 'candidate']
    with (DATA / 'paired_timings.csv').open(newline='') as f:
        timings = list(csv.DictReader(f))
    meta = json.loads((DATA / 'provenance.json').read_text())
    for summary in summaries:
        config = meta['paired_bootstrap'][summary['name']]
        pairs = sorted((r for r in timings if r['method'] == summary['name']),
                       key=lambda r: int(r['pair_id']))
        b = np.array([float(r['reference_s']) for r in pairs])
        c = np.array([float(r['candidate_s']) for r in pairs])
        assert len(pairs) == 36
        rng = np.random.default_rng(config['seed'])
        starts = rng.integers(0, 36, (config['draws'], 6))
        idx = ((starts[:, :, None] + np.arange(6)) % 36).reshape(config['draws'], 36)
        ci = np.quantile(np.median(b[idx], axis=1) / np.median(c[idx], axis=1), [.025, .975])
        np.testing.assert_allclose(ci, [float(summary['low']), float(summary['high'])], rtol=1e-10)
        assert ci[0] < 1 < ci[1]


def test_reliability_intervals_resample_independent_objects():
    import csv
    with (DATA / 'final_metrics.csv').open(newline='') as f:
        raw = list(csv.DictReader(f))
    with (DATA / 'results.csv').open(newline='') as f:
        summaries = [r for r in csv.DictReader(f) if r['section'] == 'reliability']
    bases = sorted({int(r['base_id']) for r in raw})
    for r in summaries:
        means = np.array([np.mean([float(x[r['metric']]) for x in raw if int(x['base_id']) == b]) for b in bases])
        rng = np.random.default_rng(2026101519)
        ci = np.quantile(means[rng.integers(0, 15, (2000, 15))].mean(axis=1), [.025, .975])
        np.testing.assert_allclose(ci, [float(r['low']), float(r['high'])], rtol=1e-10)
