"""Explicit feature-FFT counting and oneDNN primitive evidence extraction."""

import csv
import hashlib
import io
import json
import math
from pathlib import Path


def minimum_hot_speedup(fraction, target=1.10):
    denominator = 1/target-(1-fraction)
    return fraction/denominator if denominator > 0 and 0 < fraction <= 1 else math.inf


def select_prototypes(rows, hot_fraction):
    """Predeclared screen only; every selected method still needs full-model gates."""
    minimum = minimum_hot_speedup(hot_fraction)
    selected, decisions = [], {}
    for kind in ('winograd', 'tiled', 'tiled_small', 'unfold'):
        matching = [row for row in rows if row['method'] == kind
                    and json.loads(row.get('kernel_input_shape', row['input_shape'])) == [1, 256, 66, 66]]
        if not matching:
            decisions[kind] = {'eligible': False, 'reason': 'representative hot shape not measured'}
            continue
        row = max(matching, key=lambda item: int(item['calls_per_forward']))
        parity = str(row['strict_parity']).lower() == 'true'
        eligible = parity and float(row['speedup']) >= minimum and float(row['ci_low']) > 1.
        decisions[kind] = {'eligible': eligible, 'local_speedup': float(row['speedup']),
                           'minimum_local_speedup_for_network_target': minimum if math.isfinite(minimum) else None,
                           'strict_operator_parity': parity,
                           'reason': 'full-model gate warranted' if eligible else 'operator parity or credible full-model gain screen not met'}
        if eligible:
            selected.append('phase2_'+kind)
    return selected, decisions


def parse_onednn(text):
    rows = []
    for line in text.splitlines():
        if not line.startswith(('onednn_verbose,', 'dnnl_verbose,')):
            continue
        values = next(csv.reader(io.StringIO(line)))
        if 'exec' not in values:
            continue
        kind = next((v for v in ('convolution', 'reorder', 'eltwise', 'batch_normalization') if v in values), None)
        if kind is None:
            continue
        pos = values.index(kind)
        try:
            elapsed = float(values[-1])
        except ValueError:
            continue
        rows.append({'primitive': kind, 'implementation': values[pos+1], 'elapsed_ms': elapsed,
                     'details': ','.join(values[pos+2:-1]), 'raw': line})
    return rows


def profile_report(folder, previous=None):
    folder = Path(folder)
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    with (folder/'operators.csv').open(encoding='utf-8', newline='') as handle:
        operators = list(csv.DictReader(handle))
    repeats = summary['repeats']
    rfft = sum(int(row['count']) for row in operators if row['name'] == 'FU.rfft2')/repeats
    irfft = sum(int(row['count']) for row in operators if row['name'] == 'FU.irfft2')/repeats
    if summary['fourier_calls_per_forward'] != 42 or rfft != 42 or irfft != 42:
        raise ValueError(f'Expected 42 FourierUnits / 84 feature FFTs; found {rfft} rfft2 and {irfft} irfft2')
    if not summary['bitwise_instrumentation']:
        raise ValueError('Instrumented output differs from untouched reference')
    stage_ms = {k: v/1000 for k, v in summary['scope_times_us'].items() if k.startswith('stage.')}
    total = sum(float(row['self_cpu_us']) for row in operators)
    convs = [row for row in operators if row['name'] in ('aten::mkldnn_convolution', 'aten::_slow_conv2d_forward')]
    hottest = max(convs, key=lambda row: float(row['self_cpu_us'])) if convs else None
    target_rows = []
    for row in convs:
        try:
            shapes = json.loads(row['input_shapes'])
        except ValueError:
            continue
        if shapes[:2] == [[1, 256, 66, 66], [256, 256, 3, 3]]:
            target_rows.append(row)
    report = {'scope': 'instrumented diagnostics, not deployment timing or a speedup experiment',
              'instrumentation_bitwise_identical': True, 'fourier_units_per_forward': 42,
              'rfft2_per_forward': rfft, 'irfft2_per_forward': irfft, 'feature_ffts_per_forward': rfft+irfft,
              'physics_fft_events_not_in_feature_count': True, 'disjoint_stage_ms': stage_ms,
              'disjoint_stage_total_ms': sum(stage_ms.values()),
              'fft_self_fraction': summary['fft_self_fraction'],
              'fft_only_ideal_speedup_bound': summary['fft_only_ideal_speedup_bound'],
              'hottest_convolution': None if hottest is None else {
                  'operator': hottest['name'], 'shapes': hottest['input_shapes'],
                  'calls_per_forward': int(hottest['count'])/repeats,
                  'operator_self_fraction': float(hottest['self_cpu_us'])/total},
              'target_3x3_operator': None if not target_rows else {
                  'operators': [row['name'] for row in target_rows],
                  'calls_per_forward': sum(int(row['count']) for row in target_rows)/repeats,
                  'operator_self_fraction': sum(float(row['self_cpu_us']) for row in target_rows)/total,
                  'input_shape': [1, 256, 66, 66], 'weight_shape': [256, 256, 3, 3]},
              'source_profile': str(folder),
              'source_sha256': {name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
                                for name in ('summary.json', 'operators.csv')}}
    if previous:
        earlier = json.loads((Path(previous)/'summary.json').read_text(encoding='utf-8'))
        report['previous_profile'] = {'path': str(previous), 'fourier_units': earlier['fourier_calls_per_forward'],
                                      'fft_self_fraction': earlier['fft_self_fraction'],
                                      'disjoint_stage_ms': {k: v/1000 for k, v in earlier['scope_times_us'].items() if k.startswith('stage.')}}
        report['comparison_note'] = 'Independent instrumented sessions; differences are not optimization gains'
    return report
