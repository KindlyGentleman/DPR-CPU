"""Replay extension artifacts and archive native text sources; never run a model."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import zipfile

import numpy as np
import psutil

from .reporting import load_campaigns, sha
from ..quality import compare_outputs
from ..statistics import paired_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[2]
    home = root/'cpu_optimization'
    evidence = home/'phase2_extension'
    results = home/'results'
    records, commands = load_campaigns([results/f'phase2_extension_20261016_v{i}' for i in (2, 3, 4, 5)])
    deadline = datetime.fromisoformat(json.loads((evidence/'execution_budget_v5.json').read_text())['absolute_cpu_deadline_utc'])
    errors, sources, workers = [], {}, []
    def record(path):
        sources[str(path)] = sha(path)
    for item in records:
        record(item['path']/'campaign.json')
        if datetime.fromisoformat(item['state']['finished_utc']) > deadline:
            errors.append(f"Campaign exceeded combined deadline: {item['path']}")
    parity_cases, violations = 0, {}
    for command in commands:
        folder = results/command['id']
        worker_path = folder/'workers.json'
        if worker_path.is_file():
            record(worker_path)
            trial_workers = json.loads(worker_path.read_text())
            for worker in trial_workers:
                workers.append(worker['instance_id'])
                if psutil.pid_exists(worker['pid']):
                    process = psutil.Process(worker['pid'])
                    if abs(process.create_time()-worker['process_created_s']) < 1:
                        errors.append(f"Owned worker still alive: {worker['pid']}")
        predictions, validation = folder/'predictions.npz', folder/'validation.json'
        if predictions.is_file() and validation.is_file():
            record(predictions); record(validation)
            cases = json.loads(validation.read_text())['cases']
            with np.load(predictions, allow_pickle=False) as saved:
                for i, case in enumerate(cases):
                    actual = compare_outputs(saved[f'reference_{i}'], saved[f'candidate_{i}'])
                    for key, expected in case['parity'].items():
                        if actual[key] != expected:
                            errors.append(f"Parity replay mismatch: {command['id']} / {i} / {key}")
                    parity_cases += 1
            violations[command['id']] = {
                'strict_failing_cases': sum(not case['parity']['allclose'] for case in cases),
                'violating_elements': sum(case['parity']['violating_elements'] for case in cases)}
    if len(workers) != len(set(workers)):
        errors.append('A worker instance identity was reused between trials')
    conv = results/'phase2_convs_20261016_v3'
    with (conv/'raw_convs.csv').open(newline='') as handle:
        raw = list(csv.DictReader(handle))
    with (conv/'convs.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    record(conv/'raw_convs.csv'); record(conv/'convs.csv'); record(conv/'shapes.json')
    stats_keys = ('baseline_median_s', 'baseline_p95_s', 'candidate_median_s', 'candidate_p95_s',
                  'speedup', 'median_pair_ratio', 'ci_low', 'ci_high')
    for row in rows:
        def cohort(method):
            values = sorted((value for value in raw if value['shape_id'] == row['shape_id']
                             and value['method'] == method), key=lambda value: int(value['repeat']))
            if [int(value['repeat']) for value in values] != list(range(int(row['pairs']))):
                errors.append(f"Missing/duplicate repeat: {row['shape_id']} / {method}")
            return [float(value['seconds']) for value in values]
        actual = paired_summary(cohort('functional'), cohort(row['method']),
                                int(row['bootstrap_seed']), int(row['bootstrap_draws']), int(row['bootstrap_block']))
        for key in stats_keys:
            if not np.isclose(actual[key], float(row[key]), rtol=1e-12, atol=1e-15):
                errors.append(f"Convolution statistic mismatch: {row['shape_id']} / {row['method']} / {key}")
    shapes = json.loads((conv/'shapes.json').read_text())
    if len({row['shape_id'] for row in rows}) != len(shapes):
        errors.append('Incomplete unique-shape characterization')
    research = root.parent/'dpr-research'
    history = json.loads((research/'provenance/pilot_prechange_hashes.json').read_text())
    historical_count = 0
    for name, expected in history.items():
        if name.replace('\\', '/').startswith('results/'):
            historical_count += 1
            if sha(research/name) != expected:
                errors.append(f'Research historical result changed: {name}')
    partial_path = research/'provenance/pilot_v2_partial_audit.json'
    partial = json.loads(partial_path.read_text())
    record(partial_path)
    if partial['partial_evidence_integrity_passed'] is not True or partial['full_grid_gating_conclusion_available']:
        errors.append('Partial pilot audit status invalid')
    cache = Path(records[-1]['state']['native_cache'])
    inventory, text_count = [], 0
    native_zip = args.output.with_name('native_text_sources.zip')
    with zipfile.ZipFile(native_zip, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(cache.rglob('*')):
            if path.is_file():
                inventory.append({'path': path.relative_to(cache).as_posix(), 'bytes': path.stat().st_size,
                                  'sha256': sha(path)})
                if path.suffix in ('.cpp', '.h', '.py', '.txt'):
                    archive.write(path, path.relative_to(cache).as_posix())
                    text_count += 1
    cache_path = args.output.with_name('native_cache_inventory.json')
    cache_path.write_text(json.dumps({'cache': str(cache), 'files': inventory,
                                     'bytes': sum(row['bytes'] for row in inventory),
                                     'scope': 'Shared v4/v5 cache pool, not a per-trial kernel attribution'}, indent=2)+'\n')
    record(cache_path); record(native_zip)
    output = {'passed': not errors, 'errors': errors, 'combined_cpu_deadline_utc': deadline.isoformat(),
              'last_cpu_finished_utc': records[-1]['state']['finished_utc'],
              'unique_worker_instances': len(workers), 'saved_extension_parity_cases_replayed': parity_cases,
              'strict_parity_outcomes': violations, 'convolution_summary_rows_replayed': len(rows),
              'unique_convolution_shapes': len(shapes), 'research_historical_result_files_unchanged': historical_count,
              'native_cache_files': len(inventory), 'native_text_files_archived': text_count,
              'sources_sha256': sources, 'verifier_sha256': sha(Path(__file__)),
              'scope': 'Parent artifact replay/preservation; no model execution or independent scientific validation. '
                       'Whole-model timing/source ZIP replay is in post_execution_audit.json. '
                       'Cache inventory covers the retained v4/v5 pool; native binaries are hashed, text sources archived.'}
    args.output.write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in output.items() if key != 'sources_sha256'}))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
