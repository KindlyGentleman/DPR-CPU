"""Serialize verified CPU experiments after the separately authorized pilot exits.

Invoke this file directly so waiting does not import Torch or hold a model.
Every command, timeout, skipped prerequisite and partial trial is retained.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
HOME = ROOT/'cpu_optimization'


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def terminate_owned(child):
    try:
        processes = [psutil.Process(child.pid), *psutil.Process(child.pid).children(recursive=True)]
    except psutil.NoSuchProcess:
        return
    for process in reversed(processes):
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=3)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, survivors = psutil.wait_procs(alive, timeout=3)
    if survivors:
        raise RuntimeError('Owned child processes survived cleanup; refuse overlapping CPU experiments: '+
                           ', '.join(str(process.pid) for process in survivors))


def promising_screen(summary):
    return (summary.get('quality_pass') is True and summary.get('memory_gate') is True
            and summary.get('speedup', 0) >= 1.10 and summary.get('ci_low', 0) > 1
            and summary['candidate_p95_s'] <= 1.05*summary['baseline_p95_s'])


def native_cache_path(suffix, requested=None):
    path = Path(requested or Path(tempfile.gettempdir())/('dpr_phase2_'+suffix)).resolve()
    if any(character.isspace() for character in str(path)):
        raise ValueError('Native cache path must contain no whitespace for installed Torch 2.5.1 compiler commands')
    return path


def previous_characterization(folder):
    folder = Path(folder).resolve()
    path = folder/'campaign.json'
    prior = json.loads(path.read_text(encoding='utf-8'))
    if prior['status'] != 'executed_pending_report_review':
        raise ValueError('A completed terminal prior campaign is required for compiler recovery')
    finished = {item['name'] for item in prior['commands'] if item.get('exit_code') == 0}
    if not {'baseline_t4', 'profile', 'onednn', 'convs'} <= finished:
        raise ValueError('Prior baseline/profile/oneDNN/convolution characterization must have finished')
    return {'path': str(folder), 'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'scope': 'Prior characterization cited for context only; all new quality/timing arms run fresh',
            'timing_denominators_reused': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-run', type=Path, required=True)
    parser.add_argument('--pilot-pid', type=int, required=True)
    parser.add_argument('--suffix', default='20261016_v1')
    parser.add_argument('--max-cpu-seconds', type=float, default=7200)
    parser.add_argument('--native-cache', type=Path, help='Fresh cache without spaces; default is a task-specific system-temp folder')
    parser.add_argument('--compiler-only-evidence', type=Path,
                        help='Completed earlier characterization campaign; recover compiler trials with fresh paired arms')
    parser.add_argument('--compiler-candidates', nargs='+', choices=['compile_default', 'compile_selective'],
                        default=['compile_default', 'compile_selective'], help='Explicit compiler subset for a bounded recovery')
    args = parser.parse_args()
    if len(args.compiler_candidates) != len(set(args.compiler_candidates)):
        raise ValueError('Duplicate compiler candidates would reuse a trial ID; refused before artifact creation')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,48}', args.suffix) or not 0 < args.max_cpu_seconds <= 7200:
        raise ValueError('Safe unique suffix and a CPU campaign bound <=7200 seconds required')
    folder = HOME/'results'/('phase2_extension_'+args.suffix)
    folder.mkdir(parents=True, exist_ok=False)
    state_path = folder/'campaign.json'
    state = {'status': 'waiting_for_pilot', 'created_utc': datetime.now(timezone.utc).isoformat(),
             'driver_pid': os.getpid(),
             'pilot_run': str(args.pilot_run.resolve()), 'pilot_pid': args.pilot_pid,
             'max_cpu_seconds': args.max_cpu_seconds, 'commands': [], 'not_run': [],
             'native_cache': str(native_cache_path(args.suffix, args.native_cache)),
             'native_cache_existed_at_start': native_cache_path(args.suffix, args.native_cache).exists(),
             'requested_compiler_candidates': args.compiler_candidates,
             'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(state_path, state)
    last_cohort = -1
    try:
        pilot_created = psutil.Process(args.pilot_pid).create_time()
    except psutil.NoSuchProcess:
        pilot_created = None
    while True:
        pilot = json.loads(args.pilot_run.read_text(encoding='utf-8'))
        cohort = pilot.get('raw_rows', 0)//256
        if cohort != last_cohort:
            state['pilot_progress'] = {key: pilot.get(key) for key in ('status', 'raw_rows', 'expected_rows', 'wall_seconds_excluding_model_setup')}
            write(state_path, state)
            print('Pilot: '+json.dumps(state['pilot_progress']), flush=True)
            last_cohort = cohort
        try:
            process = psutil.Process(args.pilot_pid)
            active = process.is_running() and process.create_time() == pilot_created
        except psutil.NoSuchProcess:
            active = False
        if not active:
            break
        time.sleep(5)
    state['pilot_terminal_status'] = pilot['status']
    state['status'] = 'running_cpu_extension'
    state['cpu_started_utc'] = datetime.now(timezone.utc).isoformat()
    write(state_path, state)
    deadline = time.monotonic()+args.max_cpu_seconds
    try:
        if pilot['status'] == 'running':
            raise RuntimeError('Pilot PID disappeared without terminal state; refuse potentially overlapping model work')
        evidence = load_file('phase2_evidence', HOME/'phase2_extension/evidence.py')
        toolchain = load_file('phase2_toolchain', HOME/'phase2_extension/toolchain.py')
        diagnostic = toolchain.diagnose()
        write(folder/'toolchain_preflight.json', diagnostic)
        environment = dict(os.environ)
        sdk_versions = sorted(set(diagnostic['windows_sdk_include_versions']) & set(diagnostic['windows_sdk_lib_versions']))
        if diagnostic['cl_executables'] and sdk_versions:
            environment = toolchain.explicit_vc_environment(diagnostic['cl_executables'][-1], diagnostic['windows_sdk_root'], sdk_versions[-1])
        environment = toolchain.python_development_environment(environment)
        environment.update(PYTHONUTF8='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                           KMP_BLOCKTIME='0', TORCHINDUCTOR_COMPILE_THREADS='1', CXX='cl')
        state['compiler_environment'] = {key: environment.get(key) for key in ('CXX', 'INCLUDE', 'LIB', 'TORCHINDUCTOR_COMPILE_THREADS')}
        if args.compiler_only_evidence:
            state['prior_characterization'] = previous_characterization(args.compiler_only_evidence)
            write(folder/'prior_characterization.json', state['prior_characterization'])
        write(state_path, state)
    except BaseException as exc:
        state.update(status='failed', exception=repr(exc), finished_utc=datetime.now(timezone.utc).isoformat())
        write(state_path, state)
        raise

    def command(name, options, limit=600, interpreter=None, cache=False):
        remaining = deadline-time.monotonic()
        if remaining < 20:
            raise TimeoutError('Declared CPU extension wall budget exhausted before '+name)
        ident = 'phase2_'+name+'_'+args.suffix
        log_path = HOME/'results'/(ident+'.log')
        env = dict(environment)
        if cache:
            env['TORCHINDUCTOR_CACHE_DIR'] = state['native_cache']
            env['TORCH_LOGS'] = 'dynamo'
        cmd = [str(interpreter or sys.executable), '-u', *map(str, options)]
        entry = {'name': name, 'id': ident, 'command': cmd, 'log': str(log_path),
                 'started_utc': datetime.now(timezone.utc).isoformat(), 'timeout_s': min(limit, remaining)}
        if cache:
            entry['native_cache'] = env['TORCHINDUCTOR_CACHE_DIR']
            entry['compiler_logs'] = 'TORCH_LOGS=dynamo; backend compilation metrics retained in log'
        state['commands'].append(entry)
        write(state_path, state)
        print('Running '+subprocess.list2cmdline(cmd), flush=True)
        start = time.perf_counter()
        with log_path.open('x', encoding='utf-8') as log:
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                entry['exit_code'] = child.wait(timeout=min(limit, remaining))
            except BaseException as exc:
                terminate_owned(child)
                entry['exit_code'] = None
                entry['exception'] = repr(exc)
                log.write('\nOwned process tree stopped; partial files preserved: '+repr(exc)+'\n')
                if not isinstance(exc, subprocess.TimeoutExpired):
                    raise
        entry['wall_s'] = time.perf_counter()-start
        summary = HOME/'results'/ident/'summary.json'
        if summary.exists():
            entry['summary'] = json.loads(summary.read_text(encoding='utf-8'))
        elif (HOME/'results'/ident/'run.json').exists():
            metadata_path = HOME/'results'/ident/'run.json'
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            metadata.update(status='blocked', terminal_reason='Owned process did not finish; see campaign/log',
                            finished_utc=datetime.now(timezone.utc).isoformat())
            write(metadata_path, metadata)
        write(state_path, state)
        print(f"Finished {name}: exit={entry['exit_code']} wall={entry['wall_s']:.1f}s", flush=True)
        time.sleep(2)
        return entry

    def module(name, module_name, extra=(), limit=600, cache=False):
        ident = 'phase2_'+name+'_'+args.suffix
        return command(name, ['-m', module_name, '--id', ident, *extra], limit=limit, cache=cache)

    try:
        research = args.pilot_run.resolve().parents[2]
        research_python = research/'.venv312/Scripts/python.exe'
        if pilot['status'] == 'completed':
            replay = command('pilot_replay', [research/'scripts/audit_pilot.py', '--results', args.pilot_run.parent,
                                     '--output', folder/'pilot_replay_audit.json'], limit=300, interpreter=research_python)
            state['pilot_replay_passed'] = replay['exit_code'] == 0
        else:
            state['not_run'].append({'task': 'full pilot policy inference', 'reason': 'baseline is '+pilot['status']})
        verified = command('scoped_tests', ['-m', 'pytest', 'cpu_optimization/tests', 'tests/test_cpu_study.py', '-q'], limit=600)
        if verified['exit_code'] != 0:
            raise RuntimeError('Scoped CPU tests failed; do not run unverified candidate benchmarks')
        selected = []
        if args.compiler_only_evidence:
            state['not_run'].append({'task': 'repeat characterization/prototype/reference-only trials',
                                     'reason': 'Completed prior evidence retained; this is compiler recovery with fresh paired reference arms'})
        else:
            module('baseline_t4', 'cpu_optimization.benchmarks.baseline', limit=300)
            prof = module('profile', 'cpu_optimization.benchmarks.profile', ['--repeats', '3'], limit=300)
            profile_folder = HOME/'results'/prof['id']
            report = evidence.profile_report(profile_folder, HOME/'results/profile_v2')
            write(folder/'profiling_phase2.json', report)
            verbose = module('onednn', 'cpu_optimization.phase2_extension.verbose', limit=180)
            primitive_rows = evidence.parse_onednn(Path(verbose['log']).read_text(encoding='utf-8', errors='replace'))
            write(folder/'onednn_primitives.json', primitive_rows)
            if not any(row['primitive'] == 'convolution' for row in primitive_rows):
                state['not_run'].append({'task': 'oneDNN implementation attribution', 'reason': 'no convolution primitive lines captured'})
            micro = module('convs', 'cpu_optimization.phase2_extension.convs', ['--profile', profile_folder], limit=900)
            with (HOME/'results'/micro['id']/'convs.csv').open(encoding='utf-8', newline='') as handle:
                local_rows = list(csv.DictReader(handle))
            selected, decisions = evidence.select_prototypes(local_rows, (report['target_3x3_operator'] or {}).get('operator_self_fraction', 0.))
            write(folder/'prototype_screen.json', decisions)
        probe_path = folder/'small_cnn_probe.json'
        command('small_compile', ['-m', 'cpu_optimization.phase2_extension.toolchain', '--probe', '--output', probe_path],
                limit=300, cache=True)
        probe = json.loads(probe_path.read_text(encoding='utf-8')) if probe_path.exists() else {}
        candidates = list(selected)
        if probe.get('small_cnn', {}).get('status') == 'compiled_and_executed':
            candidates += args.compiler_candidates
        else:
            state['not_run'].append({'task': 'full/selective DPR Inductor', 'reason': 'small native compile probe did not pass'})
        screens = []
        for candidate in candidates:
            item = module(candidate+'_screen', 'cpu_optimization.benchmarks.threading',
                          ['--candidate', candidate, '--candidate-threads', '4',
                           '--worker-startup-timeout', '600'], limit=1800, cache=True)
            screens.append(item)
        if not args.compiler_only_evidence:
            module('selected_reference_repeat72', 'cpu_optimization.benchmarks.threading',
                   ['--candidate', 'reference', '--candidate-threads', '6', '--affinity', 'p_only', '--repeats', '72'], limit=1200)
        promising = [item for item in screens if promising_screen(item.get('summary', {}))]
        if promising:
            best = max(promising, key=lambda item: item['summary']['speedup'])['summary']['candidate']
            for session in ('confirm72', 'independent72'):
                module(best+'_'+session, 'cpu_optimization.benchmarks.threading',
                       ['--candidate', best, '--candidate-threads', '4', '--repeats', '72',
                        '--worker-startup-timeout', '600'], limit=1200, cache=True)
            module(best+'_versus_selected72', 'cpu_optimization.benchmarks.threading',
                   ['--candidate', best, '--threads', '6', '--reference-affinity', 'p_only',
                    '--candidate-threads', '4', '--affinity', 'default', '--repeats', '72',
                    '--worker-startup-timeout', '600'], limit=1200, cache=True)
        state['status'] = 'executed_pending_report_review'
    except BaseException as exc:
        state['status'] = 'bounded_stopped' if isinstance(exc, TimeoutError) else 'failed'
        state['exception'] = repr(exc)
        raise
    finally:
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write(state_path, state)
        print(json.dumps({key: value for key, value in state.items() if key != 'commands'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
