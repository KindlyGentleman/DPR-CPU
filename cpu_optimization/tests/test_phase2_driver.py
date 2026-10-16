"""The serial driver must stop safely before starting unverified work."""

import importlib.util
import json
from pathlib import Path

import pytest


def test_preflight_failure_has_terminal_state_and_does_not_launch_children(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1]/'phase2_extension/run_after_pilot.py'
    spec = importlib.util.spec_from_file_location('phase2_serial_driver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path/'cpu'
    home = root/'cpu_optimization'
    monkeypatch.setattr(module, 'ROOT', root)
    monkeypatch.setattr(module, 'HOME', home)
    pilot = tmp_path/'research/results/pilot/run.json'
    pilot.parent.mkdir(parents=True)
    pilot.write_text(json.dumps({'status':'completed','raw_rows':24,'expected_rows':24}))
    monkeypatch.setattr('sys.argv', ['driver', '--pilot-run', str(pilot), '--pilot-pid', '99999999', '--suffix', 'fixture'])
    with pytest.raises(FileNotFoundError):
        module.main()
    state = json.loads((home/'results/phase2_extension_fixture/campaign.json').read_text())
    assert state['status'] == 'failed'
    assert state['commands'] == []
    assert 'exception' in state


def test_compiled_worker_can_use_explicit_bounded_startup_timeout(monkeypatch):
    from cpu_optimization.benchmarks.threading import start_worker
    calls = []
    class Connection:
        def poll(self, value):
            calls.append(value); return True
        def recv(self):
            return {'ready': True}
        def close(self):
            pass
    class Process:
        def start(self):
            pass
    class Context:
        def Pipe(self):
            return Connection(), Connection()
        def Process(self, **kwargs):
            return Process()
    start_worker(Context(), {'threads': 4, 'affinity': [0], 'backend': 'compile_default'},
                 'checkpoint', 'development', 10, startup_timeout=600)
    assert calls == [600]


def test_driver_refuses_to_continue_when_owned_process_survives_cleanup(monkeypatch):
    path = Path(__file__).resolve().parents[1]/'phase2_extension/run_after_pilot.py'
    spec = importlib.util.spec_from_file_location('phase2_cleanup_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class Process:
        pid = 123
        def children(self, recursive):
            return []
        def terminate(self):
            pass
        def kill(self):
            pass
    child = Process()
    monkeypatch.setattr(module.psutil, 'Process', lambda pid: child)
    monkeypatch.setattr(module.psutil, 'wait_procs', lambda processes, timeout: ([], processes))
    with pytest.raises(RuntimeError, match='surviv'):
        module.terminate_owned(child)


def test_screen_with_failed_memory_gate_cannot_consume_finalist_confirmation_budget():
    path = Path(__file__).resolve().parents[1]/'phase2_extension/run_after_pilot.py'
    spec = importlib.util.spec_from_file_location('phase2_finalist_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = dict(quality_pass=True, memory_gate=True, speedup=1.2, ci_low=1.15,
                  candidate_p95_s=.8, baseline_p95_s=1.)
    assert module.promising_screen(result)
    assert not module.promising_screen({**result, 'memory_gate': False})
    assert not module.promising_screen({**result, 'memory_gate': None})


def test_native_cache_avoids_installed_torch_unquoted_source_path_bug(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1]/'phase2_extension/run_after_pilot.py'
    spec = importlib.util.spec_from_file_location('phase2_cache_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.tempfile, 'gettempdir', lambda: str(tmp_path))
    assert module.native_cache_path('fixture') == tmp_path/'dpr_phase2_fixture'
    with pytest.raises(ValueError, match='space|whitespace'):
        module.native_cache_path('fixture', tmp_path/'path with spaces')


def test_compiler_recovery_requires_completed_characterization_context(tmp_path):
    path = Path(__file__).resolve().parents[1]/'phase2_extension/run_after_pilot.py'
    spec = importlib.util.spec_from_file_location('phase2_recovery_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prior = tmp_path/'prior'
    prior.mkdir()
    state = {'status': 'running_cpu_extension', 'commands': [dict(name=name, exit_code=0)
             for name in ('baseline_t4', 'profile', 'onednn', 'convs')]}
    (prior/'campaign.json').write_text(json.dumps(state))
    with pytest.raises(ValueError, match='terminal|completed'):
        module.previous_characterization(prior)
    state['status'] = 'executed_pending_report_review'
    (prior/'campaign.json').write_text(json.dumps(state))
    result = module.previous_characterization(prior)
    assert result['timing_denominators_reused'] is False
    assert result['manifest_sha256']
    state['commands'].pop()
    (prior/'campaign.json').write_text(json.dumps(state))
    with pytest.raises(ValueError, match='characterization'):
        module.previous_characterization(prior)


def test_duplicate_compiler_subset_is_rejected_before_creating_trial_artifacts(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1]/'phase2_extension/run_after_pilot.py'
    spec = importlib.util.spec_from_file_location('phase2_compiler_subset_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'HOME', tmp_path/'outputs')
    monkeypatch.setattr('sys.argv', ['driver', '--pilot-run', str(tmp_path/'unused.json'),
                                   '--pilot-pid', '99999999', '--suffix', 'duplicate',
                                   '--compiler-candidates', 'compile_default', 'compile_default'])
    with pytest.raises(ValueError, match='Duplicate compiler'):
        module.main()
    assert not module.HOME.exists()
