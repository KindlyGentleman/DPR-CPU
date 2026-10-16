import csv
import json

import pytest


def test_compiler_discovery_does_not_accept_incomplete_installation(tmp_path):
    from cpu_optimization.phase2_extension.toolchain import discover_vc
    folder = tmp_path/'BuildTools'
    folder.mkdir()
    instance = {'installationPath': str(folder), 'isComplete': False, 'isLaunchable': False}
    result = discover_vc([instance])
    assert result['cl_executables'] == []
    assert result['vcvars64'] == []
    compiler = folder/'VC/Tools/MSVC/14.29/bin/Hostx64/x64/cl.exe'
    compiler.parent.mkdir(parents=True)
    compiler.write_bytes(b'fixture only; never executed')
    assert discover_vc([instance])['cl_executables'] == [str(compiler)]


def test_activation_uses_child_environment_without_mutating_parent(monkeypatch, tmp_path):
    from cpu_optimization.phase2_extension.toolchain import activated_environment
    from types import SimpleNamespace
    import os
    script = tmp_path/'vcvars64.bat'
    script.write_text('@echo off')
    monkeypatch.setenv('PATH', 'original')
    def activate(command, **kwargs):
        assert str(script) in command[-1]
        return SimpleNamespace(returncode=0, stdout='PATH=compiler-bin\nINCLUDE=compiler-headers\nLIB=compiler-lib\n')
    monkeypatch.setattr('subprocess.run', activate)
    result = activated_environment(script)
    assert result['PATH'] == 'compiler-bin'
    assert result['INCLUDE'] == 'compiler-headers'
    assert os.environ['PATH'] == 'original'


def test_explicit_vc_environment_uses_only_verified_installed_layout(tmp_path):
    from cpu_optimization.phase2_extension.toolchain import explicit_vc_environment
    vc = tmp_path/'VC/Tools/MSVC/14.29'
    sdk = tmp_path/'Windows Kits/10'
    compiler = vc/'bin/Hostx64/x64/cl.exe'
    for path in (compiler.parent, vc/'include', vc/'lib/x64', sdk/'Include/10.0/ucrt',
                 sdk/'Include/10.0/shared', sdk/'Include/10.0/um', sdk/'Lib/10.0/ucrt/x64',
                 sdk/'Lib/10.0/um/x64', sdk/'bin/10.0/x64'):
        path.mkdir(parents=True, exist_ok=True)
    compiler.write_bytes(b'not executed')
    result = explicit_vc_environment(compiler, sdk, '10.0')
    assert str(vc/'include') in result['INCLUDE']
    assert str(sdk/'Lib/10.0/um/x64') in result['LIB']
    assert result['VSCMD_ARG_TGT_ARCH'] == 'x64'
    assert result['DISTUTILS_USE_SDK'] == '1'
    compiler.unlink()
    with pytest.raises(ValueError, match='compiler'):
        explicit_vc_environment(compiler, sdk, '10.0')


def test_python_import_library_is_added_to_child_linker_search_only(tmp_path, monkeypatch):
    from cpu_optimization.phase2_extension.toolchain import python_development_environment
    import os
    import sys
    prefix = tmp_path/'Python'
    (prefix/'include').mkdir(parents=True)
    (prefix/'libs').mkdir()
    (prefix/'include/Python.h').write_text('fixture only')
    library = prefix/'libs'/f'python{sys.version_info.major}{sys.version_info.minor}.lib'
    library.write_bytes(b'fixture only')
    monkeypatch.setenv('LIB', 'parent-library-path')
    result = python_development_environment({'LIB': 'vc;sdk'}, prefix)
    assert str(prefix/'libs') in result['LIB'].split(';')
    assert result['LIB'].endswith('vc;sdk')
    assert os.environ['LIB'] == 'parent-library-path'
    library.unlink()
    with pytest.raises(ValueError, match='Python|python'):
        python_development_environment({'LIB': 'vc;sdk'}, prefix)


def test_onednn_parser_retains_actual_implementation_and_reorders():
    from cpu_optimization.phase2_extension.evidence import parse_onednn
    log = '\n'.join([
        'onednn_verbose,info,oneDNN v3.5.3',
        'onednn_verbose,primitive,exec,cpu,convolution,jit:avx2,forward_inference,src_f32,a:0,alg:convolution_direct,mb1_ic256_oc256,12.5',
        'onednn_verbose,primitive,exec,cpu,reorder,jit:uni,undef,src_f32,dst_f32,0.3',
        'unrelated stdout'])
    rows = parse_onednn(log)
    assert len(rows) == 2
    assert rows[0]['implementation'] == 'jit:avx2'
    assert rows[0]['primitive'] == 'convolution'
    assert rows[0]['elapsed_ms'] == 12.5
    assert rows[1]['primitive'] == 'reorder'


def test_profile_report_checks_feature_fft_counts_and_keeps_physics_separate(tmp_path):
    from cpu_optimization.phase2_extension.evidence import profile_report
    summary = {'repeats': 3, 'fourier_calls_per_forward': 42, 'bitwise_instrumentation': True,
               'fft_self_fraction': .02, 'fft_only_ideal_speedup_bound': 1/.98,
               'scope_times_us': {'stage.wpc': 180000., 'stage.ffc_merge': 520000.}}
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    fields = ['name', 'self_cpu_us', 'inclusive_cpu_us', 'count', 'input_shapes']
    rows = [dict(zip(fields, values)) for values in [
        ('FU.rfft2', 0., 30000., 126, '[]'), ('FU.irfft2', 0., 60000., 126, '[]'),
        ('aten::_fft_c2c', 10000., 10000., 6, '[]'),
        ('aten::mkldnn_convolution', 1000000., 1000000., 54, '[[1, 256, 66, 66], [256, 256, 3, 3]]')]]
    with (tmp_path/'operators.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    report = profile_report(tmp_path)
    assert report['feature_ffts_per_forward'] == 84
    assert report['fourier_units_per_forward'] == 42
    assert report['physics_fft_events_not_in_feature_count'] is True
    assert report['hottest_convolution']['calls_per_forward'] == 18
    assert report['target_3x3_operator']['calls_per_forward'] == 18
    assert report['disjoint_stage_total_ms'] == 700.
    rows[0]['count'] = 123
    with (tmp_path/'operators.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match='FFT'):
        profile_report(tmp_path)


def test_prototype_screen_uses_amdahl_bound_and_never_relaxes_parity():
    from cpu_optimization.phase2_extension.evidence import select_prototypes, minimum_hot_speedup
    assert minimum_hot_speedup(.426, 1.1) == pytest.approx(1.2713, rel=1e-3)
    rows = [dict(method=kind, input_shape='[1, 256, 66, 66]', speedup=speed, ci_low=low,
                 strict_parity=parity, calls_per_forward=18)
            for kind, speed, low, parity in [('winograd', 1.5, 1.3, True),
                                            ('tiled', 1.1, 1.05, True),
                                            ('unfold', 1.6, 1.4, False)]]
    selected, decisions = select_prototypes(rows, .426)
    assert selected == ['phase2_winograd']
    assert decisions['unfold']['eligible'] is False
    assert decisions['tiled']['eligible'] is False


def test_compiler_probe_rejects_nonfinite_and_wrong_precision_outputs():
    from cpu_optimization.phase2_extension.toolchain import output_statistics
    import torch
    reference = torch.tensor([0., 1.], dtype=torch.float32)
    invalid = output_statistics(reference, torch.tensor([float('nan'), 1.]))
    assert invalid['passed'] is False
    assert invalid['violating_elements'] == 1
    assert invalid['max_absolute_error'] is None
    assert output_statistics(reference, reference.double())['passed'] is False
    assert output_statistics(reference, reference.clone())['passed'] is True
