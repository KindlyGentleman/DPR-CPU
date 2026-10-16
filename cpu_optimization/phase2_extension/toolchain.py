"""Read-only compiler preflight and an explicit small-CNN Inductor execution probe."""

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback


def discover_vc(instances):
    compilers, scripts = [], []
    for instance in instances:
        root = Path(instance['installationPath'])
        compilers.extend(str(p) for p in root.glob('VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe'))
        path = root/'VC/Auxiliary/Build/vcvars64.bat'
        if path.is_file():
            scripts.append(str(path))
    return {'cl_executables': sorted(compilers), 'vcvars64': sorted(scripts)}


def activated_environment(script):
    """Activate installed VC tools in a child process; never mutate global settings."""
    script = Path(script).resolve()
    if not script.is_file() or any(c in str(script) for c in ('"', '&', '|', '\n', '\r')):
        raise ValueError('An existing, safely quoted vcvars64.bat path is required')
    # Some partially registered VS installations initialize VC successfully but
    # leave a nonzero status from optional tools. Retain that status and inspect
    # the resulting environment; native compilation remains the actual gate.
    command = f'call "{script}" >nul 2>&1 & set "_DPR_VCVARS_EXIT=!errorlevel!" & set'
    completed = subprocess.run([os.environ.get('COMSPEC', 'cmd.exe'), '/d', '/v:on', '/s', '/c', command],
                               capture_output=True, text=True, timeout=60)
    if completed.returncode:
        raise RuntimeError(f'VC activation failed with exit {completed.returncode}; no system settings changed')
    environment = dict(os.environ)
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition('=')
        if separator and key and not key.startswith('='):
            environment[key] = value
    return environment


def explicit_vc_environment(compiler, sdk, version):
    """Local fallback for installed VC/SDK files when installer registration fails."""
    compiler, sdk = Path(compiler).resolve(), Path(sdk).resolve()
    if not compiler.is_file():
        raise ValueError('Installed compiler executable required')
    vc = compiler.parents[3]
    includes = [vc/'include', sdk/f'Include/{version}/ucrt', sdk/f'Include/{version}/shared', sdk/f'Include/{version}/um']
    libraries = [vc/'lib/x64', sdk/f'Lib/{version}/ucrt/x64', sdk/f'Lib/{version}/um/x64']
    if not all(path.is_dir() for path in [*includes, *libraries]):
        raise ValueError('Complete installed VC headers and Windows SDK libraries required')
    environment = dict(os.environ)
    environment.update(PATH=os.pathsep.join([str(compiler.parent), str(sdk/f'bin/{version}/x64'), environment.get('PATH', '')]),
                       INCLUDE=';'.join(map(str, includes)), LIB=';'.join(map(str, libraries)),
                       VSCMD_ARG_TGT_ARCH='x64', DISTUTILS_USE_SDK='1', MSSdk='1')
    return environment


def python_development_environment(environment, prefix=None):
    """Expose the base interpreter's verified import library to a child linker."""
    prefix = Path(prefix or sys.base_prefix).resolve()
    library = prefix/'libs'/f'python{sys.version_info.major}{sys.version_info.minor}.lib'
    if not library.is_file() or not (prefix/'include/Python.h').is_file():
        raise ValueError('Installed Python development header/import library required: '+str(library))
    result = dict(environment)
    result['LIB'] = ';'.join(filter(None, [str(library.parent), result.get('LIB', '')]))
    return result


def diagnose():
    vswhere = Path(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'))/'Microsoft Visual Studio/Installer/vswhere.exe'
    instances, error = [], None
    if vswhere.is_file():
        try:
            instances = json.loads(subprocess.check_output([str(vswhere), '-all', '-products', '*', '-format', 'json'],
                                                           text=True, encoding='utf-8-sig', timeout=30))
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            error = repr(exc)
    sdk = Path(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'))/'Windows Kits/10'
    tools = {name: shutil.which(name) for name in ('cl', 'clang-cl', 'icx-cl', 'link', 'nvcc')}
    versions = {}
    for package in ('torch', 'numpy', 'sympy', 'psutil'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {'platform': platform.platform(), 'python': sys.version, 'executable': sys.executable,
            'python_base_prefix': sys.base_prefix,
            'python_import_library': str(Path(sys.base_prefix)/'libs'/f'python{sys.version_info.major}{sys.version_info.minor}.lib'),
            'child_linker_LIB': os.environ.get('LIB'),
            'versions': versions, 'PATH_tools': tools, 'CXX': os.environ.get('CXX'),
            'vswhere': str(vswhere) if vswhere.exists() else None, 'vswhere_error': error,
            'visual_studio_instances': instances, **discover_vc(instances),
            'windows_sdk_include_versions': sorted(p.name for p in (sdk/'Include').glob('*') if p.is_dir()),
            'windows_sdk_lib_versions': sorted(p.name for p in (sdk/'Lib').glob('*') if p.is_dir()),
            'windows_sdk_root': str(sdk),
            'administrator_token': bool(ctypes.windll.shell32.IsUserAnAdmin()) if sys.platform == 'win32' else None,
            'cpu_only': True, 'nvcc_required': False,
            'compiler_note': 'Windows Torch 2.5.1 requires MSVC-style cl/clang-cl/icx-cl and compatible VC runtime/headers; missing PATH alone is not proof of missing installation',
            'free_disk_gib': shutil.disk_usage(Path.cwd()).free/2**30,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def output_statistics(reference, actual):
    import torch
    same_shape = actual.shape == reference.shape
    finite = bool(torch.isfinite(actual).all())
    same_dtype = actual.dtype == reference.dtype
    if not same_shape:
        return {'passed': False, 'shape_matches': False, 'dtype_matches': same_dtype,
                'max_absolute_error': None, 'violating_elements': int(actual.numel())}
    delta = (actual-reference).abs()
    bad = ~torch.isfinite(actual) | (delta > 1e-5+1e-4*reference.abs())
    return {'passed': finite and same_dtype and not bool(bad.any()), 'shape_matches': True,
            'dtype_matches': same_dtype, 'max_absolute_error': float(delta.max()) if finite else None,
            'violating_elements': int(bad.sum()), 'nonfinite_elements': int((~torch.isfinite(actual)).sum())}


def small_cnn_probe():
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    from torch._dynamo import config
    config.suppress_errors = False
    torch.manual_seed(20261016)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(),
                                torch.nn.Conv2d(8, 8, 3, padding=1), torch.nn.Sigmoid()).eval()
    x = torch.randn(1, 3, 16, 16)
    record = {'torch': torch.__version__, 'backend': 'inductor', 'fullgraph': True,
              'dynamic': False, 'suppress_errors': False, 'input_shape': list(x.shape), 'cpu_only': True}
    started = time.perf_counter()
    try:
        with torch.inference_mode(), torch.autocast('cpu', enabled=False):
            reference = model(x)
            compiled = torch.compile(model, backend='inductor', fullgraph=True, dynamic=False)
            first = time.perf_counter()
            actual = compiled(x)
            record['compile_and_first_call_s'] = time.perf_counter()-first
            from torch._inductor import metrics
            record['generated_kernel_count'] = metrics.generated_kernel_count
            statistics = output_statistics(reference, actual)
            record.update(statistics)
            if not statistics['passed']:
                raise AssertionError('Small-CNN compiled output fails reference-anchored strict gate')
            for _ in range(3):
                compiled(x)
            record['status'] = 'compiled_and_executed'
            record['scope'] = 'small CNN toolchain probe only; no DPR speedup claim'
    except Exception as exc:
        record.update(status='blocked', exception=repr(exc), traceback=traceback.format_exc())
    record['total_probe_s'] = time.perf_counter()-started
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnose', action='store_true')
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--activate', action='store_true', help='Use existing vcvars64 only for this child process')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output path; preserve prior diagnostics')
    result = diagnose()
    if args.activate:
        if not result['vcvars64']:
            raise RuntimeError('No installed vcvars64.bat discovered')
        environment = activated_environment(result['vcvars64'][-1])
        result['activation_method'] = 'existing vcvars64 child environment'
        result['vcvars_exit_status'] = environment.get('_DPR_VCVARS_EXIT')
        if not shutil.which('cl', path=environment.get('PATH', '')):
            versions = sorted(set(result['windows_sdk_include_versions']) & set(result['windows_sdk_lib_versions']))
            if not result['cl_executables'] or not versions:
                raise RuntimeError('Activation did not expose a complete installed compiler/SDK')
            environment = explicit_vc_environment(result['cl_executables'][-1], result['windows_sdk_root'], versions[-1])
            result['activation_method'] = 'explicit child VC/SDK paths; no system changes'
        os.environ.update(environment)  # This CLI process only; no registry/system writes.
        result['activated_cl'] = shutil.which('cl')
        result['activated_vc_include'] = os.environ.get('INCLUDE')
        result['activated_vc_lib'] = os.environ.get('LIB')
    if args.probe:
        os.environ.update(python_development_environment(os.environ))  # Probe process only.
        result['child_linker_LIB'] = os.environ.get('LIB')
        result['small_cnn'] = small_cnn_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
