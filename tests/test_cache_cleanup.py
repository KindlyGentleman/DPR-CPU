import importlib.util
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def cleanup():
    path=ROOT/'scripts/cache_cleanup.py'
    assert path.is_file(), 'Missing cache cleanup implementation'
    spec=importlib.util.spec_from_file_location('cache_cleanup',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

@pytest.fixture
def repo(tmp_path):
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    (tmp_path/'.gitignore').write_text('__pycache__/\n.pytest_cache/\n')
    cache=tmp_path/'nested/__pycache__'; cache.mkdir(parents=True)
    (cache/'x.pyc').write_bytes(b'12345')
    (tmp_path/'data').mkdir(); (tmp_path/'data/keep').write_text('evidence')
    return tmp_path

def test_dry_run_and_apply_remove_only_allowlisted_cache(cleanup,repo):
    report=cleanup.clean(repo,apply=False,active=[])
    assert report['removed_bytes']==0
    assert report['caches'][0]['bytes']==5
    assert (repo/'nested/__pycache__/x.pyc').exists()
    result=cleanup.clean(repo,apply=True,active=[])
    assert result['removed_bytes']==5
    assert (repo/'data/keep').exists()

def test_tracked_cache_is_refused(cleanup,repo):
    subprocess.run(['git','-C',str(repo),'add','-f','nested/__pycache__/x.pyc'],check=True)
    with pytest.raises(ValueError,match='tracked'):
        cleanup.clean(repo,apply=True,active=[])
    assert (repo/'nested/__pycache__/x.pyc').exists()

def test_active_benchmark_blocks_deletion(cleanup,repo):
    with pytest.raises(ValueError,match='active'):
        cleanup.clean(repo,apply=True,active=[{'pid':123}])

def test_symlink_escape_refused(cleanup,repo,tmp_path):
    outside=repo.parent/'outside'; outside.mkdir(exist_ok=True)
    (outside/'keep').write_text('evidence')
    link=repo/'__pycache__'
    # Junctions work without Windows symlink privilege.
    if __import__('os').name=='nt':
        result=subprocess.run(['cmd','/c','mklink','/J',str(link),str(outside)],capture_output=True)
        if result.returncode: pytest.skip('junction creation unavailable')
    else: link.symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='link|reparse'):
        cleanup.clean(repo,apply=True,active=[])
    assert (outside/'keep').exists()

def test_external_cache_requires_separate_optin_and_proof(cleanup,repo):
    with pytest.raises(ValueError,match='external'):
        cleanup.clean(repo,apply=True,active=[],external=repo.parent/'cache')

def test_root_itself_and_outside_target_are_refused(cleanup,repo):
    for target in (repo,repo.parent/'__pycache__'):
        with pytest.raises(ValueError): cleanup.validate_target(repo,target)

def test_native_compiler_and_uninspectable_python_block_cleanup(cleanup,repo,monkeypatch):
    from types import SimpleNamespace
    import psutil
    for info in ({'pid':321,'name':'cl.exe','cmdline':['cl.exe','/c','generated.cpp'],'cwd':str(repo)},
                 {'pid':322,'name':'python.exe','cmdline':None,'cwd':None}):
        monkeypatch.setattr(psutil,'process_iter',lambda attrs:iter([SimpleNamespace(pid=info['pid'],info=info)]))
        assert cleanup.owned_processes(repo)

def test_compiler_names_do_not_match_unrelated_services(cleanup,repo,monkeypatch):
    from types import SimpleNamespace
    import psutil
    process=SimpleNamespace(pid=321,info={'pid':321,'name':'OneApp.IGCC.WinService.exe','cmdline':None,'cwd':None})
    monkeypatch.setattr(psutil,'process_iter',lambda attrs:iter([process]))
    assert cleanup.owned_processes(repo)==[]
