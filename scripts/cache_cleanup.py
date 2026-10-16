"""Conservative repository cache inventory/deletion engine for cleanup_cache.ps1."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

ALLOWLIST={'__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','.hypothesis',
           '.mplconfig','.ipynb_checkpoints'}
SKIP={'.git','node_modules','build','dist'}
COMPILERS={'cl','cl.exe','link','link.exe','ninja','ninja.exe','clang','clang.exe',
           'clang++','clang++.exe','gcc','gcc.exe','g++','g++.exe','cc','cc.exe'}

def linked(path):
    info=path.lstat()
    return path.is_symlink() or bool(getattr(info,'st_file_attributes',0)&stat.FILE_ATTRIBUTE_REPARSE_POINT)

def validate_target(root,target):
    root=Path(root).absolute();target=Path(target).absolute()
    if root.resolve()!=root or target.resolve()==root.resolve() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'outside approved repository or root itself: {target}')
    current=target
    while current!=root:
        if current.exists() and linked(current):raise ValueError(f'link/reparse point refused: {current}')
        current=current.parent
    if linked(root):raise ValueError(f'link/reparse root refused: {root}')
    return target

def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()

def owned_processes(root,external=None):
    import psutil
    roots=[str(Path(root).resolve()).lower()]
    if external:roots.append(str(Path(external).absolute()).lower())
    matches=[]
    for process in psutil.process_iter(['pid','name','cmdline','cwd']):
        if process.pid==os.getpid():continue
        try:
            name=(process.info['name'] or '').lower()
            compiler=name in COMPILERS
            if not (compiler or name.startswith(('python','pypy'))):continue
            if process.info['cmdline'] is None or process.info['cwd'] is None:
                matches.append({'pid':process.pid,'state':'uninspectable compiler/Python process'})
                continue
            command=' '.join(process.info['cmdline'] or []).lower()
            cwd=(process.info['cwd'] or '').lower()
            scoped=any(r in command or cwd==r or cwd.startswith(r+os.sep) for r in roots)
            workload=any(s in command for s in ('cpu_optimization','dpr_research','benchmark','inductor'))
            if scoped and (workload or compiler):matches.append({'pid':process.pid,'name':name,'owned_benchmark':True})
        except (psutil.AccessDenied,psutil.NoSuchProcess):
            # Unknown relevant processes cannot prove idle ownership.
            name=(process.info.get('name') or '').lower()
            if name in COMPILERS or name.startswith(('python','pypy')):
                matches.append({'pid':process.pid,'state':'uninspectable compiler/Python process'})
    return matches

def cache_candidates(root):
    candidates=[]
    for parent,dirs,_ in os.walk(root,followlinks=False):
        retained=[]
        for name in dirs:
            path=Path(parent)/name
            if name in ALLOWLIST:
                validate_target(root,path);candidates.append(path)
            elif name not in SKIP and not name.startswith('.venv') and not linked(path):
                retained.append(name)
            elif linked(path) and name not in SKIP and not name.startswith('.venv'):
                raise ValueError(f'link/reparse traversal refused: {path}')
        dirs[:]=retained
    return sorted(candidates)

def file_inventory(root,path,tracked):
    validate_target(root,path)
    files=[]
    for parent,dirs,names in os.walk(path,followlinks=False):
        for name in dirs+names:
            p=Path(parent)/name
            validate_target(root,p)
            if linked(p):raise ValueError(f'link/reparse descendant refused: {p}')
        files.extend(Path(parent)/name for name in names)
    rel=path.relative_to(root).as_posix()
    tracked_here=any(f==rel or f.startswith(rel+'/') for f in tracked)
    ignored=subprocess.run(['git','-C',str(root),'check-ignore','-q',str(path)],capture_output=True).returncode==0
    return dict(path=str(path),cache_type=path.name,files=len(files),
                bytes=sum(p.stat().st_size for p in files),tracked=tracked_here,ignored=ignored,
                inside_repository=True)

def external_inventory(external):
    """External caches are inventory-only until a separately reviewed ownership policy exists."""
    rows=[]
    for kind,value in [('TORCHINDUCTOR_CACHE_DIR',os.environ.get('TORCHINDUCTOR_CACHE_DIR')),
                       ('TORCH_EXTENSIONS_DIR',os.environ.get('TORCH_EXTENSIONS_DIR')),
                       ('explicit owned native scratch',str(external) if external else None),
                       ('default Torch extensions',str(Path.home()/'.cache/torch_extensions')),
                       ('Windows Torch extensions',str(Path(os.environ.get('LOCALAPPDATA',Path.home()))/'torch_extensions'))]:
        if value:
            p=Path(value).absolute()
            rows.append({'cache_type':kind,'path':str(p),'exists':p.exists(),
                         'inside_repository':False,'deletion':'not authorized by repository ownership'})
    return rows

def clean(root,apply=False,active=None,external=None,include_owned_external=False):
    root=Path(root).absolute()
    if not root.is_dir() or root.resolve()!=root or linked(root):raise ValueError('repository root contains link/reparse')
    top=Path(git(root,'rev-parse','--show-toplevel')).resolve()
    if top!=root.resolve():raise ValueError('exact Git repository root required')
    inspect_active=active is None
    active=owned_processes(root,external) if inspect_active else active
    tracked=git(root,'ls-files','-z').split('\0')
    caches=[file_inventory(root,p,tracked) for p in cache_candidates(root)]
    report={'mode':'apply' if apply else 'dry-run','repository':str(root),'active_owned_processes':active,
            'caches':caches,'external_caches':external_inventory(external),
            'removed_bytes':0,'removed_files':0,'removed_paths':[]}
    # Always emit the full inventory before validating any destructive request.
    print(json.dumps(report,indent=2),flush=True)
    if not apply:return report
    if active:raise ValueError('active or uninspectable owned benchmark/compiler process; deletion refused')
    if external or include_owned_external:
        raise ValueError('external deletion requires separately reviewed exact ownership proof, idle check, and preserved compiler logs/source evidence; inventory only')
    if any(c['tracked'] for c in caches):raise ValueError('tracked cache file; backup and reviewed index removal required')
    for item in caches:
        path=Path(item['path'])
        # Recheck ownership, tracking, contents and process state immediately before each removal.
        current=file_inventory(root,path,set(git(root,'ls-files','-z').split('\0')))
        if current!=item:raise ValueError(f'cache changed during inventory: {path}')
        if inspect_active and owned_processes(root):raise ValueError('active benchmark appeared')
        shutil.rmtree(validate_target(root,path))
        report['removed_bytes']+=item['bytes'];report['removed_files']+=item['files']
        report['removed_paths'].append(str(path))
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--include-owned-external-cache',action='store_true')
    parser.add_argument('--owned-external-cache',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    try:
        report=clean(args.repository,args.apply,external=args.owned_external_cache,
                     include_owned_external=args.include_owned_external_cache)
    except (ValueError,OSError,subprocess.SubprocessError,ImportError) as exc:
        parser.exit(1,f'Cleanup refused: {exc}\n')
    if args.report:
        with args.report.open('x',encoding='utf-8') as handle:json.dump(report,handle,indent=2);handle.write('\n')
    print(f"Removed {report['removed_bytes']} bytes in {report['removed_files']} files; mode={report['mode']}")

if __name__=='__main__':main()
