"""Audit the compact research release; --verify runs bounded reproduction/tests."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import unquote
import zlib

if not __debug__:
    raise SystemExit('Release audit refuses optimized Python: verification assertions must be enabled.')

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
CACHE_NAMES={'__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','.hypothesis','.mplconfig','.ipynb_checkpoints'}
ARCHIVAL_PATHS={f'cpu_optimization/config/{name}.json' for name in
                ('selection','development_provenance','environment_primary_packages','environment_secondary_packages','external_source')}
ARCHIVAL_PATHS.add('cpu_study/evidence/config/selection.json')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def protected_mismatches(root,manifest):
    def allowed(value):
        return {value} if isinstance(value,str) else {value['git_content_sha256'],value['original_worktree_sha256']}
    return [name for name,digest in manifest.items() if not (root/name).is_file() or sha(root/name) not in allowed(digest)]

def readme_links(root):
    links=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',(root/'README.md').read_text(encoding='utf-8'))
    return [link for link in links if not re.match(r'[a-z]+:|#',link)
            and not (root/unquote(link.split('#')[0])).exists()]

def run(command,cwd,env=None,timeout=240):
    result=subprocess.run(command,cwd=cwd,env=env,text=True,capture_output=True,timeout=timeout)
    return {'exit_code':result.returncode,'output':result.stdout+result.stderr}

def regression_environment(root):
    env={k:v for k,v in os.environ.items() if not k.startswith('REG_')
         and k not in ('PYTEST_ADDOPTS','PYTEST_PLUGINS','PYTHONOPTIMIZE','PYTHONPATH')}
    env.update(REG_IMPL='dpr_modern',REG_CODE_ROOT=str(root),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    return env

def known_regression_signature(text):
    expected={'out_param_dpr0_seed21':1.802e-5,'out_param_dpr1_seed22':1.866e-5}
    matches=re.findall(r'(out_param_dpr[01]_seed\d+): rel L2 diff ([\deE.+-]+) > ([\deE.+-]+)',text)
    if not matches or {m[0] for m in matches}!=set(expected):
        raise ValueError('network_synthetic failure is not the recorded numerical mismatch')
    for key,value,tolerance in matches:
        if float(tolerance)!=1e-5 or abs(float(value)-expected[key])>5e-9:
            raise ValueError('network_synthetic mismatch differs from recorded inherited signature')
    if 'AssertionError' not in text:raise ValueError('not the inherited assertion failure')
    return True

def pdf_content(path):
    streams=[]
    for raw in re.findall(rb'stream\r?\n(.*?)\r?\nendstream',Path(path).read_bytes(),re.S):
        try:streams.append(zlib.decompress(raw))
        except zlib.error:streams.append(raw)
    return streams

def load_plot():
    spec=importlib.util.spec_from_file_location('plot_cpu_update',ROOT/'cpu_study/plot_cpu_update.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def figure_metadata(root):
    from PIL import Image
    png=root/'cpu_study/figure_cpu_update.png';pdf=root/'cpu_study/figure_cpu_update.pdf'
    with Image.open(png) as image:
        size=image.size;dpi=image.info.get('dpi',(0,0))
    data=pdf.read_bytes()
    media=re.search(rb'/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',data)
    width=float(media[3])-float(media[1]) if media else 0
    streams=[]
    for raw in re.findall(rb'stream\r?\n(.*?)\r?\nendstream',data,re.S):
        try:streams.append(zlib.decompress(raw))
        except zlib.error:pass
    charts=b'\n'.join(streams)
    info={'png_size':size,'png_dpi':dpi,'pdf_width_points':width,
          'embedded_truetype':b'/FontFile2' in data,'vector_paths':bool(re.search(rb'\d+\.?\d* \d+\.?\d* m\s',charts)),
          'chart_raster_images':b'/Subtype /Image' in data}
    if shutil.which('pdffonts'):
        fonts=run(['pdffonts',str(pdf)],root)
        info['pdffonts']=fonts['output']
        lines=fonts['output'].splitlines()[2:]
        info['all_fonts_embedded']=fonts['exit_code']==0 and bool(lines) and all(re.search(r'\s+yes\s+yes\s+',line) for line in lines)
    else:info['all_fonts_embedded']=info['embedded_truetype']
    assert size==(4296,2130) and all(abs(v-600)<1 for v in dpi),info
    assert abs(width-7.16*72)<.01 and info['embedded_truetype'] and info['vector_paths'] and not info['chart_raster_images'] and info['all_fonts_embedded'],info
    return info

def audit(output,verify=False,require_clean=False):
    if not __debug__:raise RuntimeError('optimized Python disables verification')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    checks={};failures=[]
    def check(name,fn):
        try:
            value=fn();checks[name]={'passed':True,'details':value}
        except Exception as exc:
            checks[name]={'passed':False,'error':str(exc)};failures.append(name)
    def require(condition,details):
        if not condition:raise ValueError(str(details))
        return details
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    files=sorted(set(git('ls-files').splitlines()+git('ls-files','--others','--exclude-standard').splitlines()))
    files=[f for f in files if (ROOT/f).is_file()]
    protected=json.loads((ROOT/'scripts/protected_files.json').read_text())
    check('git_state',lambda:require(not require_clean or not git('status','--porcelain'),
          {'branch':git('branch','--show-current'),'head':git('rev-parse','HEAD'),'status':git('status','--short')}))
    check('protected_hashes',lambda:require(not protected_mismatches(ROOT,protected),
          {'checked':len(protected),'mismatches':protected_mismatches(ROOT,protected)}))
    from cpu_optimization.api import CHECKPOINT_SHA256
    check('checkpoint',lambda:require(sha(ROOT/'pretrained/param_dpr1.pt')==CHECKPOINT_SHA256,CHECKPOINT_SHA256))
    plot=load_plot();rows=[]
    def sources():
        nonlocal rows
        rows,profile=plot.load_evidence(ROOT/'cpu_study/evidence')
        manifest=json.loads((ROOT/'cpu_study/evidence/manifest.json').read_text())
        selection=json.loads((ROOT/'cpu_study/evidence/config/selection.json').read_text())
        for trial,entries in selection['source_artifacts'].items():
            for name,digest in entries.items():
                path=ROOT/'cpu_study/evidence/results'/trial/name
                if path.exists():assert sha(path)==digest,(trial,name)
        for trial,_,_ in plot.TRIALS:
            v=json.loads((ROOT/'cpu_study/evidence/results'/trial/'validation.json').read_text())
            row=next(r for r in rows if r['experiment_id']==trial)
            assert v['passed']==row['quality_pass'],trial
        return {'source_records':len(manifest['sources']),'paired_experiments':rows,'profile':profile,
                'unavailable_archive_references':'Original campaign/source ZIP hashes identify preserved local records; not all are public.'}
    check('manifest_statistics_quality',sources)
    def convolution_replay():
        from cpu_optimization.statistics import paired_summary
        folder=ROOT/'cpu_study/evidence/results/phase2_convs_20261016_v3'
        with (folder/'convs.csv').open(newline='') as f:cohorts=list(csv.DictReader(f))
        with (folder/'raw_convs.csv').open(newline='') as f:raw=list(csv.DictReader(f))
        groups={}
        for row in raw:groups.setdefault((row['shape_id'],row['method']),[]).append(row)
        for row in cohorts:
            values=lambda method:[float(v['seconds']) for v in sorted(groups[(row['shape_id'],method)],key=lambda v:int(v['repeat']))]
            replay=paired_summary(values('functional'),values(row['method']),seed=int(row['bootstrap_seed']),
                                  draws=int(row['bootstrap_draws']),block=int(row['bootstrap_block']))
            for key in ('baseline_median_s','candidate_median_s','speedup','ci_low','ci_high'):
                assert abs(replay[key]-float(row[key]))<1e-12,(row['shape_id'],row['method'],key)
        prototypes=[r for r in cohorts if r['method'] in ('winograd','tiled','tiled_small','unfold')]
        assert len(cohorts)==80 and len(prototypes)==8
        assert all(float(r['speedup'])<1 for r in prototypes)
        assert {r['method'] for r in prototypes if r['strict_parity']=='True'}=={'tiled'}
        return {'primitive_cohorts_replayed':len(cohorts),'prototype_rows':len(prototypes),'strict_fp32_passing_prototype':'tiled (tile-16)','all_prototypes_slower':True}
    check('convolution_claims_raw_replay',convolution_replay)
    def claims():
        text=(ROOT/'README.md').read_text(encoding='utf-8')
        for row in rows:
            precision=6 if row['pairs']==72 else 5
            for key in ('speedup','ci_low','ci_high'):
                assert f"{row[key]:.{precision}f}" in text,(row['experiment_id'],key)
        validation=json.loads((ROOT/'cpu_study/evidence/results/phase2_compile_default_screen_20261016_v5/validation.json').read_text())
        failing=sum(not c['parity']['allclose'] for c in validation['cases'])
        pixels=sum(c['parity']['violating_elements'] for c in validation['cases'])
        assert failing==2 and pixels==11 and 'two measurements / 11 pixels' in text
        assert 'four intra-op threads' in text and 'not a controlled cross-session ranking' in text
        return {'claims_match_source_precision':True,'full_compile_failing_cases':failing,'violating_pixels':pixels}
    check('readme_claims',claims)
    def tables():
        with tempfile.TemporaryDirectory() as temp:
            expected=Path(temp)/'table.csv'
            r,p=plot.load_evidence(ROOT/'cpu_study/evidence');plot.write_table(r,p,expected)
            assert expected.read_bytes()==(ROOT/'cpu_study/cpu_update_results.csv').read_bytes()
        return 'CSV exactly matches verified source-derived table'
    check('figure_data_consistency',tables)
    check('figure_metadata',lambda:figure_metadata(ROOT))
    def gif_check():
        from PIL import Image
        path=ROOT/'cpu_study/reconstruction.gif'
        with Image.open(path) as im:
            assert im.info.get('loop')==0 and im.n_frames==5
            duration=[]
            for i in range(im.n_frames):im.seek(i);duration.append(im.info.get('duration'))
            assert duration==[2000]*5
        value=protected['cpu_study/reconstruction.gif']
        assert sha(path)==(value if isinstance(value,str) else value['git_content_sha256'])
        return {'frames':5,'loop':0,'duration_ms':duration,'bytes':path.stat().st_size,'historical_bytes_preserved':True}
    check('preserved_gif',gif_check)
    check('readme_links',lambda:require(not readme_links(ROOT),readme_links(ROOT)))
    def commands():
        text=(ROOT/'README.md').read_text(encoding='utf-8');targets=[]
        for language,body in re.findall(r'```(\w*)\n(.*?)```',text,re.S):
            if language=='python':compile(body,'README example','exec')
            for target in re.findall(r'(?m)^python\s+((?:cpu_study|scripts)/[\w./-]+\.py)',body):
                assert (ROOT/target).is_file(),target;targets.append(target)
        assert targets
        return {'script_targets_checked':targets,'python_example_compiled':True,'dependency_installs':'Documented recipe; wheel install separately verified without re-downloading dependencies.'}
    check('markdown_commands',commands)
    def inventory():
        caches=[f for f in git('ls-files').splitlines() if CACHE_NAMES.intersection(Path(f).parts) or f.endswith(('.pyc','.pyo'))]
        large=[{'path':f,'bytes':(ROOT/f).stat().st_size} for f in files if (ROOT/f).stat().st_size>10_000_000]
        unexpected=[r for r in large if r['path'] not in ('pretrained/param_dpr0.pt','pretrained/param_dpr1.pt','cpu_study/examples.npz','cpu_optimization/config/development_inputs.npz')]
        assert not caches and not unexpected,{'caches':caches,'large_generated':unexpected}
        return {'tracked_caches':caches,'large_files':large,'file_count':len(files),'total_working_file_bytes':sum((ROOT/f).stat().st_size for f in files)}
    check('cache_and_large_files',inventory)
    def secrets_paths():
        secrets=[];absolute=[]
        for f in files:
            if Path(f).suffix not in ('.py','.md','.json','.toml','.txt','.yml','.ps1','.csv'):continue
            text=(ROOT/f).read_text(encoding='utf-8',errors='replace')
            if re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',text):secrets.append(f)
            if re.search(r'[A-Z]:[\\/](?:Users|03_Proyek)|/(?:Users|home)/[^\s/]+',text) and f not in ARCHIVAL_PATHS:absolute.append(f)
        assert not secrets and not absolute,{'credential_patterns':secrets,'accidental_absolute_paths':absolute}
        return {'credential_pattern_hits':[],'accidental_absolute_paths':[],
                'intentional_immutable_provenance_path_files':sorted(ARCHIVAL_PATHS),
                'scope':'Pattern scan of release text, not proof that no credential exists in all history.'}
    check('credentials_and_portability',secrets_paths)
    check('license_citation',lambda:require(all((ROOT/f).is_file() for f in protected if f.startswith('LICENSE'))
          and '10.1038/s41524-025-01569-7' in (ROOT/'docs/CITATION_AND_LICENSE.md').read_text(),
          'Original license hashes covered by protected manifest; original DPR BibTeX and attribution table retained'))
    if verify:
        def regeneration():
            with tempfile.TemporaryDirectory(prefix='dpr-release-') as temp:
                clean=Path(temp)/'checkout';clean.mkdir()
                for folder in ('cpu_study','deeppr','pretrained'):
                    shutil.copytree(ROOT/folder,clean/folder,ignore=shutil.ignore_patterns('__pycache__'))
                commands=[['cpu_study/plot.py','--figure','main'],['cpu_study/plot.py','--figure','diagnostics'],
                          ['cpu_study/animate.py'],['cpu_study/reproduce.py','--all']]
                for i,command in enumerate(commands):
                    result=run([sys.executable,*command],clean)
                    (output/f'reproduction_{i}.log').write_text(result['output'],encoding='utf-8')
                    assert result['exit_code']==0,command
                generated=Path(temp)/'update';plot.generate(ROOT/'cpu_study/evidence',generated)
                assert (generated/'cpu_update_results.csv').read_bytes()==(ROOT/'cpu_study/cpu_update_results.csv').read_bytes()
                from PIL import Image,ImageChops
                assert ImageChops.difference(Image.open(generated/'figure_cpu_update.png').convert('RGB'),
                                            Image.open(ROOT/'cpu_study/figure_cpu_update.png').convert('RGB')).getbbox() is None
                assert pdf_content(generated/'figure_cpu_update.pdf')==pdf_content(ROOT/'cpu_study/figure_cpu_update.pdf'),'Regenerated PDF content differs'
                return {'commands':commands,'new_update_pixels_identical':True,'new_pdf_content_identical':True,'historical_outputs_generated_in_temporary_copy':True}
        check('figure_and_model_reproduction',regeneration)
        def tests():
            result=run([sys.executable,'-m','pytest','-o','addopts=','cpu_optimization/tests','tests/test_cpu_study.py',
                        'tests/test_publication.py','tests/test_cache_cleanup.py','-q'],ROOT,regression_environment(ROOT))
            (output/'scoped_tests.log').write_text(result['output'],encoding='utf-8')
            assert result['exit_code']==0,result['output'][-2000:]
            return result['output'].strip().splitlines()[-1]
        check('scoped_tests',tests)
        def original_tests():
            import xml.etree.ElementTree as ET
            env=regression_environment(ROOT)
            junit=output/'original_regression.xml'
            result=run([sys.executable,'-m','pytest','-o','addopts=','tests/regression','-q',f'--junitxml={junit}'],ROOT,env)
            (output/'original_regression.log').write_text(result['output'],encoding='utf-8')
            failed=re.findall(r'(?m)^FAILED ([^\s]+)',result['output'])
            expected=['tests/regression/test_regression.py::test_case[network_synthetic]']
            assert (result['exit_code']==0 and not failed) or (result['exit_code']==1 and failed==expected),result['output'][-2000:]
            if failed:known_regression_signature(result['output'])
            cases=ET.parse(junit).findall('.//testcase')
            # The immutable upstream suite has 19 float32 cases plus five opt-in float64 cases.
            assert len(cases)==24 and sum(c.attrib['name'].startswith('test_case[') for c in cases)==19
            skipped=[c for c in cases if c.find('skipped') is not None]
            assert len(skipped)==9 and all(c.attrib['name'].startswith('test_case_float64[') or
                c.find('skipped').attrib.get('type')=='pytest.xfail' for c in skipped),'Unexpected skipped float32 case'
            return {'classification':'known inherited network_synthetic failure' if failed else 'all passed',
                    'summary':result['output'].strip().splitlines()[-1],'tolerance_unchanged':True}
        check('original_regression_separate',original_tests)
        def wheel():
            with tempfile.TemporaryDirectory(prefix='dpr-wheel-') as temp:
                temp=Path(temp);wheels=temp/'wheels';installed=temp/'installed'
                result=run([sys.executable,'-m','pip','wheel','--no-deps','--wheel-dir',str(wheels),str(ROOT)],temp)
                assert result['exit_code']==0,result['output'][-2000:]
                artifact=next(wheels.glob('*.whl'))
                install=run([sys.executable,'-m','pip','install','--no-deps','--target',str(installed),str(artifact)],temp)
                assert install['exit_code']==0,install['output'][-2000:]
                code='''import sys,json,numpy as np
from pathlib import Path
import deeppr,cpu_optimization
assert Path(deeppr.__file__).is_relative_to(sys.argv[1])
assert Path(cpu_optimization.__file__).is_relative_to(sys.argv[1])
model=cpu_optimization.DPRCPU.from_pretrained(checkpoint=sys.argv[2],threads=4)
with np.load(sys.argv[3],allow_pickle=False) as saved:
    out=model.predict(saved['intensity_0'],saved['valid_0'])
    np.testing.assert_allclose(out,saved['reference_0'],rtol=1e-4,atol=1e-5)
    print(json.dumps({'shape':list(out.shape),'dtype':str(out.dtype),'max_abs':float(abs(out-saved['reference_0']).max()),'installed_imports':True}))
'''
                env=os.environ.copy();env['PYTHONPATH']=str(installed)
                smoke=run([sys.executable,'-c',code,str(installed),str(ROOT/'pretrained/param_dpr1.pt'),str(ROOT/'cpu_study/examples.npz')],temp,env)
                (output/'wheel_smoke.log').write_text(smoke['output'],encoding='utf-8')
                assert smoke['exit_code']==0,smoke['output'][-2000:]
                return {'wheel_sha256':sha(artifact),'inference':json.loads(smoke['output'].strip().splitlines()[-1]),
                        'dependencies':'Existing pinned environment reused; no clean-machine dependency-resolution claim.'}
        check('installed_package_inference',wheel)
    else:
        checks['execution_checks']={'passed':None,'details':'Not run; use --verify for regeneration, tests and installed wheel inference.'}
    report={'passed':not failures,'verification_executed':verify,'failures':failures,'checks':checks,
            'tested_source_sha256':{f:sha(ROOT/f) for f in files if Path(f).suffix in ('.py','.toml')},
            'scope':'Engineering release audit and compact-data replay, not independent scientific validation. Research 84-test suite is a separate historical result, not included in CPU test totals.'}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    summary='# Release audit\n\n'+f"Result: {'PASS' if report['passed'] else 'FAIL'}; verify={verify}.\n\n"
    summary+='\n'.join(f"- {name}: {'passed' if value['passed'] else 'not run' if value['passed'] is None else 'FAILED'}" for name,value in checks.items())+'\n'
    summary+='\nOriginal network_synthetic regression is classified separately; its tolerance is preserved.\n'
    (output/'summary.md').write_text(summary,encoding='utf-8')
    print(summary);return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true');parser.add_argument('--require-clean',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'build/release-audit')
    args=parser.parse_args();report=audit(args.output,args.verify,args.require_clean)
    raise SystemExit(0 if report['passed'] else 1)

if __name__=='__main__':main()
