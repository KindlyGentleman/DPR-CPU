import importlib.util
import json
from pathlib import Path
import shutil
import pytest

ROOT = Path(__file__).resolve().parents[1]

def load_script(name, path):
    assert path.is_file(), f'Missing implementation: {path.name}'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_compact_evidence_replays_each_paired_session():
    plot = load_script('plot_cpu_update', ROOT/'cpu_study/plot_cpu_update.py')
    rows, profile = plot.load_evidence(ROOT/'cpu_study/evidence')
    assert [r['pairs'] for r in rows] == [72, 36, 36]
    assert rows[0]['speedup'] == pytest.approx(1.1020263143246039)
    assert rows[1]['ci_low'] == pytest.approx(.7286737668640134)
    assert rows[2]['quality_pass'] is False
    assert profile['convolution_percent'] == pytest.approx(1029147.4/2415575*100)
    assert profile['fft_percent'] == pytest.approx(2.288270908582735)
    assert sum(profile[k] for k in ('convolution_percent','fft_percent','remaining_percent')) == pytest.approx(100)

def test_source_tampering_fails_before_plot(tmp_path):
    plot = load_script('plot_cpu_update', ROOT/'cpu_study/plot_cpu_update.py')
    evidence=tmp_path/'evidence'
    shutil.copytree(ROOT/'cpu_study/evidence',evidence)
    next(evidence.rglob('paired_timings.csv')).write_text('tampered')
    with pytest.raises(ValueError,match='SHA-256'):
        plot.load_evidence(evidence)

def test_audit_detects_broken_readme_link(tmp_path):
    audit=load_script('release_audit',ROOT/'scripts/release_audit.py')
    (tmp_path/'README.md').write_text('[lost](missing.md)')
    assert audit.readme_links(tmp_path)==['missing.md']

def test_audit_detects_changed_protected_file(tmp_path):
    audit=load_script('release_audit',ROOT/'scripts/release_audit.py')
    (tmp_path/'file').write_text('changed')
    assert audit.protected_mismatches(tmp_path,{'file':'0'*64})==['file']

def test_unlisted_consumed_source_is_rejected(tmp_path):
    plot=load_script('plot_cpu_update',ROOT/'cpu_study/plot_cpu_update.py')
    evidence=tmp_path/'evidence'; shutil.copytree(ROOT/'cpu_study/evidence',evidence)
    path=evidence/'manifest.json';manifest=json.loads(path.read_text())
    del manifest['sources']['results/fixed_t6_p_only_repeat72_v1/summary.json']
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='unlisted'):
        plot.load_evidence(evidence)

def test_release_audit_sanitizes_regression_selectors(monkeypatch):
    audit=load_script('release_audit',ROOT/'scripts/release_audit.py')
    monkeypatch.setenv('REG_FLOAT64','1');monkeypatch.setenv('REG_CASES','other')
    monkeypatch.setenv('REG_REFERENCES','wrong');monkeypatch.setenv('PYTEST_ADDOPTS','-k missing')
    env=audit.regression_environment(ROOT)
    assert env['REG_IMPL']=='dpr_modern' and env['REG_CODE_ROOT']==str(ROOT)
    assert not any(k in env for k in ('REG_FLOAT64','REG_CASES','REG_REFERENCES','PYTEST_ADDOPTS'))

def test_release_audit_rejects_unrelated_inherited_node_failure():
    audit=load_script('release_audit',ROOT/'scripts/release_audit.py')
    with pytest.raises(ValueError):
        audit.known_regression_signature('FAILED tests/regression/test_regression.py::test_case[network_synthetic]\nRuntimeError: checkpoint missing')

def test_release_audit_refuses_optimized_python(tmp_path):
    import subprocess,sys
    result=subprocess.run([sys.executable,'-O',str(ROOT/'scripts/release_audit.py'),
                           '--output',str(tmp_path/'audit')],capture_output=True,text=True)
    assert result.returncode!=0 and 'optimized Python' in result.stdout+result.stderr
