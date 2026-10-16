"""Generate the CPU update from hash-verified, session-specific source evidence."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cpu_optimization.statistics import paired_summary

TRIALS = (
    ('fixed_t6_p_only_repeat72_v1', 'Six P cores', 'accepted configuration'),
    ('phase2_compile_selective_screen_20261016_v4', 'Selective Inductor', 'quality passed; no detectable gain'),
    ('phase2_compile_default_screen_20261016_v5', 'Full Inductor', 'strict numerical parity failed'),
)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_evidence(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder/'manifest.json').read_text())
    required={'results/profile_v2/summary.json','results/profile_v2/operators.csv','config/selection.json'}
    required.update(f'results/{trial}/{name}' for trial,_,_ in TRIALS
                    for name in ('summary.json','paired_timings.csv','validation.json'))
    missing=required-manifest['sources'].keys()
    if missing:raise ValueError(f'unlisted consumed evidence: {sorted(missing)}')
    for name, source in manifest['sources'].items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder) or sha256(path) != source['sha256']:
            raise ValueError(f'SHA-256 mismatch or unsafe source: {name}')
    rows=[]
    for trial, label, decision in TRIALS:
        path=folder/'results'/trial
        summary=json.loads((path/'summary.json').read_text())
        with (path/'paired_timings.csv').open(newline='') as f:
            pairs=list(csv.DictReader(f))
        replay=paired_summary([float(p['reference_s']) for p in pairs],
                              [float(p['candidate_s']) for p in pairs],
                              seed=summary['bootstrap_seed'],draws=summary['bootstrap_draws'],
                              block=summary['bootstrap_block'])
        for key,value in replay.items():
            if isinstance(value, float) and abs(value-summary[key])>1e-12:
                raise ValueError(f'Paired replay mismatch: {trial}/{key}')
            if not isinstance(value,float) and value!=summary[key]:
                raise ValueError(f'Paired replay mismatch: {trial}/{key}')
        rows.append(dict(experiment_id=trial,label=label,decision=decision,**replay,
                         quality_pass=summary['quality_pass'],
                         summary_sha256=sha256(path/'summary.json'),
                         timings_sha256=sha256(path/'paired_timings.csv')))
    profile_path=folder/'results/profile_v2'
    summary=json.loads((profile_path/'summary.json').read_text())
    with (profile_path/'operators.csv').open(newline='') as f:
        operators=list(csv.DictReader(f))
    total=sum(float(r['self_cpu_us']) for r in operators)
    if abs(total-summary['self_cpu_total_us'])>1e-5:
        raise ValueError('Operator self CPU time denominator mismatch')
    hot=sum(float(r['self_cpu_us']) for r in operators
            if r['name']=='aten::mkldnn_convolution'
            and ast.literal_eval(r['input_shapes'])[:2]==[[1,256,66,66],[256,256,3,3]])
    fft=sum(float(r['self_cpu_us']) for r in operators if r['name'].startswith('aten::_fft_'))
    if abs(fft/total-summary['fft_self_fraction'])>1e-12 or hot<=0:
        raise ValueError('Profiler composition mismatch')
    profile=dict(experiment_id='profile_v2',convolution_percent=100*hot/total,
                 fft_percent=100*fft/total,remaining_percent=100*(total-hot-fft)/total,
                 self_cpu_total_us=total,summary_sha256=sha256(profile_path/'summary.json'),
                 operators_sha256=sha256(profile_path/'operators.csv'))
    return rows,profile

def write_table(rows,profile,path):
    records=[dict(section='paired_speedup',**r) for r in rows]
    records += [dict(section='operator_self_cpu_time',experiment_id='profile_v2',label=key,
                     percent=profile[key],summary_sha256=profile['summary_sha256'],
                     operators_sha256=profile['operators_sha256'])
                for key in ('convolution_percent','fft_percent','remaining_percent')]
    fields=list(dict.fromkeys(k for r in records for k in r))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)

def generate(evidence,output):
    import matplotlib
    matplotlib.use('Agg')
    import scienceplots  # noqa: F401
    import matplotlib.pyplot as plt
    import numpy as np

    rows,profile=load_evidence(evidence)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    write_table(rows,profile,output/'cpu_update_results.csv')
    plt.style.use(['science','ieee','no-latex'])
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,
                         'ytick.labelsize':7.5,'pdf.fonttype':42,'ps.fonttype':42,
                         'font.family':'DejaVu Sans','savefig.bbox':None})
    fig,axes=plt.subplots(1,2,figsize=(7.16,3.55),gridspec_kw={'width_ratios':[1.25,1]})
    fig.subplots_adjust(left=.19,right=.97,bottom=.27,top=.83,wspace=.65)
    ax=axes[0]
    colors=['#0072B2','#00725A','#A44C00']
    markers=['o','s','X']
    for i,(r,color,marker) in enumerate(zip(rows,colors,markers)):
        ax.errorbar(r['speedup'],2-i,
                    xerr=[[r['speedup']-r['ci_low']],[r['ci_high']-r['speedup']]],
                    fmt=marker,color=color,markersize=5,capsize=3,elinewidth=1.2)
    ax.axvline(1,color='.25',lw=.8,ls='--')
    ax.set(yticks=[2,1,0],yticklabels=['Six P cores\n72 timing pairs','Selective Inductor\n36 timing pairs',
                                    'Full Inductor\n36 timing pairs'],
           xlim=(.68,1.17),ylim=(-.55,2.55),xlabel='Paired speedup (ratio of medians)')
    ax.set_title('(a) Separate paired experiments',loc='left',fontsize=8.5,pad=12)
    ax.text(.02,-.28,'95% circular block-bootstrap CIs.\nSeparate sessions and paired baselines.',
            transform=ax.transAxes,fontsize=7,va='top')
    ax=axes[1]
    values=[profile[k] for k in ('convolution_percent','fft_percent','remaining_percent')]
    ax.barh([2,1,0],values,color=['#0072B2','#00725A','#777777'],height=.5)
    ax.set(yticks=[2,1,0],yticklabels=['256→256\n3×3 convolution','FFT kernels','Remaining'],
           xlim=(0,75),ylim=(-.55,2.55),xlabel='Operator self CPU time (%)')
    for y,v in zip([2,1,0],values):ax.text(v+1,y,f'{v:.2f}%',va='center',fontsize=7.5)
    ax.set_title('(b) Operator self CPU time',loc='left',fontsize=8.5,pad=12)
    ax.text(.02,-.28,'Three instrumented forwards.\nDisjoint operator self-time;\nnot inclusive stage timing.',
            transform=ax.transAxes,fontsize=7,va='top')
    fig.suptitle('Frozen FP32 DPR1 on Intel Core i7-13620H',fontsize=10,y=.97)
    for ax in axes:ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    for suffix in ('png','pdf'):
        fig.savefig(output/f'figure_cpu_update.{suffix}',dpi=600,facecolor='white',
                    metadata={'Creator':'DPR-CPU verified evidence plotting'})
    plt.close(fig)
    return rows,profile

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,default=ROOT/'cpu_study/evidence')
    p.add_argument('--output',type=Path,default=ROOT/'cpu_study')
    args=p.parse_args();generate(args.evidence,args.output)
    print(f'Verified CSV, PNG and PDF written to {args.output}')

if __name__=='__main__':main()
