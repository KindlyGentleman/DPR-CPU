"""Source-backed extension tables and provisional scientific figures."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path


def promotion_checks(summary, independent_repeat=False):
    estimates = [summary.get(key) for key in ('speedup', 'ci_low', 'ci_high', 'candidate_p95_s', 'baseline_p95_s')]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in estimates):
        return False
    return (summary['ci_low'] <= summary['ci_high']
            and summary.get('quality_pass') is True and summary.get('speedup', 0) >= 1.10
            and summary.get('ci_low', 0) > 1 and summary.get('pairs', 0) >= 72
            and summary.get('candidate_p95_s', math.inf) <= 1.05*summary.get('baseline_p95_s', 0)
            and summary.get('memory_gate') is True and independent_repeat)


def derived_decision(summary, independent_repeat=False):
    if summary.get('quality_pass') is False:
        return 'numerical_failure'
    if summary.get('status') == 'blocked':
        return 'blocked'
    if 'speedup' not in summary:
        return 'characterization_or_no_timing'
    if summary['ci_high'] < 1:
        return 'regression'
    if summary['speedup'] < 1.10 and summary['ci_high'] < 1.10:
        return 'beneficial_below_target' if summary['ci_low'] > 1 else 'no_detectable_gain'
    if promotion_checks(summary, independent_repeat):
        return 'meets_development_promotion_checks'
    if summary['speedup'] >= 1.10 and summary['ci_low'] > 1:
        if summary['candidate_p95_s'] > 1.05*summary['baseline_p95_s']:
            return 'p95_gate_failure'
        if summary.get('memory_gate') is False:
            return 'memory_gate_failure'
        return 'confirmation_required'
    return 'inconclusive_target'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows, fields):
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def load_campaigns(paths):
    """Combine terminal evidence without merging or reusing timing arms."""
    records, commands, seen = [], [], set()
    for path in paths:
        path = Path(path).resolve()
        state = json.loads((path/'campaign.json').read_text(encoding='utf-8'))
        if state.get('status') not in ('completed', 'executed_pending_report_review', 'bounded_stopped', 'failed'):
            raise ValueError(f'Report only a terminal campaign: {path}')
        records.append({'path': path, 'state': state})
        for command in state['commands']:
            if command['id'] in seen:
                raise ValueError(f"duplicate command identity: {command['id']}")
            seen.add(command['id'])
            commands.append({**command, 'campaign_path': str(path)})
    commands.sort(key=lambda command: (command.get('started_utc', ''), command['id']))
    return records, commands


def memory_label(row):
    peak = row.get('peak_memory', {})
    if isinstance(peak, dict) and all(peak.get(key) is not None for key in
                                    ('reference_worker_bytes', 'candidate_worker_bytes')):
        return f"{peak['reference_worker_bytes']/2**20:.1f} / {peak['candidate_worker_bytes']/2**20:.1f} (workers)"
    if isinstance(peak, dict) and peak.get('peak_wset') is not None:
        return f"{peak['peak_wset']/2**20:.1f} (whole process)"
    return 'not measured'


def confirmed_configurations(trials):
    """Keep reference, source, input and fresh-worker identities in replication."""
    groups = {}
    direct = []
    for item in trials:
        summary, folder = item['summary'], item['folder']
        files = [folder/name for name in ('workers.json', 'run.json', 'inputs.json')]
        if not all(path.is_file() for path in files) or summary.get('candidate') == 'reference':
            continue
        workers, metadata, inputs = [json.loads(path.read_text()) for path in files]
        settings = summary.get('worker_settings', [])
        ids = frozenset(worker.get('instance_id') for worker in workers)
        if len(settings) != 2 or len(ids) != 2 or None in ids:
            continue
        identity = json.dumps({'checkpoint': metadata['checkpoint_sha256'],
                               'source': metadata['source_files'], 'inputs': inputs}, sort_keys=True)
        candidate = json.dumps(settings[1], sort_keys=True)
        reference = json.dumps(settings[0], sort_keys=True)
        if promotion_checks(summary, independent_repeat=True):
            groups.setdefault((summary['candidate'], candidate, reference, identity), []).append((item['command']['id'], ids))
        if (summary.get('reference_threads') == 6 and summary.get('reference_affinity_name') == 'p_only'
                and summary.get('quality_pass') is True and summary.get('pairs', 0) >= 72
                and summary.get('ci_low', 0) > 1 and summary.get('speedup', 0) > 1
                and summary.get('memory_gate') is True
                and summary['candidate_p95_s'] <= 1.05*summary['baseline_p95_s']):
            direct.append((candidate, identity, item['command']['id'], ids))
    qualified = []
    for (name, candidate, reference, identity), runs in groups.items():
        if not any(first[1].isdisjoint(second[1]) for i, first in enumerate(runs) for second in runs[i+1:]):
            continue
        direct_passes = [run_id for config, source, run_id, ids in direct
                         if config == candidate and source == identity
                         and all(ids.isdisjoint(run[1]) for run in runs)]
        qualified.append({'candidate': name, 'candidate_settings': json.loads(candidate),
                          'reference_settings': json.loads(reference),
                          'evidence': [item[0] for item in runs],
                          'scope': 'replicated development checks versus the specified reference',
                          'beats_selected_reference_checks': bool(direct_passes),
                          'direct_selected_reference_evidence': direct_passes,
                          'acceptance': 'human scientific review and locked replay still required'})
    return qualified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--include-campaign', type=Path, action='append', default=[],
                        help='Additional terminal evidence; commands remain separate trials')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    root = Path(__file__).resolve().parents[2]
    records, commands = load_campaigns([campaign, *args.include_campaign])
    state = records[0]['state']
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_files = [path for record in records for path in record['path'].glob('*.json')]
    def context_artifact(name):
        return next((record['path']/name for record in records if (record['path']/name).is_file()), campaign/name)
    trials = []
    for command in commands:
        folder = root/'cpu_optimization/results'/command['id']
        summary_file = folder/'summary.json'
        summary = json.loads(summary_file.read_text(encoding='utf-8')) if summary_file.exists() else command.get('summary', {})
        if summary_file.exists():
            source_files.append(summary_file)
        trials.append({'command': command, 'folder': folder, 'summary': summary})
    qualified = confirmed_configurations(trials)
    rows = []
    for item in trials:
        summary, command = item['summary'], item['command']
        row = {'id': command['id'], 'task': command['name'], 'candidate': summary.get('candidate'),
               'raw_status': summary.get('status', 'see command log'),
               'campaign': Path(command['campaign_path']).name,
               'derived_decision': derived_decision(summary), 'exit_code': command.get('exit_code'),
               'reason': summary.get('reason', command.get('exception', '')), **summary}
        rows.append(row)
        # Hash the raw evidence as well as the summary that drives this report.
        if item['folder'].exists():
            source_files += [path for path in item['folder'].rglob('*') if path.is_file()]
        log = Path(command['log']) if command.get('log') else None
        if log is not None and log.is_file():
            source_files.append(log)
    fields = ['id', 'campaign', 'task', 'candidate', 'raw_status', 'derived_decision', 'exit_code', 'pairs', 'median_s', 'p95_s',
              'baseline_median_s', 'candidate_median_s', 'baseline_p95_s', 'candidate_p95_s',
              'speedup', 'ci_low', 'ci_high', 'quality_pass', 'max_abs', 'max_nrmse_increase',
              'max_ssim_loss', 'new_failures', 'memory_gate', 'peak_memory', 'worker_settings',
              'reference_threads', 'candidate_threads', 'reason']
    write_csv(output/'experiments.csv', rows, fields)
    figures = []
    os.environ.setdefault('MPLCONFIGDIR', str(root/'build/phase2_mpl_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import scienceplots
    from PIL import Image

    def export(fig, name, caption, alt, sources):
        files = []
        for extension in ('pdf', 'svg', 'png'):
            path = output/(name+'.'+extension)
            fig.savefig(path, dpi=600, facecolor='white')
            if extension == 'png':
                with Image.open(path) as image:
                    image.convert('RGB').save(path, dpi=(600, 600))
            files.append({'file': path.name, 'sha256': sha(path)})
        plt.close(fig)
        figures.append({'name': name, 'caption': caption, 'alt_text': alt, 'files': files,
                        'sources': [str(path) for path in sources], 'raster_export': 'opaque RGB; no data/image resampling'})

    style = {'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
             'pdf.fonttype': 42, 'svg.fonttype': 'none'}
    micro_rows = []
    with plt.style.context(['science', 'ieee', 'no-latex']), plt.rc_context(style):
        profile_path = context_artifact('profiling_phase2.json')
        if profile_path.exists():
            profile = json.loads(profile_path.read_text())
            source_files.append(profile_path)
            stage = profile['disjoint_stage_ms']
            stage_rows = [{'stage': key.removeprefix('stage.'), 'milliseconds': value} for key, value in stage.items()]
            write_csv(output/'bottleneck_source.csv', stage_rows, ['stage', 'milliseconds'])
            fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.5), layout='constrained')
            axes[0].barh([r['stage'] for r in stage_rows], [r['milliseconds'] for r in stage_rows], color='#4477AA', edgecolor='black')
            axes[0].invert_yaxis(); axes[0].set(xlabel='Instrumented stage time (ms)', title='Disjoint DPR stages', xlim=(0, max(stage.values())*1.12))
            target = (profile['target_3x3_operator'] or {}).get('operator_self_fraction', 0)
            fft = profile['fft_self_fraction']
            axes[1].barh(['Target spatial 3x3', 'All FFT kernels'], [100*target, 100*fft], color=['#332288', '#117733'], edgecolor='black', hatch='/')
            axes[1].set(xlabel='Operator self-time (%)', xlim=(0, 100), title='Kernel self-time fractions')
            axes[1].text(.98, .96, f"42 FourierUnits / 84 feature FFTs\nFFT-only ideal bound: {profile['fft_only_ideal_speedup_bound']:.3f}x",
                         transform=axes[1].transAxes, ha='right', va='top', fontsize=8,
                         bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 3})
            export(fig, 'bottlenecks',
                   'Three instrumented forwards of the first original development input. Stage scopes are disjoint; operator fractions use the separate operator-self denominator. Physics FFTs are excluded from the 84 feature-transform count but included in the all-FFT kernel fraction. Session differences from old profiles are not speedup measurements.',
                   'Horizontal bars compare actual disjoint stages and separate spatial-convolution/FFT operator fractions.', [profile_path])
        measured = [row for row in rows if 'speedup' in row]
        fig, ax = plt.subplots(figsize=(8.1, max(2.8, .42*len(measured)+1.3)), layout='constrained')
        if measured:
            labels = [row['id'].replace('phase2_', '').replace('_20261016_', ' / ').replace('_', ' ')
                      + (' [parity fail]' if row.get('quality_pass') is False else '') for row in measured]
            means = np.array([row['speedup'] for row in measured])
            lower = np.array([row['ci_low'] for row in measured]); upper = np.array([row['ci_high'] for row in measured])
            positions = np.arange(len(measured))
            # A percentile bootstrap interval can exclude its point estimate.
            # Preserve both endpoints instead of clipping negative error lengths.
            ax.hlines(positions, lower, upper, color='#333333')
            for position, mean, row in zip(positions, means, measured):
                passed = row.get('quality_pass') is True
                ax.scatter(mean, position, color='#332288' if passed else '#AA3377',
                           marker='o' if passed else 'X', s=25, zorder=3)
            ax.set_yticks(np.arange(len(measured)), labels); ax.invert_yaxis()
        else:
            ax.text(.5, .5, 'No completed paired timing with passing numerical gates', ha='center', va='center', transform=ax.transAxes)
        ax.axvline(1, color='black', linewidth=.8); ax.axvline(1.10, color='#117733', linestyle='--', linewidth=.8)
        ax.set(xlabel='Paired ratio of arm medians (speedup)', title='Fresh prepared-network measurements')
        export(fig, 'candidate_speedups',
               'Every completed paired trial is plotted; crosses retain numerical failures and do not indicate eligible candidates. Blocked trials have no timing point. Each point uses its own nearby paired reference. Intervals are 95% circular moving-block bootstrap intervals (block six, 5,000 draws); timing pairs are not independent scientific objects. Solid line: no gain; dashed: original 1.10 target. Memory gates and scopes are in experiments.csv.',
               'Actual speedup points and confidence intervals; a cross marks the full compiler numerical failure. Separate rows identify campaign and reference replication.', [output/'experiments.csv'])
        micro_files = [item['folder']/'convs.csv' for item in trials if item['command']['name'] == 'convs' and (item['folder']/'convs.csv').exists()]
        if micro_files:
            source_files += micro_files
            with micro_files[0].open(newline='') as handle:
                micro_rows = list(csv.DictReader(handle))
            write_csv(output/'convs.csv', micro_rows, list(micro_rows[0]))
            hot = [row for row in micro_rows if json.loads(row.get('kernel_input_shape', row['input_shape'])) == [1, 256, 66, 66]
                   and json.loads(row['weight_shape']) == [256, 256, 3, 3]]
            if hot:
                fig, ax = plt.subplots(figsize=(8.1, max(3.4, .35*len(hot)+1.3)), layout='constrained')
                for i, row in enumerate(hot):
                    passed = str(row['strict_parity']).lower() == 'true'
                    mean, low, high = (float(row[key]) for key in ('speedup', 'ci_low', 'ci_high'))
                    color = '#332288' if passed else '#AA3377'
                    ax.hlines(i, low, high, color=color)
                    ax.scatter(mean, i, marker='o' if passed else 'X', color=color, s=25, zorder=3)
                ax.set_yticks(range(len(hot)), [f"shape {row['shape_id']} ({row['calls_per_forward']} calls): {row['method']}"
                    +(' [parity fail]' if str(row['strict_parity']).lower() != 'true' else '') for row in hot])
                ax.invert_yaxis(); ax.axvline(1, color='black', linewidth=.8)
                ax.set(xlabel='Local paired median speedup versus F.conv2d', title='Captured hot 256x256x3x3 primitive')
                export(fig, 'conv_prototypes',
                       'Two captured layout cohorts, with three and 15 calls per forward; logical input 64x64, native internally reflect-padded input 66x66. One seeded input and one actual frozen checkpoint weight tensor per shape. Randomized method order, 36 warm repeat blocks; 95% paired moving-block intervals (2,000 draws, block six). Padding, transforms, tiling, materialization and reassembly are timed; frozen weight setup is separate. Cross markers retain strict FP32 failures. Local gains are not full-network gains.',
                       'Measured local convolution speedups; cross markers identify strict numerical failures.', micro_files)
    table = ['| Trial | Paired speedup [95% CI] | Candidate median / p95 (s) | Peak memory (MiB, reference / candidate) | Quality | Derived annotation |',
             '|---|---:|---:|---:|---|---|']
    for row in rows:
        if 'speedup' not in row and row['task'] in ('pilot_replay', 'scoped_tests', 'small_compile'):
            continue
        effect = f"{row['speedup']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]" if 'speedup' in row else 'not measured'
        if 'candidate_median_s' in row:
            latency = f"{row['candidate_median_s']:.6f} / {row['candidate_p95_s']:.6f}"
        elif 'median_s' in row:
            latency = f"{row['median_s']:.6f} / {row['p95_s']:.6f} (baseline characterization)"
        else:
            latency = 'not measured'
        table.append(f"| {row['id']} | {effect} | {latency} | {memory_label(row)} | {row.get('quality_pass', 'not evaluated')} | {row['derived_decision']} |")
    status = json.loads(Path(state['pilot_run']).read_text())
    source_files.append(Path(state['pilot_run']))
    probes = []
    for record in sorted(records, key=lambda record: record['state'].get('created_utc', '')):
        probe_file = record['path']/'small_cnn_probe.json'
        if probe_file.is_file():
            source_files.append(probe_file)
            small = json.loads(probe_file.read_text())['small_cnn']
            cache_state = record['state'].get('native_cache_existed_at_start')
            cache_context = ('reused native cache; not a cold compilation measurement' if cache_state is True else
                             'fresh task cache at campaign start' if cache_state is False else
                             'initial cache state not recorded in this campaign manifest')
            exception = small.get('exception', '')
            diagnostic = ('Linker LNK1104: python312.lib unavailable to the child linker.' if 'LNK1104' in exception else
                          exception.splitlines()[0][:200] if exception else '')
            probes.append(f"{record['path'].name}: small-CNN native Inductor {small['status']}; "
                f"compile/load plus first call {small.get('compile_and_first_call_s', 'unavailable')} s; "
                f"generated kernels {small.get('generated_kernel_count', 'unavailable')}; "
                f"numerical gate {small.get('passed', False)}; {cache_context}. {diagnostic} "
                f"Full diagnostic: {probe_file}.")
    compiler = '\n\n'.join(probes) or 'Small-CNN probe not completed.'
    replay = ('passed' if state.get('pilot_replay_passed') is True else
              'failed' if state.get('pilot_replay_passed') is False else 'not run')
    screen_path = context_artifact('prototype_screen.json')
    screen = json.loads(screen_path.read_text()) if screen_path.exists() else 'not run'
    primitive_path = context_artifact('onednn_primitives.json')
    primitives = json.loads(primitive_path.read_text()) if primitive_path.exists() else []
    implementations = sorted({row.get('implementation', '') for row in primitives if row.get('primitive') == 'convolution'})
    skipped = json.dumps([{'campaign': record['path'].name, **item}
                          for record in records for item in record['state'].get('not_run', [])], indent=2)
    interrupted = [dict(task=item['name'], exit_code=item.get('exit_code'), reason=item.get('exception', 'see preserved log'))
                   for item in commands if item.get('exit_code') != 0]
    report = '\n'.join([
        '# Phase 2 CPU extension results', '',
        '**No new convolution or compiler backend qualifies for acceptance.** Selective compilation passes strict quality but has no detectable gain; the completed full-model compile is slower and fails strict parity. All four local prototypes are slower than native convolution; only tile-16 passes strict FP32 parity. The accepted six-P-core reference configuration and portable four-thread default remain unchanged.', '',
        f"Synthetic pilot status: **{status['status']}**, {status.get('raw_rows', 0)}/{status.get('expected_rows', 8192)} pairs. Full-grid gating inference is unavailable unless the run completed and replay passed. This is separate from prepared-network CPU optimization.", '',
        f"Terminal campaigns: {', '.join(record['path'].name+': '+record['state']['status'] for record in records)}. Whole-model quality population: original 18 measurements from seven development objects. Timing repeats do not enlarge that population. No new experimental validation or model training.", '',
        *table, '',
        compiler, '',
        f'Completed-grid pilot replay audit: **{replay}**. A bounded partial pilot has no full-grid policy inference.', '',
        'oneDNN convolution implementations captured: '+json.dumps(implementations)+f'; {len(primitives)} parsed primitive/reorder records. Original untimed logs remain authoritative.', '',
        'Prototype eligibility (strict FP32 parity and measured Amdahl screen): '+json.dumps(screen)+'.', '',
        'Not-run prerequisites or tasks:\n```json\n'+skipped+'\n```', '',
        'Nonzero or interrupted commands: '+json.dumps(interrupted)+'. Zero exit can still contain a documented numerical rejection or compiler blocker; see table and probe.', '',
        'Replicated development checks: '+json.dumps(qualified)+'. These require two independent worker trials against the same specified reference, identical source/checkpoint/inputs, original numerical/scientific/p95 gates, at least 72 pairs and acceptable lifetime one-model peak memory. Beating the accepted six-P-core reference additionally requires a passing direct comparison; four-thread comparisons alone cannot establish that. Human scientific review and selection-locked replay are still required for acceptance. This report does not change the immutable historical selection or portable auto default.', '',
        'The table reports MiB; experiments.csv retains original byte dictionaries. Paired rows compare isolated model-worker lifetime peaks; the baseline characterization peak covers its whole process and is not an isolated candidate/reference ratio.', '',
        'The existing accepted reference backend at six P-core threads remains a separately locked machine-specific configuration. Its new repetition is replication evidence; it does not rewrite the original acceptance. A credible smaller gain is reported as beneficial below target, rather than incorrectly called no improvement.', '',
        'Compiler startup and warmup are excluded from model clocks. workers.json and post_validation_warmup.json retain cold/setup/warmup times and compilation diagnostics; TORCH_LOGS=dynamo and the shared native cache retain backend compile evidence. Do not equate graph capture with native compilation. Untimed oneDNN logs identify actual primitive implementations and reorder activity.', '',
        'Negative numerical results cannot be rescued by more timings. Blocked/not-run work has no fabricated bars or timing values. These are same-machine session results with observed background/resource variation and unavailable temperature; instrumented profiles are not latency benchmarks.', '',
        'Source paths, raw results, exact commands and timeouts are preserved in each campaign.json and EXPERIMENT_LEDGER.md. V3 supplies characterization; v4 retains the initial compiler attempts; v5 is the explicitly warm-cache full-compiler recovery. Trials are not pooled and historical latency is never a new denominator. Earlier v2 diagnostics remain separate in the raw results.', '',
        'Methods references: [PyTorch Windows CPU Inductor](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html), [installed-version debugging guidance](https://docs.pytorch.org/tutorials/intermediate/inductor_debug_cpu.html), [Lavin and Gray, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Lavin_Fast_Algorithms_for_CVPR_2016_paper.pdf). Winograd is an established algorithm implemented locally; no external performance number is evidence for DPR.', '',
        'Figures are provisional research report exports, with no journal-specific compliance claim. SVG text is editable and fonts are not embedded; PDF uses TrueType fonts.'])
    (output/'FINAL_RESULTS.md').write_text(report+'\n', encoding='utf-8')
    ledger = ['# Phase 2 extension experiment ledger', '', *table, '',
              'Raw trial status is preserved in experiments.csv; derived annotations do not rewrite it.', '',
              '## Exact commands and execution bounds', '']
    for item in commands:
        ledger += [f"### {item['id']}", '', '```json', json.dumps(item, indent=2), '```', '']
    ledger += ['## Not run', '', '```json', skipped, '```', '',
               f'Pilot replay audit: {replay}.', '', 'Prototype screen: '+json.dumps(screen)+'.']
    (output/'EXPERIMENT_LEDGER.md').write_text('\n'.join(ledger)+'\n', encoding='utf-8')
    prototype = ['# Convolution prototypes', '',
                 'The local implementation uses true F(2x2,3x3) input/filter/output transforms, a separate unfold/GEMM method, and spatial tile-16 plus materialized tile-2 convolution. Frozen transformed weights are cached; activations are recomputed. Transforms, materialization, halo reads and reassembly are timed. No grouped-convolution or architecture substitution is made.', '',
                 'At the captured batch-one FP32 shape, tile-16 input plus halo is about 324 KiB. Tile-2 materializes a contiguous 16 KiB input and 4 KiB output footprint; weight streaming/packing is separate and cache residency is not demonstrated. Small toy float64 transform tests establish mathematical semantics; actual FP32 strict parity and whole-model gates decide eligibility.', '',
                 'Eligibility decisions: '+json.dumps(screen)+'.', '',
                 '| Shape / calls / method | Candidate median (s) | Speedup [95% CI] | Strict parity | Violating elements |',
                 '|---|---:|---:|---|---:|']
    for row in micro_rows:
        if (json.loads(row.get('kernel_input_shape', row['input_shape'])) == [1, 256, 66, 66]
                and json.loads(row['weight_shape']) == [256, 256, 3, 3]):
            prototype.append(f"| {row['shape_id']} / {row['calls_per_forward']} / {row['method']} | {float(row['candidate_median_s']):.6f} | {float(row['speedup']):.3f} [{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}] | {row['strict_parity']} | {row['violating_elements']} |")
    prototype += ['', 'All captured unique shapes and raw timings are retained in convs.csv and the original trial directory. These primitive measurements use seeded random inputs and real frozen checkpoint weights. They cannot establish full-network gains. Negative and ineligible prototypes remain evidence rather than being silently omitted. No backward-pass claim is made for this inference-only study.']
    (output/'conv_prototypes.md').write_text('\n'.join(prototype)+'\n', encoding='utf-8')
    manifest = {'scope': 'Parent source-backed extension reporting; not an independent final scientific audit',
                'campaign': str(campaign), 'included_campaigns': [str(record['path']) for record in records],
                'qualified_configurations': qualified, 'figures': figures,
                'source_files_sha256': {str(path): sha(path) for path in source_files},
                'reporting_script_sha256': sha(Path(__file__)), 'matplotlib': matplotlib.__version__,
                'known_limits': 'same-session timing noise, seven development bases, compiler setup separate, no full pilot conclusion from partial grid'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
