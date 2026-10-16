import pytest


def summary(**overrides):
    return {'quality_pass': True, 'speedup': 1.03, 'ci_low': 1.01, 'ci_high': 1.06,
            'candidate_p95_s': .9, 'baseline_p95_s': 1., 'pairs': 72,
            'memory_gate': True, **overrides}


def test_reporting_does_not_call_significant_below_target_gain_no_improvement():
    from cpu_optimization.phase2_extension.reporting import derived_decision
    assert derived_decision(summary()) == 'beneficial_below_target'
    assert derived_decision(summary(speedup=.98, ci_low=.96, ci_high=.99)) == 'regression'
    assert derived_decision(summary(ci_low=.99, ci_high=1.08)) == 'no_detectable_gain'
    assert derived_decision(summary(ci_low=.99, ci_high=1.12)) == 'inconclusive_target'
    assert derived_decision(summary(quality_pass=False)) == 'numerical_failure'


def test_reporting_requires_fresh_replication_memory_and_p95_for_promotion():
    from cpu_optimization.phase2_extension.reporting import promotion_checks
    assert promotion_checks(summary(speedup=1.15, ci_high=1.19), independent_repeat=True)
    assert not promotion_checks(summary(speedup=1.15, ci_high=1.19), independent_repeat=False)
    assert not promotion_checks(summary(speedup=1.15, ci_high=1.19, memory_gate=None), independent_repeat=True)
    assert not promotion_checks(summary(speedup=1.15, ci_high=1.19, candidate_p95_s=1.1), independent_repeat=True)
    # Percentile intervals may exclude the estimate: do not change the locked point/CI-low gate.
    assert promotion_checks(summary(speedup=1.11, ci_low=1.02, ci_high=1.09), independent_repeat=True)
    from cpu_optimization.phase2_extension.reporting import derived_decision
    assert derived_decision(summary(speedup=1.11, ci_low=1.02, ci_high=1.09)) == 'confirmation_required'


@pytest.mark.parametrize('overrides', [dict(ci_low=float('nan')), dict(ci_high=.99),
                                      dict(speedup=float('inf')), dict(baseline_p95_s=0)])
def test_reporting_rejects_invalid_or_below_target_promotion_evidence(overrides):
    from cpu_optimization.phase2_extension.reporting import promotion_checks
    assert not promotion_checks(summary(speedup=1.15, **overrides) if 'speedup' not in overrides
                                else summary(**overrides), independent_repeat=True)


def test_portable_replication_does_not_establish_beating_selected_reference(tmp_path):
    import json
    from cpu_optimization.phase2_extension.reporting import confirmed_configurations
    trials = []
    for name, reference_threads, reference_affinity, speedup, low in [
            ('first', 4, 'default', 1.15, 1.11), ('second', 4, 'default', 1.16, 1.12),
            ('direct', 6, 'p_only', .98, .95)]:
        folder = tmp_path/name
        folder.mkdir()
        (folder/'workers.json').write_text(json.dumps([{'instance_id': name+'B'}, {'instance_id': name+'C'}]))
        (folder/'run.json').write_text(json.dumps({'checkpoint_sha256': 'frozen', 'source_files': {'model.py': 'same'}}))
        (folder/'inputs.json').write_text(json.dumps([{'sample': 'same'}]))
        trials.append({'folder': folder, 'command': {'id': name}, 'summary': summary(
            candidate='new', speedup=speedup, ci_low=low, ci_high=1.19,
            reference_threads=reference_threads, reference_affinity_name=reference_affinity,
            worker_settings=[dict(threads=reference_threads, affinity=[0, 1], backend='reference'),
                             dict(threads=4, affinity=[0, 1], backend='new')])})
    result = confirmed_configurations(trials)
    assert len(result) == 1
    assert not result[0]['beats_selected_reference_checks']
    trials[-1]['summary'].update(speedup=1.04, ci_low=1.01)
    assert confirmed_configurations(trials)[0]['beats_selected_reference_checks']
    (trials[1]['folder']/'inputs.json').write_text(json.dumps([{'sample': 'different'}]))
    assert confirmed_configurations(trials) == []


def test_report_campaigns_require_terminal_unique_evidence_in_execution_order(tmp_path):
    import json
    from cpu_optimization.phase2_extension.reporting import load_campaigns
    folders = []
    for suffix, started in [('retry', '2026-10-16T11:00:00Z'), ('context', '2026-10-16T10:00:00Z')]:
        folder = tmp_path/suffix
        folder.mkdir()
        (folder/'campaign.json').write_text(json.dumps({
            'status': 'completed', 'created_utc': started,
            'commands': [{'id': suffix, 'started_utc': started}]}))
        folders.append(folder)
    records, commands = load_campaigns(folders)
    assert [command['id'] for command in commands] == ['context', 'retry']
    assert len(records) == 2
    with pytest.raises(ValueError, match='duplicate'):
        load_campaigns([folders[0], folders[0]])
    (folders[1]/'campaign.json').write_text(json.dumps({'status': 'running_cpu_extension'}))
    with pytest.raises(ValueError, match='terminal'):
        load_campaigns(folders)


def test_report_memory_distinguishes_isolated_workers_and_process_scope():
    from cpu_optimization.phase2_extension.reporting import memory_label
    assert memory_label({'peak_memory': {'reference_worker_bytes': 2**20,
                                        'candidate_worker_bytes': 2*2**20}}) == '1.0 / 2.0 (workers)'
    assert memory_label({'peak_memory': {'peak_wset': 3*2**20}}) == '3.0 (whole process)'
    assert memory_label({}) == 'not measured'
