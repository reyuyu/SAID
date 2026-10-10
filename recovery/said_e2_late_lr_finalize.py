"""Report-only final review; no model loading, optimizer, or GPU computation."""
import json
import statistics

from recovery import said_e2_late_lr_fourarm as c


def quality(result):
    values = c.protocol.common.quality(result)
    return dict(values, Urban_Mean=(values['Urban_I2T'] + values['Urban_T2I']) / 2)


def comparison(result, baseline):
    q, b = quality(result), quality(baseline)
    return dict(absolute=q, delta_pp={k: q[k] - b[k] for k in q},
                recall_delta_pp=c.protocol.common.compare(result, baseline)['recall_delta_pp'],
                Urban_both_directions_improved=all(q[k] > b[k] + 1e-6
                    for k in ('Urban_I2T', 'Urban_T2I')),
                Score5_and_Urban_Mean_improved=q['Score5'] > b['Score5'] + 1e-6
                    and q['Urban_Mean'] > b['Urban_Mean'] + 1e-6)


def distribution(values):
    return dict(min=min(values), median=statistics.median(values),
                mean=statistics.fmean(values), max=max(values))


def runtime_summary(arm):
    records = c.rows(c.arm_run(arm) / 'training/steps.jsonl')
    cycles = c.rows(c.arm_run(arm) / 'training/cycle_timing.jsonl')
    assert len(records) == len(cycles) == 1217
    log = (c.arm_run(arm) / 'train.log').read_text(errors='replace')
    errors = [s for s in ('Image failure sample=', 'Missing local sample=',
        'Input/output error', 'CUDA out of memory', 'NCCL error') if s in log]
    assert not errors
    assert all(r['nonfinite'] == 0 and all(h['gradients_finite']
        for h in r['rank_health']) for r in records)
    with (c.PARENT_RUN / 'step4868/training/steps.jsonl').open() as handle:
        original_first = json.loads(next(handle))
    first = records[0]
    assert first['step'] == original_first['step'] == 3652
    loss_difference = first['loss'] - original_first['loss']
    norm_difference = max(abs(h['gradient_norms'][group] - ref['gradient_norms'][group])
        for h, ref in zip(first['rank_health'], original_first['rank_health'])
        for group in c.GROUPS)
    return dict(updates=1217, full_cycle_seconds=distribution([
        r['four_rank_max_seconds'] for r in cycles]),
        actual_first_update_preoptimization=dict(loss_difference=loss_difference,
            max_group_gradient_norm_difference=norm_difference,
            loss_and_group_gradient_norms_exact=loss_difference == norm_difference == 0,
            note='Norm equality supplements the saved full-vector preupdate probe; it does not establish vector equality alone.'),
        GPU_peak_allocated_GiB={str(rank): max(h['peak_allocated_gib']
            for r in records for h in r['rank_health'] if h['rank'] == rank)
            for rank in range(4)}, errors=errors,
        nonfinite_updates=0,
        mask_all_closed_fraction_max={view: max(r[prefix + '_all_closed']
            for r in records) for view, prefix in (('F', 'F'), ('Dall', 'O'), ('D3', 'E'))},
        mask_all_open_fraction_max={view: max(r[prefix + '_all_open']
            for r in records) for view, prefix in (('F', 'F'), ('Dall', 'O'), ('D3', 'E'))},
        density_order_reversal_updates=sum(not (
            r['HNS_F_keep'] >= r['HNS_Dall_keep'] >= r['HNS_D3_keep']) for r in records))


def immutability():
    snap = c.read(c.EXP / 'BASELINE_IMMUTABILITY.json')
    checks = {str(c.PARENT): c.sha(c.PARENT) == c.PARENT_SHA,
        str(c.BASE_EXP / 'step4868/RESULTS.json'):
            c.sha(c.BASE_EXP / 'step4868/RESULTS.json') == snap['results_sha256']}
    for key in ('baseline_checkpoint', 'baseline_bare'):
        record = snap[key]
        checks[record['path']] = c.sha(record['path']) == record['sha256']
    checks.update({path: c.sha(path) == expected
        for path, expected in snap['original_eval_files'].items()})
    assert all(checks.values()), checks
    from train.train_nested_semantic_mask import code_manifest
    assert code_manifest() == c.read(c.EXP / 'PARENT_IDENTITY.json')['sources']
    assert c.protocol.common.evaluator_proof() == c.read(c.EXP / 'FOUR_ARM_PLAN.json')['evaluator_manifest']
    return dict(passed=True, files=checks, production_manifest_unchanged=True,
                evaluator_manifest_unchanged=True, original_visual_and_text_detach_preserved=True)


def evaluator_equivalence():
    base = c.read(c.BASE_EXP / 'step4868/evaluations/EVAL_PARALLEL_RUN.json')
    def invariant_command(command):
        result, index = [], 0
        while index < len(command):
            if command[index] in ('--checkpoint', '--output', '--output-dir'):
                index += 2
            else:
                result.append(command[index])
                index += 1
        return result
    receipts = {}
    for arm in c.ARMS:
        run = c.read(c.arm_exp(arm) / 'evaluations/EVAL_PARALLEL_RUN.json')
        assert run['status'] == 'COMPLETED'
        for key in ('evaluator_source_sha256', 'scheduler_source_sha256', 'batch_size', 'gpu_mapping'):
            assert run[key] == base[key], (arm, key)
        assert all(value == 0 for value in run['returncodes'].values())
        for dataset, job in run['jobs'].items():
            assert invariant_command(job['command']) == invariant_command(base['jobs'][dataset]['command'])
        manifests = {}
        for dataset in ('DOCCI', 'Long-DCI', 'Flickr30k-test1k'):
            value = c.read(c.arm_exp(arm) / 'evaluations' / (dataset + '.json'))
            original = c.read(c.BASE_EXP / 'step4868/evaluations' / (dataset + '.json'))
            assert all(value[k] == original[k] for k in ('manifest_sha256', 'n_images', 'n_captions'))
            manifests[dataset] = value['manifest_sha256']
        receipts[arm] = dict(passed=True, source_hashes_exact=True,
            native_commands_same_except_weight_and_output_paths=True,
            batch_and_GPU_mapping_exact=True, extended_manifest_hashes=manifests,
            all_process_returncodes_zero=True)
    return dict(passed=True, arms=receipts,
        limitation='COCO/Urban evaluators do not record per-image content hashes; unchanged source and input paths are verified. No retrospective per-image immutability claim is made.')


def main():
    from tools.eval_five_parallel import require_gpu_idle
    state = c.read(c.EXP / 'STATE.json')
    assert state['status'] == 'COMPLETED_GPU_IDLE'
    assert state['completed_arms'] == list(c.ARMS)
    require_gpu_idle({0, 1, 2, 3})
    baseline = c.read(c.BASE_EXP / 'step4868/RESULTS.json')
    models = {arm: c.read(c.arm_exp(arm) / 'RESULTS.json') for arm in c.ARMS}
    proofs = {arm: c.read(c.arm_exp(arm) / 'FULL_STREAM_AND_LR_PROOF.json')
              for arm in c.ARMS}
    assert all(p['passed'] and p['optimizer_updates'] == 1217 for p in proofs.values())
    assert len({p['full_stage_stream_sha256'] for p in proofs.values()}) == 1
    analyses = {arm: comparison(r, baseline) for arm, r in models.items()}
    resources = {arm: runtime_summary(arm) for arm in c.ARMS}
    original_records = c.rows(c.PARENT_RUN / 'step4868/training/steps.jsonl')
    mask_keys = ('HNS_F_keep', 'HNS_Dall_keep', 'HNS_D3_keep',
        'HNS_DF_hard_violation_ratio', 'HNS_3D_hard_violation_ratio',
        'HNS_DF_IoU', 'HNS_3D_IoU')
    baseline_structure = dict(last50={k: statistics.fmean(r[k]
        for r in original_records[-50:]) for k in mask_keys},
        density_order_reversal_updates=sum(not (
            r['HNS_F_keep'] >= r['HNS_Dall_keep'] >= r['HNS_D3_keep'])
            for r in original_records))
    intact = immutability()
    evaluator = evaluator_equivalence()
    for arm, result in models.items():
        assert c.sha(result['checkpoint']['path']) == result['checkpoint']['sha256']
        assert c.sha(result['bare']['path']) == result['bare']['sha256']
        assert result['strict_export']['passed']
    axes = {}
    for name, lower, higher in (
        ('backbone', 'B1-BB085', 'B2-BB115'),
        ('mask', 'B3-MASK085', 'B4-MASK115')):
        lo, hi, base = quality(models[lower]), quality(models[higher]), quality(baseline)
        axes[name] = {k: dict(lower=lo[k], baseline=base[k], higher=hi[k],
            higher_minus_lower_pp=hi[k]-lo[k],
            monotonic_increase=lo[k] <= base[k] <= hi[k],
            monotonic_decrease=lo[k] >= base[k] >= hi[k],
            observed_pattern=('UNCHANGED' if max(lo[k], base[k], hi[k])
                - min(lo[k], base[k], hi[k]) < 1e-6 else
                'NONDECREASING' if lo[k] <= base[k] <= hi[k] else
                'NONINCREASING' if lo[k] >= base[k] >= hi[k] else
                'BASELINE_LOCAL_PEAK' if base[k] > max(lo[k], hi[k]) else
                'BASELINE_LOCAL_TROUGH' if base[k] < min(lo[k], hi[k]) else
                'NONMONOTONIC')) for k in base}
    evidence = dict(passed=True, baseline_immutability=intact, evaluator_equivalence=evaluator, comparisons=analyses,
        LR_axis_descriptive_trends=axes, resources=resources,
        baseline_epoch4_structure=baseline_structure,
        all_four_stream_hashes_identical=True, GPU_idle=True,
        scientific_status='EXPLORATORY_REPEATED_PUBLIC_BENCHMARKS',
        independent_validation_verified=False, no_final_model_selected=True)
    c.dump(c.EXP / 'FINAL_REVIEW.json', evidence)
    report = c.EXP / 'FOUR_ARM_4868_REPORT.md'
    # Refresh the generated report from its frozen inputs, then add final review.
    c.combined(list(c.ARMS))
    native = c.read(c.EXP / 'FOUR_ARM_NATIVE_RESULTS.json')
    native.update(comparisons_including_Urban_Mean=analyses,
        selection_using_test_results_only=False, no_formal_model_selected=True,
        interpretation='Descriptive comparison of all four preregistered arms; no unbiased model selection claimed.')
    c.dump(c.EXP / 'FOUR_ARM_NATIVE_RESULTS.json', native)
    lines = [report.read_text(), '\n## Original E2@4868 complete recall reference\n',
        '| Dataset | I2T R@1/5/10 (%) | T2I R@1/5/10 (%) |',
        '|---|---|---|']
    for dataset, directions in baseline['metrics'].items():
        lines.append('| ' + dataset + ' | ' + ' | '.join(' / '.join(
            f'{100*directions[dr][k]:.6f}' for k in ('R@1', 'R@5', 'R@10'))
            for dr in ('I2T', 'T2I')) + ' |')
    lines += ['\n## Final scientific and engineering review\n',
        'This experiment reports all four arms without selecting a new formal model. '
        'The held-out candidate screen found one caption duplicated in training; '
        'image-content and semantic overlap remain unverified. The public five-set '
        'results cannot establish unbiased model selection or a new SOTA.\n',
        '| Arm | Δ Score5 | Δ J_long3 | Δ J_long | Δ Short4 | Δ Urban I2T | Δ Urban T2I | Δ Urban Mean |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    keys = ('Score5', 'J_long3', 'J_long', 'Short4', 'Urban_I2T', 'Urban_T2I', 'Urban_Mean')
    for arm, a in analyses.items():
        lines.append('| ' + arm + ' | ' + ' | '.join(
            f'{a["delta_pp"][k]:+.6f}' for k in keys) + ' |')
    lines += ['', 'All deltas are percentage points. Urban is reported to three '
        'decimal places in the absolute table; 0.100pp is one correct query.', '']
    lines += ['### Last50 structure compared with original E2', '',
        '| Model | F keep | Dall keep | D3 keep | DF violation | 3D violation | DF IoU | 3D IoU |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    structure = {'E2-Uniform': baseline_structure['last50'], **{arm:
        c.read(c.arm_exp(arm) / 'TRAINING_DIAGNOSTICS.json')['last50'] for arm in c.ARMS}}
    for name, last in structure.items():
        lines.append('| ' + name + ' | ' + ' | '.join(f'{last[k]:.6f}'
            for k in mask_keys) + ' |')
    lines += ['', 'Original E2 density-order reversal updates: '
        f'{baseline_structure["density_order_reversal_updates"]}/1217. '
        'Interpret arm reversals relative to this existing structure, rather than '
        'treating the ordering alone as an engineering failure.', '']
    for arm in c.ARMS:
        a = analyses[arm]
        diag = c.read(c.arm_exp(arm) / 'TRAINING_DIAGNOSTICS.json')
        last = diag['last50']
        groups = diag['gradient_probe']['group_diagnostics']
        regressions = [(ds, dr, delta['R@1'])
            for ds, directions in a['recall_delta_pp'].items()
            for dr, delta in directions.items() if delta['R@1'] < -1e-6]
        lines += [f'### {arm}', '',
            f'Both Urban directions improve: {a["Urban_both_directions_improved"]}. '
            f'Score5 and Urban Mean both improve: {a["Score5_and_Urban_Mean_improved"]}.',
            'R@1 regressions (dataset, direction, pp): ' + json.dumps(regressions) + '.',
            'Last50 F/Dall/D3 keep: ' + ' / '.join(f'{last["HNS_"+v+"_keep"]:.6f}'
                for v in ('F', 'Dall', 'D3')) + '.',
            'Last50 DF/3D hard violations and IoU: ' + ' / '.join(
                f'{last[k]:.6f}' for k in ('HNS_DF_hard_violation_ratio',
                'HNS_3D_hard_violation_ratio', 'HNS_DF_IoU', 'HNS_3D_IoU')) + '.',
            f'Density order maintained in last50 mean: {diag["density_order"]}; '
            f'reversal updates: {resources[arm]["density_order_reversal_updates"]}/1217.',
            'Gradient diagnostics (weighted norms and hierarchy/sparsity cosine):', '',
            '| Group | Alignment | Sparsity | Hierarchy | Total | H/S cosine |',
            '|---|---:|---:|---:|---:|---:|']
        for group, g in groups.items():
            values = [g[k] for k in ('weighted_alignment_norm', 'weighted_sparsity_norm',
                'weighted_hierarchy_norm', 'total_norm', 'weighted_cosine_hierarchy_sparsity')]
            lines.append('| ' + group + ' | ' + ' | '.join('N/A' if v is None
                else f'{v:.6f}' for v in values) + ' |')
        lines += ['', 'Resource summary: ' + json.dumps(resources[arm]) + '.', '']
    lines += ['### LR-axis trends and limits', '']
    for axis, values in axes.items():
        lines.append(axis + ': ' + json.dumps(values) + '.')
    lines += ['', '### Experimental outcome', '',
        'All four arms leave Urban I2T/T2I R@1 and Urban Mean unchanged. '
        'Every arm has lower Score5 than original E2. No arm meets the proposed '
        'joint improvement objective; no new formal model is selected.',
        'B4 is closest to original E2: Score5 -0.004800pp, J_long3 -0.006667pp, '
        'J_long -0.010000pp, Short4 -0.002000pp. B1/B2/B3 Score5 deltas are '
        '-0.020338/-0.042524/-0.053377pp. These single-seed observations do '
        'not establish statistically reliable gains or losses.',
        'Among the two perturbations, increasing backbone LR reduces Score5 '
        'relative to decreasing it; increasing mask LR performs better than '
        'decreasing it. Original LR remains the local peak for both axes, '
        'and neither axis changes Urban R@1. There is no common beneficial trend.',
        'No nonfinite update or complete closed mask was observed. Last50 mask '
        'structure remains close to original E2, including its existing F<Dall '
        'density order. The frozen-batch hierarchy/sparsity cosine is negative '
        'in visual masks (-0.357 to -0.345) and positive in text masks/shared '
        'pool (0.226 to 0.253). This is descriptive mechanism evidence, '
        'not evidence of improved semantic selection.',
        'CPU validation: 32 tests passed; see FINAL_CPU_TESTS.json. '
        'Frozen full-batch preupdate validation, four complete stream/LR '
        'proofs, strict exports, and final source/checkpoint immutability '
        'checks all passed. Full weights and raw logs remain local.']
    lines += ['', 'Gradient probes are frozen-checkpoint measurements on the first '
        'seed0 epoch0 batch; they describe mechanism, not held-out retrieval quality. '
        'Density order reversals are structural observations rather than failures. '
        'No semantic-evidence improvement follows from lower density or violation alone.',
        '', 'All original parent and baseline checkpoint/bare/results/evaluation hashes '
        'remain unchanged. All four full-stage sample/token/index summaries match '
        'the independent original E2 epoch4 run. Production manifest and evaluator '
        'mathematics remain unchanged. GPU processes have exited. No new training is scheduled.']
    report.write_text(('\n'.join(lines) + '\n').replace(
        '`.\n| Dataset', '`.\n\n| Dataset'))
    print(json.dumps(dict(passed=True, comparisons=analyses, GPU_idle=True)))


if __name__ == '__main__':
    main()
