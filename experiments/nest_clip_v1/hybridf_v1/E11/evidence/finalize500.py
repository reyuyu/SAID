"""Render the user-selected E11 step500 checkpoint and its real native evaluations."""
import gzip
import hashlib
import json
from pathlib import Path


PUBLIC = Path(__file__).resolve().parents[1]
EVIDENCE = PUBLIC / 'evidence'
ROOT = Path('/root/lk_projects/SAID-nest-clip-v1/hybridf_v1/E11')
REPO = PUBLIC.parents[3]
DATASETS = ('coco', 'urban', 'flickr_test1k', 'docci')
SIZES = ((5000, 25000), (1000, 1000), (1000, 5000), (5000, 5000))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def recall(payload):
    if 'urban1k' in payload:
        payload = payload['urban1k']
    if 'image2text_R1' in payload:
        return {(direction, f'R@{k}'): payload[f'{prefix}_R{k}']
                for direction, prefix in (('I2T', 'image2text'), ('T2I', 'text2image'))
                for k in (1, 5, 10)}
    if 'image2text' in payload:
        return {(direction, f'R@{k}'): payload[key][f'R{k}']
                for direction, key in (('I2T', 'image2text'), ('T2I', 'text2image'))
                for k in (1, 5, 10)}
    return recall(payload['metrics']) if 'image2text_R1' in payload.get('metrics', {}) else {
        (direction, f'R@{k}'): payload['metrics'][f'R@{k}'][direction]
        for direction in ('I2T', 'T2I') for k in (1, 5, 10)}


def main():
    audit = read(ROOT / 'evidence/formal500-audit.json')
    smoke = read(ROOT / 'evidence/smoke-retry-1-audit.json')
    export = read(ROOT / 'step500/export-check.json')
    assert audit['passed'] and smoke['passed']
    assert audit['audited_snapshot_step'] == 500 and audit['horizon'] == 3651
    assert audit['actual_last_logged_step'] > 500 and audit['compared_rank_streams'] == 2000
    assert audit['sample_F_P_R_K_streams_identical_to_A3'] and audit['step0_model_optimizer_rng_equal']
    assert audit['execution']['exit_code'] == 1
    assert export['passed'] and export['strict_load'] and export['optimizer_steps'] == [500]
    assert export['image_max_abs'] == export['text_max_abs'] == 0
    assert export['checkpoint_sha256'] == audit['checkpoints'][-1]['sha256']
    assert export['bare_sha256'] == sha(ROOT / 'step500/student_step500.pt')
    assert read(EVIDENCE / 'export500.execution.json')['exit_code'] == 0
    assert read(EVIDENCE / 'verify500.execution.json')['exit_code'] == 0
    raw500 = PUBLIC / 'step500_steps.jsonl.gz'
    with gzip.open(raw500, 'rt') as handle:
        first500 = [json.loads(line) for line in handle]
    assert len(first500) == 500 and [x['step'] for x in first500] == list(range(1, 501))
    peaks = [dict(rank=rank,
                  peak_allocated_gib=max(x['rank_health'][rank]['peak_allocated_gib']
                                         for x in first500),
                  peak_reserved_gib=max(x['rank_health'][rank]['peak_reserved_gib']
                                        for x in first500)) for rank in range(4)]
    observed_step_seconds = sum(max(h['seconds'] for h in row['rank_health'])
                                for row in first500)

    evaluated, executions = {}, {}
    for name, (n_images, n_texts) in zip(DATASETS, SIZES):
        execution = read(EVIDENCE / f'eval500-{name}.execution.json')
        assert execution['exit_code'] == 0, name
        assert 'Traceback' not in (EVIDENCE / f'eval500-{name}.console.txt').read_text(), name
        path = (ROOT / 'step500/evaluation' / f'{name}_native.json' if name in ('coco', 'urban')
                else ROOT / 'step500/evaluation' / name / f'{name}.json')
        row = read(path)
        assert row['checkpoint_sha256'] == export['bare_sha256'], name
        if name == 'coco':
            assert (n_images, n_texts) == (5000, 25000)
        else:
            assert (row['n_images'], row.get('n_texts', row.get('n_captions'))) == (n_images, n_texts)
        assert len(recall(row)) == 6 and all(0 <= value <= 1 for value in recall(row).values())
        evaluated[name], executions[name] = row, execution

    old = read(PUBLIC.parents[1] / 'randomk500/results.json')
    assert old['baseline_verification']['clean']['asset_verification'] == 'PASS'
    fixed = read(PUBLIC.parents[1] / 'formal500_results.json')['raw_evaluations']['A3']
    a3 = read(EVIDENCE / 'references/randomk3651.json')['evaluations']['500']
    s0 = read(PUBLIC.parents[2] / 's0_dualmask_masked_3epoch/evidence/masked_formal500_report.json')['baseline_S0']
    refs = {'A3-RandomK@500': a3, 'Fixed-A3@500': fixed,
            'Clean@500': old['baselines']['clean'],
            'Full@500': old['baselines']['full'],
            'S0@500': {'coco': s0['coco'], 'urban': s0['urban1k']}}
    rows, head = [], []
    for name in DATASETS:
        for direction in ('I2T', 'T2I'):
            for metric in ('R@1', 'R@5', 'R@10'):
                value = recall(evaluated[name])[(direction, metric)]
                baseline = {key: recall(payload[name])[(direction, metric)]
                            for key, payload in refs.items() if name in payload}
                row = dict(dataset=name, direction=direction, metric=metric, E11=value,
                           reference=baseline, delta_pp={key: 100 * (value - ref)
                                                          for key, ref in baseline.items()})
                rows.append(row)
                if name in ('coco', 'urban'):
                    head.append(row)
    result = dict(status='COMPLETE_STEP500_USER_STOPPED',
                  description='Only saved step500 checkpoint used. User interrupted original 3651-step run after it had reached step1123.',
                  experiment='NEST-HybridF-E11', training_commit=audit['code_commit'],
                  initial_sha256=audit['config']['init_sha256'],
                  initial_smoke_failure=read(EVIDENCE / 'blocked-audit.json'),
                  clean_smoke=dict(passed=True, path='evidence/smoke-retry-1-audit.json',
                                   sha256=sha(ROOT / 'evidence/smoke-retry-1-audit.json')),
                  training_audit=audit, training_audit_sha256=sha(ROOT / 'evidence/formal500-audit.json'),
                  first500_evidence=dict(path=str(raw500.relative_to(PUBLIC)),
                                         sha256=sha(raw500), logged_steps=500,
                                         observed_step_seconds=observed_step_seconds,
                                         peak_gpu_memory=peaks),
                  evaluator_sha256={name: sha(REPO / name) for name in (
                      'tools/eval_nest_native.py', 'eval/retrieval/coco_retrieval.py',
                      'tools/urban1k_retrieval.py',
                      'experiments/s0_dualmask_full_v01/evidence/step2000/new_evaluations/eval_extended_real.py')},
                  export=dict(**export,
                              export_execution=read(EVIDENCE / 'export500.execution.json'),
                              verification_execution=read(EVIDENCE / 'verify500.execution.json')),
                  evaluations=evaluated, evaluation_executions=executions,
                  comparisons=rows, j_long=None,
                  j_long_unavailable='User excluded DCI and Long-DCI from this step500 evaluation.',
                  reference_provenance=dict(A3_RandomK='randomk500/results.json and randomk3epoch/results.json at update500',
                                            Fixed_A3='formal500_results.json',
                                            Clean_Full='randomk500/results.json verified existing step500 exports',
                                            S0='s0_dualmask_masked_3epoch/evidence/masked_formal500_report.json (historical, original checkpoint unavailable here)'),
                  limits='One seed; S0/Clean/Full change methodology. No completed 3651-step E11 run or final four-rank agreement at precisely step500; evaluation uses strict rank0 step500 checkpoint. No DCI or Long-DCI evaluation this round.')
    (PUBLIC / 'step500_results.json').write_text(json.dumps(result, indent=2) + '\n')

    def pct(x):
        return f'{100 * x:.3f}'

    def table(data, names):
        lines = ['| Dataset | Direction | Metric | E11 (%) | ' + ' | '.join(
            f'{name} (%)' for name in names) + ' | Δ vs A3-RandomK (pp) |',
            '|---|---|---|---:|' + '---:|' * len(names) + '---:|']
        for row in data:
            lines.append('| ' + ' | '.join((row['dataset'], row['direction'], row['metric'],
                pct(row['E11']), *(pct(row['reference'].get(name, float('nan'))) for name in names),
                f"{row['delta_pp']['A3-RandomK@500']:+.3f}")) + ' |')
        return '\n'.join(lines)

    main_table = table(rows, ('A3-RandomK@500',))
    same_budget = table(head, ('A3-RandomK@500', 'Fixed-A3@500',
                               'S0@500', 'Clean@500', 'Full@500'))
    keys = ('loss', 'common_loss', 'F_mask_i2t', 'F_mask_t2i', 'F_native_i2t',
            'F_native_t2i', 'F_hybrid', 'O_i2t', 'O_t2i', 'E_i2t', 'E_t2i',
            'F_sparse', 'O_sparse', 'E_sparse', 'inc', 'inc_weight',
            'hard_inclusion_violation', 'F_keep_ratio', 'O_keep_ratio', 'E_keep_ratio',
            'F_all_open', 'O_all_open', 'E_all_open', 'F_all_closed',
            'O_all_closed', 'E_all_closed', 'oe_iou', 'valid_global')
    reference = old['mechanism']['RandomK']['last50']
    diagnostics = ['| Last 50 updates | E11@500 | A3-RandomK@500 |', '|---|---:|---:|']
    for key in keys:
        prior = reference.get(key, reference.get(key.replace('F_mask_', 'F_')))
        if key.startswith('F_native_') or key == 'F_hybrid':
            prior = None
        diagnostics.append(f"| {key} | {audit['last50'][key]:.6f} | " +
                           (f'{prior:.6f}' if prior is not None else 'n/a') + ' |')
    positive = sum(row['delta_pp']['A3-RandomK@500'] > 0 for row in rows)
    report = f'''# NEST-HybridF-E11: step500 original retrieval

**Only the saved step500 checkpoint is evaluated.** The user stopped the previously authorized 3651-step run after it had already logged step{audit['actual_last_logged_step']}; the torchrun process received SIGINT and exited 1 by design. No updates after 500 contribute to this report. This is not a completed 3-epoch run, and the final cross-rank parameter-agreement check did not run at step500.

Training code commit `{audit['code_commit']}`; seed0, 4×A100 80GB, batch256/rank, no accumulation, shared init SHA256 `{audit['config']['init_sha256']}`, eta0.25, arm=A3, random K unchanged, epochs3 and cosine horizon3651. From the common step0, all four ranks logged steps1–500 with finite loss and gradients; all {audit['compared_rank_streams']} rank-step sample/F/P/R/K streams match the A3-RandomK reference. Maximum full-loss formula reconstruction error {audit['max_loss_reconstruction_abs_error']:.3g}. The step500 checkpoint SHA256 is `{export['checkpoint_sha256']}`; strictly loaded bare student SHA256 `{export['bare_sha256']}`. Optimizer step500 and exact image/text embedding equality passed; all four native evaluation commands exited 0. The [interrupted-run audit](evidence/formal500-audit.json) records the intentional nonzero exit and later unused checkpoints.

The first 5-step smoke was rejected because rank3 reported an NCCL error after its updates. A new four-rank 5-step smoke with `NCCL_DEBUG=INFO` for diagnostics passed the strict audit; both original failure and retry remain recorded. The first error's cause was not established.

**Primary same-budget comparison: E11@500 vs A3-RandomK@500.** The following 24 values are percentages. The difference is `100×(E11 recall−A3 recall)` in percentage points, not relative percent. {positive} of 24 values are higher for E11.

{main_table}

On COCO and Urban, additional existing 500-step references are available. Fixed-A3/Clean/Full results were verified in their original reports. S0 numbers come from a historical step500 report; its original checkpoint is not present on this server, so S0 was not re-evaluated here. These other methods are context, not a one-variable controlled comparison.

{same_budget}

Per the user's latest instruction, DCI and Long-DCI were not evaluated. Prespecified `J_long` includes DCI, so it cannot be computed from this round's evaluations. No historical DCI score is substituted.

Existing training-log diagnostics, arithmetic mean of updates451–500 (O=P and E=R). A3's F uses masked-only CE; E11 combines masked and native CE. Different total losses do not measure retrieval quality; `common_loss=loss−inc_weight×inc`.

{chr(10).join(diagnostics)}

The observed first500 step times sum to {observed_step_seconds:.1f}s (excludes checkpoint writes and startup); the interrupted full command ran for {audit['execution']['wall_seconds']:.1f}s before the user-requested SIGINT. Highest per-rank GPU allocated memory through step500: {max(p['peak_allocated_gib'] for p in peaks):.3f}GiB; reserved: {max(p['peak_reserved_gib'] for p in peaks):.3f}GiB. The source training log and all checkpoints, including unused step600–1100 files, remain on the server at `/root/lk_projects/SAID-nest-clip-v1/hybridf_v1/E11/formal`. Only the step500 bare student under `step500/` was scored. The [first500 step logs](step500_steps.jsonl.gz), [machine-readable scores and exit codes](step500_results.json), [snapshot audit](evidence/formal500-audit.json) and [evaluation JSONs](step500_evaluation) retain the evidence. Only one seed was run; no significance, broad superiority, or independent effect of the inclusion term is claimed. The native-F mix also reduces the masked-F weight, so this experiment does not isolate those two changes.
'''
    (PUBLIC / 'STEP500_REPORT.md').write_text(report)
    print(json.dumps(dict(passed=True, stop=audit['actual_last_logged_step'],
                          evaluated_step=500, comparisons=len(rows), positive=positive,
                          j_long=None)))


if __name__ == '__main__':
    main()
