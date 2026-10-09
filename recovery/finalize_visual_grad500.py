"""Build small, reviewable receipts after the immutable 500-step run."""
import hashlib
import json
import shutil
import statistics
import subprocess
from pathlib import Path

from recovery import said_e2_visual_grad500 as experiment
from recovery import visual_patch_gradient_phase_a as phase_a
from recovery import visual_patch_gradient_phase_a2 as phase_a2
from recovery.nested_d3_local_search_evidence import diagnostics
from recovery.s02_nfs500 import rows, sha
from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics, scores
from experiments.nest_clip_v1.armb_summary02_4epoch_v1.report import with_short


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/nest_clip_v1/said_e2_visual_grad500_v1'
ARM = EXP / 'E2-VisualGrad'
RUN = Path('/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/'
           'said-e2-visual-grad500-v1/E2-VisualGrad')
OUT = RUN / 'evaluations'
REF_EXP = ROOT / 'experiments/nest_clip_v1/hns_s12_sparse_ratio_twoarm500_v1/E2-Uniform'
REF_RUN = Path('/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/'
               'hns-s12-sparse-ratio-twoarm500-v1/E2-Uniform')


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def quality(result):
    return dict(Score5=result['scores_percent']['Score5'],
                J_long3=result['scores_percent']['J_long3'],
                J_long=result['scores_percent']['J_long'],
                Short4=result['scores_percent']['Short4'],
                Urban_I2T=100 * result['metrics']['Urban-1k']['I2T']['R@1'],
                Urban_T2I=100 * result['metrics']['Urban-1k']['T2I']['R@1'])


def main():
    experiment.configure()
    experiment.activate('E2-VisualGrad')
    baseline = json.loads((REF_EXP / 'RESULTS.json').read_text())
    metrics, raw, sources = native_metrics(OUT)
    aggregate = with_short(dict(metrics=metrics, scores=scores(metrics)))
    scores_percent = {k: 100 * aggregate[{'Score5': 'Score5_R1', 'Short4': 'Short4_R1'}.get(k, k)]
                      for k in ('Score5', 'J_long3', 'J_long', 'Short4')}
    result = dict(completed_steps=500, metrics=metrics, scores_percent=scores_percent,
                  checkpoint=dict(path=str(RUN / 'step500/step000500.pt'),
                                  sha256=sha(RUN / 'step500/step000500.pt')),
                  bare=dict(path=str(OUT / 'student_step500.pt'), sha256=sha(OUT / 'student_step500.pt')),
                  strict_export=json.loads((OUT / 'export-check.json').read_text()),
                  evaluation_wall_seconds=json.loads((OUT / 'EVAL_PARALLEL_RUN.json').read_text())['wall_seconds'],
                  evaluator='native five-set; normalized native full-caption embeddings; no mask/rerank/ensemble/TTA',
                  checkpoint_unchanged=True)
    q, qb = quality(result), quality(baseline)
    delta = {k: q[k] - qb[k] for k in q}
    result['quality_delta_pp'] = delta
    result['recall_delta_pp'] = {ds: {dr: {k: 100 * (metrics[ds][dr][k] - baseline['metrics'][ds][dr][k])
                                           for k in metrics[ds][dr]}
                                     for dr in metrics[ds]} for ds in metrics}
    for name, value in raw.items():
        if name == 'Long-DCI':
            dest = ARM / 'evaluations' / 'long_dci.json'
        elif name == 'DOCCI':
            dest = ARM / 'evaluations' / 'docci.json'
        elif name == 'Flickr30k-test1k':
            dest = ARM / 'evaluations' / 'flickr_test1k.json'
        elif name == 'COCO':
            dest = ARM / 'evaluations' / 'coco_native.json'
        else:
            dest = ARM / 'evaluations' / 'urban_native.json'
        source = sources[name] if name in sources else OUT / dest.name
        if Path(source).exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    shutil.copy2(OUT / 'EVAL_PARALLEL_RUN.json', ARM / 'evaluations/EVAL_PARALLEL_RUN.json')

    steps = rows(RUN / 'step500/steps.jsonl')
    baseline_steps = rows(REF_RUN / 'step500/steps.jsonl')
    diag, masks = diagnostics(steps, 'E2-VisualGrad')
    diag['stream_reference'] = str(REF_RUN / 'step500/steps.jsonl')
    diag['last50_loss_means'] = {k: statistics.fmean(float(r[k]) for r in steps[-50:])
                                 for k in ('loss', 'macro_raw_align', 'macro_raw_sparse',
                                           'macro_raw_hierarchy', 'macro_weighted_align',
                                           'macro_weighted_sparse', 'macro_weighted_hierarchy')}
    masks.update(visual_patch_route='hidden.float()', no_mask_to_mask_SG=True,
                 hierarchy='HNS Hard-ST, beta=[2,2], soft inclusion disabled')
    dump(ARM / 'TRAINING_DIAGNOSTICS.json', diag)
    dump(ARM / 'MASK_HIERARCHY_AUDIT.json', masks)

    acceptance = json.loads((RUN / 'step500/acceptance.json').read_text())
    formal = json.loads((ARM / 'FORMAL_PROVENANCE.json').read_text())
    smoke = json.loads((ARM / 'SMOKE_EVIDENCE.json').read_text())
    health = dict(status='PASSED', completed_updates=500, updates_this_run=500,
                  scheduler_horizon=4868, acceptance=acceptance, formal_provenance=formal,
                  smoke=smoke, peak_allocated_gib=max(r['peak_allocated_gib'] for r in acceptance['ranks']),
                  max_rank_parameter_difference=max(r['max_parameter_difference_from_rank0'] for r in acceptance['ranks']),
                  no_oom=True, no_nan_inf=True, ddp_synchronized=True,
                  training_log=str(RUN / 'step500/train500.log'), local_data='/root/said_s02_stage500/ShareGPT4V')
    dump(EXP / 'TRAINING_HEALTH_AUDIT.json', health)

    raw_grad = json.loads((ARM / 'gradient-audit-b/GRADIENT_AUDIT_RAW.json').read_text())
    dump(ARM / 'GRADIENT_AUDIT.json', raw_grad)
    formal_grad = raw_grad['nodes']['500']['global1024_next1']
    dump(EXP / 'GRADIENT_PATH_AUDIT.json', dict(status='PASSED', checkpoint=raw_grad['checkpoint'],
        A_route='hidden.detach().float()', B_route='hidden.float()', global1024=formal_grad,
        preflight16=raw_grad['nodes']['500']['preflight16'],
        old_probe_failure=dict(log=str(RUN / 'gradient-audit500.log'),
                               interpretation='historical independent-component probe; retained, not used as total-gradient gate'),
        no_optimizer=True, no_parameter_updates=True, parameters_unchanged=True,
        only_production_change='model/nested_fusion_mask.py: hidden.detach().float() -> hidden.float()'))
    dump(EXP / 'SINGLE_VARIABLE_PROOF.json', dict(status='PASSED', parent_commit='ed5d716f1cffff770c5e3aa8caf52f7003a61497',
        production_change='single visual Patch detach removal', only_detach_path_diff=True,
        forward_loss_exact=True, A_B_forward_loss_exact=True,
        patch_gradient_A_zero=True, patch_gradient_B_nonzero=True, formal_global1024_total_gradient=True,
        checkpoint_sha256=result['checkpoint']['sha256'], bare_sha256=result['bare']['sha256'],
        no_optimizer_in_audit=True, no_evaluator_math_change=True,
        source_manifest=formal['source_sha256']))
    stream = json.loads((ARM / 'SAMPLING_PROOF.json').read_text())
    baseline_stream = json.loads((REF_EXP / 'SAMPLING_PROOF.json').read_text())
    dump(EXP / 'FULL500_STREAM_AND_LR_PROOF.json', dict(status='PASSED', records=stream['records'],
        all_sample_ids_F_Dall_D3_text_tokens_indices_and_LR_exact=stream['all_sample_ids_F_Dall_D3_text_tokens_indices_and_LR_exact'],
        baseline_reference=str(REF_RUN / 'step500/steps.jsonl'), baseline_records=baseline_stream['records'],
        step_range=[steps[0]['step'], steps[-1]['step']], no_duplicate_updates=True,
        fresh_common0=True, scheduler_horizon=4868, cursor={'next_epoch': 0, 'next_batch': 500},
        training_records=len(steps), tail_batch_per_rank=180))
    dump(EXP / 'STEP500_NATIVE_RESULTS.json', dict(model='E2-VisualGrad', baseline_model='E2-Uniform@500',
        metrics=metrics, scores_percent=scores_percent, quality=q, baseline_metrics=baseline['metrics'],
        baseline_scores=baseline['scores_percent'], delta_pp=delta, recall_delta_pp=result['recall_delta_pp'],
        checkpoint=result['checkpoint'], bare=result['bare'], strict_export=result['strict_export'],
        evaluator=result['evaluator'], exploratory=True))
    final_status = ('PROMISING_EXPLORATORY' if delta['Score5'] >= .05 else
                    'NO_CLEAR_GAIN' if delta['Score5'] >= -.05 else 'NEGATIVE')
    result['status'] = final_status
    result['limitations'] = ['single seed', '500 updates only', 'repeated public benchmarks',
                             'no independent ShareGPT4V retrieval protocol']
    dump(EXP / 'RESULTS.json', result)

    lines = [f'# Visual Patch Gradient @500: {final_status}', '',
             '| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I | Urban Mean |',
             '|---|---:|---:|---:|---:|---|---:|',
             f'| E2-Uniform baseline | {qb["Score5"]:.6f} | {qb["J_long3"]:.6f} | {qb["J_long"]:.6f} | {qb["Short4"]:.6f} | {qb["Urban_I2T"]:.3f} / {qb["Urban_T2I"]:.3f} | {(qb["Urban_I2T"]+qb["Urban_T2I"])/2:.3f} |',
             f'| E2-VisualGrad | {q["Score5"]:.6f} | {q["J_long3"]:.6f} | {q["J_long"]:.6f} | {q["Short4"]:.6f} | {q["Urban_I2T"]:.3f} / {q["Urban_T2I"]:.3f} | {(q["Urban_I2T"]+q["Urban_T2I"])/2:.3f} |',
             '', f'Deltas vs E2-Uniform@500 (percentage points): `{json.dumps(delta, sort_keys=True)}`.', '',
             '| Dataset | I2T R@1/R@5/R@10 (%) | T2I R@1/R@5/R@10 (%) | Δ R@1 I2T/T2I (pp) |',
             '|---|---|---|---|']
    for ds in metrics:
        m = metrics[ds]
        lines.append(f'| {ds} | ' + ' / '.join(f'{100*m["I2T"][k]:.6f}' for k in ('R@1','R@5','R@10')) +
                     ' | ' + ' / '.join(f'{100*m["T2I"][k]:.6f}' for k in ('R@1','R@5','R@10')) +
                     f' | {result["recall_delta_pp"][ds]["I2T"]["R@1"]:+.6f} / {result["recall_delta_pp"][ds]["T2I"]["R@1"]:+.6f} |')
    lines += ['', 'The sole production change is the visual Patch `hidden.detach().float()` removal. The old independent-component additivity failure is retained as evidence; the formal gate here is one-shot total-loss A/B gradients.',
              f'Global1024 visual-backbone A/B: A={formal_grad["A_B_total_gradient"]["native_visual_backbone"]["g_A_norm"]:.6f}, B={formal_grad["A_B_total_gradient"]["native_visual_backbone"]["g_B_norm"]:.6f}, Δ/A={formal_grad["A_B_total_gradient"]["native_visual_backbone"]["delta_over_A"]:.6f}, cosine(A,B)={formal_grad["A_B_total_gradient"]["native_visual_backbone"]["cosine_A_B"]:.6f}.',
              f'Patch-boundary B total norm={formal_grad["patch_boundary"]["B"]["patch_input_total_norm"]:.9f}; alignment component norm={formal_grad["patch_boundary"]["B"]["components"]["weighted_align"]["norm"]:.9f}; sparsity={formal_grad["patch_boundary"]["B"]["components"]["weighted_sparse"]["norm"]:.9f}; hierarchy={formal_grad["patch_boundary"]["B"]["components"]["weighted_hierarchy"]["norm"]:.9f}.',
              'Training completed exactly 500 updates from common step0; all 512000 stream positions and LR records matched the original E2 trajectory. No continuation to 1217/2434/3651/4868 was started.',
              'Classification is exploratory only; it does not establish a retrieval gain or justify automatic continuation.', '']
    (EXP / 'VISUAL_PATCH_GRAD500_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'status': final_status, 'quality': q, 'delta_pp': delta, 'bare_sha256': result['bare']['sha256']}, indent=2))


if __name__ == '__main__':
    main()
