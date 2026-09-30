"""Final Balanced native comparison with budget-matched TI/VCP and its own step500."""
import datetime as dt
import json
from pathlib import Path
import shutil
import statistics
import subprocess

from experiments.nest_clip_v1.stack_crossscore_v1.summarize import (
    DATASETS, DIRECTIONS, RECALLS, compact, j_long, load, metrics, rows, sha)


EXP = Path(__file__).resolve().parent
REPO = EXP.parents[2]
OUT = Path('/root/lk_projects/SAID-nest-clip-v1/balanced_stack_3epoch_v1/formal/Balanced-Stack-Patch')
PARENT_EXP = Path('/root/lk_projects/SAID-mask-balance-cosine-v1/experiments/nest_clip_v1/mask_balance_cosine_v1')
LONG_SHA = '8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b'
ALL_DATASETS = (*DATASETS, 'Long-DCI')


def difference(left, right):
    return {dataset: {direction: {recall: 100 * (left[dataset][direction][recall] -
                                                right[dataset][direction][recall]) for recall in RECALLS}
                      for direction in DIRECTIONS} for dataset in left if dataset in right}


def main():
    evidence = EXP / 'evidence'
    parent_path = PARENT_EXP / 'RESULTS.json'
    long_parent_path = PARENT_EXP / 'LONG_DCI_RESULTS.json'
    ti_path = REPO / 'experiments/nest_clip_v1/ti_fast_3epoch_v1/FULL3EPOCH_RESULTS.json'
    vcp_path = REPO / 'experiments/nest_clip_v1/vcp_mask_3epoch_v1/FULL3EPOCH_RESULTS.json'
    parent, parent_long, ti, vcp = [load(path) for path in (parent_path, long_parent_path, ti_path, vcp_path)]
    check = load(OUT / 'export-check.json')
    assert check['passed'] and check['strict_load'] and check['optimizer_steps'] == [3651]
    assert check['image_max_abs'] == check['text_max_abs'] == 0
    current, raw, sources = metrics(OUT)
    long_path = OUT / 'long_dci/long_dci.json'
    long_result = load(long_path)
    assert long_result['n_images'] == long_result['n_captions'] == 7602
    assert long_result['manifest_sha256'] == LONG_SHA
    assert ti['long_dci_evaluation']['manifest_sha256'] == LONG_SHA
    assert parent_long['manifest_sha256'] == LONG_SHA
    sources['Long-DCI'] = long_path
    raw['Long-DCI'] = long_result
    assert all(value['checkpoint_sha256'] == check['bare_sha256'] for value in raw.values())
    assert raw['Urban-1k']['n_images'] == raw['Urban-1k']['n_captions'] == 1000
    assert raw['Flickr30k-test1k']['n_images'] == 1000 and raw['Flickr30k-test1k']['n_captions'] == 5000
    assert raw['DOCCI']['n_images'] == raw['DOCCI']['n_captions'] == 5000
    current['Long-DCI'] = {direction: {recall: long_result['metrics'][recall][direction]
                                     for recall in RECALLS} for direction in DIRECTIONS}
    all_metrics = {
        'Balanced@3651': current,
        'Balanced@500': {**parent['metrics']['Balanced-Stack-Patch'],
                         'Long-DCI': parent_long['metrics']['Balanced-Stack-Patch']},
        'TI-fast@3651': {**ti['metrics_step3651'], 'Long-DCI': ti['long_dci_evaluation']['step3651']},
        'VCP@3651': vcp['metrics']['VCP@3651'],
    }
    deltas = {group: difference(current, target) for group, target in all_metrics.items()
              if group != 'Balanced@3651'}
    long_values = {group: j_long(m) for group, m in all_metrics.items()}
    records = rows(OUT / 'steps.jsonl')
    assert [record['step'] for record in records] == list(range(501, 3652))
    assert all(record['nonfinite'] == 0 and all(h['gradients_finite'] for h in record['rank_health'])
               for record in records)
    acceptance = load(OUT / 'acceptance.json')
    assert acceptance['passed'] and acceptance.get('resource_failure') is None
    assert all(r['completed_updates'] == 3651 and r['updates_this_run'] == 3151 and
               r['max_parameter_difference_from_rank0'] == 0 for r in acceptance['ranks'])
    cycles = rows(OUT / 'cycle_timing.jsonl')
    assert [r['step'] for r in cycles] == list(range(501, 3652))
    regular = [r['four_rank_max_seconds'] for r in cycles if 502 <= r['step'] <= 3650]
    last50 = [compact(r) for r in records[-50:]]
    means = {key: statistics.fmean(r[key] for r in last50 if key in r)
             for key, value in last50[-1].items() if isinstance(value, (int, float))}
    training = dict(config=load(OUT / 'config.json'), acceptance=acceptance,
                    continuous=True, records=len(records), loss_and_gradients_finite=True,
                    timing_scope='real DataLoader complete cycles, maximum over four ranks, ordinary logs included; disk checkpoint separate',
                    regular_window='steps 502-3650; excludes first resumed update and final tail',
                    mean_seconds=statistics.fmean(regular), median_seconds=statistics.median(regular),
                    p95_seconds=statistics.quantiles(regular, n=100)[94], max_seconds=max(regular),
                    slow_steps=[r for r in cycles if r['step'] > 501 and r['four_rank_max_seconds'] > 3],
                    resumed_first_update_seconds=cycles[0]['four_rank_max_seconds'],
                    loop_seconds=max(r['seconds'] for r in acceptance['ranks']),
                    checkpoint_timing=rows(OUT / 'checkpoint_timing.jsonl'),
                    fallback_steps=sum(r['valid_global'] < 2 for r in records),
                    duplicate_id_steps=sum(bool(r['duplicate_image_ids']) for r in records),
                    F_candidate_range=[min(r['F_candidates'] for r in records),max(r['F_candidates'] for r in records)],
                    view_valid_range=[min(r['valid_global'] for r in records),max(r['valid_global'] for r in records)])
    for name, source in sources.items():
        shutil.copy2(source, evidence / f'{name.lower().replace("-", "_")}.json')
    for name in ('config.json','acceptance.json','export-check.json','cycle_timing.jsonl','checkpoint_timing.jsonl'):
        shutil.copy2(OUT / name, evidence / name)
    result = dict(schema_version=1, generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  experiment='Balanced-Stack-Patch full three epochs', completed_updates=3651, horizon=3651,
                  training_git_head=training['config']['git_head'],
                  evaluation_git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                  checkpoint_sha256=sha(OUT / 'step003651.pt'), bare_sha256=sha(OUT / 'student_step3651.pt'),
                  export_check=check, training=training,
                  reference_files={name: dict(path=str(path),sha256=sha(path)) for name,path in
                                   [('Balanced500',parent_path),('Balanced500Long',long_parent_path),
                                    ('TI3651',ti_path),('VCP3651',vcp_path)]},
                  metrics=all_metrics, deltas_pp=deltas, J_long=long_values,
                  delta_J_long_pp={group:100*(long_values['Balanced@3651']-value)
                                   for group,value in long_values.items() if group!='Balanced@3651'},
                  mechanism_last50=means, mechanism_parent500=parent['groups']['Balanced-Stack-Patch']['last50_mean'],
                  protocol=dict(datasets=ALL_DATASETS, batch_size=64, device='cuda:0', coco_similarity_chunk=512,
                                native_only=True, scoring='normalized student image/text embeddings, plain inner product',
                                long_dci_manifest_sha256=LONG_SHA,
                                J_long='mean Urban/DOCCI I2T/T2I R@1; Long-DCI reported separately', excluded=['DCI Full']),
                  execution={p.name:load(p) for p in evidence.glob('*.execution.json')},
                  limitations=['one seed; previously inspected benchmarks',
                               'Balanced500 is a budget extension comparison; only TI3651/VCP3651 match final budget',
                               'VCP3651 has no Long-DCI result; absent measurements are not inferred',
                               'gate/logit diagnostics were preset at updates up to500; no final gate quantiles were collected'])
    (EXP / 'FULL3EPOCH_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT / 'FINAL_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Balanced-Stack-Patch: Final Three-Epoch Results','',
           f"Generated UTC: {result['generated_utc']}",'',
           f"Training commit: `{result['training_git_head']}`. Evaluation commit: `{result['evaluation_git_head']}`.",'',
           'The audited step500 formal state continued to3651, restoring the model, channel gate, optimizer and '
           'four rank RNG states. The original three-epoch schedule and method were unchanged. '
           'Only the final full training checkpoint was saved in the continuation.','',
           '## J_long','', '| Model | J_long % | Balanced3651 minus model pp |', '|---|---:|---:|']
    for group,value in long_values.items():
        lines.append(f'| {group} | {value*100:.3f} | {100*(long_values["Balanced@3651"]-value):+.3f} |')
    lines += ['', 'J_long is the frozen Urban/DOCCI bidirectional R@1 average. Long-DCI does not change its definition. '
              'Balanced@500 shows the budget extension, while TI-fast@3651 and VCP@3651 provide matched-budget '
              'method comparisons. One seed does not establish statistical significance.','',
              '## Native Retrieval','',
              'All recall values are percentages. Bare-student normalized image/text embeddings use plain inner product '
              'without mask, fusion, reranking or ensemble. Differences in JSON are percentage points.','',
              '| Dataset | Direction | Recall | Balanced@500 | Balanced@3651 | TI-fast@3651 | VCP@3651 | Balanced-TI pp |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for dataset in ALL_DATASETS:
        for direction in DIRECTIONS:
            for recall in RECALLS:
                values=[f'{all_metrics[group][dataset][direction][recall]*100:.2f}'
                        if dataset in all_metrics[group] else 'not evaluated'
                        for group in ('Balanced@500','Balanced@3651','TI-fast@3651','VCP@3651')]
                lines.append('| '+' | '.join([dataset,direction,recall,*values,
                                              f'{deltas["TI-fast@3651"][dataset][direction][recall]:+.2f}'])+' |')
    lines += ['', '## Cost and Integrity','',
              f"Training loop time: {training['loop_seconds']/3600:.3f} hours. Real DataLoader full-cycle "
              f"mean/median/P95/max: {training['mean_seconds']:.3f}/{training['median_seconds']:.3f}/"
              f"{training['p95_seconds']:.3f}/{training['max_seconds']:.3f}s. First resumed update: "
              f"{training['resumed_first_update_seconds']:.3f}s. Slow regular updates are preserved in JSON.", '',
              'All four ranks completed3151 continuation updates, with consecutive logs501-3651, finite losses and '
              'gradients, final parameter max difference0 and successful NCCL all-reduce. Export strictly loads '
              'at optimizer step3651 with native image/text embedding error0.','',
              '| Rank | Peak allocated GiB | Peak reserved GiB |', '|---|---:|---:|']
    for rank in acceptance['ranks']:
        lines.append(f"| {rank['rank']} | {rank['peak_allocated_gib']:.3f} | {rank['peak_reserved_gib']:.3f} |")
    lines += ['', '## Mechanism','', 'The values below average the last50 updates. O/E are P/R in the implementation.', '',
              '| Metric | Balanced@500 | Balanced@3651 |', '|---|---:|---:|']
    for key in ('common_loss','F_i2t','F_t2i','O_i2t','O_t2i','E_i2t','E_t2i',
                'F_keep_ratio','O_keep_ratio','E_keep_ratio','inc','inc_weight','hard_inclusion_violation','oe_iou'):
        lines.append(f"| {key} | {result['mechanism_parent500'][key]:.6f} | {means[key]:.6f} |")
    lines += ['', 'Gate/logit diagnostics were preset only through update500, so final gate quantiles, logit moments '
              'and replacement-image switching are not claimed here. No additional mechanism benchmark was run.','',
              '## Protocol and Artifacts','',
              'COCO canonical5000 images/25000 texts (similarity chunk512), Urban1000 pairs, Flickr test1K1000 '
              'images/5000 texts, DOCCI5000 pairs, Long-DCI7602 pairs. Image batch64 on cuda:0. '
              'No DCI Full evaluation was run. VCP has no Long-DCI measurement in this comparison.','',
              f"Training checkpoint SHA256: `{result['checkpoint_sha256']}`.",'',
              f"Bare student SHA256: `{result['bare_sha256']}`.",'',
              f"Long-DCI manifest SHA256: `{LONG_SHA}`.",'',
              'Raw evaluator JSON, actual code commits, commands, stage exit codes, timing, hash and strict-export '
              'evidence are preserved under evidence/. Weights, data and full training/token logs remain server-local. '
              'All training and evaluation stop at the requested final model; no new seeds or parameter scans are started.','']
    (EXP / 'FULL3EPOCH_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(J_long=long_values,deltas_pp=result['delta_J_long_pp'],
                         report=str(EXP/'FULL3EPOCH_REPORT.md')),indent=2))


if __name__ == '__main__':
    main()
