"""Summarize frozen VCP@3651 native protocols against VCP@500 and TI-fast@3651."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess


EXP = Path(__file__).resolve().parent
REPO = EXP.parents[2]
OUT = Path('/root/lk_projects/SAID-nest-clip-v1/vcp_mask_3epoch_v1/formal/VCP-Mask')
DATASETS = ('COCO', 'Urban-1k', 'Flickr30k-test1k', 'DOCCI')
DIRECTIONS = ('I2T', 'T2I')
RECALLS = ('R@1', 'R@5', 'R@10')


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def j_long(metrics):
    return statistics.fmean(metrics[name][direction]['R@1']
                            for name in ('Urban-1k', 'DOCCI') for direction in DIRECTIONS)


def compact(row):
    values = {key: value for key, value in row.items()
              if isinstance(value, (float, int, str)) and not isinstance(value, bool)}
    values['common_loss'] = row['loss'] - row['inc_weight'] * row['inc']
    return values


def main():
    reference_path = REPO / 'experiments/nest_clip_v1/ti_fast_3epoch_v1/FULL3EPOCH_RESULTS.json'
    vcp500_path = REPO / 'experiments/nest_clip_v1/vcp_mask_v1/FORMAL500_RESULTS.json'
    reference = load(reference_path)
    vcp500 = load(vcp500_path)
    check = load(OUT / 'export-check.json')
    assert check['passed'] and check['optimizer_steps'] == [3651]
    sources = {
        'COCO': OUT / 'coco_native.json', 'Urban-1k': OUT / 'urban_native.json',
        'Flickr30k-test1k': OUT / 'flickr_test1k/flickr_test1k.json',
        'DOCCI': OUT / 'docci/docci.json',
    }
    raw = {name: load(path) for name, path in sources.items()}
    assert all(value['checkpoint_sha256'] == check['bare_sha256'] for value in raw.values())
    assert raw['Urban-1k']['n_images'] == raw['Urban-1k']['n_captions'] == 1000
    assert raw['Flickr30k-test1k']['n_images'] == 1000 and raw['Flickr30k-test1k']['n_captions'] == 5000
    assert raw['DOCCI']['n_images'] == raw['DOCCI']['n_captions'] == 5000
    metrics = {'COCO': {direction: {f'R@{k}': raw['COCO'][f'{prefix}_R{k}'] for k in (1, 5, 10)}
                        for direction, prefix in [('I2T', 'image2text'), ('T2I', 'text2image')]},
               'Urban-1k': {direction: {f'R@{k}': raw['Urban-1k'][field][f'R{k}'] for k in (1, 5, 10)}
                            for direction, field in [('I2T', 'image2text'), ('T2I', 'text2image')]}}
    for name in ('Flickr30k-test1k', 'DOCCI'):
        metrics[name] = {direction: {recall: raw[name]['metrics'][recall][direction] for recall in RECALLS}
                         for direction in DIRECTIONS}
    all_metrics = {'VCP@3651': metrics, 'VCP@500': vcp500['metrics']['VCP-Mask'],
                   'TI-fast@3651': {name: reference['metrics_step3651'][name] for name in DATASETS}}
    deltas = {name: {dataset: {direction: {recall: 100 * (metrics[dataset][direction][recall] -
                                                                       value[dataset][direction][recall])
                                         for recall in RECALLS} for direction in DIRECTIONS}
                    for dataset in DATASETS}
              for name, value in all_metrics.items() if name != 'VCP@3651'}
    values = {name: j_long(value) for name, value in all_metrics.items()}
    rows = [json.loads(line) for line in (OUT / 'steps.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == list(range(501, 3652))
    assert all(row['nonfinite'] == 0 and all(h['gradients_finite'] for h in row['rank_health']) for row in rows)
    acceptance = load(OUT / 'acceptance.json')
    assert acceptance['passed'] and all(r['completed_updates'] == 3651 for r in acceptance['ranks'])
    step_times = {row['step']: max(h['seconds'] for h in row['rank_health']) for row in rows}
    regular = [step_times[step] for step in range(502, 3651)]
    window = [compact(row) for row in rows[-50:]]
    last50 = {key: statistics.fmean(row[key] for row in window) for key, value in window[-1].items()
              if isinstance(value, (int, float))}
    training = dict(records=len(rows), continuous=True, loss_and_gradients_finite=True,
                    acceptance=acceptance,
                    regular_timing_window='steps 502-3650; excludes first resumed step and final tail',
                    timing_scope='existing per-rank update/log diagnostic; excludes DataLoader waiting and disk save; total loop below includes them',
                    regular_step_mean_seconds=statistics.fmean(regular),
                    regular_step_median_seconds=statistics.median(regular),
                    regular_step_p95_seconds=statistics.quantiles(regular, n=100)[94],
                    regular_step_max_seconds=max(regular),
                    regular_steps_over_3_seconds={str(step): seconds for step, seconds in step_times.items()
                                                 if 502 <= step < 3651 and seconds > 3},
                    startup_update_seconds=step_times[501],
                    loop_seconds=max(r['seconds'] for r in acceptance['ranks']),
                    global_F_candidates_min=min(r['F_candidates'] for r in rows),
                    global_F_candidates_max=max(r['F_candidates'] for r in rows),
                    fallback_steps=sum(r['valid_global'] < 2 for r in rows),
                    duplicate_id_steps=sum(bool(r['duplicate_image_ids']) for r in rows))
    evidence = EXP / 'evidence'
    evidence.mkdir(exist_ok=True)
    for name, source in sources.items():
        shutil.copy2(source, evidence / f'{name.lower().replace("-", "_")}.json')
    for name in ('export-check.json', 'config.json', 'acceptance.json'):
        shutil.copy2(OUT / name, evidence / name)
    execution = {p.stem: load(p) for p in evidence.glob('*.execution.json')}
    result = dict(schema_version=1, generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  experiment='VCP-Mask full three epochs', training_git_head=load(OUT / 'config.json')['git_head'],
                  evaluation_git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  completed_updates=3651, scheduler_horizon=3651, checkpoint_sha256=sha(OUT / 'step003651.pt'),
                  bare_sha256=sha(OUT / 'student_step3651.pt'), export_check=check,
                  source_hashes={'TI_reference': sha(reference_path), 'VCP500_reference': sha(vcp500_path)},
                  protocol=dict(datasets=DATASETS, batch_size=64, device='cuda:0', coco_images=5000,
                                coco_captions=25000, coco_similarity_chunk=512, native_only=True,
                                scoring='normalized native image/text embedding inner product',
                                excluded=['DCI', 'long-DCI']),
                  metrics=all_metrics, deltas_pp=deltas, J_long=values,
                  training=training, mechanism_last50=last50,
                  mechanism_VCP500_last50=vcp500['groups']['VCP-Mask']['last50_mean'],
                  execution=execution, limitations=['one seed; previously inspected benchmarks',
                                                   'longer budget is compared separately from step500 controls'])
    (EXP / 'FULL3EPOCH_RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT / 'FINAL_RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    wins = sum(value > 1e-12 for dataset in deltas['TI-fast@3651'].values()
               for direction in dataset.values() for value in direction.values())
    ties = sum(abs(value) <= 1e-12 for dataset in deltas['TI-fast@3651'].values()
               for direction in dataset.values() for value in direction.values())
    lines = [
        '# VCP-Mask: Full Three-Epoch Native Results', '',
        f"Training commit: `{result['training_git_head']}`. Evaluation commit: `{result['evaluation_git_head']}`.", '',
        'The audited VCP@500 formal state was continued to update 3651 with the unchanged three-epoch cosine schedule. '
        'Only the final training checkpoint was saved in this continuation. All four ranks completed 3151 new updates.', '',
        f"J_long: VCP@500 {values['VCP@500']*100:.3f}%, VCP@3651 {values['VCP@3651']*100:.3f}%, "
        f"TI-fast@3651 {values['TI-fast@3651']*100:.3f}%. VCP change from step500: "
        f"{100*(values['VCP@3651']-values['VCP@500']):+.3f} pp; difference from TI-fast@3651: "
        f"{100*(values['VCP@3651']-values['TI-fast@3651']):+.3f} pp.", '',
        f'Against TI-fast at the same budget, VCP has {wins} wins, {ties} ties and {24-wins-ties} losses '
        'across the 24 recall values. This single-seed experiment does not establish statistical significance.', '',
        '## Native Retrieval', '',
        'All recall values are percentages; differences are percentage points. Bare-student normalized '
        'image/text vectors are scored by inner product, without masks, adapters, fusion or reranking.', '',
        '| Dataset | Direction | Metric | VCP@500 | VCP@3651 | TI-fast@3651 | VCP3651-500 | VCP-TI3651 |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for dataset in DATASETS:
        for direction in DIRECTIONS:
            for recall in RECALLS:
                items = [dataset, direction, recall]
                items += [f'{all_metrics[group][dataset][direction][recall]*100:.2f}'
                          for group in ('VCP@500', 'VCP@3651', 'TI-fast@3651')]
                items += [f'{deltas[group][dataset][direction][recall]:+.2f}'
                          for group in ('VCP@500', 'TI-fast@3651')]
                lines.append('| ' + ' | '.join(items) + ' |')
    lines += ['', '## Training and Export', '',
              f"Training-loop wall time: {training['loop_seconds']/3600:.3f} hours. "
              f"Logged regular step mean/median/P95/max: {statistics.fmean(regular):.3f}/"
              f"{statistics.median(regular):.3f}/{training['regular_step_p95_seconds']:.3f}/{max(regular):.3f} seconds. "
              'These per-rank update diagnostics exclude DataLoader waiting and disk writes; the complete loop time includes them.', '',
              'All 3151 continuation records are consecutive. Losses and gradients are finite; final rank parameter '
              'difference is 0 and NCCL all-reduce passes. Strict export loads at optimizer step 3651 and native '
              'image/text embedding maximum absolute errors are 0.', '',
              '| Rank | Peak allocated GiB | Peak reserved GiB | Final parameter difference |',
              '|---|---:|---:|---:|']
    for rank in acceptance['ranks']:
        lines.append(f"| {rank['rank']} | {rank['peak_allocated_gib']:.3f} | {rank['peak_reserved_gib']:.3f} | "
                     f"{rank['max_parameter_difference_from_rank0']:.1f} |")
    lines += ['', f"Regular logged updates over 3s: `{training['regular_steps_over_3_seconds']}`. "
              f"First resumed update: {step_times[501]:.3f}s. All slow regular entries are retained.", '',
              '## Mechanism', '',
              'The following values average the last 50 updates. O/E are the implementation names for P/R.', '',
              '| Quantity | VCP@500 | VCP@3651 |', '|---|---:|---:|']
    for key in ('common_loss', 'F_i2t', 'F_t2i', 'O_i2t', 'O_t2i', 'E_i2t', 'E_t2i',
                'F_keep_ratio', 'O_keep_ratio', 'E_keep_ratio', 'inc', 'inc_weight',
                'hard_inclusion_violation', 'oe_iou', 'q_minus_w_norm', 'w_norm',
                'F_hard_mask_switch_fraction', 'O_hard_mask_switch_fraction', 'E_hard_mask_switch_fraction'):
        lines.append(f"| {key} | {vcp500['groups']['VCP-Mask']['last50_mean'][key]:.6f} | {last50[key]:.6f} |")
    lines += ['', '## Protocol and Evidence', '',
              'COCO canonical: 5000 images/25000 captions, similarity chunk 512. Urban-1k: 1000 pairs. '
              'Flickr30k test1K: 1000 images/5000 captions. DOCCI test5K: 5000 pairs. Image batch 64 on cuda:0. '
              'DCI and Long-DCI were not run.', '',
              f"Final full-checkpoint SHA256: `{result['checkpoint_sha256']}`.", '',
              f"Bare-student SHA256: `{result['bare_sha256']}`.", '',
              'Original evaluator JSON, commands, evaluator commit, stage timings, exit codes and export/acceptance '
              'evidence are under `evidence/`. Full training checkpoints, student weights, data and large logs '
              'remain on the server. Results use one seed on previously examined benchmarks. '
              'The 3651-update model is not ranked against unrelated 500-update experiments.', '']
    (EXP / 'FULL3EPOCH_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'report': str(EXP / 'FULL3EPOCH_REPORT.md'), 'J_long': values}))


if __name__ == '__main__':
    main()
