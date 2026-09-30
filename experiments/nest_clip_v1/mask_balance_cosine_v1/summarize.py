"""Two fixed repairs: exact 500-step references, native recalls and mechanism evidence."""
import datetime as dt
import json
from pathlib import Path
import shutil
import statistics
import subprocess

from experiments.nest_clip_v1.stack_crossscore_v1.summarize import (
    DATASETS, DIRECTIONS, RECALLS, compact, difference, j_long, load, metrics, rows, sha, signatures)


EXP = Path(__file__).resolve().parent
REPO = EXP.parents[2]
RUN = Path('/root/lk_projects/SAID-nest-clip-v1/mask_balance_cosine_v1')
GROUPS = ('Balanced-Stack-Patch', 'Cosine-CrossScore-CLS')
PARENTS = {'Balanced-Stack-Patch': 'S-PATCH', 'Cosine-CrossScore-CLS': 'C-CLS'}
CONFIGS = {'Balanced-Stack-Patch': 'nest_balanced_stack_patch.json',
           'Cosine-CrossScore-CLS': 'nest_cosine_crossscore_cls.json'}
MILESTONES = (1, 100, 200, 201, 300, 400, 500)


def main():
    evidence = EXP / 'evidence'
    parent_path = REPO / 'experiments/nest_clip_v1/stack_crossscore_v1/RESULTS.json'
    ti_path = REPO / 'experiments/nest_clip_v1/jointmask_fast_v1/FORMAL500_RESULTS.json'
    parent, ti = load(parent_path), load(ti_path)
    assert parent['groups']['S-PATCH']['completed_updates'] == 500
    assert parent['groups']['C-CLS']['completed_updates'] == 500
    resources = load(RUN / 'resource-results.json')
    reference_records = rows('/root/lk_projects/SAID-nest-clip-v1/jointmask_fast_v1/formal/TI-fast/steps.jsonl')
    reference_streams = signatures(reference_records)
    all_metrics = {name: {dataset: parent['metrics'][name][dataset] for dataset in DATASETS}
                   for name in ('S-PATCH', 'C-CLS', 'TI-fast')}
    groups = {}
    for group in GROUPS:
        item = dict(resource=resources[group], config=load(REPO / 'configs' / CONFIGS[group]),
                    completed_updates=0, metrics=None)
        probe = RUN / 'probe' / group
        if (probe / 'config.json').exists():
            item['probe_config'] = load(probe / 'config.json')
            item['parameter_counts'] = item['probe_config']['parameter_counts']
            item['component_initialization'] = item['probe_config']['component_initialization']
        if (probe / 'probe_timing.jsonl').exists():
            item['probe_timing'] = rows(probe / 'probe_timing.jsonl')
        if (probe / 'steps.jsonl').exists():
            item['initial_diagnostics'] = compact(rows(probe / 'steps.jsonl')[0])
        item['smoke_acceptance'] = load(RUN / 'smoke' / group / 'acceptance.json')
        assert item['smoke_acceptance']['passed']
        if item['resource']['resource_status'] != 'passed':
            item['formal_status'] = 'not_run_resource_infeasible'
            groups[group] = item
            continue
        root = RUN / 'formal' / group
        records = rows(root / 'steps.jsonl')
        assert [r['step'] for r in records] == list(range(1, 501))
        assert all(r['nonfinite'] == 0 and all(h['gradients_finite'] for h in r['rank_health']) for r in records)
        assert signatures(records) == reference_streams
        acceptance, check = load(root / 'acceptance.json'), load(root / 'export-check.json')
        assert acceptance['passed'] and all(r['completed_updates'] == 500 for r in acceptance['ranks'])
        assert check['passed'] and check['optimizer_steps'] == [500]
        m, raw, sources = metrics(root)
        assert all(r['checkpoint_sha256'] == check['bare_sha256'] for r in raw.values())
        cycles = rows(root / 'cycle_timing.jsonl')
        stable = [r['four_rank_max_seconds'] for r in cycles if r['step'] >= 6]
        window = [compact(r) for r in records[-50:]]
        mean = {key: statistics.fmean(r[key] for r in window if key in r)
                for key, value in window[-1].items() if isinstance(value, (float, int))}
        item.update(formal_status='completed', completed_updates=500, config=load(root / 'config.json'),
                    acceptance=acceptance, export_check=check, metrics=m, J_long=j_long(m),
                    stream_equal_TI500=True, steps_continuous=True, losses_and_gradients_finite=True,
                    checkpoint_sha256={f'step{s:06d}.pt': sha(root / f'step{s:06d}.pt') for s in (0,100,200,300,400,500)},
                    last50_mean=mean, key_steps={str(r['step']): compact(r) for r in records if r['step'] in MILESTONES},
                    diagnostic_steps={str(r['step']): compact(r) for r in records if r['step'] in (1,100,200,300,400,500)},
                    candidate_ranges={v: [min(r[v+'_candidates'] for r in records), max(r[v+'_candidates'] for r in records)]
                                      for v in ('F','O','E')},
                    fallback_steps=sum(r['valid_global'] < 2 for r in records),
                    duplicate_id_steps=sum(bool(r['duplicate_image_ids']) for r in records),
                    timing=dict(scope='complete real DataLoader cycle, four-rank maximum; checkpoints timed separately',
                                stable_steps='6-500', mean_seconds=statistics.fmean(stable),
                                median_seconds=statistics.median(stable), p95_seconds=statistics.quantiles(stable,n=100)[94],
                                max_seconds=max(stable), full_loop_seconds=max(r['seconds'] for r in acceptance['ranks']),
                                slow_steps=[r for r in cycles if r['step'] > 1 and r['four_rank_max_seconds'] > 3]))
        destination = evidence / 'native_results' / group
        destination.mkdir(parents=True, exist_ok=True)
        for name, source in sources.items():
            shutil.copy2(source, destination / f'{name.lower().replace("-", "_")}.json')
        for name in ('config.json','acceptance.json','export-check.json','cycle_timing.jsonl','checkpoint_timing.jsonl'):
            shutil.copy2(root / name, destination / name)
        groups[group] = item
        all_metrics[group] = m
    comparisons = {}
    for group in GROUPS:
        if group not in all_metrics:
            continue
        for reference in (PARENTS[group], 'TI-fast'):
            comparisons[f'{group}_minus_{reference}'] = dict(
                delta_pp=difference(all_metrics[group], all_metrics[reference]),
                delta_J_long_pp=100 * (j_long(all_metrics[group])-j_long(all_metrics[reference])))
    initializations = {}
    for group, item in groups.items():
        if 'component_initialization' not in item:
            continue
        before = parent['groups'][PARENTS[group]]['component_initialization']
        after = item['component_initialization']
        initializations[group] = {key: before[key] == after[key]
                                 for key in ('clip','text_blocks','visual_blocks','visual_adapter')}
        assert all(initializations[group].values())
        if group == 'Cosine-CrossScore-CLS':
            assert before['crossscore_query'] == after['crossscore_query']
            assert before['crossscore_key'] == after['crossscore_key']
    values = {name: j_long(m) for name, m in all_metrics.items()}
    result = dict(schema_version=1, generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  branch='codex/nest-mask-balance-cosine-v1', report_source_head=subprocess.check_output(
                      ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                  reference_commit='40d6f44d68a6df89b3e47d78adbcaeda5a5b049c',
                  reference_hashes=dict(parent=sha(parent_path),TI500=sha(ti_path)),
                  initialization_sha256='54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6',
                  protocol=dict(updates=500,horizon=3651,world_size=4,batch_per_rank=256,global_batch=1024,
                                seed=0,datasets=DATASETS,batch_size=64,native_only=True,coco_similarity_chunk=512,
                                excluded=['DCI','Long-DCI'],J_long='mean Urban/DOCCI I2T/T2I R@1'),
                  groups=groups,metrics=all_metrics,comparisons=comparisons,J_long=values,
                  shared_initializations_vs_parent=initializations,
                  parent_mechanism={name:parent['groups'][name]['key_steps'] for name in ('S-PATCH','C-CLS')},
                  validation=load(evidence/'ddp-reference.json'),
                  execution={p.name:load(p) for p in evidence.glob('*.execution.json')},
                  limitations=['one seed; previously inspected benchmarks; no independent blind development set',
                               'cosine arm changes normalization and readout standard initialization including bias; cannot attribute all change only to normalization'])
    (EXP / 'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Balanced Stack / Cosine CrossScore: 500-Step Results','',
           f"Date: {result['generated_utc']}",'',
           'This fixed experiment repairs separate modality competition and uncontrolled QK scale. '
           'Both new arms retain A3-RandomK, full candidates, losses, optimizer groups, seed and horizon3651. '
           'All rankings compare formal step500 only. TI-fast is reused, not retrained.','',
           '## Main Summary','',
           '| Model | J_long % | Change from original pp | Change from TI pp |',
           '|---|---:|---:|---:|']
    for name in ('TI-fast','S-PATCH','C-CLS',*GROUPS):
        if name in groups and name not in all_metrics:
            lines.append(f'| {name} | not run | not available | not available |')
        else:
            original=(f'{100*(values[name]-values[PARENTS[name]]):+.3f}' if name in PARENTS else 'reference')
            lines.append(f'| {name} | {values[name]*100:.3f} | {original} | {100*(values[name]-values["TI-fast"]):+.3f} |')
    lines += ['', '## Native Retrieval','', 'All recalls are percentages; differences use percentage points. '
              'Scores are normalized bare-student image/text embedding inner products, without mask reranking.','',
              '| Dataset | Direction | Metric | TI-fast | Original Stack-Patch | Balanced | Original CrossScore-CLS | Cosine |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for dataset in DATASETS:
        for direction in DIRECTIONS:
            for recall in RECALLS:
                numbers=[f'{all_metrics[g][dataset][direction][recall]*100:.2f}' if g in all_metrics else 'not run'
                         for g in ('TI-fast','S-PATCH',GROUPS[0],'C-CLS',GROUPS[1])]
                lines.append('| '+' | '.join([dataset,direction,recall,*numbers])+' |')
    lines += ['', '## Mechanism','',
              'Gate diagnostics refer to global valid positive examples. Quantiles and cross-view differences '
              'are sampled at updates 1/100/200/300/400/500; logits/saturation and cosine QK include all valid '
              'candidate pairs. Image replacement uses one fixed rank-order roll of conditions, without changing scoring labels.','']
    for group in GROUPS:
        if group not in all_metrics:
            continue
        first,last=groups[group]['key_steps']['1'],groups[group]['key_steps']['500']
        lines += [f'### {group}','', '| Metric | Step1 | Step500 |', '|---|---:|---:|']
        keys=['F_sigmoid_saturation','O_sigmoid_saturation','E_sigmoid_saturation','F_logit_mean','F_logit_variance',
              'F_keep_ratio','O_keep_ratio','E_keep_ratio','hard_inclusion_violation','oe_iou',
              'F_replaced_image_mask_switch','O_replaced_image_mask_switch','E_replaced_image_mask_switch']
        if group == GROUPS[0]:
            keys += ['F_g_mean','F_g_variance','F_g_q05','F_g_q25','F_g_q50','F_g_q75','F_g_q95',
                     'O_g_mean','E_g_mean','g_F_P_abs_difference','g_F_R_abs_difference','g_P_R_abs_difference']
        else:
            keys += ['F_qk_mean','F_qk_variance','F_qk_abs_max','O_qk_abs_max','E_qk_abs_max']
        for key in keys:
            lines.append(f'| {key} | {first[key]:.6f} | {last[key]:.6f} |')
        delta=100*(values[group]-values[PARENTS[group]])
        if group == GROUPS[0]:
            lines += ['', f'Balanced changes J_long by {delta:+.3f} pp from original Stack-Patch. '
                      'The gate/condition-switch results above determine whether the modal collapse was repaired; '
                      'retrieval differences alone do not identify a single causal mechanism.','']
        else:
            lines += ['', f'Cosine changes J_long by {delta:+.3f} pp from original CrossScore-CLS. '
                      'The bounded QK and saturation measurements test the numerical mechanism directly. '
                      'This arm also uses the requested standard Linear readout initialization, including a '
                      'uniform bias instead of the original zero bias, so improvement is not a pure normalization-only ablation.','']
    lines += ['## Cost and Verification','',
              '| Arm | Full-update mean/median/P95/max s | Peak allocated/reserved GiB | Training-loop minutes |',
              '|---|---:|---:|---:|']
    for group,item in groups.items():
        if group not in all_metrics:
            lines.append(f"| {group} | resource excluded | not trained | not trained |")
            continue
        timing=item['timing']; ranks=item['acceptance']['ranks']
        lines.append(f"| {group} | {timing['mean_seconds']:.3f}/{timing['median_seconds']:.3f}/"
                     f"{timing['p95_seconds']:.3f}/{timing['max_seconds']:.3f} | "
                     f"{max(r['peak_allocated_gib'] for r in ranks):.2f}/{max(r['peak_reserved_gib'] for r in ranks):.2f} | "
                     f"{timing['full_loop_seconds']/60:.2f} |")
    lines += ['', 'The real-DataLoader gate runs 5 warmup + 30 consecutive measured updates, each <=3s. '
              'Each successful arm has separate 5-step four-rank smoke, formal step0/100/200/300/400/500 '
              'checkpoints and uninterrupted logs. Every rank/sample/F/P/R/K stream matches TI-fast@500. '
              'CLIP/B_T/B_V/A_V initialization digests match original counterparts; cosine Q/K match C-CLS. '
              'Final rank parameter difference is zero; losses and gradients are finite. Strict native export '
              'loads at optimizer step500 and has image/text embedding maximum absolute error zero.','',
              '## Limitations and Delivery','',
              'This is a single-seed exploration on previously inspected benchmarks, not evidence of statistical '
              'significance or a final best architecture. Restoring condition sensitivity does not itself prove '
              'semantic grounding. No temperature, learning rate, rank or mechanism sweep was performed. '
              'No mixed Balanced/Cosine architecture or 3651-step continuation was launched.','',
              'Raw JSON, runtime configs, initialization/checkpoint hashes, commands, exit codes, smoke/gate '
              'acceptance and sampled mechanism data are in RESULTS.json and evidence/. Large model weights, '
              'data and full token logs remain on the server. COCO uses 5000/25000 candidates and similarity '
              'chunk512; Urban uses1000 pairs; Flickr test1K uses1000/5000; DOCCI uses5000 pairs. Image batch64. '
              'DCI and Long-DCI are excluded.','']
    (EXP / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(J_long=values,comparisons={k:v['delta_J_long_pp'] for k,v in comparisons.items()},
                         report=str(EXP/'REPORT.md')),indent=2))


if __name__ == '__main__':
    main()
