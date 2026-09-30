"""Freeze all native 500-step results, diagnostics and resource exclusions."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess


EXP=Path(__file__).resolve().parent
REPO=EXP.parents[2]
RUN=Path('/root/lk_projects/SAID-nest-clip-v1/stack_crossscore_v1')
GROUPS=('S-CLS','C-CLS','S-PATCH','C-PATCH')
DATASETS=('COCO','Urban-1k','Flickr30k-test1k','DOCCI')
DIRECTIONS=('I2T','T2I')
RECALLS=('R@1','R@5','R@10')


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def signatures(records):
    keys=('sample_id_sha256','full_view_sha256','local_views_sha256','split_sha256',
          'fixed_first_reference_stream_sha256')
    return [[dict(rank=h['rank'],stream=h['stream_sha256'],
                  sampling={k:h['sampling'][k] for k in keys}) for h in r['rank_health']]
            for r in records]


def metrics(root):
    sources={'COCO':root/'coco_native.json','Urban-1k':root/'urban_native.json',
             'Flickr30k-test1k':root/'flickr_test1k/flickr_test1k.json','DOCCI':root/'docci/docci.json'}
    raw={name:load(path) for name,path in sources.items()}
    result={'COCO':{direction:{f'R@{k}':raw['COCO'][f'{prefix}_R{k}'] for k in (1,5,10)}
                    for direction,prefix in [('I2T','image2text'),('T2I','text2image')]},
            'Urban-1k':{direction:{f'R@{k}':raw['Urban-1k'][field][f'R{k}'] for k in (1,5,10)}
                        for direction,field in [('I2T','image2text'),('T2I','text2image')]}}
    for name in ('Flickr30k-test1k','DOCCI'):
        result[name]={direction:{recall:raw[name]['metrics'][recall][direction] for recall in RECALLS}
                      for direction in DIRECTIONS}
    return result,raw,sources


def j_long(m):
    return statistics.fmean(m[name][direction]['R@1'] for name in ('Urban-1k','DOCCI') for direction in DIRECTIONS)


def difference(left,right):
    return {name:{direction:{recall:100*(left[name][direction][recall]-right[name][direction][recall])
                             for recall in RECALLS} for direction in DIRECTIONS} for name in DATASETS}


def compact(row):
    out={k:v for k,v in row.items() if isinstance(v,(float,int,str))}
    out['common_loss']=row['loss']-row['inc_weight']*row['inc']
    return out


def main():
    evidence=EXP/'evidence'
    resources=load(RUN/'resource-results.json')
    reference_path=REPO/'experiments/nest_clip_v1/jointmask_fast_v1/FORMAL500_RESULTS.json'
    reference=load(reference_path)
    reference_metrics={name:reference['metrics']['TI-fast'][name] for name in DATASETS}
    old_stream=signatures(rows('/root/lk_projects/SAID-nest-clip-v1/jointmask_fast_v1/formal/TI-fast/steps.jsonl'))
    reference_groups={name:{dataset:reference['metrics'][name][dataset] for dataset in DATASETS}
                      for name in ('TI-fast','T-fast')}
    vcp_path=REPO/'experiments/nest_clip_v1/vcp_mask_v1/FORMAL500_RESULTS.json'
    reference_groups['VCP-Mask']=load(vcp_path)['metrics']['VCP-Mask']
    groups={}
    all_metrics=dict(reference_groups)
    config_paths={'S-CLS':'nest_stack_cls.json','C-CLS':'nest_crossscore_cls.json',
                  'S-PATCH':'nest_stack_patch.json','C-PATCH':'nest_crossscore_patch.json'}
    for group in GROUPS:
        cfg=load(REPO/'configs'/config_paths[group])
        probe=RUN/'probe'/group
        item=dict(resource=resources[group],config=cfg,completed_updates=0,metrics=None)
        if (probe/'config.json').exists():
            item['probe_config']=load(probe/'config.json')
            item['parameter_counts']=item['probe_config']['parameter_counts']
            item['component_initialization']=item['probe_config']['component_initialization']
        if (probe/'probe_timing.jsonl').exists():
            item['probe_cycle_timing']=rows(probe/'probe_timing.jsonl')
        if (probe/'steps.jsonl').exists():
            item['initial_diagnostics']=compact(rows(probe/'steps.jsonl')[0])
        if resources[group]['resource_status'] != 'passed':
            item['formal_status']='not_run_resource_infeasible'
            groups[group]=item
            continue
        root=RUN/'formal'/group
        accepted=load(root/'acceptance.json')
        export=load(root/'export-check.json')
        assert accepted['passed'] and all(r['completed_updates']==500 for r in accepted['ranks'])
        assert export['passed'] and export['optimizer_steps']==[500]
        records=rows(root/'steps.jsonl')
        assert [r['step'] for r in records]==list(range(1,501))
        assert all(r['nonfinite']==0 and all(h['gradients_finite'] for h in r['rank_health']) for r in records)
        streams_equal=signatures(records)==old_stream
        assert streams_equal,f'Data drift {group}'
        m,raw,sources=metrics(root)
        assert all(r['checkpoint_sha256']==export['bare_sha256'] for r in raw.values())
        all_metrics[group]=m
        local_dest=evidence/'native_results'/group
        local_dest.mkdir(parents=True,exist_ok=True)
        for name,source in sources.items():
            shutil.copy2(source,local_dest/f'{name.lower().replace("-","_")}.json')
        for filename in ('export-check.json','config.json','acceptance.json','cycle_timing.jsonl','checkpoint_timing.jsonl'):
            shutil.copy2(root/filename,local_dest/filename)
        window=[compact(r) for r in records[-50:]]
        last50={k:statistics.fmean(r[k] for r in window if k in r) for k,v in window[-1].items()
                if isinstance(v,(int,float))}
        cycles=rows(root/'cycle_timing.jsonl')
        regular=[r['four_rank_max_seconds'] for r in cycles if 6 <= r['step'] <= 500]
        item.update(formal_status='completed',completed_updates=500,metrics=m,J_long=j_long(m),
                    delta_J_long_TI_pp=100*(j_long(m)-j_long(reference_metrics)),
                    config=load(root/'config.json'),acceptance=accepted,export_check=export,
                    checkpoint_sha256={f'step{s:06d}.pt':sha(root/f'step{s:06d}.pt')
                                       for s in (0,100,200,300,400,500)},
                    streams_equal_TI500=streams_equal,steps_continuous=True,finite=True,
                    F_candidates_range=[min(r['F_candidates'] for r in records),max(r['F_candidates'] for r in records)],
                    valid_candidates_range=[min(r['valid_global'] for r in records),max(r['valid_global'] for r in records)],
                    fallback_steps=sum(r['valid_global']<2 for r in records),
                    duplicate_id_steps=sum(bool(r['duplicate_image_ids']) for r in records),
                    timing=dict(scope='real DataLoader full update, slowest rank, normal logs included; checkpoint writes separate',
                                stable_steps='6-500',mean_seconds=statistics.fmean(regular),
                                median_seconds=statistics.median(regular),p95_seconds=statistics.quantiles(regular,n=100)[94],
                                max_seconds=max(regular),
                                all_regular_slow_steps=[r for r in cycles if r['step']>1 and r['four_rank_max_seconds']>3],
                                full_loop_seconds=max(r['seconds'] for r in accepted['ranks'])),
                    last50_mean=last50,key_steps={str(r['step']):compact(r) for r in records
                                               if r['step'] in (1,100,200,201,300,400,500)})
        groups[group]=item
    components=[v['component_initialization'] for v in groups.values() if 'component_initialization' in v]
    shared={key:len({v[key] for v in components})==1 for key in ('clip','text_blocks','visual_blocks','visual_adapter')}
    assert all(shared.values()),shared
    cross=[groups[g]['component_initialization'] for g in ('C-CLS','C-PATCH') if 'component_initialization' in groups[g]]
    if len(cross)==2:
        assert cross[0]['crossscore_query']==cross[1]['crossscore_query']
        assert cross[0]['crossscore_key']==cross[1]['crossscore_key']
    comparisons={}
    for left,right in [('C-CLS','S-CLS'),('C-PATCH','S-PATCH'),('S-PATCH','S-CLS'),('C-PATCH','C-CLS')]:
        comparisons[f'{left}_minus_{right}']=(dict(delta_pp=difference(all_metrics[left],all_metrics[right]),
                                                  delta_J_long_pp=100*(j_long(all_metrics[left])-j_long(all_metrics[right])))
                                             if left in all_metrics and right in all_metrics else
                                             dict(status='not_available_resource_exclusion'))
    for group in GROUPS:
        if group in all_metrics:
            comparisons[f'{group}_minus_TI-fast']=dict(delta_pp=difference(all_metrics[group],reference_metrics),
                                                       delta_J_long_pp=100*(j_long(all_metrics[group])-j_long(reference_metrics)))
    eligible=[g for g in GROUPS if g in all_metrics]
    better=[g for g in eligible if groups[g]['delta_J_long_TI_pp']>0]
    quality=max(better,key=lambda g:groups[g]['J_long']) if better else None
    cost=min(better,key=lambda g:groups[g]['timing']['mean_seconds']) if better else None
    recommended=list(dict.fromkeys(g for g in (quality,cost) if g is not None))[:2]
    recommendation=dict(descriptive_followups=recommended,retain_TI_reference=True,
                        criterion='at most two descriptive positive-J_long candidates, one by quality and one by measured cost; small differences remain uncertain')
    result=dict(schema_version=1,generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                branch='codex/nest-stack-crossscore-v1',report_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                reference_commit='6bafa8a009af4ffee1100027d5171d78dbeb96d3',reference_sha256=sha(reference_path),
                initialization_sha256='54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6',
                protocol=dict(updates=500,horizon=3651,world_size=4,batch_per_rank=256,global_batch=1024,
                              datasets=DATASETS,native_only=True,image_batch=64,coco_similarity_chunk=512,
                              excluded=['DCI','Long-DCI'],J_long='mean Urban/DOCCI I2T/T2I R@1'),
                groups=groups,metrics=all_metrics,comparisons=comparisons,shared_initializations=shared,
                J_long={g:j_long(m) for g,m in all_metrics.items()},recommendation=recommendation,
                validation=load(evidence/'ddp-reference.json'),
                limitations=['one seed, previously examined benchmarks, no independent development set',
                             'patch changes token count/cost and CrossScore head capacity',
                             'new branches add capacity relative to TI; this is a method comparison'],
                execution={p.name:load(p) for p in evidence.glob('*.execution.json')})
    (EXP/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Dual-Branch Mask Fusion: 500-Update Exploration','',
           f"Generated UTC: {result['generated_utc']}",'',
           '## Outcome','',
           ('Retain TI-fast as the current reference: no new feasible candidate improves the native long-text summary.'
            if not better else 'Candidates with a descriptive long-text gain for controlled follow-up: '+', '.join(recommended)+'.'),'',
           'All rankings use step500, seed0. Resource-excluded candidates have no inferred retrieval scores. '
           'One seed and previously examined benchmarks do not establish significance or a final best model.','',
           '| Candidate | Visual slots | Fusion | Total / added parameters | Formal updates | Resource | Full step mean/P95/max s | Peak allocated GiB | Urban I2T/T2I R1 | DOCCI I2T/T2I R1 | J_long | vs TI pp |',
           '|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for group,item in groups.items():
        params=item.get('parameter_counts',{})
        count=f"{params.get('total','unknown')}/{params.get('added','unknown')}"
        if group in all_metrics:
            m=item['metrics'];t=item['timing'];a=item['acceptance']['ranks']
            details=[f"{t['mean_seconds']:.3f}/{t['p95_seconds']:.3f}/{t['max_seconds']:.3f}",
                     f"{max(x['peak_allocated_gib'] for x in a):.2f}",
                     f"{m['Urban-1k']['I2T']['R@1']*100:.2f}/{m['Urban-1k']['T2I']['R@1']*100:.2f}",
                     f"{m['DOCCI']['I2T']['R@1']*100:.2f}/{m['DOCCI']['T2I']['R@1']*100:.2f}",
                     f"{item['J_long']*100:.3f}",f"{item['delta_J_long_TI_pp']:+.3f}"]
        else:
            details=['not run']*6
        lines.append('| '+' | '.join([group,str(1 if item['config']['visual']=='cls' else 196),
                                      item['config']['fusion'],count,str(item['completed_updates']),
                                      item['resource']['resource_status'],*details])+' |')
    lines += ['',f"TI-fast@500 J_long: {j_long(reference_metrics)*100:.3f}%.",'',
              '## Native Retrieval','', 'Recall is shown as percent; differences are percentage points.', '',
              '| Dataset | Direction | Metric | TI-fast | T-fast | VCP background | S-CLS | C-CLS | S-PATCH | C-PATCH |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in DATASETS:
        for direction in DIRECTIONS:
            for recall in RECALLS:
                values=[f'{all_metrics[g][name][direction][recall]*100:.2f}' if g in all_metrics else 'not run'
                        for g in ('TI-fast','T-fast','VCP-Mask',*GROUPS)]
                lines.append('| '+' | '.join([name,direction,recall,*values])+' |')
    lines += ['', '## Factor Comparisons', '',
              '| Difference | J_long delta pp |', '|---|---:|']
    for name,value in comparisons.items():
        text=f"{value['delta_J_long_pp']:+.3f}" if 'delta_J_long_pp' in value else 'not available'
        lines.append(f'| {name} | {text} |')
    lines += ['', '## Resource Gates and Exclusions', '',
              'Each successful gate has 5 warmup and 30 consecutive real-DataLoader full updates. '
              'Reported time is the maximum across all four ranks. A failed measured update stops that gate; '
              'no slow entries are dropped and no formal training starts for failures.', '']
    for group,item in groups.items():
        r=item['resource']['resource_result'];gate=r.get('speed_gate')
        lines.append(f"- {group}: {item['resource']['resource_status']}; gate `{gate}`; reason `{r.get('resource_failure')}`.")
    lines += ['', '## Mechanism and Integrity', '',
              '| Group | Common loss | F/P/R positive keep | F/P/R negative keep | Hard violation | P/R IoU |',
              '|---|---:|---:|---:|---:|---:|']
    for group in eligible:
        m=groups[group]['last50_mean']
        lines.append(f"| {group} | {m['common_loss']:.4f} | "+
                     '/'.join(f"{m[v+'_positive_keep_ratio']:.4f}" for v in ('F','O','E'))+' | '+
                     '/'.join(f"{m[v+'_negative_keep_ratio']:.4f}" for v in ('F','O','E'))+
                     f" | {m['hard_inclusion_violation']:.4f} | {m['oe_iou']:.4f} |")
    lines += ['', 'Initial and steps100/200/300/400/500 logits/saturation, all-open/all-closed ratios, '
              'Stack modality pooling mass, CrossScore raw QK moments and replaced-image mask switches '
              'are preserved in RESULTS.json. These are mechanism diagnostics, not evidence of semantic grounding.','',
              'For completed groups, all 500 updates and every rank/sample/F/P/R/K stream digest match the '
              'TI-fast@500 reference. Shared CLIP, B_T, B_V and A_V initial state digests are identical. '
              'CrossScore Q/K initial tensors also match between CLS and patch. Optimizer assignment is unique. '
              'Strict exports load at optimizer step500 with native image/text embedding error 0.', '',
              'The new blocks are training-only. Stack uses a scalar mix of modality summaries, not additional '
              'token cross-attention. CrossScore uses no softmax, value aggregation or old logits residual. '
              'Patch readout remains dense at 24,887,296 weights plus 512 bias parameters.','',
              '## Validation Caveats','',
              '45 unit/regression cases and 20 two-rank cases passed. Named gradient checks include None states. '
              'The existing VCP gradient tolerance is retained (absolute 1.5e-3, relative 1e-4). AdamW differences '
              'above 2e-6 are accepted only for gradients below 2e-5 where the exact first-update AdamW formula '
              'explains the amplification within FP32 parameter rounding. No gradients or parameters are removed. '
              'Theoretical K bias and single-CLS visual Q/K null directions are explicitly distinguished.','',
              'Early 8-channel synthetic patch fixtures could yield all-closed masks and very large 1/eps gradients. '
              'Those failure logs are retained. A separately labeled nonzero-bias state passed, and final reference '
              'tests use 32 channels with the original zero-bias initialization. All four production groups still '
              'use 512 channels and the frozen initialization rules.','',
              '## Artifacts','',
              'Source, configs, reproduction commands, actual training commits, initialization and checkpoint hashes, '
              'resource/smoke/strict-export evidence, raw native JSON and execution exit codes are preserved here. '
              'Large checkpoints, full token logs, images and caches stay server-local. DCI and Long-DCI are excluded. '
              'This pipeline stops after the authorized 500-update exploration; it launches no full epochs, extra '
              'seeds, shuffle controls or combined candidates.','']
    (EXP/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(results=str(EXP/'RESULTS.json'),recommendation=recommendation,
                         J_long=result['J_long']),indent=2))


if __name__=='__main__':
    main()
