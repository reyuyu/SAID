"""Reproducible post-training audits and one final native evaluation, no training."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

from recovery.hns_v1_pure3 import (ROOT, ORIGINAL, RUN, EXP, TRAIN, STEP0, STEP0_SHA,
    INDEX, IMAGES, CONFIG, HORIZON, GROUPS, REFERENCE500, REFERENCEFULL,
    read, sha, dump, now, analytic_lrs, normalize_json, validate_config, source_proof)

BARE = TRAIN / 'student_step3651.pt'
FINAL = TRAIN / 'step003651.pt'
E2_ROOT = Path('/opt/data/private/lklk/said-e2-worktrees/hns-s12-uniform-full4868-v1/experiments/nest_clip_v1/hns_s12_uniform_full4868_v1')
E2_FULL = Path('/opt/data/private/lklk/said-e2-worktrees/hyfl-data-completion-v1/experiments/nest_clip_v1/said_e2_hyfl_data_completion_v1/FLICKR_FULL_RESULTS.json')
FULL_JOB = Path('/root/said_hyfl_data_completion_v1/full_protocol/JOB.json')
DATASETS = ('COCO', 'Flickr30k-test1k', 'DOCCI', 'Long-DCI', 'Urban-1k')
COUNTS = {'COCO':(5000,25000), 'Flickr30k-test1k':(1000,5000),
          'DOCCI':(5000,5000), 'Long-DCI':(7602,7602), 'Urban-1k':(1000,1000)}


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip(): yield json.loads(line)


def digest(value):
    return hashlib.sha256(json.dumps(normalize_json(value), ensure_ascii=False,
                                    sort_keys=True).encode()).hexdigest()


def aggregate(metrics):
    from experiments.nest_clip_v1.balanced_hparam_search_v1.search import scores
    result = {k.replace('_R1', ''):v*100 for k,v in scores(metrics).items()}
    result['Short4'] = statistics.fmean(metrics[d][r]['R@1']*100
        for d in ('COCO','Flickr30k-test1k') for r in ('I2T','T2I'))
    result['Urban I2T'] = metrics['Urban-1k']['I2T']['R@1']*100
    result['Urban T2I'] = metrics['Urban-1k']['T2I']['R@1']*100
    result['Urban Mean'] = (result['Urban I2T']+result['Urban T2I'])/2
    return result


def training_audit():
    import torch
    from train.train_nested_semantic_mask import state_digest
    torch.set_num_threads(4)
    assert sha(STEP0) == STEP0_SHA
    production = source_proof()
    steps = list(rows(TRAIN / 'steps.jsonl'))
    assert [r['step'] for r in steps] == list(range(1,HORIZON+1))
    phase = Path(read(EXP / 'DDP_ACCEPTANCE.json')['phase'])
    refs = {rank:list(rows(RUN / f'reference-rank{rank}.jsonl')) for rank in range(4)}
    recorded = {rank:list(rows(phase / f'stream-rank{rank}.jsonl')) for rank in range(4)}
    for rank in range(4):
        assert [r['step'] for r in recorded[rank]] == list(range(1,HORIZON+1))
    positions = 0; lr_records=[]; health_norms={g:[] for g in GROUPS}
    for row in steps:
        s=row['step']; assert row['s']==s-1 and row['epoch']==(s-1)//1217
        expected=dict(zip(GROUPS, analytic_lrs(s-1)))
        assert row['actual_lrs']==expected and row['nonfinite']==0
        assert math.isfinite(row['loss'])
        assert sorted(h['rank'] for h in row['rank_health'])==[0,1,2,3]
        for h in row['rank_health']:
            rank=h['rank']; ref=refs[rank][s-1]; actual=recorded[rank][s-1]
            assert h['batch']==ref['batch']==actual['batch']==(180 if s%1217==0 else 256)
            assert h['updates']==s and h['gradients_finite']
            assert h['stream_sha256']==ref['stream_sha256']==actual['stream_sha256']
            assert normalize_json(h['sampling'])==normalize_json(ref['sampling'])
            assert actual['full_sampling_digest']==digest(ref['sampling'])
            assert actual['epoch']==row['epoch'] and actual['batch_cursor']==(s-1)%1217
            positions+=h['batch']
            for g,v in h['gradient_norms'].items():
                assert math.isfinite(v); health_norms[g].append(v)
        lr_records.append(dict(step=s, s=s-1, actual_lrs=expected))
    assert positions==3737712
    dump(EXP / 'DATA_STREAM_AUDIT.json', dict(passed=True, updates=HORIZON,
        unique_updates=HORIZON, missing_updates=[], duplicate_updates=[], ranks=4,
        compared_actual_historical_logs=True, reference=read(EXP/'DATA_PREFLIGHT.json')['actual_history_reference'],
        all_sample_ID_F_Dall_D3_text_tokens_K_indices_match=True,
        sampling_dictionary_comparison='Strict complete JSON normalization; collisions/nonfinite rejected',
        batch_positions=positions, positions_per_epoch=1245904, records_per_epoch=1245901,
        DistributedSampler_padding_per_epoch=3, tail_batch_per_rank=180, epochs=[0,1,2],
        native_log_sha256=sha(TRAIN/'steps.jsonl'),
        full_stage_stream_sha256={str(r):digest([x['stream_sha256'] for x in refs[r]]) for r in range(4)},
        actual_stream_receipt_sha256={str(r):sha(phase/f'stream-rank{r}.jsonl') for r in range(4)},
        no_fourth_epoch_training=True, model_gradient_equality_to_old_trajectory_not_required=True,
        augmentation='Unchanged byte-identical dataset preprocessing and historical worker seed construction; augmented tensors are not present in historical logs'))
    dump(EXP / 'SCHEDULER_AUDIT.json', dict(passed=True, horizon=HORIZON,
        actual_update_indices=[0,HORIZON-1], warmup_updates=200,
        backbone_peak=1e-6, text_mask_peak=1e-3, visual_mask_peak=1e-3, fusion_peak=2e-4,
        checkpoints={str(s):lr_records[s-1] for s in (1,100,200,500,1217,2434,3651)},
        all_updates=lr_records, terminal_after_training=analytic_lrs(HORIZON),
        final_update_is_small_positive='Native update3651 uses s3650; s3651 endpoint is zero without an extra update',
        not_a_truncated4868_schedule=True))
    checkpoint_proofs=[]
    for step in (0,500,1217,2434,3651):
        path=TRAIN/f'step{step:06d}.pt'; before=sha(path)
        p=torch.load(path,map_location='cpu',weights_only=False)
        validate_config(p['config'])
        assert p['completed_steps']==p['global_step']==step
        assert p['scheduler_horizon']==p['scheduler']['horizon']==HORIZON
        assert p['data_cursor']==dict(next_epoch=step//1217,next_batch=step%1217)
        assert p['config']['code_sha256']==__import__('train.train_nested_semantic_mask',fromlist=['code_manifest']).code_manifest()
        assert p['config']['init_sha256']==STEP0_SHA and p['trajectory_root']==str(RUN)
        assert p['sampler']['world_size']==4 and p['sampler']['seed']==0
        assert len(p['rng_per_rank'])==4
        for r in p['rng_per_rank']:
            assert set(('python','numpy','cpu','cuda','loader_generator')) <= set(r)
            assert r['cpu'].numel()>0 and r['cuda'].numel()>0 and r['loader_generator'].numel()>0
        counters=sorted({int(v['step']) for v in p['optimizer']['state'].values()})
        assert counters==([step] if step else [])
        finite=True
        for state in p['optimizer']['state'].values():
            for k in ('exp_avg','exp_avg_sq'):
                assert torch.isfinite(state[k]).all() and state[k].dtype==torch.float32
        for sd in (p['model'],p['adapter']):
            assert all(torch.isfinite(v).all() for v in sd.values())
        checkpoint_proofs.append(dict(passed=True,step=step,path=str(path),sha256=before,
            bytes=path.stat().st_size,model_sha256=state_digest(p['model']),adapter_sha256=state_digest(p['adapter']),
            optimizer_steps=counters,optimizer_state_count=len(p['optimizer']['state']),
            scheduler=p['scheduler'],sampler=p['sampler'],cursor=p['data_cursor'],rng_ranks=4,
            complete_restore_fields_present=True,parameters_moments_finite_FP32=finite,
            control_source_sha256=p['control_source_sha256']))
        del p; assert sha(path)==before
    dump(EXP/'CHECKPOINT_AUDIT.json',dict(passed=True,checkpoints=checkpoint_proofs,
        all_new_trajectory_only=True,original_common0_sha256=sha(STEP0),original_common0_immutable=True,
        actual_full_restore_check='Final tools.nest_clip verify-export loads the full native model+adapter+AdamW; no optimizer update'))
    smoke=read(RUN/'smoke-STEP0_IDENTITY.json'); formal=read(RUN/'formal-STEP0_IDENTITY.json')
    assert smoke['passed'] and formal['passed']
    for k in ('model_sha256','adapter_sha256','adapter_components_exact','optimizer_groups'):
        assert smoke[k]==formal[k]
    dump(EXP/'STEP0_IDENTITY.json',dict(passed=True,common=read(EXP/'COMMON0_IDENTITY.json'),
        smoke=smoke,formal=formal,independent_initializations_exact=True))
    timing=list(rows(TRAIN/'cycle_timing.jsonl'))
    assert len(timing)==HORIZON
    times=[r['four_rank_max_seconds'] for r in timing]
    native_accept=read(TRAIN/'acceptance.json')
    dump(EXP/'RUNTIME_STATS.json',dict(passed=True,acceptance=native_accept,
        update_cycle_mean_seconds=statistics.fmean(times),update_cycle_median_seconds=statistics.median(times),
        update_cycle_p95_seconds=sorted(times)[math.ceil(.95*len(times))-1],update_cycle_max_seconds=max(times),
        warnings_over3s=sum(t>3 for t in times),original_hard_stop60s_preserved=True,
        checkpoint_times=list(rows(TRAIN/'checkpoint_timing.jsonl')),
        group_gradient_norms={g:dict(min=min(v),mean=statistics.fmean(v),max=max(v)) for g,v in health_norms.items()},
        phase_local=str(phase),resource_telemetry=str(RUN/'full-resource-telemetry.jsonl'),
        resource_telemetry_sha256=sha(RUN/'full-resource-telemetry.jsonl'),
        supervisor_stage_env='Legacy SAID_S02_STAGE=step4868 is a monitor label only; saved configuration and horizon are3651'))
    old=[r for r in rows(REFERENCEFULL/'steps.jsonl') if 3602<=r['step']<=3651]
    assert len(old)==50
    keys=['loss','HNS_align','HNS_original_sparse','HNS_surcharge','lambda_h',
          'HNS_F_keep','HNS_Dall_keep','HNS_D3_keep','HNS_DF_hard_violation_ratio',
          'HNS_3D_hard_violation_ratio','HNS_DF_IoU','HNS_3D_IoU',
          'F_i2t','F_t2i','O_i2t','O_t2i','E_i2t','E_t2i','F_sparse','O_sparse','E_sparse']
    def summary(data):
        return {k:dict(mean=statistics.fmean(r[k] for r in data),first=data[0][k],last=data[-1][k]) for k in keys}
    recent=steps[-50:]
    dump(EXP/'TRAINING_DIAGNOSTICS.json',dict(passed=True,
        last50_Pure3=summary(recent),last50_historical_HNS3651=summary(old),
        mean_difference={k:statistics.fmean(r[k] for r in recent)-statistics.fmean(r[k] for r in old) for k in keys},
        final50_curves=[{k:r[k] for k in ['step',*keys]} for r in recent],
        final50_group_gradient_norms={g:statistics.fmean(h['gradient_norms'][g] for r in recent for h in r['rank_health']) for g in GROUPS},
        density_order_reversals=sum(not(r['HNS_F_keep']>=r['HNS_Dall_keep']>=r['HNS_D3_keep']) for r in steps),
        density_order_is_descriptive_not_stop_condition=True,
        per_loss_gradient_cosines='UNMEASURED; no mechanism or retrieval benefit inferred from unmeasured cosines'))


def evaluate(supervisor):
    from tools.eval_five_parallel import require_gpu_idle
    from tools import eval_hns_pure3_flickrfull as full
    from tools.eval_resource_queue import Queue
    from tools.audit_hyfl_data import image_inventory
    from tools.flickr_full_protocol import read_official
    from tools.retrieval_bounded import digest as frozen_digest
    python=str(ROOT/'.venv/bin/python')
    before=sha(FINAL); supervisor.train=TRAIN
    supervisor.execute('export3651',[python,'-m','tools.nest_clip','export','--checkpoint',str(FINAL),
        '--output',str(BARE),'--expect-updates','3651'])
    supervisor.execute('verify3651',[python,'-m','tools.nest_clip','verify-export','--checkpoint',str(FINAL),
        '--bare',str(BARE),'--output',str(TRAIN/'export-check.json'),'--index-dir',str(INDEX),'--image-root',str(IMAGES)])
    export=read(TRAIN/'export-check.json')
    assert export['passed'] and export['strict_load'] and export['checkpoint_sha256']==before
    assert export['image_max_abs']==export['text_max_abs']==0
    assert export['optimizer_steps']==[HORIZON]
    dump(EXP/'EXPORT_AUDIT.json',export)
    supervisor.execute('eval3651-five-parallel',[python,'-m','tools.eval_five_parallel',
        '--checkpoint',str(BARE),'--training-checkpoint',str(FINAL),'--output-dir',str(TRAIN),
        '--assets-root',str(ROOT/'local_assets')])
    from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics
    metrics,raw,sources=native_metrics(TRAIN)
    assert sha(FINAL)==before and sha(BARE)==export['bare_sha256']
    evidence=EXP/'native'; evidence.mkdir(exist_ok=False)
    for name,value in raw.items(): dump(evidence/(name+'.json'),value)
    dump(EXP/'RESULTS.json',dict(model='HNS-v1 Pure3',completed_steps=HORIZON,scheduler_horizon=HORIZON,
        metrics=metrics,scores_percent=aggregate(metrics),checkpoint=str(FINAL),checkpoint_sha256=before,
        bare=str(BARE),bare_sha256=export['bare_sha256'],strict_export=export,
        historical_five_protocol_unchanged=True,evaluation_checkpoint_immutable=True,
        native_result_sources={n:dict(path=str(p),sha256=sha(p)) for n,p in sources.items()},
        parallel_evaluation=read(TRAIN/'EVAL_PARALLEL_RUN.json'),FlickrFull_excluded_from_Score5=True))
    require_gpu_idle((0,1,2,3))
    job=read(FULL_JOB); full.validate_job(job,BARE); full.bind(BARE)
    assert sha(job['manifest'])==job['manifest_sha256']
    manifest,images=full.native.rows_and_images(job['manifest'])
    token=ORIGINAL/'runtime/SAID-nest-clip-v1/said-e2-hyfl-eval-unified-v1/references/flickr30k_annotations.tar.gz'
    annotation,official=read_official(token)
    assert hashlib.sha256(annotation).hexdigest()==job['annotation_sha256'] and manifest==official
    inv=image_inventory(images,job['image_root'])
    assert frozen_digest(inv)==job['image_content_sha256']
    assert inv==read('/root/said_hyfl_data_completion_v1/full_protocol/IMAGE_INVENTORY.json')
    dump(EXP/'FLICKR_FULL_DATA_AUDIT.json',dict(passed=True,job=job,
        all_official_captions_exact=True,all31783_image_bytes_and_PIL_verified=True,
        frozen_sources=full.FROZEN,own_entry_sha256=sha(ROOT/full.ENTRY)))
    output=RUN/'flickr_full/Flickr30k-Full/attempt1/RESULT.json'
    job['command']=[python,'-m','tools.eval_hns_pure3_flickrfull','--job',str(RUN/'flickr-full-JOB.json'),
        '--checkpoint',str(BARE),'--output',str(output),'--cache-dir',str(RUN/'flickr_full/cache/Flickr30k-Full'),'--device','cuda:0']
    dump(RUN/'flickr-full-JOB.json',job)
    plan=dict(checkpoint=str(BARE),jobs=[job],blocked={})
    assert Queue(plan,RUN/'flickr_full',gpus=(0,)).run()==0
    result=read(output)
    assert result['status']=='COMPLETED' and result['returncode']==0
    assert result['checkpoint_sha256']==export['bare_sha256']
    assert result['state_sha256_before']==result['state_sha256_after']
    assert result['n_images']==31783 and result['n_captions']==158915
    e2=read(E2_FULL)
    for k,v in e2['identity'].items():
        if k not in ('checkpoint_sha256','encoder_sources'): assert result['identity'][k]==v,('Full protocol mismatch',k)
    for k,v in e2['identity']['encoder_sources'].items(): assert result['identity']['encoder_sources'][k]==v
    import torch
    from tools.retrieval_bounded import ShardCache
    cache=ShardCache(RUN/'flickr_full/cache/Flickr30k-Full',result['identity'])
    features={}
    for kind,records in [('image',images),('text',manifest)]:
        count=0; error=0.; shards=0
        for start in range(0,len(records),64):
            subset=records[start:start+64]
            ids=[r[0] for r in subset] if kind=='image' else [r['caption_id'] for r in subset]
            tensor=cache.read(kind,start,ids)
            assert tensor is not None and tensor.dtype==torch.float32 and list(tensor.shape)==[len(ids),512]
            assert torch.isfinite(tensor).all()
            error=max(error,float((tensor.norm(dim=1)-1).abs().max())); count+=len(tensor); shards+=1
        assert count==len(records) and error<1e-6
        features[kind]=dict(count=count,shards=shards,tail=len(records)%64,max_normalization_error=error)
    detail=torch.load(output.with_suffix('.queries.pt'),map_location='cpu',weights_only=True)
    for direction,n in [('I2T',31783),('T2I',158915)]:
        assert list(detail[direction]['top_indices'].shape)==[n,11]
        for k in ('1','5','10'):
            m=result['metrics'][direction]
            assert m['query_count']==n and sum(detail[direction]['hits'][k])==m['correct'][k]
            assert m['recall_percent'][k]==100*m['correct'][k]/n
    dump(EXP/'FLICKR_FULL_RESULTS.json',result)
    dump(EXP/'FLICKR_FULL_FEATURE_AUDIT.json',dict(passed=True,features=features,
        query_details=str(output.with_suffix('.queries.pt')),sha256=sha(output.with_suffix('.queries.pt')),
        counts_recomputed_from_all_query_receipts=True,new_model_own_cache=True,
        evaluation_identity_matches_E2_except_checkpoint_and_wrapper=True))
    assert sha(FINAL)==before and sha(BARE)==export['bare_sha256'] and sha(STEP0)==STEP0_SHA
    require_gpu_idle((0,1,2,3))


def summarize():
    result=read(EXP/'RESULTS.json'); metrics=result['metrics']; measured=result['scores_percent']
    historical_path=ORIGINAL/'experiments/nest_clip_v1/epoch_curve_eval_v1/RESULTS.json'
    history=read(historical_path)['models']['HNS_v1']['3651']
    baselines={'HNS-v1 old@3651':history,'E2@3651':read(E2_ROOT/'step3651/RESULTS.json'),
               'E2@4868':read(E2_ROOT/'step4868/RESULTS.json')}
    comparisons={}
    for name,base in baselines.items():
        totals=aggregate(base['metrics'])
        recall_delta={d:{r:{k:100*(metrics[d][r][k]-base['metrics'][d][r][k])
            for k in ('R@1','R@5','R@10')} for r in ('I2T','T2I')} for d in DATASETS}
        query_delta={d:{r:{k:round(metrics[d][r][k]*COUNTS[d][i])-round(base['metrics'][d][r][k]*COUNTS[d][i])
            for k in ('R@1','R@5','R@10')} for i,r in enumerate(('I2T','T2I'))} for d in DATASETS}
        comparisons[name]=dict(scores_percent=totals,
            aggregate_delta_pp={k:measured[k]-totals[k] for k in measured},
            all30_recall_delta_pp=recall_delta,correct_query_count_delta=query_delta)
    full=read(EXP/'FLICKR_FULL_RESULTS.json'); e2_full=read(E2_FULL)
    fd={r:{k:full['metrics'][r]['recall_percent'][k]-e2_full['metrics'][r]['recall_percent'][k]
           for k in ('1','5','10')} for r in ('I2T','T2I')}
    fc={r:{k:full['metrics'][r]['correct'][k]-e2_full['metrics'][r]['correct'][k]
           for k in ('1','5','10')} for r in ('I2T','T2I')}
    dump(EXP/'COMPARISON.json',dict(new_model=result,baselines=comparisons,
        FlickrFull_E2_delta_pp=fd,FlickrFull_correct_query_delta=fc,
        baseline_sources={str(p):sha(p) for p in (historical_path,E2_ROOT/'step3651/RESULTS.json',E2_ROOT/'step4868/RESULTS.json',E2_FULL)},
        query_count_interpretation='Net correct-query count changes, not paired-query corrections; no significance inferred'))
    lines=['# HNS-v1 Pure3 comparison','',
        '| Model | Horizon | Updates | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I | Urban Mean |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    keys=('Score5','J_long3','J_long','Short4','Urban I2T','Urban T2I','Urban Mean')
    for name,scores,horizon,updates in [('HNS-v1 Pure3',measured,3651,3651),
        *[(n,c['scores_percent'],4868,4868 if n=='E2@4868' else 3651) for n,c in comparisons.items()]]:
        lines.append('| '+name+f' | {horizon} | {updates} | '+' | '.join(f'{scores[k]:.6f}' for k in keys)+' |')
    for name,c in comparisons.items():
        lines+=['',f'## Pure3 minus {name} (percentage points)','',
            '| Metric | Delta pp |','|---|---:|',*['| '+k+f" | {c['aggregate_delta_pp'][k]:+.6f} |" for k in keys],
            '', '| Dataset / direction | Pure3 R@1 | R@5 | R@10 | ΔR@1 | ΔR@5 | ΔR@10 | Correct-query Δ @1/5/10 |',
            '|---|---:|---:|---:|---:|---:|---:|---|']
        for d in DATASETS:
            for r in ('I2T','T2I'):
                raw=metrics[d][r]; delta=c['all30_recall_delta_pp'][d][r]
                q=c['correct_query_count_delta'][d][r]
                lines.append(f'| {d} {r} | '+' | '.join(f'{raw[k]*100:.6f}' for k in ('R@1','R@5','R@10'))+' | '+
                    ' | '.join(f'{delta[k]:+.6f}' for k in ('R@1','R@5','R@10'))+' | '+
                    '/'.join(f'{q[k]:+d}' for k in ('R@1','R@5','R@10'))+' |')
    lines+=['','## Flickr30k-Full (excluded from Score5)','',
        '| Direction | Pure3 R@1/5/10 (%) | E2 R@1/5/10 (%) | Delta pp @1/5/10 | Correct-query Δ @1/5/10 |',
        '|---|---|---|---|---|']
    for r in ('I2T','T2I'):
        lines.append('| '+r+' | '+' / '.join(f"{full['metrics'][r]['recall_percent'][k]:.6f}" for k in ('1','5','10'))+
            ' | '+' / '.join(f"{e2_full['metrics'][r]['recall_percent'][k]:.6f}" for k in ('1','5','10'))+
            ' | '+' / '.join(f'{fd[r][k]:+.6f}' for k in ('1','5','10'))+
            ' | '+' / '.join(f'{fc[r][k]:+d}' for k in ('1','5','10'))+' |')
    (EXP/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    runtime=read(EXP/'RUNTIME_STATS.json'); diag=read(EXP/'TRAINING_DIAGNOSTICS.json')
    proof=read(EXP/'CHECKPOINT_AUDIT.json'); final=proof['checkpoints'][-1]
    olddelta=comparisons['HNS-v1 old@3651']['aggregate_delta_pp']
    report=['# HNS-v1 Pure 3-Epoch final report','',
        'Completed one new common0 trajectory, 3 epochs, exactly3651 updates. Only training length and cosine horizon changed. No intermediate public-test selection, fourth epoch, other arm, or DCI evaluation.',
        '',f"Pure3 Score5={measured['Score5']:.6f}; Δ against old HNS-v1@3651={olddelta['Score5']:+.6f}pp. The shortened schedule "+('improved' if olddelta['Score5']>0 else 'did not improve')+' the measured five-set aggregate in this seed.',
        '',f'Final full checkpoint: `{FINAL}`; SHA256 `{final["sha256"]}`.',
        f'Bare student: `{BARE}`; SHA256 `{result["bare_sha256"]}`.',
        '', '## Engineering evidence','',
        '- Common0 SHA verified before/after. Initial model tensors equal common0. Original common0 lacks adapter; frozen deterministic constructor supplies adapter/visual blocks/gate. All component digests equal historical HNS initialization. Smoke and formal state0 are exact and independent; AdamW state empty.',
        '- Historical production model, objective, preprocessing, tokenizer, optimizer and native evaluator sources byte-identical. HNS sparse weights[1,2,2], Hard-ST beta2/2 ramp200, original detach preserved. No E2 macro coefficients.',
        '- CPU and original four-rank gloo objective/gradient/AdamW synchronization tests pass. Four-GPU real smoke5 and formal DDP acceptance pass; final rank parameter max difference0 and NCCL all-reduce10.',
        '- Every3651 update and all3737712 sample positions match actual historical stream hashes and complete sampling dictionaries. Three padded positions/epoch; last180/rank. No missing/duplicate optimizer updates. Only epochs0..2 trained.',
        '- Native LR checked for every update. Index0..3650, horizon3651, warmup200 and all group peaks unchanged. Update3651 has small positive LR; theoretical next index3651 is0. No extra optimizer update.',
        '- Full checkpoints0/500/1217/2434/3651 include model, fusion, FP32 AdamW moments, four-rank Python/NumPy/CPU/CUDA RNG, loader generator, sampler/cursor and3651 scheduler metadata. Final cursor3,0; all optimizer counters3651. Final native model+adapter+optimizer strict restore and image/text bare embedding equality checked.',
        f"- Mean update cycle {runtime['update_cycle_mean_seconds']:.3f}s; median {runtime['update_cycle_median_seconds']:.3f}s; p95 {runtime['update_cycle_p95_seconds']:.3f}s; max {runtime['update_cycle_max_seconds']:.3f}s. Allocated peaks per rank: "+', '.join(f"{r['peak_allocated_gib']:.3f}GiB" for r in runtime['acceptance']['ranks'])+'.',
        '- Final-only historical five evaluation uses original algorithms/normalization/top-k. Full Flickr uses frozen manifest31783images/158915captions, own new cache, exact source fingerprints, FP32 inference,248tokens and frozen tie rules. No inference masks/rerank/ensemble/TTA. Original full checkpoint and common0 immutable.',
        '', '## Mask and training diagnostics','',
        '| Statistic | Pure3 last50 mean | Old HNS@3651 last50 mean | Delta |','|---|---:|---:|---:|']
    for k in ('HNS_F_keep','HNS_Dall_keep','HNS_D3_keep','HNS_DF_hard_violation_ratio','HNS_3D_hard_violation_ratio','HNS_DF_IoU','HNS_3D_IoU','HNS_align','HNS_original_sparse','HNS_surcharge'):
        report.append(f"| {k} | {diag['last50_Pure3'][k]['mean']:.9f} | {diag['last50_historical_HNS3651'][k]['mean']:.9f} | {diag['mean_difference'][k]:+.9f} |")
    report+=['','## Scientific limits','',
        'Single seed and one schedule candidate. Public benchmarks were repeatedly observed in previous experiments; these results are exploratory, not unbiased SOTA/generalization evidence. No reliable independent held-out protocol was introduced. Small net query-count differences are reported without claiming significance or assuming no query regressions.',
        'Density reversals are descriptive. Parameter-group gradients are observed and finite; per-loss gradient cosines were not measured here. Mask structure does not prove retrieval causality. Historical logs prove sample/text/token/sampling equivalence, but do not retain per-sample augmented pixels for tensorwise comparison. Dataset preprocessing and worker seed logic remain unchanged.',
        'Original3651 checkpoint used a4868 horizon, so model weights are expected to differ. Full Flickr is excluded from Score5. No checkpoint selected by intermediate scores.',
        '', 'Full36recalls, all baseline deltas and exact net correct-query counts are in COMPARISON.md / COMPARISON.json. Checkpoint and raw large logs/caches remain local. GitHub synchronization and final process/GPU evidence are recorded separately in DELIVERY_AUDIT.json.']
    (EXP/'FINAL_REPORT.md').write_text('\n'.join(report)+'\n')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['training-audit','summarize'])
    a=p.parse_args()
    if a.phase=='training-audit': training_audit()
    else: summarize()
