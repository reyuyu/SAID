"""Only E2-Uniform: native complete-state resume500 ->1217, evaluate, stop."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess

from recovery import hns_s12_2434_validation as protocol
from recovery.s02_nfs500 import ROOT, STEP0, STEP0_SHA, dump, rows, sha, now

PROJECT=protocol.PROJECT
BRANCH='experiment/hns-s12-uniform-e1-1217-v1'
PARENT_COMMIT='9880a296fb6552c66cf45f9646c6f8a2f2e802e4'
MOTHER='origin/experiment/hns-s12-sparse-ratio-twoarm500-v1'
BASE_EXP=ROOT/'experiments/nest_clip_v1/hns_s12_sparse_ratio_twoarm500_v1/E2-Uniform'
BASE_RUN=PROJECT/'runtime/SAID-nest-clip-v1/hns-s12-sparse-ratio-twoarm500-v1/E2-Uniform'
PARENT=BASE_RUN/'step500/step000500.pt'
PARENT_SHA='74271df5298525f924834b3c74fa228eceff623020291c775808f1994214cb9a'
EXP=ROOT/'experiments/nest_clip_v1/hns_s12_uniform_e1_1217_v1'
RUN=PROJECT/'runtime/SAID-nest-clip-v1/hns-s12-uniform-e1-1217-v1'
ENTRY='recovery.hns_s12_uniform_e1_1217'
TARGETS=(1217,)
BASELINE_BRANCH='origin/experiment/hns-s12-2434-validation-v1'
BASELINE_FILE='experiments/nest_clip_v1/hns_s12_2434_validation_v1/step1217/RESULTS.json'
BASELINE_RUN=PROJECT/'runtime/SAID-nest-clip-v1/hns-s12-2434-validation-v1/step1217'
CODE=('recovery/hns_s12_uniform_e1_1217.py','tests/test_hns_s12_uniform_e1_1217.py')
ORIGINAL_IDENTITY=protocol.identity
ORIGINAL_WORKER=protocol.worker
ORIGINAL_REPORT=protocol.report
ORIGINAL_RESUME_GATE=protocol.resume_gate
ORIGINAL_STATE=protocol.state


def frozen(cfg):
    expected=protocol.common.read(BASE_EXP/'config.json')
    assert all(cfg.get(k)==v for k,v in expected.items()),'E2-Uniform configuration drift'
    assert tuple(cfg[k] for k in protocol.common.MACRO_KEYS)==(10.,1.2,1.)
    assert cfg['view_sparsity_weights']==[5/3,5/3,5/3]
    assert cfg['view_weights']==[1.35,1.35,.3] and cfg['sparsity_scale']==1
    assert cfg['hns_enabled'] and cfg['inclusion_max']==0
    assert cfg['sampling_mode']=='nested_detail_d3' and cfg['inclusion_hierarchy']=='detail_chain'
    assert cfg.get('hns_beta',[2,2])==[2,2] and not cfg.get('hns_detach_child',False)
    assert not cfg.get('hns_half_after500',False)


def predecessor(target):
    assert target==1217,'Only stop1217 authorized'
    return PARENT,500


def phase(target):
    assert target==1217
    return protocol.local.IMAGES.parent/'formal-hns-s12-uniform-e1-1217-v1'


def identity(path,completed):
    if Path(path)==PARENT:
        assert completed==500 and path.is_file() and sha(path)==PARENT_SHA,'Invalid original500 checkpoint'
    proof=ORIGINAL_IDENTITY(path,completed)
    if completed==1217:
        import torch
        p=torch.load(path,map_location='cpu',weights_only=False)
        assert p['config']['start_updates']==500 and p['config']['updates_planned_this_run']==717
        assert p['config']['resume']==str(PARENT) and p['data_cursor']==dict(next_epoch=1,next_batch=0)
        assert p['resume_lineage']['sha256']==PARENT_SHA
        proof.update(start_updates=500,updates_this_run=717,parent_sha256=PARENT_SHA,complete_epoch=True)
    return proof


def archive(branch,path):
    blob=subprocess.check_output(['git','show',branch+':'+path],cwd=ROOT)
    return json.loads(blob),dict(branch=branch,commit=protocol.common.git('rev-parse',branch),
        path=path,sha256=hashlib.sha256(blob).hexdigest())


def local_suffix_proof():
    """Validate actual local image paths within the unused epoch0 suffix."""
    import torch
    from torch.utils.data import DistributedSampler
    from recovery.s02_full_local_data import FullLocalDataset
    dataset=FullLocalDataset(protocol.local.INDEX,protocol.local.IMAGES,'nested_detail_d3',0)
    before=torch.get_rng_state().clone();rng=random.Random(0);evidence=[]
    for rank in range(4):
        sampler=DistributedSampler(dataset,4,rank,seed=0,shuffle=True,drop_last=False);sampler.set_epoch(0)
        positions=rng.sample(range(500*256,len(sampler)),1250);indices=list(sampler)
        for pos in positions:
            index=indices[pos];path=dataset.resolved_path(index)
            evidence.append(dict(rank=rank,sample_id=index+1000,position=pos,path=str(path)))
    assert torch.equal(before,torch.get_rng_state()) and len(evidence)==5000
    return dict(passed=True,count=5000,scope='Unused epoch0 sampler positions500*256..end, each rank',
        local_only=True,NFS_fallback=False,global_RNG_unchanged=True,rows=evidence)


def prepare():
    import torch
    from tools.eval_five_parallel import require_gpu_idle
    torch.set_num_threads(4)
    require_gpu_idle({0,1,2,3});assert protocol.common.git('branch','--show-current')==BRANCH
    assert not EXP.exists() and not RUN.exists(),'Refuse overwrite/implicit retry'
    assert protocol.common.git('rev-parse',MOTHER)==PARENT_COMMIT
    proof=identity(PARENT,500);assert sha(STEP0)==STEP0_SHA
    old,old_receipt=archive(PARENT_COMMIT,str((BASE_EXP/'RESULTS.json').relative_to(ROOT)))
    assert old==protocol.common.read(BASE_EXP/'RESULTS.json')
    assert old['checkpoint_unchanged'] and old['checkpoint']['sha256']==PARENT_SHA
    assert old['strict_export']['checkpoint_sha256']==PARENT_SHA
    cfg,config_receipt=archive(PARENT_COMMIT,str((BASE_EXP/'config.json').relative_to(ROOT)))
    assert cfg==protocol.common.read(BASE_EXP/'config.json');frozen(cfg)
    for path,digest in proof['sources'].items():
        for commit in (PARENT_COMMIT,proof['git_head']):
            assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)).hexdigest()==digest
    ready=protocol.common.read(protocol.local.IMAGES.parent/'full-ready.json')
    assert ready['status']=='LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
    evaluator=protocol.common.evaluator_proof()
    baseline,ref_receipt=archive(BASELINE_BRANCH,BASELINE_FILE)
    assert baseline['completed_steps']==1217 and baseline['evaluation_checkpoint_immutable']
    expected=dict(Score5=72.906355,J_long3=76.999258,J_long=85.760003,Short4=66.767,
                  Urban_I2T=93.2,Urban_T2I=91.1)
    assert all(abs(protocol.common.quality(baseline)[k]-v)<1e-5 for k,v in expected.items())
    suffix=local_suffix_proof();reference=protocol.resume_reference(500)
    EXP.mkdir(parents=True);RUN.mkdir(parents=True)
    dump(EXP/'config.json',cfg)
    dump(EXP/'RESUME_PROVENANCE.json',dict(**proof,parent_branch=MOTHER,parent_commit=PARENT_COMMIT,
        parent_results=old_receipt,config_provenance=config_receipt,evaluator_sources=evaluator,
        production_source_changes=[],local_only=True,NFS_fallback=False))
    dump(EXP/'REFERENCES.json',dict(models={'HNS_S12':{'1217':baseline},'E2_Uniform_500':{'1217':old}},
        provenance=[ref_receipt,old_receipt],same_node_primary='HNS_S12',
        E2_500_comparison_is_training_progress_not_same_node_ablation=True))
    dump(RUN/'local-suffix-path-proof-5000.json',suffix)
    dump(EXP/'LOCAL_ONLY_PROOF.json',{k:v for k,v in suffix.items() if k!='rows'})
    protocol.segment(1217).mkdir();(EXP/'step1217').mkdir()
    dump(protocol.segment(1217)/'resume-reference.json',reference)
    dump(EXP/'PLAN.json',dict(order=['native resume500->1217','audit/export/verify/five-eval/report','STOP'],
        target=1217,start_updates=500,updates_this_run=717,horizon=4868,macro=[10,1.2,1],
        view_weights=[1.35,1.35,.3],raw_sparsity_ratio=[1,1,1],normalized_sparsity_weights=[5/3]*3,
        effective_sparsity_coefficients=[2,2,2],beta=[2,2],Hard_ST=True,no_SG=True,soft_inclusion=0,
        K=3,ramp200_unchanged=True,local_only=True,full_restore=True,resume_checkpoint_sha256=PARENT_SHA,
        command=protocol.training_command(1217),first_update=501,last_update=1217,
        replay_first500_batches_without_optimizer_updates=True,tail_batch_per_rank=180,
        epoch0_suffix_sample_records=733904,initial_cursor=dict(next_epoch=0,next_batch=500),
        final_cursor=dict(next_epoch=1,next_batch=0),evaluation_mapping=dict(coco=0,docci=1,long_dci=2,flickr=3,urban=3),
        automatic2434=False,automatic3651=False,automatic4868=False,E1_continuation=False,other_arms=False))
    cpu_tests()
    state('PREPARED')


def cpu_tests():
    tests=[str(PROJECT/'.venv/bin/python'),'-m','pytest','-q','tests/test_hns_s12_uniform_e1_1217.py']
    with (RUN/'cpu-tests.log').open('w') as handle:
        check=subprocess.run(tests,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
    dump(EXP/'CPU_TESTS.json',dict(passed=check.returncode==0,returncode=check.returncode,command=tests,
        summary=(RUN/'cpu-tests.log').read_text(),checked_utc=now()))
    assert check.returncode==0,'Resume controller tests failed; do not train'


def worker(target):
    """Reuse the reviewed resume hooks; add read-only update501..505 audits."""
    import torch
    import torch.distributed as dist
    from train import train_nested_semantic_mask as trainer
    from recovery import nested_d3_local_search as search
    context={};original_optimizer=trainer.build_optimizer;original_rates=trainer.optimizer_learning_rates
    original_observe=search.observe_selection
    def optimizer(module):
        context['optimizer']=original_optimizer(module);return context['optimizer']
    def rates(module,completed,horizon):
        context['module']=module;context['completed_before_update']=completed
        return original_rates(module,completed,horizon)
    def observe(batch):
        step=context['completed_before_update']+1
        if 501<=step<=505:
            assert {int(s['step']) for s in context['optimizer'].state.values()}=={step}
            assert all(torch.isfinite(p).all() for p in context['module'].parameters())
            difference=trainer.parameter_agreement(context['module']);assert difference==0
            dump(protocol.segment(target)/f'update-{step}-rank{dist.get_rank()}.json',dict(passed=True,
                step=step,optimizer_counters=[step],updates_this_run=step-500,
                parameter_difference=difference,parameters_finite=True))
        return original_observe(batch)
    trainer.build_optimizer=optimizer;trainer.optimizer_learning_rates=rates;search.observe_selection=observe
    ORIGINAL_WORKER(target)


def resume_gate(target):
    proof=ORIGINAL_RESUME_GATE(target)
    update_checks=[protocol.common.read(protocol.segment(target)/f'update-{s}-rank{r}.json')
                   for s in range(501,506) for r in range(4)]
    assert all(v['passed'] and v['optimizer_counters']==[v['step']] and v['parameter_difference']==0 for v in update_checks)
    proof['post_update_checks']=update_checks;proof['no_repeated_first500_optimizer_updates']=True
    dump(EXP/'step1217/RESUME_GATE.json',proof);return proof


def stream_proof(actual,baseline):
    assert [r['step'] for r in actual]==[r['step'] for r in baseline]==list(range(501,1218))
    count=0
    for a,b in zip(actual,baseline):
        assert a['epoch']==b['epoch']==0 and a['actual_lrs']==b['actual_lrs']
        assert a['nonfinite']==0
        old={h['rank']:h for h in b['rank_health']};assert set(old)=={0,1,2,3}
        for h in a['rank_health']:
            ref=old[h['rank']]
            assert h['updates']==a['step']-500 and h['batch']==ref['batch']
            assert h['gradients_finite'] and h['stream_sha256']==ref['stream_sha256']
            assert h['sampling']==ref['sampling'],'Text/token/K/indices/sample order drift'
            count+=h['batch']
    assert count==733904 and all(h['batch']==180 for h in actual[-1]['rank_health'])
    return dict(passed=True,updates=717,first_update=501,last_update=1217,records=count,
        all_sample_ids_images_F_Dall_D3_tokens_K_indices_and_LR_match_original_epoch0_suffix=True,
        final_batch_per_rank=180,optimizer_no_repeated_updates=True,
        reference='Original HNS-S121217 data trajectory only; model/AdamW tensors intentionally not compared',
        parameter_equality_to_original_1_2_2_not_required=True)


def report(target,supervisor,result):
    ORIGINAL_REPORT(target,supervisor,result)
    dest=EXP/'step1217';runtime=protocol.segment(target)
    records=rows(runtime/'training/steps.jsonl')
    proof=stream_proof(records,rows(BASELINE_RUN/'training/steps.jsonl'))
    dump(dest/'SAMPLING_AND_LR_PROOF.json',proof)
    assert sha(PARENT)==PARENT_SHA
    masks=protocol.common.read(dest/'MASK_HIERARCHY_AUDIT.json')
    diag=protocol.common.read(dest/'TRAINING_DIAGNOSTICS.json');means=diag['last50_loss_components']
    masks.update(keep_ratios_valid_population={v:means['HNS_'+v+'_keep'] for v in ('F','Dall','D3')},
        valid_population_hierarchy={k:means[k] for k in ('HNS_DF_IoU','HNS_3D_IoU',
            'HNS_DF_hard_violation_ratio','HNS_3D_hard_violation_ratio',
            'HNS_DF_exact_equality_ratio','HNS_3D_exact_equality_ratio','HNS_triple_exact_equality_ratio')},
        density_order_F_Dall_D3=means['HNS_F_keep']>=means['HNS_Dall_keep']>=means['HNS_D3_keep'])
    for row in records:
        assert all(row['HNS_sparse_coeff_'+v]==5/3 for v in ('F','Dall','D3'))
    diag['weighted_sparsity_share_percent']={v:100*means['HNS_sparse_weighted_'+v]/means['macro_weighted_sparse'] for v in ('F','Dall','D3')}
    for name,value in [('MASK_HIERARCHY_AUDIT',masks),('TRAINING_DIAGNOSTICS',diag)]:dump(dest/(name+'.json'),value)
    for c in result['comparisons'].values():
        d=c['quality_delta_pp'];d['Urban_Mean']=(d['Urban_I2T']+d['Urban_T2I'])/2
    result.update(start_updates=500,updates_this_run=717,parent500_checkpoint_unchanged=True,
        parent500_sha256=PARENT_SHA,full_stream_and_LR_proof=proof)
    dump(dest/'RESULTS.json',result)


def combined(completed):
    assert completed==[1217]
    result=protocol.common.read(EXP/'step1217/RESULTS.json')
    masks=protocol.common.read(EXP/'step1217/MASK_HIERARCHY_AUDIT.json')
    gradient=protocol.common.read(EXP/'step1217/GRADIENT_AUDIT.json')
    refs=protocol.common.read(EXP/'REFERENCES.json')['models']
    baseline_diag=ROOT/'experiments/nest_clip_v1/hns_s12_2434_validation_v1/step1217'
    old_masks=protocol.common.read(BASE_EXP/'MASK_HIERARCHY_AUDIT.json')
    s12_masks=protocol.common.read(baseline_diag/'MASK_HIERARCHY_AUDIT.json')
    old_gradient=protocol.common.read(BASE_EXP/'GRADIENT_AUDIT.json')
    d=result['comparisons']['HNS_S12']['quality_delta_pp']
    scientific=dict(Score5_improved=d['Score5']>0,Urban_I2T_improved=d['Urban_I2T']>0,
        Urban_T2I_improved=d['Urban_T2I']>0,Urban_Mean_improved=d['Urban_Mean']>0,
        weak_single_seed_Score5_signal=0<d['Score5']<.05,
        matched_node_deltas=result['comparisons']['HNS_S12'],
        progress_vs_own500=result['comparisons']['E2_Uniform_500'],
        masks1217=masks,masks500=old_masks,HNS_S12_1217_masks=s12_masks,
        gradients1217=gradient['group_diagnostics'],gradients500=old_gradient['group_diagnostics'],
        view_gradients1217=gradient['view_gradients'],
        visual_mask_HNS_sparse_cosine1217=gradient['group_diagnostics']['visual_mask']['weighted_cosine_hierarchy_sparsity'],
        visual_mask_HNS_sparse_cosine500=old_gradient['group_diagnostics']['visual_mask']['weighted_cosine_hierarchy_sparsity'],
        full_results_do_not_follow_from_one_epoch=True,lower_keep_or_IoU_alone_does_not_prove_alignment_quality=True)
    dump(EXP/'SCIENTIFIC_DIAGNOSTICS.json',scientific)
    dump(EXP/'RESULTS.json',dict(status='COMPLETED',nodes={'1217':result},references=refs,
        scientific_diagnostics=scientific,stop_updates=1217,updates_this_run=717,automatic_continuation=False))
    lines=['# E2-Uniform: native resume500 ->1217','',
        'Original evaluated500 checkpoint resumed without parameter/optimizer/RNG/scheduler reset;717 updates, horizon4868. Original500 SHA: `'+PARENT_SHA+'`.','',
        '| Model | Step | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I | Urban Mean |',
        '|---|---:|---:|---:|---:|---:|---|---:|']
    for name,n,r in [('E2 Uniform',500,refs['E2_Uniform_500']['1217']),('HNS-S12',1217,refs['HNS_S12']['1217']),('E2 Uniform',1217,result)]:
        q=protocol.common.quality(r)
        lines.append(f'| {name} | {n} | '+' | '.join(f'{q[k]:.6f}' for k in ('Score5','J_long3','J_long','Short4'))+
            f' | {q["Urban_I2T"]:.3f} / {q["Urban_T2I"]:.3f} | {(q["Urban_I2T"]+q["Urban_T2I"])/2:.3f} |')
    lines+=['','| Dataset | I2T R1/R5/R10 (%) | T2I R1/R5/R10 (%) |','|---|---|---|']
    for ds,rec in result['metrics'].items():
        lines.append('| '+ds+' | '+' | '.join(' / '.join(f'{100*rec[dr][k]:.6f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I'))+' |')
    for name,c in result['comparisons'].items():
        lines+=['',name+' deltas(pp): '+json.dumps(c['quality_delta_pp'])+'.',
            '| Dataset | Delta I2T R1/R5/R10(pp) | Delta T2I R1/R5/R10(pp) |','|---|---|---|']
        for ds,rec in c['recall_delta_pp'].items():
            lines.append('| '+ds+' | '+' | '.join(' / '.join(f'{rec[dr][k]:+.6f}' for k in ('R@1','R@5','R@10')) for dr in ('I2T','T2I'))+' |')
    lines+=['','Matched-node interpretation: '+json.dumps({k:scientific[k] for k in ('Score5_improved','Urban_I2T_improved','Urban_T2I_improved','Urban_Mean_improved','weak_single_seed_Score5_signal')})+'.',
        'Mask coverage/order/violations/IoU and500/same-node baselines are in SCIENTIFIC_DIAGNOSTICS.json. Density inversions do not stop training and do not establish retrieval quality.',
        'Visual-mask hierarchy/sparsity gradient cosine500='+str(scientific['visual_mask_HNS_sparse_cosine500'])+',1217='+str(scientific['visual_mask_HNS_sparse_cosine1217'])+'. Other groups/norms/cosines/additivity and CE/share are in node diagnostics.',
        'Original HNS-S121217 sampler/text/token/K/index and LR suffix matches all733904 actual records, including180/rank tail. Different sparse allocation intentionally means model parameters are not required to match the1:2:2 baseline.',
        'Model/fusion/AdamW, four-rank RNG, sampler/cursor and DataLoader generator restore audits plus501..505 parameter agreement/optimizer counters/finite checks passed. Production code remains byte-identical.',
        'Five native datasets only; no mask inference/rerank/ensemble/TTA. Training and gradient processes exit before parallel evaluation. Parent and final checkpoint hashes remain unchanged through evaluation.',
        'Only E2 Uniform; stop1217, no2434/3651/4868, E1 or other arm. Checkpoints/bare/large raw logs/data stay local. Urban0.1pp is one query; small single-seed differences are exploratory.']
    (EXP/'REPORT.md').write_text('\n'.join(lines)+'\n')


def state(status,**extra):
    if status=='NODE_COMPLETED':status='COMPLETED_GPU_IDLE'
    ORIGINAL_STATE(status,**extra)


def configure():
    protocol.BRANCH=BRANCH;protocol.MOTHER=MOTHER;protocol.BASE_EXP=BASE_EXP;protocol.BASE_RUN=BASE_RUN
    protocol.PARENT=PARENT;protocol.PARENT_SHA=PARENT_SHA;protocol.EXP=EXP;protocol.RUN=RUN
    protocol.ENTRY=ENTRY;protocol.TARGETS=TARGETS;protocol.CODE=CODE
    protocol.frozen=frozen;protocol.predecessor=predecessor;protocol.phase=phase
    protocol.identity=identity;protocol.prepare=prepare;protocol.worker=worker
    protocol.resume_gate=resume_gate;protocol.report=report;protocol.combined=combined;protocol.state=state


def main():
    configure();protocol.main()


if __name__=='__main__':main()
