"""One authorized W20+KR234 combination, fresh0->500; native eval, then stop."""
import ast
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

from recovery import nested_d3_local_search as search
from recovery import nested_d3_followup500 as runner
from recovery.s02_nfs500 import ROOT, STEP0, STEP0_SHA, dump, sha, rows, now

EXP = ROOT/'experiments/nest_clip_v1/nested_d3_w20_kr234_500_v1'
RUN_ROOT = ROOT/'runtime/SAID-nest-clip-v1/nested-d3-w20-kr234-500-v1'
BRANCH = 'experiment/nested-d3-w20-kr234-500-v1'
ENTRY = 'recovery.w20_kr234_500'
ARM = 'W20-KR234'
CANONICAL_CONFIG = ROOT/'configs/nested_d3_w20_kr234_500.json'
ANCHOR_EXP = ROOT/'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1'
W20_EXP = ROOT/'experiments/nest_clip_v1/nested_d3_local_search500_v1/W20'
KR_EXP = ROOT/'experiments/nest_clip_v1/nested_d3_local_search500_v1/KR234'
KR_RUN = ROOT/'runtime/SAID-nest-clip-v1/nested-d3-local-search500-20261007/KR234'
W20_RUN = KR_RUN.parent/'W20'
REMOTE_SEARCH = 'origin/experiment/nested-d3-local-search500-v1'
REMOTE_ANCHOR = 'origin/experiment/nested-detail-d3-balanced500'
ARMS = {ARM:dict(axis='authorized_alignment_plus_granularity',weights=[1.4,1.4,.2],r=2.,
    mode='nested_detail_kr234',strict_lowest_reference=True,experiment_dir=str(EXP))}
STATIC = ('BASELINE_PROVENANCE.json','MATCHED_PREFLIGHT.json','CORRECTNESS.md')
CODE = {'recovery/w20_kr234_500.py','tests/test_w20_kr234_500.py',
        'recovery/check_stage500_publish.py','recovery/nested_d3_followup500.py',
        str(CANONICAL_CONFIG.relative_to(ROOT))}
ORIGINAL_INVARIANTS = search.checkpoint_invariants
ORIGINAL_MATCHED = search.matched_stream
ORIGINAL_FROZEN = search.frozen_config


def read(path):
    return json.loads(Path(path).read_text())


def git_bytes(ref,path):
    return subprocess.check_output(['git','show',ref+':'+str(Path(path).relative_to(ROOT))],cwd=ROOT)


def expected_config():
    return dict(read(ANCHOR_EXP/'config.json'),view_weights=[1.4,1.4,.2],sampling_mode='nested_detail_kr234')


def frozen_config(config,arm):
    ORIGINAL_FROZEN(config,arm)
    assert not config.get('hns_enabled',False), 'HNS forbidden'
    assert config.get('summary_t2i_weight',1.)==1.
    assert config['inclusion_max']==1. and config['inclusion_hierarchy']=='detail_chain'
    assert config['sparsity_scale']==1. and config.get('view_sparsity_weights',[1.,2.,2.])==[1.,2.,2.]


def reconstruction(row):
    import math
    assert not row.get('HNS_enabled',False) and not any(k.startswith('HNS_') for k in row)
    assert row['inclusion_enabled'] and row['inc_weight']==min(1.,(row['step']-1)/200.)
    ce=[row[p+'_i2t']+row[p+'_t2i'] for p in ('F','O','E')]
    align=10/3*sum(w*c for w,c in zip([1.4,1.4,.2],ce))
    sparse=(row['F_sparse']+2*row['O_sparse']+2*row['E_sparse'])/3
    inc=row['inc_weight']*row['inc']
    assert math.isclose(row['inclusion_loss'],inc,rel_tol=5e-6,abs_tol=2e-6)
    assert math.isclose(row['loss'],align+sparse+inc,rel_tol=5e-6,abs_tol=2e-6), 'Literal W20 objective reconstruction failed'


def matched_stream(actual,reference,arm):
    proof=ORIGINAL_MATCHED(actual,reference,arm)
    w20=rows(W20_RUN/'step500/steps.jsonl')[:len(actual)]
    assert len(w20)==len(actual)
    for a,b in zip(actual,w20):
        reconstruction(a)
        assert a['actual_lrs']==b['actual_lrs']
        old={h['rank']:h for h in b['rank_health']}
        for h in a['rank_health']:
            assert all(h['sampling'][k]==old[h['rank']]['sampling'][k] for k in search.SHARED_STREAM)
    proof.update(all_lowest_text_tokens_K_indices_exact_KR234=True,all_F_Dall_LR_exact_W20=True,
        all_total_losses_reconstructed=True,alignment=[1.4,1.4,.2],sparsity=[1,2,2],
        inclusion='Original detached-child soft chain, ramp200/max1',HNS=False)
    return proof


def checkpoint_invariants(current,reference):
    """Retain strict inherited initialization/AdamW/shape/source gates.

    The only runtime metadata addition since KR234 is an inactive HNS constructor;
    current non-HNS loss/every gradient/AdamW are exact against fetched old code.
    """
    proof=read(EXP/'BASELINE_PROVENANCE.json')
    cfg=current['config']; old=reference['config'];frozen_config(cfg,ARM)
    assert cfg['code_sha256']==proof['current_production_sources']
    assert old['code_sha256']==proof['KR234_production_sources']
    new=copy.deepcopy(current['config']);runtime=new['runtime_model']
    assert runtime.get('hns_enabled',False) is False
    assert runtime.get('hns_beta',[2.,2.])==[2.,2.]
    for key in ('hns_enabled','hns_beta'):
        if key not in old['runtime_model']:runtime.pop(key,None)
    normalized=dict(current,config=new)
    value=ORIGINAL_INVARIANTS(normalized,reference)
    value.update(HNS_inactive=True,CPU_fetched_KR234_forward_all_gradients_AdamW_exact=True,
        literal_W20_weights=[1.4,1.4,.2],source_migration=proof['source_migration'])
    return value


def fetched_data(ref):
    path=ROOT/'train/nested_semantic_data.py'
    ns={'__name__':'train._w20_kr234_fetched','__file__':str(path),'__package__':'train'}
    exec(compile(git_bytes(ref,path),'fetched-KR234-dataset','exec'),ns)
    return ns


def function_digest(source,name):
    nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name]
    assert len(nodes)==1
    return hashlib.sha256(ast.dump(nodes[0],include_attributes=False).encode()).hexdigest()


def isolation():
    from train.train_nested_semantic_mask import code_manifest
    references={}
    for name,exp,ref in [('Anchor',ANCHOR_EXP,REMOTE_ANCHOR),('W20',W20_EXP,REMOTE_SEARCH),('KR234',KR_EXP,REMOTE_SEARCH)]:
        result=read(exp/'RESULTS.json');config=read(exp/'config.json')
        assert git_bytes(ref,exp/'RESULTS.json')==(exp/'RESULTS.json').read_bytes()
        assert git_bytes(ref,exp/'config.json')==(exp/'config.json').read_bytes()
        assert result['completed_steps']==500 and result['evaluation_checkpoint_immutable']
        references[name]=dict(remote_ref=ref,remote_commit=subprocess.check_output(['git','rev-parse',ref],cwd=ROOT,text=True).strip(),
            config=config,result_sha256=sha(exp/'RESULTS.json'),scores_percent=result['scores_percent'],
            checkpoint_sha256=result['checkpoint_sha256'])
    assert references['W20']['config']==dict(references['Anchor']['config'],view_weights=[1.4,1.4,.2])
    assert references['KR234']['config']==dict(references['Anchor']['config'],sampling_mode='nested_detail_kr234')
    assert read(EXP/'config.json')==read(CANONICAL_CONFIG)==expected_config()
    frozen_config(read(EXP/'config.json'),ARM)
    kr=read(KR_EXP/'RESULTS.json');current=code_manifest()
    old={p:h for p,h in kr['launch_provenance']['source_sha256'].items() if p in current}
    changed={p for p in old if old[p]!=current[p]}
    assert changed==search.EDITED
    assert set(current)-set(old)=={'model/hard_nested_sparsity.py'}
    frozen={}
    for p in current:
        if p not in changed and p in old:
            assert sha(ROOT/p)==old[p];frozen[p]=old[p]
    for p in ('recovery/s02_full_local_data.py','tools/eval_nest_native.py',
        'experiments/nest_clip_v1/armb_summary02_4epoch_v1/reproduction_full.py',
        'experiments/s0_dualmask_full_v01/evidence/step2000/new_evaluations/eval_extended_real.py'):
        assert sha(ROOT/p)==kr['launch_provenance']['source_sha256'][p];frozen[p]=sha(ROOT/p)
    trainer=ROOT/'train/train_nested_semantic_mask.py'
    old_source=git_bytes(REMOTE_SEARCH,trainer).decode();new_source=trainer.read_text()
    exact_functions={}
    for name in ('build_optimizer','learning_rates','optimizer_learning_rates','seed_all','training_horizon','consumed_batch','save_checkpoint'):
        assert function_digest(old_source,name)==function_digest(new_source,name),name
        exact_functions[name]=function_digest(new_source,name)
    assert sha(STEP0)==STEP0_SHA
    return dict(passed=True,references=references,common_step0=dict(path=str(STEP0),sha256=STEP0_SHA,resume=None),
        sole_method_config_changes={'view_weights':[1.4,1.4,.2],'sampling_mode':'nested_detail_kr234'},
        current_production_sources=current,KR234_production_sources=old,
        frozen_source_SHA256=frozen,optimizer_scheduler_rng_checkpoint_functions_AST_exact=exact_functions,
        source_migration={p:dict(previous=old.get(p),current=current[p]) for p in sorted(changed|{'model/hard_nested_sparsity.py'})},
        export_source=dict(old=kr['launch_provenance']['source_sha256']['tools/nest_clip.py'],current=sha(ROOT/'tools/nest_clip.py'),
            reason='Added constructor routing for inactive HNS; actual strict bare embedding equality required at500'),
        no_new_production_source_edits=True,HNS=False,checked_utc=now())


def matched_preflight():
    import numpy as np
    import torch
    from torch.utils.data import DistributedSampler
    from train.nested_semantic_data import sampled_text_views
    from recovery.s02_full_local_data import FullLocalDataset
    assert not RUN_ROOT.exists(), 'Preflight cannot overwrite a launched run'
    evidence_dir=RUN_ROOT.parent/(RUN_ROOT.name+'.preflight')
    assert not evidence_dir.exists(), 'No silent preflight overwrite'
    evidence_dir.mkdir()
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    historical=fetched_data(REMOTE_SEARCH)
    dataset=FullLocalDataset(search.local.INDEX,search.IMAGES,'nested_detail_kr234',0)
    previous=read(KR_RUN/'sampling-audit-1000.json');evidence=[]
    for rank in range(4):
        sampler=DistributedSampler(dataset,num_replicas=4,rank=rank,seed=0,shuffle=True,drop_last=False)
        sampler.set_epoch(0)
        for index in itertools.islice(iter(sampler),250):
            path=dataset.resolved_path(index)
            record=json.loads(dataset._records[dataset._offsets[index]:dataset._offsets[index+1]])
            sid=index+1000
            current=sampled_text_views(record['caption'],'nested_detail_kr234',0,0,sid)
            old=historical['sampled_text_views'](record['caption'],'nested_detail_kr234',0,0,sid)
            audit=previous[len(evidence)]
            assert audit['sample_id']==sid and audit['rank']==rank and current['views']==old['views']
            for k in ('n','K','valid','reason','detail_indices','dall_indices','detail_pool_size','untruncated_lengths'):
                assert current[k]==old[k],k
            assert current['detail_indices']==audit['sentence_indices']['lowest']
            assert current['dall_indices']==audit['sentence_indices']['Dall']
            assert current['K']==audit['K'] and current['detail_pool_size']==audit['m']
            for label,key,pos in [('F','tokens_f',0),('Dall','tokens_o',1),('Dk','tokens_e',2)]:
                assert torch.equal(current[key],old[key])
                assert current['views'][pos]==audit['strings'][label]
                assert current[key].tolist()==audit['token_ids'][label]
            evidence.append(dict(rank=rank,sample_id=sid,local_path=str(path),
                pool=current['dall_indices'],K=current['K'],selected_indices=current['detail_indices'],
                text=current['views'],token_ids={k:current[k].tolist() for k in ('tokens_f','tokens_o','tokens_e')}))
    after=np.random.get_state()
    assert before[0]==random.getstate() and torch.equal(before[2],torch.get_rng_state())
    assert before[1][0]==after[0] and np.array_equal(before[1][1],after[1]) and before[1][2:]==after[2:]
    raw=evidence_dir/'sampling-1000.json';dump(raw,evidence)
    pathproof=search.local.path_proof();dump(evidence_dir/'local-path-proof-5000.json',pathproof)
    result=dict(passed=True,records=1000,sample_ids_pool_K_indices_text_tokens_exact_historical_KR234=True,
        fetched_original_sampling_exact=True,private_RNG_global_python_numpy_torch_unchanged=True,
        local_path_proof=dict(passed=True,count=5000),image_transforms_frozen_source=True,
        raw_evidence=dict(path=str(raw),bytes=raw.stat().st_size,sha256=sha(raw),uploaded=False),
        historical_audit=dict(path=str(KR_RUN/'sampling-audit-1000.json'),sha256=sha(KR_RUN/'sampling-audit-1000.json')))
    dump(EXP/'MATCHED_PREFLIGHT.json',result)
    return result


def classify(q,base,anchor,w20):
    """Raw Score5 controls candidacy; tolerance is explicitly only the long guard."""
    candidate=q['Score5']>base['Score5']
    if candidate and q['J_long3']>=base['J_long3']-.002:
        return 'STRONG_POSITIVE',True
    if q['Score5']>max(anchor['Score5'],w20['Score5'],base['Score5']):
        return 'POSITIVE',True
    if any(q[k]>base[k] for k in ('J_long3','J_long','Short4')):
        return 'MIXED',candidate
    return 'NEGATIVE',candidate


def summarize():
    from recovery.nested_d3_local_search_evidence import quality,recall_delta
    from recovery.nested_d3_hns500 import quality_raw
    from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics
    assert read(EXP/'VALIDATION.json')['passed']
    result=read(EXP/'RESULTS.json');q=quality_raw(result)
    bases={n:read(p/'RESULTS.json') for n,p in [('Anchor',ANCHOR_EXP),('W20',W20_EXP),('KR234',KR_EXP)]}
    # Compare raw fractions for Score5; never use rounded chat/Markdown values.
    raw={n:quality_raw(r) for n,r in bases.items()}
    status,candidate=classify(q,raw['KR234'],raw['Anchor'],raw['W20'])
    comparisons={n:dict(quality_delta_pp={k:100*(q[k]-v) for k,v in raw[n].items()},
        recall_delta_pp=recall_delta(result,r),quality_percent=quality(r)) for n,r in bases.items()}
    diagnostic=read(EXP/'TRAINING_DIAGNOSTICS.json');gradient=read(EXP/'GRADIENT_SPOTCHECK.json')
    def view(d):return d['last50_views'].get('Dk',d['last50_views'].get('D3'))
    def gradient_ratio(g):
        if 'weighted_lowest_Dall_ratio' in g:return g['weighted_lowest_Dall_ratio']
        return g['weighted_mean_gradient_norms']['D3']/g['weighted_mean_gradient_norms']['Dall']
    compare_diagnostics={}
    for n,p in [('Anchor',ANCHOR_EXP),('W20',W20_EXP),('KR234',KR_EXP)]:
        old_d=read(p/'TRAINING_DIAGNOSTICS.json');old_g=read(p/'GRADIENT_SPOTCHECK.json')
        old_v=view(old_d)
        share=old_v.get('alignment_share_percent',100*old_v.get('weighted_alignment_loss_share',0))
        compare_diagnostics[n]=dict(lowest_alignment_share_percent=share,
            weighted_lowest_Dall_gradient_ratio=gradient_ratio(old_g),
            raw_gradient_norms=old_g['mean_gradient_norms'],
            weighted_gradient_norms=old_g['weighted_mean_gradient_norms'])
    known={}
    for n,p in [('INC0',ROOT/'experiments/nest_clip_v1/nested_d3_inc0_500_v1/RESULTS.json'),
                ('HNS-v1',ROOT/'experiments/nest_clip_v1/nested_d3_hns500_v1/RESULTS.json')]:
        old=read(p);value=quality_raw(old)['Score5'];known[n]=dict(Score5=100*value,
            delta_Score5_pp=100*(q['Score5']-value),exceeds=q['Score5']>value,not_used_to_change_selection_rule=True)
    interactions={k:100*(q[k]-raw['W20'][k]-raw['KR234'][k]+raw['Anchor'][k]) for k in q}
    result.update(classification=status,next_full_candidate=candidate,
        next_action='NEXT_FULL_CANDIDATE' if candidate else 'NO_FULL_CANDIDACY',automatic_full=False,
        authorized_combination=['W20 alignment','KR234 granularity'],no_additional_combination=True,
        comparisons=comparisons,quality_raw_fraction=q,known_500_references=known,
        interaction_vs_additive_baseline_pp=interactions,
        alignment_share_lowest_percent=view(diagnostic)['alignment_share_percent'],
        weighted_lowest_Dall_gradient_ratio=gradient['weighted_lowest_Dall_ratio'],
        diagnostic_baselines=compare_diagnostics,
        questions=dict(Q1='Descriptive interaction deltas recorded; one seed cannot establish statistical synergy',
            Q2_exceeds_KR234_raw_Score5=candidate,Q3_exceeds_known_500={n:v['exceeds'] for n,v in known.items()},
            Q4_directional_tradeoffs_vs_KR234=comparisons['KR234']['recall_delta_pp'],
            Q5_long_rich_vs_W20=comparisons['W20']['quality_delta_pp']),
        quality_delta_vs_anchor_pp=comparisons['Anchor']['quality_delta_pp'],
        recall_delta_vs_anchor_pp=comparisons['Anchor']['recall_delta_pp'],
        trajectory_reference='KR234 exact512000 IDs/F/Dall/K/indices/Dk text/tokens/LR; Anchor and W20 shared F/Dall/LR',
        HNS=False,resume=None,stopped_at_500=True)
    _,_,sources=native_metrics(search.RUN/'step500')
    for name,path in sources.items():
        dest=EXP/'evaluations'/(name+'.json');dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest)
    dump(EXP/'COMMANDS.json',read(search.RUN/'commands.json'))
    stats=read(EXP/'RUNTIME_STATS.json')
    raw=Path(read(EXP/'MATCHED_PREFLIGHT.json')['raw_evidence']['path'])
    pathproof=raw.parent/'local-path-proof-5000.json'
    stats['additional_local_artifacts']=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),uploaded=False,
        time_range_utc=[read(EXP/'SEARCH_PLAN.json')['prepared_utc'],read(EXP/'SEARCH_PLAN.json')['prepared_utc']],
        time_scope='Prelaunch evidence, timestamp is preparation receipt') for p in (raw,pathproof)]
    dump(EXP/'RUNTIME_STATS.json',stats)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    result['GPU_idle_after_evaluation']=True
    dump(EXP/'RESULTS.json',result)
    validation=read(EXP/'VALIDATION.json');validation.update(matched_preflight=read(EXP/'MATCHED_PREFLIGHT.json'),
        only_two_authorized_method_changes=True,HNS=False,all500_literal_loss_reconstruction=True)
    dump(EXP/'VALIDATION.json',validation)
    provenance=read(EXP/'BASELINE_PROVENANCE.json')
    lines=['# W20 + KR234: one authorized combination at500','',f'Classification: `{status}`; full candidacy: `{result["next_action"]}`.','',
        'Fresh common0, exactly500 updates, horizon4868, local-only. Only W20[1.4,1.4,.2] and historical KR234 sampling are combined. Original soft detached-child inclusion ramp200/max1 and sparsity[1,2,2] remain active. HNS is disabled.','',
        '| Method | Alignment | K | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    keys=('Score5','J_long3','J_long','Short4','Urban_I2T','Urban_T2I')
    for n,r in list(bases.items())+[(ARM,result)]:
        scores=quality(r);weights='1.4/1.4/.2' if n in ('W20',ARM) else '1.35/1.35/.30'
        k='KR234' if n in ('KR234',ARM) else 'fixed3'
        lines.append('| '+n+' | '+weights+' | '+k+' | '+' | '.join(f'{scores[v]:.6f}' for v in keys)+' |')
    lines+=['','| Reference | ΔScore5 | ΔJ_long3 | ΔJ_long | ΔShort4 | ΔUrban I2T | ΔUrban T2I |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for n,comp in comparisons.items():lines.append('| '+n+' | '+' | '.join(f'{comp["quality_delta_pp"][v]:+.6f}' for v in keys)+' |')
    lines+=['','| Dataset | I2T R@1/5/10 | T2I R@1/5/10 |','|---|---|---|']
    for ds,m in result['metrics'].items():
        lines.append('| '+ds+' | '+' | '.join(' / '.join(f'{100*m[d][k]:.6f}' for k in ('R@1','R@5','R@10')) for d in ('I2T','T2I'))+' |')
    lines+=['',f'Lowest-view alignment share: {result["alignment_share_lowest_percent"]:.6f}%; weighted lowest/Dall gradient norm ratio: {gradient["weighted_lowest_Dall_ratio"]:.6f}.',
        'Identical eight-global-batch, fixed500-weight gradient spotcheck; raw directional CE on exact backbone group, all-reduce/4. Diagnostic norms are not signed net-gradient contributions.',
        'Baseline gradient/CE/share comparisons, all five-set directional deltas, K statistics, mask hierarchy, gate statistics and runtime are recorded in JSON.','',
        'Q1: interaction = combination - W20 - KR234 + Anchor, in RESULTS.json. A positive value is a descriptive single-seed result, not proof of causal/statistical synergy.',
        f'Q2: raw Score5 exceeds KR234: {candidate}. Q3: exceeds known500 points: '+str(result['questions']['Q3_exceeds_known_500'])+'.',
        'Q4: Urban T2I, Short4, Flickr T2I and Long-DCI deltas vsKR234 identify whether W20 eases tradeoffs. Q5: J_long3/J_long/Long-DCI deltas vsW20 identify any rich-text benefit.',
        'Selection uses raw Score5. STRONG_POSITIVE applies when aboveKR234 and J_long3 decline is at most0.20pp, a predeclared interpretation of “not markedly worse”; positiveScore5 cannot be labeledNEGATIVE by this guard. Full candidacy requires strictly greater rawScore5, with no rounding tolerance.',
        'Github fetched configs/results and pinned source migrations are in BASELINE_PROVENANCE.json. Current non-HNS production loss/every gradient/AdamW were compared exactly to fetched historicalKR234. Native evaluator sources are unchanged, optimizer/LR/scheduler/checkpoint function ASTs match.',
        'Strict bare/native full-caption inference only; no mask/gate/local-caption/rerank/ensemble/TTA.','',
        'Checkpoint SHA256: `'+result['checkpoint_sha256']+'`. Bare SHA256: `'+result['strict_export']['bare_sha256']+'`.',
        'Large checkpoints/weights/1000-sample token evidence/rawlogs remainserver-local. RUNTIME_STATS.json inventories path/size/SHA/time windows. `/root` cache is disposable; NFS originals retained.',
        'No full4868 or additional experiment launched, regardless of candidacy. Stop and wait for human decision.']
    for name in ('REPORT.md','SEARCH_SUMMARY.md'):(EXP/name).write_text('\n'.join(lines)+'\n')


def publication_paths():
    names=runner.REPORT_NAMES+('SEARCH_SUMMARY.md','COMMANDS.json',*STATIC,'CPU_TESTS.json','SEARCH_PLAN.json')
    return [ROOT/p for p in sorted(CODE)]+[EXP/n for n in names]+[
        EXP/'evaluations'/(ds+'.json') for ds in ('COCO','Urban-1k','Flickr30k-test1k','DOCCI','Long-DCI')]


def configure():
    search.EXP,search.RUN_ROOT,search.BRANCH=EXP,RUN_ROOT,BRANCH
    search.ANCHOR_EXP,search.ANCHOR_RUN=KR_EXP,KR_RUN
    search.ARMS,search.ENTRY_MODULE=ARMS,ENTRY
    search.PHASE_PREFIX='formal-w20-kr234-500-v1-'
    search.EXTRA_SOURCES=CODE|{str((EXP/n).relative_to(ROOT)) for n in STATIC}
    search.frozen_config=frozen_config;search.checkpoint_invariants=checkpoint_invariants
    search.matched_stream=matched_stream
    runner.EXP,runner.RUN_ROOT,runner.BRANCH=EXP,RUN_ROOT,BRANCH
    runner.ARMS,runner.ENTRY=ARMS,ENTRY
    runner.MAIN_LOG=RUN_ROOT.parent/(RUN_ROOT.name+'.runner.log')
    runner.IDENTITY=RUN_ROOT.parent/(RUN_ROOT.name+'.runner.json')
    runner.PUBLISH_MESSAGE='Report one authorized W20 plus KR234 fresh common0 local500 combination'
    runner.configure=configure;runner.summarize=summarize
    runner.publication_paths=publication_paths
    runner.augment_diagnostics=lambda arm: None


def prepare():
    configure();search.activate(ARM)
    assert read(EXP/'config.json')==read(CANONICAL_CONFIG)==expected_config()
    dump(EXP/'BASELINE_PROVENANCE.json',isolation())
    matched_preflight()
    dump(EXP/'SEARCH_PLAN.json',dict(arm=ARM,authorized_method_changes=['W20 alignment','KR234 granularity'],
        view_weights=[1.4,1.4,.2],sampling='Exact fetched historical KR234',sparsity=[1,2,2],
        inclusion='Original detached-child soft inclusion, ramp200/max1',HNS=False,
        common_step0_sha256=STEP0_SHA,resume=None,stop_updates=500,horizon=4868,
        classification='Raw Score5; STRONG if >KR234 and J_long3 decline<=.20pp; POSITIVE if newrelated500best withlongtradeoff; MIXED ifindependentlong/shortadvantage; otherwiseNEGATIVE',
        full_candidacy='raw Score5 strictly > historicalKR234; candidateonly, never automaticfull',
        local_image_root=str(search.IMAGES),NFS_fallback=False,automatic_full=False,additional_arms=False,
        single_seed_limit=True,baseline_results='Fetched Github JSON; no chat-only metric inputs',
        worker_entry=ENTRY,matched_preflight='1000 actual frozen samples: IDs,pool,K,indices,text,tokens exactKR234',
        formal_gate='First5 checked before6; full512000 exactKR234 stream and W20sharedstream/LR',
        resource_policy='>3s warning; actualIO/CUDA/OOM/DDP/NaN/Inf/oom_kill/>60s/pod error hardstop',
        background_queue=['fresh common0 formal500','readonlygradient8globalbatches','strict native fiveeval','report','commit push fetch'],
        prepared_utc=now()))


def publish_setup():
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()==BRANCH
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
    paths=[ROOT/p for p in sorted(CODE)]+[EXP/p for p in ('config.json','CPU_TESTS.json','SEARCH_PLAN.json',*STATIC)]
    assert all(p.is_file() and p.stat().st_size<1024*1024 for p in paths)
    subprocess.run(['git','status','--short'],cwd=ROOT,check=True)
    subprocess.run(['git','add','--',*[str(p.relative_to(ROOT)) for p in paths]],cwd=ROOT,check=True)
    from recovery.check_stage500_publish import inspect
    review=inspect();assert review['passed']
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Prepare one frozen W20 plus KR234 common0 local500 experiment'],cwd=ROOT,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    subprocess.run(['git','fetch','origin','refs/heads/'+BRANCH+':refs/remotes/origin/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    remote=subprocess.check_output(['git','rev-parse','refs/remotes/origin/'+BRANCH],cwd=ROOT,text=True).strip()
    assert remote==head==subprocess.check_output(['git','rev-parse','FETCH_HEAD'],cwd=ROOT,text=True).strip()
    dump(EXP/'SETUP_GITHUB_RECEIPT.json',dict(passed=True,branch=BRANCH,commit=head,remote_HEAD=remote,
        remote_HEAD_matches_local=True,push_success=True,checked_utc=now(),publication_check=review))


def main():
    if '--preflight' in sys.argv:
        assert sys.argv[1:]==['--preflight'];prepare();return
    if '--publish-setup' in sys.argv:
        assert sys.argv[1:]==['--publish-setup'];publish_setup();return
    configure();runner.main()


if __name__=='__main__':main()
