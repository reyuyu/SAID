"""One common0/local-only500 D3 equal-weight experiment; no automatic follow-up."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import random
import signal
import statistics
import subprocess
import sys

from recovery import nested_detail500 as parent
from recovery.s02_nfs500 import ROOT, OUT, STEP0_SHA, dump, rows, sha

EXP = ROOT/'experiments/nest_clip_v1/nested_detail_d3_equal_500_v1'
FAILED_ATTEMPT = ROOT/'runtime/SAID-nest-clip-v1/nested-detail-d3-equal500-20261007'
RUN = ROOT/'runtime/SAID-nest-clip-v1/nested-detail-d3-equal500-20261007-r1'
PHASE = parent.LOCAL/'formal-nested-detail-d3-equal500-phase-20261007-r1'
CONFIG = EXP/'config.json'
BASELINE = ROOT/'runtime/SAID-nest-clip-v1/nested-detail-equal-weight500-20261007/step500'
ATOMIC_EXP = ROOT/'experiments/nest_clip_v1/nested_detail_equal_weight_500_v1'
LOW_EXP = ROOT/'experiments/nest_clip_v1/nested_detail_500_v1'
MODE = 'nested_detail_d3'
WEIGHTS = dict(F=1.,Dall=1.,Ds=1.)  # Frozen model's internal E/Ds slot contains D3.
SCORES = dict(Score5=70.750319,J_long3=74.327865,J_long=83.315002,Short4=65.384)
BASELINES = dict(RandomDetail=OUT/'STEP500_RESULTS.json',AllDetail=OUT/'ALL_DETAIL500_RESULTS.json',
    Nested_Ds_low=LOW_EXP/'RESULTS.json',Nested_Ds_equal=ATOMIC_EXP/'RESULTS.json')
EDITED = {'train/nested_semantic_data.py','train/train_nested_semantic_mask.py'}
RELATED_TEST_COUNT = 81  # Verified prelaunch related CPU tests, 2026-10-07.
ORIGINAL_DIAGNOSTICS = parent.diagnostics


def changed_config(c):
    old = json.loads((ATOMIC_EXP/'config.json').read_text())
    # Trainer runtime config adds provenance, cursor and construction metadata.
    # Only the frozen declared config keys are shared with the input JSON.
    assert old.keys() <= c.keys()
    for k,v in old.items():
        assert c[k]==(MODE if k=='sampling_mode' else v), ('Config drift',k)


def matched_rows(actual,expected):
    assert len(actual)==len(expected)
    count = 0
    for a,b in zip(actual,expected):
        assert a['step']==b['step'] and a['actual_lrs']==b['actual_lrs']
        assert math.isfinite(a['loss']) and a['nonfinite']==0
        old = {h['rank']:h for h in b['rank_health']}
        assert {h['rank'] for h in a['rank_health']}==set(old)=={0,1,2,3}
        for h in a['rank_health']:
            s,ref = h['sampling'],old[h['rank']]['sampling']
            for k in ('sample_ids','sample_id_sha256','full_view_sha256','summary_view_sha256',
                      'fixed_first_reference_stream_sha256'):
                assert s[k]==ref[k], ('Sample/F/Dall drift',a['step'],h['rank'],k)
            assert s['nested_d3_exact'] and h['gradients_finite'] and h['batch']==256
            assert h['updates']==a['step']
            count += len(s['sample_ids'])
    return count


def compare_prefix(actual,unused):
    assert [r['step'] for r in actual]==[1,2,3,4,5]
    count = matched_rows(actual,rows(BASELINE/'steps.jsonl')[:5])
    return dict(passed=True,samples=count,exact_sample_ids_F_Dall_strings_tokens=True,
        D3_ordered_uniform_strict_subset=True,exact_LR=True,cross_run_numeric_comparison=False)


def checkpoint_invariants(current,reference):
    import torch
    c,ref = current['config'],json.loads((BASELINE/'config.json').read_text())
    changed_config(c)
    assert c['resume'] is None and c['start_updates']==0 and c['init_sha256']==STEP0_SHA
    assert current['completed_steps']==5 and current['scheduler_horizon']==4868
    for k in ('component_initialization','data','parameter_counts','horizon','batch_size','world_size','accumulation','runtime_model'):
        assert c[k]==ref[k], ('Frozen construction drift',k)
    changed = {k for k in c['code_sha256'] if c['code_sha256'][k]!=ref['code_sha256'][k]}
    assert changed==EDITED and all(c['code_sha256'][k]==sha(ROOT/k) for k in changed)
    assert current['optimizer']['param_groups']==reference['optimizer']['param_groups']
    assert current['optimizer']['state'].keys()==reference['optimizer']['state'].keys()
    assert {int(s['step']) for s in current['optimizer']['state'].values()}=={5}
    pairs=[(current['model'],reference['model']),(current['adapter'],reference['adapter'])]
    pairs += [(v,reference['optimizer']['state'][k]) for k,v in current['optimizer']['state'].items()]
    for a,b in pairs:
        assert a.keys()==b.keys()
        for k,v in a.items():
            assert v.shape==b[k].shape and v.dtype==b[k].dtype and torch.isfinite(v).all()
    return dict(passed=True,common_step0=True,horizon=4868,optimizer_steps=[5],
        model_adapter_initialization_exact=True,optimizer_groups_order_exact=True,
        objective_architecture_unchanged=True,authorized_source_changes=sorted(EDITED),
        cross_run_numeric_comparison=False)


def sampling_audit():
    import numpy as np
    import torch
    from torch.utils.data import DistributedSampler
    from train.nested_semantic_data import sampled_text_views
    from recovery.s02_full_local_data import FullLocalDataset
    dataset=FullLocalDataset(parent.base.INDEX,parent.IMAGES,MODE,0)
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    prior=json.loads((BASELINE.parent/'sampling-audit-1000.json').read_text())
    evidence=[]
    for rank in range(4):
        sampler=DistributedSampler(dataset,num_replicas=4,rank=rank,shuffle=True,seed=0,drop_last=False)
        sampler.set_epoch(0)
        for index in itertools.islice(iter(sampler),250):
            path=dataset.resolved_path(index)
            record=json.loads(dataset._records[dataset._offsets[index]:dataset._offsets[index+1]])
            sid=index+1000
            new=sampled_text_views(record['caption'],MODE,0,0,sid)
            old=sampled_text_views(record['caption'],'nested_detail',0,0,sid)
            repeat=sampled_text_views(record['caption'],MODE,0,0,sid)
            assert new['views'][:2]==old['views'][:2] and new['valid']==old['valid']
            assert torch.equal(new['tokens_f'],old['tokens_f']) and torch.equal(new['tokens_o'],old['tokens_o'])
            assert new['views']==repeat['views'] and torch.equal(new['tokens_e'],repeat['tokens_e'])
            m=new['detail_pool_size']; selected=new['detail_indices']
            assert len(selected)==(min(3,m-1) if m>=2 else m)
            assert selected==sorted(set(selected)) and set(selected)<=set(new['dall_indices']) and 0 not in selected
            if new['valid']:
                assert new['views'][2]=='. '.join(new['views'][0].split('. ')[j] for j in selected)
            else:
                assert new['views'][1:]==[None,None] and not new['tokens_o'].any() and not new['tokens_e'].any()
            row=dict(rank=rank,sample_id=sid,epoch=0,actual_path=str(path),valid=new['valid'],
                strings=dict(zip(('F','Dall','D3'),new['views'])),
                token_ids={v:new[k].tolist() for v,k in [('F','tokens_f'),('Dall','tokens_o'),('D3','tokens_e')]},
                sentence_indices=dict(Dall=new['dall_indices'],D3=selected),m=m,K_eff=len(selected),
                strict_subset=new['valid'] and len(selected)<m,degenerate=m<=1,
                token_lengths=new['untruncated_lengths'])
            ref=prior[len(evidence)]
            assert sid==ref['sample_id'] and rank==ref['rank']
            assert all(row['strings'][v]==ref['strings'][v] and row['token_ids'][v]==ref['token_ids'][v] for v in ('F','Dall'))
            evidence.append(row)
    current=np.random.get_state()
    assert before[0]==random.getstate() and torch.equal(before[2],torch.get_rng_state())
    assert before[1][0]==current[0] and np.array_equal(before[1][1],current[1]) and before[1][2:]==current[2:]
    assert len(evidence)==1000
    raw=RUN/'sampling-audit-1000.json';dump(raw,evidence)
    audit=dict(passed=True,records=1000,valid_records=sum(r['valid'] for r in evidence),
        F_Dall_strings_tokens_exact_prior_atomic_equal=True,strict_subset_all_m_ge2=True,
        global_python_numpy_torch_RNG_unchanged=True,repeated_seed_epoch_sample_deterministic=True,
        invalid_and_single_detail_fallback_unchanged=True,
        K_eff_histogram=dict(Counter(r['K_eff'] for r in evidence)),
        degenerate_samples=sum(r['degenerate'] for r in evidence),
        degenerate_ratio=sum(r['degenerate'] for r in evidence)/1000,
        sampling_algorithm='SHA256(seed:epoch:sample_id:nested_detail_d3_v1); private Random.sample; sorted indices',
        zero_based_sentence_indices=True,raw_evidence=dict(path=str(raw),bytes=raw.stat().st_size,sha256=sha(raw),uploaded=False),
        examples=evidence[:8])
    dump(EXP/'SAMPLING_AUDIT.json',audit);return audit


def gradient_pressure(g):
    means=g['mean_gradient_norms']; low='D3' if 'D3' in means else 'Ds'
    ratios=dict(D3_Dall=means[low]/means['Dall'],D3_F=means[low]/means['F'])
    votes=sum(b['gradient_norms'][low]>b['gradient_norms']['F']+b['gradient_norms']['Dall'] for b in g['batches'])
    return dict(mean_norms={('D3' if k=='Ds' else k):v for k,v in means.items()},ratios=ratios,
        D3_larger_than_F_plus_Dall_batches=votes,
        dominant=ratios['D3_Dall']>2 and ratios['D3_F']>2 and votes>=6,
        rule='Both mean ratios>2 and lowest-view norm>F+Dall norms in at least6/8 same frozen batches',
        scope='Per-view native-backbone norm magnitudes, not a signed decomposition of total gradient')


def classify(percent,metrics,evidence=None):
    e=evidence or {};urban={r:round(100*metrics['Urban-1k'][r]['R@1'],1) for r in ('I2T','T2I')}
    score,long_score=percent['Score5'],percent['J_long3']
    ce_improved=e.get('CE_ratio_vs_atomic_equal',float('inf'))<=.75
    loss_improved=e.get('alignment_share_drop_pp',-float('inf'))>=10
    gradient_improved=e.get('both_gradient_ratio_improvement',False)
    not_dominant=not e.get('gradient_dominant',True)
    strong=score>=SCORES['Score5']-1e-6 and long_score>=SCORES['J_long3']-1e-6 and urban['T2I']>=88.5 and loss_improved and not_dominant
    near=score>=SCORES['Score5']-.2 and long_score>=SCORES['J_long3']-.2 and urban['T2I']>=88.2
    optimization=near and ce_improved and loss_improved and gradient_improved and not_dominant
    degraded=score<SCORES['Score5']-.2 or long_score<SCORES['J_long3']-.2 or urban['T2I']<88.2
    negative=degraded and e.get('gradient_dominant',False)
    status=('D3_EQUAL_WEIGHT_STRONG_POSITIVE' if strong else 'D3_OPTIMIZATION_POSITIVE' if optimization else
            'D3_EQUAL_WEIGHT_NEGATIVE' if negative else 'D3_TRADEOFF')
    return dict(status=status,Urban_R1_percent=urban,retrieval_near_low_Ds=near,
        CE_significantly_lower=ce_improved,alignment_contribution_improved=loss_improved,
        gradient_balance_improved=gradient_improved,gradient_not_dominant=not_dominant,
        thresholds=dict(Score5_strong_min=SCORES['Score5'],J_long3_strong_min=SCORES['J_long3'],Urban_T2I_strong_min=88.5,
            near_max_aggregate_decline_pp=.2,near_max_Urban_T2I_decline_pp=.3,CE_reduction_min=.25,
            alignment_share_drop_min_pp=10,gradient_ratio_reduction_min=.25),
        tradeoff_scope='Mixed retrieval or optimization/retrieval results outside the positive/negative gates',
        automatic_continuation=False,automatic_new_experiments=False)


def comparison_evidence(diag,g):
    old=json.loads((ATOMIC_EXP/'TRAINING_DIAGNOSTICS.json').read_text())
    pressure=gradient_pressure(g)
    view=diag['last50_views'].get('D3',diag['last50_views'].get('Ds'))
    old_view=old['last50_views']['Ds'];ratios=pressure['ratios']
    return dict(CE_ratio_vs_atomic_equal=view['raw_combined_CE']/old_view['raw_combined_CE'],
        alignment_share_drop_pp=100*(old_view['weighted_alignment_loss_share']-view['weighted_alignment_loss_share']),
        gradient_dominant=pressure['dominant'],
        both_gradient_ratio_improvement=ratios['D3_Dall']<=.75*old['weighted_Ds_Dall_gradient_norm_ratio'] and
            ratios['D3_F']<=.75*old['weighted_Ds_F_gradient_norm_ratio'],gradient_pressure=pressure)


def parent_classify(percent,metrics,unused=False):
    path=EXP/'GRADIENT_SPOTCHECK.json'
    evidence=None
    if path.exists():
        evidence=comparison_evidence(diagnostics(rows(RUN/'step500/steps.jsonl')),json.loads(path.read_text()))
    return classify(percent,metrics,evidence)


def diagnostics(steps):
    diag=ORIGINAL_DIAGNOSTICS(steps)
    total=sum(v['weighted_CE_contribution'] for v in diag['last50_views'].values())
    for v in diag['last50_views'].values():v['weighted_alignment_loss_share']=v['weighted_CE_contribution']/total
    diag['weighted_formula']='10/3*(L_F+L_Dall+L_D3), directional CE summed'
    hist=Counter();m_hist=Counter();sentence_hist=Counter();positions=Counter();records=degenerate=strict=valid=0
    for r in steps:
        for h in r['rank_health']:
            s=h['sampling']['nested_d3_statistics']
            hist.update(s['K_eff_histogram']);m_hist.update(s['m_histogram']);sentence_hist.update(s['D3_sentence_count_histogram'])
            positions.update(s['selected_sentence_position_histogram_1based'])
            records+=s['records'];valid+=s['valid_samples'];degenerate+=s['degenerate_samples'];strict+=s['strict_subset_samples']
    diag['sampling'].update(total_records=records,D3_mean_sentences=sum(int(k)*v for k,v in sentence_hist.items())/valid,
        K_eff_histogram=dict(hist),m_histogram=dict(m_hist),degenerate_samples=degenerate,
        degenerate_ratio=degenerate/records,strict_subset_samples=strict,
        D3_selected_sentence_position_histogram_1based=dict(positions))
    diag['sampling'].pop('Ds_sentence_position_histogram_1based',None)
    diag['alignment_loss_share_definition']='Last50 mean weighted CE divided by sum of view mean weighted CE, excluding sparsity/inclusion'
    return diag


def configure():
    parent.EXP,parent.RUN,parent.PHASE,parent.CONFIG,parent.BASELINE=EXP,RUN,PHASE,CONFIG,BASELINE
    parent.WEIGHTS,parent.EDITED_SOURCES=WEIGHTS,EDITED
    parent.changed_config,parent.compare_prefix,parent.checkpoint_invariants=changed_config,compare_prefix,checkpoint_invariants
    parent.sampling_audit,parent.diagnostics,parent.classify=sampling_audit,diagnostics,parent_classify
    parent.configure()


class Supervisor(parent.Supervisor):
    def execute(self,name,command,training=False):
        command=list(command)
        if name=='train500':
            assert RELATED_TEST_COUNT>0
            launch=json.loads((RUN/'launch-provenance.json').read_text())
            baseline=json.loads((ATOMIC_EXP/'RESULTS.json').read_text())
            previous=json.loads((BASELINE/'config.json').read_text())['code_sha256']
            assert all(sha(ROOT/p)==v for p,v in previous.items() if p not in EDITED)
            changed_config(json.loads(CONFIG.read_text()))
            launch['source_sha256'].update({p:sha(ROOT/p) for p in previous})
            extras=['recovery/nested_detail_d3_equal500.py','recovery/nested_detail_d3_gradients.py',
                'recovery/review_nested_detail_d3_equal500.py','tests/test_nested_detail_d3.py']
            launch['source_sha256'].update({p:sha(ROOT/p) for p in extras})
            launch.update(method_changes=dict(only='lowest_text_view_sampling',old='Ds:one sentence',
                new='D3:ordered uniform min(3,m-1) subset for m>=2; original m<=1 fallback',weights=[1,1,1]),
                related_unit_tests=dict(passed=RELATED_TEST_COUNT),
                isolation_proof=dict(config_only_sampling_mode_changed=True,objective_model_sources_unchanged=True,
                    loader_optimizer_architecture_unchanged=True,baseline_checkpoint_used_for_resume=False,
                    baseline_checkpoint_sha256=baseline['checkpoint_sha256'],baseline_steps_path=str(BASELINE/'steps.jsonl'),
                    baseline_steps_sha256=sha(BASELINE/'steps.jsonl')))
            failed=json.loads((FAILED_ATTEMPT/'supervisor-result.json').read_text())
            assert failed['completed_steps']==5 and failed['status']=='INCOMPLETE_HARD_STOP'
            launch['prior_aborted_attempt']=dict(path=str(FAILED_ATTEMPT),completed_steps=5,
                reason='Admission checker required runtime-expanded config keys to equal input-config keys',
                sampling_prefix_passed=True,method_changed=False,resumed=False,
                fresh_common0_restart=True,supervisor_result_sha256=sha(FAILED_ATTEMPT/'supervisor-result.json'),
                checkpoint5_path=str(FAILED_ATTEMPT/'step500/step000005.pt'))
            old_command=json.loads((FAILED_ATTEMPT/'commands.json').read_text())[0]
            launch['prior_aborted_attempt']['local_assets']=[dict(path=str(p),bytes=p.stat().st_size,
                sha256=sha(p),time_range_utc=[old_command['started_utc'],old_command['ended_utc']],uploaded=False)
                for p in (FAILED_ATTEMPT/'train500.log',FAILED_ATTEMPT/'step500/step000005.pt')]
            dump(RUN/'launch-provenance.json',launch)
            command[command.index('-m')+1]='recovery.nested_detail_d3_equal500'
        if name=='gradient500':command[command.index('-m')+1]='recovery.nested_detail_d3_gradients'
        return super().execute(name,command,training)

    def full_stream_proof(self):
        actual,expected=rows(self.train/'steps.jsonl'),rows(BASELINE/'steps.jsonl')
        assert [r['step'] for r in actual]==list(range(1,501))
        count=matched_rows(actual,expected);assert count==512000
        return dict(passed=True,records=count,all500_sample_ids_F_Dall_strings_tokens_exact=True,
            all500_D3_ordered_strict_subset_or_original_fallback=True,exact_LR=True,
            baseline_steps_path=str(BASELINE/'steps.jsonl'),baseline_steps_sha256=sha(BASELINE/'steps.jsonl'),
            first5_gate='BEFORE_UPDATE6')

    def report(self):
        super().report();update_reports()


def rename_lowest(value):
    if isinstance(value,dict):return {k.replace('Ds','D3'):rename_lowest(v) for k,v in value.items()}
    if isinstance(value,list):return [rename_lowest(v) for v in value]
    if isinstance(value,str):return value.replace('Ds','D3')
    return value


def update_reports():
    result=json.loads((EXP/'RESULTS.json').read_text())
    diag=rename_lowest(json.loads((EXP/'TRAINING_DIAGNOSTICS.json').read_text()))
    hierarchy=rename_lowest(json.loads((EXP/'MASK_HIERARCHY_AUDIT.json').read_text()))
    gradient_path=EXP/'GRADIENT_SPOTCHECK.json'
    if gradient_path.exists():
        raw=json.loads(gradient_path.read_text());g=rename_lowest(raw)
        if 'Ds' in raw['mean_gradient_norms']:dump(RUN/'gradient-spotcheck-internal.json',raw)
        pressure=gradient_pressure(g);g['mean_norm_ratios']=pressure['ratios']
        g['D3_larger_than_F_plus_Dall_batches']=pressure['D3_larger_than_F_plus_Dall_batches']
        old_g=json.loads((ATOMIC_EXP/'GRADIENT_SPOTCHECK.json').read_text())
        assert [b['global_sample_ids_sha256'] for b in g['batches']]==[b['global_sample_ids_sha256'] for b in old_g['batches']]
        assert all(g[k]==old_g[k] for k in ('fixed_global_batches','global_batch_size','selection','backbone_scope','gradients','native_reference'))
        diag['gradient_protocol_exact_atomic_equal']=True
        diag['comparison_vs_atomic_equal']=comparison_evidence(diag,g)
        diag['gradient_pressure']=pressure
        dump(gradient_path,g)
    for label,old_path in [('Nested_Ds_equal',ATOMIC_EXP),('Nested_Ds_low',LOW_EXP)]:
        old=rename_lowest(json.loads((old_path/'TRAINING_DIAGNOSTICS.json').read_text()))
        hierarchy.setdefault('delta_vs_baselines',{})[label]={k:v-old['last50_hierarchy'][k] for k,v in diag.get('last50_hierarchy',{}).items()}
        hierarchy.setdefault('keep_ratio_delta_vs_baselines',{})[label]={k:v['keep_ratio']-old['last50_views'][k]['keep_ratio'] for k,v in diag.get('last50_views',{}).items()}
    if result.get('metrics'):
        result['baseline_scores_percent']={k:json.loads(p.read_text())['scores_percent'] for k,p in BASELINES.items()}
        result['delta_vs_baselines_pp']={k:{s:result['scores_percent'][s]-old[s] for s in SCORES} for k,old in result['baseline_scores_percent'].items()}
        result['dataset_delta_vs_baselines_pp']={k:{d:{direction:{s:100*(v[direction][s]-old['metrics'][d][direction][s])
            for s in ('R@1','R@5','R@10')} for direction in ('I2T','T2I')} for d,v in result['metrics'].items()}
            for k,p in BASELINES.items() for old in [json.loads(p.read_text())]}
        result['decision']=classify(result['scores_percent'],result['metrics'],diag['comparison_vs_atomic_equal'])
        result['status']=result['decision']['status']
    result['internal_slot_mapping']=dict(F='F',O='Dall',E='D3',legacy_Ds_metrics='D3 in this run only')
    dump(EXP/'RESULTS.json',result);dump(EXP/'TRAINING_DIAGNOSTICS.json',diag);dump(EXP/'MASK_HIERARCHY_AUDIT.json',hierarchy)
    audit=json.loads((EXP/'SAMPLING_AUDIT.json').read_text())
    audit['full500_sampling_statistics']=diag.get('sampling');dump(EXP/'SAMPLING_AUDIT.json',audit)
    write_report(result,diag,hierarchy)


def write_report(r,d,h):
    lines=['# Nested D3 equal-weight matched500 experiment','',f'Status: `{r["status"]}`; {r["completed_steps"]}/500 updates, horizon4868; no automatic follow-up.',
        'Relative to atomic equal-weight: only lowest view Ds -> D3 changes. F/Dall packing, tokens, sample order, all model/objective/optimizer/LR/loader settings frozen.',
        'D3 uses ordered uniform sampling without replacement: K=min(3,m-1) for m>=2; m=1 retains its single detail; m=0 retains invalid local-view masking. No fabricated/duplicated/split sentences.',
        'Private RNG: SHA256(seed:epoch:sample_id:nested_detail_d3_v1), Random.sample, sorted selected indices. No global RNG advancement.',
        f'Fresh common0 SHA256: `{STEP0_SHA}`. Local-only `{parent.IMAGES}`; missing/symlink/escape fails; no fallback. Ephemeral cache; NFS originals retained.',
        'Alignment10/3*(L_F+L_Dall+L_D3), each directional-summed CE. Detached-child chain Dall->F,D3->Dall only,200-step ramp,max1. Sparsity(Omega_F+2*Omega_Dall+2*Omega_D3)/3.',
        'Frozen model logs use O/E and legacy Ds names internally; reports map lowest slot to D3 without changing the graph.',
        'An initial five-update attempt stopped because the admission checker rejected trainer-added runtime metadata. All five sample/F/Dall checks passed. Checker fixed and regression-tested; the formal r1 run restarted from common0, never resumed that checkpoint. Prior artifacts retained locally in launch_provenance.prior_aborted_attempt.',
        '', '| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |','|---|---|---|']
    for dataset,v in r.get('metrics',{}).items():
        values=[' / '.join(f'{100*v[q][k]:.6f}' for k in ('R@1','R@5','R@10')) for q in ('I2T','T2I')]
        lines.append(f'| {dataset} | {values[0]} | {values[1]} |')
    lines += ['', '| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |','|---|---:|---:|---:|---:|---:|---:|']
    for name,path in BASELINES.items():
        old=json.loads(path.read_text());p=old['scores_percent'];u=old['metrics']['Urban-1k']
        lines.append('| '+name+' | '+' | '.join(f'{p[k]:.6f}' for k in SCORES)+f' | {100*u["I2T"]["R@1"]:.3f} | {100*u["T2I"]["R@1"]:.3f} |')
    if r.get('metrics'):
        p=r['scores_percent'];u=r['decision']['Urban_R1_percent']
        lines.append('| Nested D3 equal | '+' | '.join(f'{p[k]:.6f}' for k in SCORES)+f' | {u["I2T"]:.3f} | {u["T2I"]:.3f} |')
        lines += ['', '| Baseline | ΔScore5 | ΔJ_long3 | ΔJ_long | ΔShort4 | ΔUrban I2T | ΔUrban T2I |','|---|---:|---:|---:|---:|---:|---:|']
        for name,dd in r['delta_vs_baselines_pp'].items():
            uu=r['dataset_delta_vs_baselines_pp'][name]['Urban-1k']
            lines.append('| '+name+' | '+' | '.join(f'{dd[k]:+.6f}' for k in SCORES)+f' | {uu["I2T"]["R@1"]:+.3f} | {uu["T2I"]["R@1"]:+.3f} |')
        for name,dd in r['dataset_delta_vs_baselines_pp'].items():
            lines += ['',f'## All recall deltas vs {name}','', '| Dataset | ΔI2T R@1 / R@5 / R@10 (pp) | ΔT2I R@1 / R@5 / R@10 (pp) |','|---|---|---|']
            for dataset,v in dd.items():
                values=[' / '.join(f'{v[q][k]:+.6f}' for k in ('R@1','R@5','R@10')) for q in ('I2T','T2I')]
                lines.append(f'| {dataset} | {values[0]} | {values[1]} |')
    if d.get('last50_views'):
        lines += ['', '| View | I2T CE | T2I CE | Combined CE | Alignment share | Keep ratio |','|---|---:|---:|---:|---:|---:|']
        for v,s in d['last50_views'].items():lines.append(f'| {v} | {s["i2t"]:.6f} | {s["t2i"]:.6f} | {s["raw_combined_CE"]:.6f} | {100*s["weighted_alignment_loss_share"]:.3f}% | {s["keep_ratio"]:.6f} |')
        lines += ['', 'CE_D3/CE_Dall: '+str(d['D3_CE_ratio_to_Dall'])+'. Shares exclude inclusion/sparsity.',
            'Atomic equal-weight comparison: `'+json.dumps(d.get('comparison_vs_atomic_equal'))+'`.',
            'Full500 sampling: `'+json.dumps(d['sampling'])+'`.',
            'Last50 masks: `'+json.dumps(h.get('last50'))+'`.',
            'Mask deltas: `'+json.dumps(h.get('delta_vs_baselines'))+'`.',
            'Keep deltas: `'+json.dumps(h.get('keep_ratio_delta_vs_baselines'))+'`.',
            'Soft chain objective unchanged; nonzero hard-mask violations are measured, never projected away.',
            'Gradient protocol: same8 fixed seed0 epoch0 global batches, step500, exact native-backbone optimizer group; raw per-view gradients and native full-caption reference. No optimizer updates.']
    lines += ['', 'Decision: `'+json.dumps(r.get('decision'))+'`.',
        'Significant CE reduction>=25%; contribution decrease>=10pp; gradient ratio reduction>=25%; same prior dominance criterion. Retrieval near guard:Score5/J_long3 down<=0.2pp, Urban T2I down<=0.3pp vs low-Ds.',
        'Native inference only normalized image @ normalized full-caption text transpose. No mask/gate/detail/rerank/ensemble.',
        'Strict export: `'+json.dumps(r.get('strict_export'))+'`.',
        'Full stream proof: `'+json.dumps(r.get('stream_proof'))+'`.',
        'Checkpoint completeness, immutable SHA, evaluation manifests/protocols and launch Git source snapshot checked in VALIDATION.json/RESULTS.json.',
        'Raw logs/checkpoints/bare/cache/1000-record raw audit remain local; path/bytes/SHA256/time inventories in RUNTIME_STATS.json/SAMPLING_AUDIT.json. Cgroup memory includes page cache, not RSS or OOM evidence.',
        'GitHub:experiment/nested-detail-d3-equal500, reviewed small code/config/tests/reports only. No full4868, K/weight/sparsity/sampling/inclusion sweeps.']
    (EXP/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true')
    args,remaining=parser.parse_known_args();configure()
    if args.worker:
        from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_train_gate as gate
        gate.compare_prefix,gate.checkpoint_invariants=compare_prefix,checkpoint_invariants
        sys.argv=[sys.argv[0],*remaining];parent.base.worker()
    else:
        assert not remaining
        def stop(sig,frame):raise RuntimeError('HARD_STOP: supervisor signal '+str(sig))
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        Supervisor().run()


if __name__=='__main__':main()
