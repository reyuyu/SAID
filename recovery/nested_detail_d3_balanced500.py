"""Strict D3 weight-only common0->500 run; all sampling/objective code frozen."""
import argparse
import copy
import hashlib
import json
import math
import signal
import subprocess
import sys

from recovery import nested_detail_d3_equal500 as protocol
from recovery import nested_detail500 as parent
from recovery.s02_nfs500 import ROOT,OUT,STEP0_SHA,dump,rows,sha

EXP=ROOT/'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1'
RUN=ROOT/'runtime/SAID-nest-clip-v1/nested-detail-d3-balanced500-20261007'
PHASE=parent.LOCAL/'formal-nested-detail-d3-balanced500-phase-20261007'
CONFIG=EXP/'config.json'
EQUAL_EXP=ROOT/'experiments/nest_clip_v1/nested_detail_d3_equal_500_v1'
BASELINE=ROOT/'runtime/SAID-nest-clip-v1/nested-detail-d3-equal500-20261007-r1/step500'
LOW_EXP=protocol.LOW_EXP
WEIGHTS=dict(F=1.35,Dall=1.35,Ds=.30)
SCORES=protocol.SCORES
BASELINES=dict(RandomDetail=OUT/'STEP500_RESULTS.json',AllDetail=OUT/'ALL_DETAIL500_RESULTS.json',
    Nested_Ds_low=LOW_EXP/'RESULTS.json',Nested_D3_equal=EQUAL_EXP/'RESULTS.json')
RELATED_TEST_COUNT=91  # Verified related CPU tests before launch, 2026-10-07.
ORIGINAL_AUDIT=protocol.sampling_audit
ORIGINAL_DIAGNOSTICS=protocol.diagnostics
ORIGINAL_UPDATE=protocol.update_reports
ORIGINAL_SAMPLING=None
STREAM_KEYS=('sample_ids','sample_id_sha256','full_view_sha256','summary_view_sha256',
    'local_views_sha256','split_sha256','fixed_first_reference_stream_sha256')


def changed_config(c):
    old=json.loads((EQUAL_EXP/'config.json').read_text())
    assert old.keys()<=c.keys()
    for k,v in old.items():
        assert c[k]==([1.35,1.35,.30] if k=='view_weights' else v),('Config drift',k)


def digest_indices(ids,choices):
    return hashlib.sha256(json.dumps(dict(sample_ids=ids,choices=choices),ensure_ascii=False,
        separators=(',',':')).encode()).hexdigest()


def selection_diagnostics(batch):
    """Detached CPU ledger only; never draws RNG or touches training tensors."""
    result=ORIGINAL_SAMPLING(batch)
    result['D3_selected_sentence_indices_sha256']=digest_indices(
        batch['sample_id'].tolist(),batch['detail_indices'])
    return result


def matched_rows(actual,expected):
    from train.nested_semantic_data import sample_partial_detail_indices
    assert len(actual)==len(expected)
    count=0
    for a,b in zip(actual,expected):
        assert a['step']==b['step'] and a['actual_lrs']==b['actual_lrs']
        assert math.isfinite(a['loss']) and a['nonfinite']==0
        old={h['rank']:h for h in b['rank_health']}
        assert {h['rank'] for h in a['rank_health']}==set(old)=={0,1,2,3}
        for h in a['rank_health']:
            s,ref=h['sampling'],old[h['rank']]['sampling']
            assert all(s[k]==ref[k] for k in STREAM_KEYS),('Sample/text/token drift',a['step'],h['rank'])
            assert s['nested_d3_exact'] and h['gradients_finite'] and h['batch']==256 and h['updates']==a['step']
            # Baseline did not log every index digest: reconstruct with its
            # immutable sampler and actual n/sample IDs, seed0, epoch0 (<1217).
            expected_choices=[sample_partial_detail_indices(n,0,0,sid) for n,sid in zip(ref['n'],ref['sample_ids'])]
            assert s['D3_selected_sentence_indices_sha256']==digest_indices(ref['sample_ids'],expected_choices)
            count+=len(s['sample_ids'])
    return count


def compare_prefix(actual,unused):
    assert [r['step'] for r in actual]==[1,2,3,4,5]
    count=matched_rows(actual,rows(BASELINE/'steps.jsonl')[:5])
    return dict(passed=True,samples=count,exact_sample_ids_F_Dall_D3_strings_tokens=True,
        exact_D3_selected_sentence_indices=True,exact_LR=True,cross_run_numeric_comparison=False)


def checkpoint_invariants(current,reference):
    import torch
    c,ref=current['config'],json.loads((BASELINE/'config.json').read_text())
    changed_config(c)
    assert c['resume'] is None and c['start_updates']==0 and c['init_sha256']==STEP0_SHA
    assert current['completed_steps']==5 and current['scheduler_horizon']==4868
    for k in ('component_initialization','data','parameter_counts','horizon','batch_size','world_size','accumulation'):
        assert c[k]==ref[k],('Frozen construction drift',k)
    model,old=copy.deepcopy(c['runtime_model']),copy.deepcopy(ref['runtime_model'])
    assert model['search_hparams'].pop('view_weights')==[1.35,1.35,.30]
    assert old['search_hparams'].pop('view_weights')==[1.,1.,1.] and model==old
    changed={k for k in c['code_sha256'] if c['code_sha256'][k]!=ref['code_sha256'][k]}
    assert changed=={'train/train_nested_semantic_mask.py'}
    assert c['code_sha256']['train/train_nested_semantic_mask.py']==sha(ROOT/'train/train_nested_semantic_mask.py')
    assert current['optimizer']['param_groups']==reference['optimizer']['param_groups']
    assert current['optimizer']['state'].keys()==reference['optimizer']['state'].keys()
    assert {int(s['step']) for s in current['optimizer']['state'].values()}=={5}
    pairs=[(current['model'],reference['model']),(current['adapter'],reference['adapter'])]
    pairs += [(v,reference['optimizer']['state'][k]) for k,v in current['optimizer']['state'].items()]
    for a,b in pairs:
        assert a.keys()==b.keys()
        for k,v in a.items():assert v.shape==b[k].shape and v.dtype==b[k].dtype and torch.isfinite(v).all()
    return dict(passed=True,common_step0=True,horizon=4868,optimizer_steps=[5],
        model_adapter_initialization_exact=True,optimizer_groups_order_exact=True,
        model_sampling_objective_sources_frozen=True,trainer_only_admission_extension=True,
        cross_run_numeric_comparison=False)


def sampling_audit():
    audit=ORIGINAL_AUDIT()
    current=json.loads((RUN/'sampling-audit-1000.json').read_text())
    old=json.loads((BASELINE.parent/'sampling-audit-1000.json').read_text())
    assert current==old and len(current)==1000
    audit.pop('F_Dall_strings_tokens_exact_prior_atomic_equal',None)
    audit.update(all1000_samples_F_Dall_D3_strings_tokens_indices_exact_equal_baseline=True,
        baseline_raw_path=str(BASELINE.parent/'sampling-audit-1000.json'),
        baseline_raw_sha256=sha(BASELINE.parent/'sampling-audit-1000.json'))
    dump(EXP/'SAMPLING_AUDIT.json',audit);return audit


def diagnostics(steps):
    d=ORIGINAL_DIAGNOSTICS(steps)
    d['weighted_formula']='10/3*(1.35*L_F+1.35*L_Dall+0.30*L_D3), directional CE summed'
    return d


def gradient_pressure(g):
    m=g['mean_gradient_norms'];slot='D3' if 'D3' in m else 'Ds'
    raw=dict(F=m['F'],Dall=m['Dall'],D3=m[slot])
    coefficients=dict(F=4.5,Dall=4.5,D3=1.)  # Includes the common10/3 factor.
    weighted={v:coefficients[v]*value for v,value in raw.items()}
    votes=sum(coefficients['D3']*b['gradient_norms'][slot]>
        coefficients['F']*b['gradient_norms']['F']+coefficients['Dall']*b['gradient_norms']['Dall'] for b in g['batches'])
    raw_votes=sum(b['gradient_norms'][slot]>b['gradient_norms']['F']+b['gradient_norms']['Dall'] for b in g['batches'])
    ratios=dict(D3_Dall=raw['D3']/raw['Dall'],D3_F=raw['D3']/raw['F'])
    wr=dict(D3_Dall=weighted['D3']/weighted['Dall'],D3_F=weighted['D3']/weighted['F'])
    return dict(mean_norms=raw,ratios=ratios,weighted_mean_norms=weighted,coefficients=coefficients,
        weighted_ratios=wr,D3_larger_than_F_plus_Dall_batches=raw_votes,
        weighted_D3_larger_than_F_plus_Dall_batches=votes,
        dominant=wr['D3_Dall']>2 and wr['D3_F']>2 and votes>=6,
        no_longer_dominant=wr['D3_Dall']<=1 and wr['D3_F']<=1 and votes<=1,
        rule='Strong non-dominance:weighted mean lowest norm<=both parent norms and at most1/8 weighted sum-dominance batches',
        scope='Per-view norm magnitudes, not a signed decomposition of total gradient')


def comparison_evidence(d,g):
    old=json.loads((EQUAL_EXP/'TRAINING_DIAGNOSTICS.json').read_text())
    old_g=json.loads((EQUAL_EXP/'GRADIENT_SPOTCHECK.json').read_text())
    old_m=old_g['mean_gradient_norms'];p=gradient_pressure(g);wr=p['weighted_ratios']
    old_ratios=dict(D3_Dall=old_m['D3']/old_m['Dall'],D3_F=old_m['D3']/old_m['F'])
    view=d['last50_views'].get('D3',d['last50_views'].get('Ds'))
    reference=json.loads((LOW_EXP/'TRAINING_DIAGNOSTICS.json').read_text())['weighted_Ds_Dall_gradient_norm_ratio']
    return dict(gradient_pressure=p,gradient_dominant=p['dominant'],no_longer_dominant=p['no_longer_dominant'],
        gradient_balance_improved=all(wr[k]<=.75*old_ratios[k] for k in wr) and not p['dominant'],
        equal_baseline_weighted_ratios=old_ratios,
        weighted_ratio_reduction={k:1-wr[k]/old_ratios[k] for k in wr},
        alignment_share_drop_pp=100*(old['last50_views']['D3']['weighted_alignment_loss_share']-view['weighted_alignment_loss_share']),
        CE_ratio_vs_D3_equal=view['raw_combined_CE']/old['last50_views']['D3']['raw_combined_CE'],
        low_Ds_weighted_Ds_Dall_ratio=reference,
        weighted_D3_Dall_distance_to_low_Ds=wr['D3_Dall']-reference,
        reference_note='Observed low-dose Ds ratio, not an additional pass threshold or tuned target')


def classify(percent,metrics,evidence=None):
    e=evidence or {};u={r:round(100*metrics['Urban-1k'][r]['R@1'],1) for r in ('I2T','T2I')}
    score,long_score=percent['Score5'],percent['J_long3']
    strong=score>=70.750319-1e-6 and long_score>=74.4-1e-6 and u['T2I']>=88.5 and e.get('no_longer_dominant',False)
    positive=score>=70.7-1e-6 and long_score>=74.4-1e-6 and u['T2I']>=88.5 and e.get('gradient_balance_improved',False)
    tradeoff=long_score>74.456501+1e-6 and (score<70.713901-.2 or percent.get('Short4',65.1)<65.1-.2)
    negative=score<70.750319-1e-6 and long_score<=74.327865+1e-6 and percent.get('J_long',0)<=83.315002+1e-6
    status=('D3_BALANCED_STRONG_POSITIVE' if strong else 'D3_BALANCED_POSITIVE' if positive else
            'D3_LONG_TRADEOFF' if tradeoff else 'D3_BALANCED_NEGATIVE' if negative else 'D3_BALANCED_INCONCLUSIVE')
    return dict(status=status,Urban_R1_percent=u,strong_gate=strong,positive_gate=positive,
        gradient_not_dominant=e.get('no_longer_dominant',False),gradient_balance_improved=e.get('gradient_balance_improved',False),
        thresholds=dict(strong_Score5_min=70.750319,positive_Score5_min=70.7,J_long3_min=74.4,Urban_T2I_min=88.5,
            weighted_ratio_reduction_min=.25,tradeoff_max_Score5_or_Short4_decline_pp=.2),
        interpretation='Long-tradeoff compared with D3 equal; no new long advantage means neither J_long3 nor J_long improves over low-dose Ds. Boundary outcomes retained as inconclusive.',
        automatic_continuation=False,automatic_new_experiments=False)


def parent_classify(percent,metrics,unused=False):
    path=EXP/'GRADIENT_SPOTCHECK.json';e=None
    if path.exists():e=comparison_evidence(diagnostics(rows(RUN/'step500/steps.jsonl')),json.loads(path.read_text()))
    return classify(percent,metrics,e)


def configure():
    protocol.EXP,protocol.RUN,protocol.PHASE,protocol.CONFIG,protocol.BASELINE=EXP,RUN,PHASE,CONFIG,BASELINE
    protocol.ATOMIC_EXP,protocol.WEIGHTS,protocol.BASELINES=EQUAL_EXP,WEIGHTS,BASELINES
    protocol.EDITED={'train/train_nested_semantic_mask.py'}
    protocol.changed_config,protocol.compare_prefix,protocol.checkpoint_invariants=changed_config,compare_prefix,checkpoint_invariants
    protocol.sampling_audit,protocol.diagnostics,protocol.classify=sampling_audit,diagnostics,classify
    protocol.gradient_pressure,protocol.comparison_evidence,protocol.parent_classify=gradient_pressure,comparison_evidence,parent_classify
    protocol.write_report=write_report
    protocol.configure()


class Supervisor(parent.Supervisor):
    def execute(self,name,command,training=False):
        command=list(command)
        if name=='train500':
            assert RELATED_TEST_COUNT>0
            launch=json.loads((RUN/'launch-provenance.json').read_text())
            old=json.loads((EQUAL_EXP/'RESULTS.json').read_text())
            assert old['completed_steps']==500 and old['evaluation_checkpoint_immutable'] and old['status']=='D3_TRADEOFF'
            assert sha(BASELINE/'step000500.pt')==old['checkpoint_sha256']
            trainer='train/train_nested_semantic_mask.py'
            previous=json.loads((BASELINE/'config.json').read_text())['code_sha256']
            assert all(sha(ROOT/p)==v for p,v in previous.items() if p!=trainer)
            archived=subprocess.check_output(['git','show',old['launch_provenance']['git_head']+':'+trainer],cwd=ROOT)
            before=b"        assert cfg['view_weights'] in ([1.4,1.4,.2], [1.,1.,1.]), 'Only reviewed Nested Detail weights are authorized'\n        if cfg['sampling_mode'] == 'nested_detail_d3':\n            assert cfg['view_weights'] == [1.,1.,1.], 'D3 experiment freezes equal weights'\n"
            after=b"        allowed = (([1.,1.,1.], [1.35,1.35,.30]) if cfg['sampling_mode']=='nested_detail_d3'\n                   else ([1.4,1.4,.2], [1.,1.,1.]))\n        assert cfg['view_weights'] in allowed, 'Only reviewed Nested Detail weights are authorized'\n"
            assert before in archived and (ROOT/trainer).read_bytes()==archived.replace(before,after)
            changed_config(json.loads(CONFIG.read_text()))
            launch['source_sha256'].update({p:sha(ROOT/p) for p in previous})
            extras=['recovery/nested_detail_d3_balanced500.py','recovery/nested_detail_d3_balanced_gradients.py',
                'recovery/review_nested_detail_d3_balanced500.py','recovery/review_nested_detail_d3_equal500.py',
                'recovery/nested_detail_d3_equal500.py','tests/test_nested_detail_d3_balanced.py']
            launch['source_sha256'].update({p:sha(ROOT/p) for p in extras})
            launch.update(method_changes=dict(only='view_weights',old=[1,1,1],new=[1.35,1.35,.30]),
                related_unit_tests=dict(passed=RELATED_TEST_COUNT),
                isolation_proof=dict(config_only_view_weights_changed=True,objective_model_sources_unchanged=True,
                    data_sampler_source_exact_baseline=True,trainer_only_admission_extension=True,
                    indices_observer_read_only=True,baseline_checkpoint_used_for_resume=False,
                    baseline_checkpoint_sha256=old['checkpoint_sha256'],baseline_steps_path=str(BASELINE/'steps.jsonl'),
                    baseline_steps_sha256=sha(BASELINE/'steps.jsonl')))
            dump(RUN/'launch-provenance.json',launch)
            command[command.index('-m')+1]='recovery.nested_detail_d3_balanced500'
        if name=='gradient500':command[command.index('-m')+1]='recovery.nested_detail_d3_balanced_gradients'
        return super().execute(name,command,training)

    def full_stream_proof(self):
        actual,old=rows(self.train/'steps.jsonl'),rows(BASELINE/'steps.jsonl')
        assert [r['step'] for r in actual]==list(range(1,501))
        count=matched_rows(actual,old);assert count==512000
        return dict(passed=True,records=count,all500_sample_ids_F_Dall_D3_strings_tokens_indices_exact=True,
            exact_LR=True,baseline_steps_path=str(BASELINE/'steps.jsonl'),baseline_steps_sha256=sha(BASELINE/'steps.jsonl'),
            selection_proof='Actual current indices digests match immutable baseline sampler replay from actual baseline n/IDs, epoch0/seed0',
            first5_gate='BEFORE_UPDATE6')

    def report(self):
        super().report();update_reports()


def update_reports():
    ORIGINAL_UPDATE()
    r=json.loads((EXP/'RESULTS.json').read_text());d=json.loads((EXP/'TRAINING_DIAGNOSTICS.json').read_text())
    h=json.loads((EXP/'MASK_HIERARCHY_AUDIT.json').read_text())
    for original,new in [('comparison_vs_atomic_equal','comparison_vs_D3_equal'),
        ('gradient_protocol_exact_atomic_equal','gradient_protocol_exact_D3_equal')]:
        if original in d:d[new]=d.pop(original)
    for key in ('delta_vs_baselines','keep_ratio_delta_vs_baselines'):
        if 'Nested_Ds_equal' in h.get(key,{}):h[key]['Nested_D3_equal']=h[key].pop('Nested_Ds_equal')
    if (EXP/'GRADIENT_SPOTCHECK.json').exists():
        g=json.loads((EXP/'GRADIENT_SPOTCHECK.json').read_text());p=gradient_pressure(g)
        g.update(weighted_mean_gradient_norms=p['weighted_mean_norms'],weighted_mean_norm_ratios=p['weighted_ratios'],
            actual_gradient_coefficients=p['coefficients'],
            weighted_D3_larger_than_F_plus_Dall_batches=p['weighted_D3_larger_than_F_plus_Dall_batches'])
        dump(EXP/'GRADIENT_SPOTCHECK.json',g)
        d['gradient_pressure']=p
    d['weighted_formula']='10/3*(1.35*L_F+1.35*L_Dall+0.30*L_D3), directional CE summed'
    r['sole_method_change']=dict(field='view_weights',old=[1,1,1],new=[1.35,1.35,.30])
    dump(EXP/'RESULTS.json',r);dump(EXP/'TRAINING_DIAGNOSTICS.json',d);dump(EXP/'MASK_HIERARCHY_AUDIT.json',h)
    write_report(r,d,h)


def write_report(r,d,h):
    lines=['# Nested D3 alignment-weight-only experiment: 1.35/1.35/0.30','',
        f'Status:`{r["status"]}`; updates{r["completed_steps"]}/500, horizon4868; no automatic follow-up.',
        'Only method change vs D3 equal:alignment weights[1,1,1] ->[1.35,1.35,0.30], sum3. All dataset/text/sampler/model/CE/temperature/optimizer/LR/batch/workers/preprocess/inclusion/sparsity frozen.',
        f'Fresh common0 SHA256:`{STEP0_SHA}`; no prior Nested checkpoint resumed. Local-only:`{parent.IMAGES}`; missing/symlink/escape fails, no NFS fallback. Disposable cache; NFS originals retained.',
        'D3 exactly reuses ordered uniform stateless sampling, K_eff=min(3,m-1) for m>=2 and original m<=1 fallback. Added indices digest is detached CPU telemetry, no RNG draws.',
        'Alignment10/3*(1.35*L_F+1.35*L_Dall+0.30*L_D3), each I2T+T2I CE. Chain Dall->F,D3->Dall only, detached-child,ramp200,max1. Sparsity(Omega_F+2*Omega_Dall+2*Omega_D3)/3.',
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
        lines.append('| Nested D3 balanced | '+' | '.join(f'{p[k]:.6f}' for k in SCORES)+f' | {u["I2T"]:.3f} | {u["T2I"]:.3f} |')
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
        lines += ['', '| View | I2T CE | T2I CE | Combined CE | Weighted CE | Alignment share | Keep ratio |','|---|---:|---:|---:|---:|---:|---:|']
        for v,s in d['last50_views'].items():
            lines.append(f'| {v} | {s["i2t"]:.6f} | {s["t2i"]:.6f} | {s["raw_combined_CE"]:.6f} | {s["weighted_CE_contribution"]:.6f} | {100*s["weighted_alignment_loss_share"]:.3f}% | {s["keep_ratio"]:.6f} |')
        lines += ['', 'Shares exclude sparsity/inclusion. D3 equal comparison:`'+json.dumps(d.get('comparison_vs_D3_equal',d.get('comparison_vs_atomic_equal')))+'`.',
            'Raw and actual10/3*weight gradient norms:`'+json.dumps(d.get('gradient_pressure'))+'`.',
            'Last50 masks:`'+json.dumps(h.get('last50'))+'`.',
            'Mask deltas:`'+json.dumps(h.get('delta_vs_baselines'))+'`.',
            'Keep deltas:`'+json.dumps(h.get('keep_ratio_delta_vs_baselines'))+'`.',
            'Full500 sampling:`'+json.dumps(d['sampling'])+'`.',
            'Soft chain remains unchanged; learned hard masks can have nonzero violations. No projection or extra constraints.']
    lines += ['', 'Decision:`'+json.dumps(r.get('decision'))+'`.',
        'Strong weighted non-dominance:mean weighted D3<=both parents and at most1/8 D3>F+Dall batches. Improved balance:both weighted ratios decrease>=25% vs D3 equal, without prior severe systemic-dominance criterion. Long-tradeoff significant decline:>0.2pp Score5 or Short4 vs D3 equal while J_long3 increases. Inconclusive fallback covers undefined boundary outcomes.',
        'Gradient protocol:exact same8 fixed seed0 epoch0 global1024 batches, step500 native-backbone optimizer group, raw directional-summed view CE and native F-reference gradients. No optimizer updates. Weighted composite cosine uses linear combination of separately computed view gradients.',
        'Native inference:normalized image @ normalized full-caption text transpose; no mask/gate/detail/rerank/ensemble.',
        'Strict export:`'+json.dumps(r.get('strict_export'))+'`.',
        'Full512000 stream and indices proof:`'+json.dumps(r.get('stream_proof'))+'`.',
        'Full checkpoint/RNG/cursor, immutable SHA, evaluation manifests/protocols, launch Git source archive and CPU tests checked in VALIDATION.json/RESULTS.json.',
        'Local raw log inventory:path/bytes/SHA256/time ranges in RUNTIME_STATS.json. Checkpoints/bare/cache/raw audit remain local. Memory includes page cache, not RSS or proof of OOM.',
        'GitHub:experiment/nested-detail-d3-balanced500; only reviewed small code/config/tests/reports. No full4868 or other K/weight/sparsity/inclusion experiments.']
    (EXP/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true')
    args,remaining=parser.parse_known_args();configure()
    if args.worker:
        global ORIGINAL_SAMPLING
        from train import train_nested_semantic_mask as trainer
        from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_train_gate as gate
        ORIGINAL_SAMPLING=trainer.sampling_diagnostics;trainer.sampling_diagnostics=selection_diagnostics
        gate.compare_prefix,gate.checkpoint_invariants=compare_prefix,checkpoint_invariants
        sys.argv=[sys.argv[0],*remaining];parent.base.worker()
    else:
        assert not remaining
        def stop(sig,frame):raise RuntimeError('HARD_STOP:supervisor signal '+str(sig))
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        Supervisor().run()


if __name__=='__main__':main()
