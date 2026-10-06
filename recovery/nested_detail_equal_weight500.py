"""One alignment-weight-only ablation; reuse the validated local500 protocol."""
import argparse
import hashlib
import json
import math
import signal
import subprocess
import sys

from recovery import nested_detail500 as parent
from recovery.s02_nfs500 import ROOT,OUT,STEP0_SHA,dump,rows,sha,now

EXP = ROOT/'experiments/nest_clip_v1/nested_detail_equal_weight_500_v1'
RUN = ROOT/'runtime/SAID-nest-clip-v1/nested-detail-equal-weight500-20261007'
PHASE = parent.LOCAL/'formal-nested-detail-equal-weight500-phase-20261007'
CONFIG = EXP/'config.json'
BASELINE_EXP = ROOT/'experiments/nest_clip_v1/nested_detail_500_v1'
BASELINE = ROOT/'runtime/SAID-nest-clip-v1/nested-detail500-20261006/step500'
WEIGHTS = dict(F=1.,Dall=1.,Ds=1.)
STREAM_KEYS = ('sample_ids','sample_id_sha256','full_view_sha256','local_views_sha256',
               'split_sha256','fixed_first_reference_stream_sha256')
SCORES = dict(Score5=70.750319,J_long3=74.327865,J_long=83.315002,Short4=65.384)
BASELINE_PATHS = dict(RandomDetail=OUT/'STEP500_RESULTS.json',AllDetail=OUT/'ALL_DETAIL500_RESULTS.json',
                      Nested_low_Ds=BASELINE_EXP/'RESULTS.json')
ORIGINAL_DIAGNOSTICS = parent.diagnostics
ORIGINAL_AUDIT = parent.sampling_audit


def changed_config(config):
    old = json.loads((BASELINE_EXP/'config.json').read_text())
    for key,value in old.items():
        assert config[key] == ([1.,1.,1.] if key=='view_weights' else value), ('Config drift',key)
    assert config['view_weights']==[1.,1.,1.]


def matched_rows(actual, expected):
    assert len(actual)==len(expected)
    count = 0
    for current,previous in zip(actual,expected):
        assert current['step']==previous['step'] and current['actual_lrs']==previous['actual_lrs']
        assert math.isfinite(current['loss']) and current['nonfinite']==0
        old = {h['rank']:h for h in previous['rank_health']}
        assert {h['rank'] for h in current['rank_health']}==set(old)=={0,1,2,3}
        for h in current['rank_health']:
            s,ref = h['sampling'],old[h['rank']]['sampling']
            assert all(s[k]==ref[k] for k in STREAM_KEYS), ('Text/token/sample drift',current['step'],h['rank'])
            assert s['nested_detail_exact'] and h['gradients_finite'] and h['batch']==256
            assert h['updates']==current['step']
            count += len(s['sample_ids'])
    return count


def compare_prefix(actual,unused_reference):
    assert [r['step'] for r in actual]==[1,2,3,4,5]
    count = matched_rows(actual,rows(BASELINE/'steps.jsonl')[:5])
    return dict(passed=True,samples=count,exact_sample_ids_F_Dall_Ds_strings_tokens=True,
        exact_sentence_draws=True,exact_LR=True,cross_run_numeric_comparison=False)


def checkpoint_invariants(current,reference):
    import torch
    c,ref = current['config'],json.loads((BASELINE/'config.json').read_text())
    changed_config(c)
    assert c['resume'] is None and c['start_updates']==0 and c['init_sha256']==STEP0_SHA
    assert current['completed_steps']==5 and current['scheduler_horizon']==4868
    for key in ('component_initialization','data','parameter_counts','horizon','batch_size','world_size','accumulation'):
        assert c[key]==ref[key], ('Construction drift',key)
    model,previous_model = dict(c['runtime_model']),dict(ref['runtime_model'])
    hp,previous_hp = dict(model.pop('search_hparams')),dict(previous_model.pop('search_hparams'))
    assert model==previous_model and hp.pop('view_weights')==[1.,1.,1.]
    previous_hp.pop('view_weights'); assert hp==previous_hp
    changed = {k for k in c['code_sha256'] if c['code_sha256'][k]!=ref['code_sha256'][k]}
    assert changed=={'train/train_nested_semantic_mask.py'}
    assert c['code_sha256']['train/train_nested_semantic_mask.py']==sha(ROOT/'train/train_nested_semantic_mask.py')
    assert current['optimizer']['param_groups']==reference['optimizer']['param_groups']
    assert current['optimizer']['state'].keys()==reference['optimizer']['state'].keys()
    assert {int(s['step']) for s in current['optimizer']['state'].values()}=={5}
    pairs = [(current['model'],reference['model']),(current['adapter'],reference['adapter'])]
    pairs += [(v,reference['optimizer']['state'][k]) for k,v in current['optimizer']['state'].items()]
    for a,b in pairs:
        assert a.keys()==b.keys()
        for k,v in a.items():
            assert v.shape==b[k].shape and v.dtype==b[k].dtype and torch.isfinite(v).all()
    return dict(passed=True,common_step0=True,horizon=4868,optimizer_steps=[5],
        model_adapter_initialization_exact=True,optimizer_groups_order_exact=True,
        model_sampling_loss_implementation_unchanged=True,validation_only_trainer_change=True,
        cross_run_numeric_comparison=False)


def sampling_audit():
    audit = ORIGINAL_AUDIT()
    baseline = json.loads((BASELINE.parent/'sampling-audit-1000.json').read_text())
    current = json.loads((RUN/'sampling-audit-1000.json').read_text())
    assert current==baseline and len(current)==1000
    audit.update(all1000_strings_tokens_indices_exact_prior_nested=True,
        prior_nested_raw_path=str(BASELINE.parent/'sampling-audit-1000.json'),
        prior_nested_raw_sha256=sha(BASELINE.parent/'sampling-audit-1000.json'))
    dump(EXP/'SAMPLING_AUDIT.json',audit)
    return audit


def gradient_pressure(gradients):
    means = gradients['mean_gradient_norms']
    ratios = dict(Ds_Dall=means['Ds']/means['Dall'],Ds_F=means['Ds']/means['F'])
    votes = sum(b['gradient_norms']['Ds']>b['gradient_norms']['F']+b['gradient_norms']['Dall']
                for b in gradients['batches'])
    return dict(mean_norms=means,ratios=ratios,
        Ds_fraction_of_sum_of_view_norms=means['Ds']/sum(means.values()),
        Ds_larger_than_F_plus_Dall_batches=votes,
        dominant=ratios['Ds_Dall']>2 and ratios['Ds_F']>2 and votes>=6,
        rule='Both mean norm ratios >2 and Ds norm > F+Dall norm in at least6 of8 fixed batches',
        scope='Native-backbone per-view CE gradients with equal coefficients; not a signed decomposition of total norm')


def classify(percent,metrics,pressure=None):
    urban = {r:round(100*metrics['Urban-1k'][r]['R@1'],1) for r in ('I2T','T2I')}
    score,long_score = percent['Score5'],percent['J_long3']
    strong = urban['T2I']>=88.7 and score>=SCORES['Score5']-1e-6 and long_score>=SCORES['J_long3']-1e-6
    guard = score>=SCORES['Score5']-.2 and long_score>=SCORES['J_long3']-.2
    positive = urban['T2I']>88.5 and guard
    degraded = score<SCORES['Score5']-1e-6 or long_score<SCORES['J_long3']-1e-6 or urban['T2I']<88.5 or urban['I2T']<90.9
    dominance = bool(pressure and pressure['dominant'])
    status = 'EQUAL_WEIGHT_STRONG_POSITIVE' if strong else 'EQUAL_WEIGHT_POSITIVE' if positive else (
        'TRADEOFF' if urban['T2I']>88.5 and not guard else
        'ATOMIC_DETAIL_OVERWEIGHTED' if dominance and degraded else 'EQUAL_WEIGHT_NO_CLEAR_GAIN')
    return dict(status=status,Urban_R1_percent=urban,positive_guard=guard,Ds_gradient_dominant=dominance,
        thresholds=dict(strong_Urban_T2I_min=88.7,strong_Score5_min=SCORES['Score5'],strong_J_long3_min=SCORES['J_long3'],
            positive_Urban_T2I_gt=88.5,positive_max_decline_pp=.2),
        precedence='Strong, positive, Urban/global tradeoff, gradient-dominated regression, no clear gain',
        automatic_continuation=False,automatic_new_experiments=False)


def parent_classify(percent,metrics,unused_atomic_flag=False):
    path = EXP/'GRADIENT_SPOTCHECK.json'
    pressure = gradient_pressure(json.loads(path.read_text())) if path.exists() else None
    return classify(percent,metrics,pressure)


def diagnostics(steps):
    result = ORIGINAL_DIAGNOSTICS(steps)
    total = sum(v['weighted_CE_contribution'] for v in result['last50_views'].values())
    for view in result['last50_views'].values():
        view['weighted_alignment_loss_share'] = view['weighted_CE_contribution']/total
    result['weighted_formula'] = '10/3*((F_i2t+F_t2i)+(Dall_i2t+Dall_t2i)+(Ds_i2t+Ds_t2i))'
    result['alignment_loss_share_definition'] = 'Ratio of last50 mean weighted CE to sum of three last50 means; excludes sparsity/inclusion'
    return result


def configure():
    parent.EXP,parent.RUN,parent.PHASE,parent.CONFIG,parent.BASELINE = EXP,RUN,PHASE,CONFIG,BASELINE
    parent.WEIGHTS = WEIGHTS
    parent.EDITED_SOURCES = {'train/train_nested_semantic_mask.py'}
    parent.BASELINE_SCORES = {**parent.BASELINE_SCORES,'Nested_low_Ds':SCORES}
    parent.changed_config,parent.compare_prefix,parent.checkpoint_invariants = changed_config,compare_prefix,checkpoint_invariants
    parent.sampling_audit,parent.diagnostics,parent.classify = sampling_audit,diagnostics,parent_classify
    parent.configure()


class Supervisor(parent.Supervisor):
    def execute(self,name,command,training=False):
        command = list(command)
        if name=='train500':
            baseline_result = json.loads((BASELINE_EXP/'RESULTS.json').read_text())
            assert baseline_result['status']=='NESTED_DETAIL_POSITIVE' and baseline_result['evaluation_checkpoint_immutable']
            assert sha(BASELINE/'step000500.pt')==baseline_result['checkpoint_sha256']
            launch = json.loads((RUN/'launch-provenance.json').read_text())
            old_launch = baseline_result['launch_provenance']
            trainer = 'train/train_nested_semantic_mask.py'
            original = subprocess.check_output(['git','show',old_launch['git_head']+':'+trainer],cwd=ROOT)
            before = b"        assert cfg.get('inclusion_hierarchy') == 'detail_chain' and cfg['view_weights'] == [1.4,1.4,.2]\n"
            after = b"        assert cfg.get('inclusion_hierarchy') == 'detail_chain'\n        assert cfg['view_weights'] in ([1.4,1.4,.2], [1.,1.,1.]), 'Only reviewed Nested Detail weights are authorized'\n"
            assert before in original and (ROOT/trainer).read_bytes()==original.replace(before,after)
            cfg = json.loads(CONFIG.read_text()); old_cfg = json.loads((BASELINE_EXP/'config.json').read_text())
            assert {k for k in cfg.keys()|old_cfg.keys() if cfg.get(k)!=old_cfg.get(k)}=={'view_weights'}
            previous_code = json.loads((BASELINE/'config.json').read_text())['code_sha256']
            assert all(sha(ROOT/p)==v for p,v in previous_code.items() if p!=trainer)
            launch['source_sha256'].update({p:sha(ROOT/p) for p in previous_code})
            extras = ['recovery/nested_detail_equal_weight500.py','recovery/nested_detail_equal_weight_gradients.py',
                      'recovery/review_nested_detail_equal_weight500.py']
            launch['source_sha256'].update({p:sha(ROOT/p) for p in extras})
            launch['method_changes'] = dict(only='view_weights',old=[1.4,1.4,.2],new=[1.,1.,1.])
            launch['related_unit_tests'] = dict(passed=56,files=[
                'tests/test_nested_detail_equal_weight.py','tests/test_nested_detail.py',
                'tests/test_balanced_hparams.py','tests/test_nested_resume.py','recovery/test_nested_detail500.py'])
            launch['alignment_weight_only_proof'] = dict(config_only_view_weights_changed=True,
                all_model_data_objective_sources_exact_prior=True,trainer_only_admission_check_extended=True,
                baseline_checkpoint_identity_verified=True,baseline_checkpoint_used_for_resume=False,
                baseline_checkpoint_sha256=baseline_result['checkpoint_sha256'],baseline_steps_path=str(BASELINE/'steps.jsonl'),
                baseline_steps_sha256=sha(BASELINE/'steps.jsonl'))
            dump(RUN/'launch-provenance.json',launch)
            command[command.index('-m')+1]='recovery.nested_detail_equal_weight500'
        if name=='gradient500':
            command[command.index('-m')+1]='recovery.nested_detail_equal_weight_gradients'
        return super().execute(name,command,training)

    def full_stream_proof(self):
        actual,expected = rows(self.train/'steps.jsonl'),rows(BASELINE/'steps.jsonl')
        assert [r['step'] for r in actual]==list(range(1,501))
        count = matched_rows(actual,expected)
        assert count==512000
        return dict(passed=True,records=count,all500_sample_ids_F_Dall_Ds_strings_tokens_exact=True,
            exact_sentence_draws=True,exact_LR=True,baseline_steps_path=str(BASELINE/'steps.jsonl'),
            baseline_steps_sha256=sha(BASELINE/'steps.jsonl'),first5_gate='BEFORE_UPDATE6')

    def report(self):
        super().report()
        update_reports()


def update_reports():
    result = json.loads((EXP/'RESULTS.json').read_text())
    diag = json.loads((EXP/'TRAINING_DIAGNOSTICS.json').read_text())
    if result.get('metrics'):
        result['dataset_delta_vs_baselines_pp'] = {label:{d:{r:{k:100*(v[r][k]-old['metrics'][d][r][k])
            for k in ('R@1','R@5','R@10')} for r in ('I2T','T2I')} for d,v in result['metrics'].items()}
            for label,path in BASELINE_PATHS.items() for old in [json.loads(path.read_text())]}
        result['decision'] = parent_classify(result['scores_percent'],result['metrics'])
        result['status'] = result['decision']['status']
    path = EXP/'GRADIENT_SPOTCHECK.json'
    if path.exists():
        gradients = json.loads(path.read_text()); pressure = gradient_pressure(gradients)
        diag['gradient_pressure'] = pressure
        diag['gradient_protocol_exact_prior'] = all(gradients[k]==json.loads((BASELINE_EXP/'GRADIENT_SPOTCHECK.json').read_text())[k]
            for k in ('fixed_global_batches','global_batch_size','selection','backbone_scope','gradients','native_reference'))
        diag['weighted_Ds_Dall_gradient_norm_ratio'] = pressure['ratios']['Ds_Dall']
        diag['weighted_Ds_F_gradient_norm_ratio'] = pressure['ratios']['Ds_F']
        diag['gradient_summary_note'] = 'Per-view norm magnitudes and ratios, all coefficients10/3. Composite cosine is the linear combination of separately recomputed view gradients; no optimizer updates.'
        result['gradient_pressure'] = pressure
    prior_diag = json.loads((BASELINE_EXP/'TRAINING_DIAGNOSTICS.json').read_text())
    hierarchy = json.loads((EXP/'MASK_HIERARCHY_AUDIT.json').read_text())
    if 'last50_hierarchy' in diag:
        hierarchy['delta_vs_nested_low_Ds'] = {k:v-prior_diag['last50_hierarchy'][k] for k,v in diag['last50_hierarchy'].items()}
        hierarchy['keep_ratio_delta_vs_nested_low_Ds'] = {v:s['keep_ratio']-prior_diag['last50_views'][v]['keep_ratio'] for v,s in diag['last50_views'].items()}
    result['sole_method_change'] = dict(old=[1.4,1.4,.2],new=[1.,1.,1.],field='view_weights')
    dump(EXP/'RESULTS.json',result); dump(EXP/'TRAINING_DIAGNOSTICS.json',diag); dump(EXP/'MASK_HIERARCHY_AUDIT.json',hierarchy)
    write_report(result,diag,hierarchy)


def write_report(result,diag,hierarchy):
    lines = ['# Nested Detail alignment-weight-only ablation: 1/1/1','',
        f'Status: `{result["status"]}`; updates{result["completed_steps"]}/500, horizon4868; no automatic continuation.',
        'Only method change: alignment weights[1.4,1.4,0.2] -> [1,1,1]. Common step0, model/adapter initialization, sample order, all F/Dall/Ds strings/tokens/draws, CE/temperature, optimizer/LR, batch/workers, mask, detached-child chain inclusion/ramp and sparsity frozen.',
        f'Common step0 SHA256: `{STEP0_SHA}`. Independent fresh start; prior nested checkpoint never used for resume.',
        f'Local-only root: `{parent.IMAGES}`. Missing/symlink/escape fails; no NFS fallback. No staging/full audit/new experiment.',
        'Alignment:10/3*(L_F+L_Dall+L_Ds), each view directional CE summed. Inclusion:Dall->F, Ds->Dall only. Sparsity:(Omega_F+2*Omega_Dall+2*Omega_Ds)/3.',
        '', '| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |','|---|---|---|']
    for d,v in result.get('metrics',{}).items():
        vals = [' / '.join(f'{v[r][k]*100:.6f}' for k in ('R@1','R@5','R@10')) for r in ('I2T','T2I')]
        lines.append(f'| {d} | {vals[0]} | {vals[1]} |')
    lines += ['', '| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |','|---|---:|---:|---:|---:|---:|---:|',
        '| S0.2 RandomDetail | 70.397054 | 73.838424 | 82.765002 | 65.235 | 89.800 | 88.200 |',
        '| S0.2 AllDetail | 70.684209 | 74.267014 | 83.375002 | 65.310 | 91.000 | 88.100 |',
        '| Nested1.4/1.4/0.2 | 70.750319 | 74.327865 | 83.315002 | 65.384 | 90.900 | 88.500 |']
    if result.get('metrics'):
        p = result['scores_percent']; u = result['decision']['Urban_R1_percent']
        lines.append('| Nested1/1/1 | '+' | '.join(f'{p[k]:.6f}' for k in SCORES)+f' | {u["I2T"]:.3f} | {u["T2I"]:.3f} |')
        lines += ['', '| Baseline | ΔScore5 | ΔJ_long3 | ΔJ_long | ΔShort4 | ΔUrban I2T | ΔUrban T2I |','|---|---:|---:|---:|---:|---:|---:|']
        for name,dd in result['delta_vs_baselines_pp'].items():
            uu = result['dataset_delta_vs_baselines_pp'][name]['Urban-1k']
            lines.append('| '+name+' | '+' | '.join(f'{dd[k]:+.6f}' for k in SCORES)+
                f' | {uu["I2T"]["R@1"]:+.3f} | {uu["T2I"]["R@1"]:+.3f} |')
        lines += ['', '| Dataset | ΔI2T R@1 / R@5 / R@10 vs nested low-Ds (pp) | ΔT2I R@1 / R@5 / R@10 (pp) |','|---|---|---|']
        for d,v in result['dataset_delta_vs_baselines_pp']['Nested_low_Ds'].items():
            vals = [' / '.join(f'{v[r][k]:+.6f}' for k in ('R@1','R@5','R@10')) for r in ('I2T','T2I')]
            lines.append(f'| {d} | {vals[0]} | {vals[1]} |')
    if 'last50_views' in diag:
        lines += ['', '| View | I2T CE | T2I CE | Combined CE | Weighted CE | Alignment-loss share | Keep ratio |','|---|---:|---:|---:|---:|---:|---:|']
        for v,s in diag['last50_views'].items():
            lines.append(f'| {v} | {s["i2t"]:.6f} | {s["t2i"]:.6f} | {s["raw_combined_CE"]:.6f} | {s["weighted_CE_contribution"]:.6f} | {100*s["weighted_alignment_loss_share"]:.3f}% | {s["keep_ratio"]:.6f} |')
        lines += ['', 'Weighted-loss shares use last50 means and exclude sparsity/inclusion. Sampling/gate/selected-step telemetry:TRAINING_DIAGNOSTICS.json.',
            'Last50 mask hierarchy: `'+json.dumps(hierarchy.get('last50'))+'`.',
            'Mask hierarchy delta vs prior nested: `'+json.dumps(hierarchy.get('delta_vs_nested_low_Ds'))+'`.',
            'Gradient pressure: `'+json.dumps(diag.get('gradient_pressure'))+'`.',
            'Soft detached-child inclusion is unchanged; learned hard masks may still violate nesting. No projection or additional loss was added.']
    lines += ['', 'Decision: `'+json.dumps(result.get('decision'))+'`.',
        'Positive “no material decline” tolerance was fixed before training at0.2pp for each of Score5/J_long3. Strong gate uses exact user thresholds; Urban uses canonical1000-hit rounding. Explicit no-clear-gain fallback handles outcomes outside the named gates.',
        'Gradient protocol: same8 seed0 epoch0 global1024 batches, fixed step500, same native-backbone optimizer group, raw per-view CE and native F-reference gradient. No optimizer updates; all first8 batch-ID hashes verified against prior run.',
        'Strict native inference uses normalized image/native full-caption embeddings and plain inner product; no mask/gate/detail/rerank/ensemble.',
        'Checkpoint/bare hashes and immutable evaluation: `'+json.dumps(result.get('strict_export'))+'`.',
        'Exact all512000 sample/text/token stream proof: `'+json.dumps(result.get('stream_proof'))+'`.',
        'Raw logs/checkpoints/bare/local mirror/full1000-record audit stay local; path/size/SHA256/time-range inventory:RUNTIME_STATS.json and SAMPLING_AUDIT.json.',
        'Ephemeral Docker overlay cache, persistent NFS originals retained. Checkpoints remain on persistent project runtime. GitHub only receives reviewed small code/config/tests/reports; branch experiment/nested-detail-equal-weight500.',
        'No automatic full4868, weight sweep, sampling/shuffle/offset/sparsity/inclusion changes.']
    (EXP/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--worker',action='store_true')
    args,remaining = parser.parse_known_args(); configure()
    if args.worker:
        from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_train_gate as gate
        gate.compare_prefix,gate.checkpoint_invariants = compare_prefix,checkpoint_invariants
        sys.argv = [sys.argv[0],*remaining]; parent.base.worker()
    else:
        assert not remaining
        def stop(sig,frame):
            raise RuntimeError('HARD_STOP: supervisor signal '+str(sig))
        signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
        Supervisor().run()


if __name__=='__main__':
    main()
