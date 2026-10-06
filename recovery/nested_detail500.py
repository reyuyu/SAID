"""Fresh common0 F/Dall/Ds chain; one local-only500 run and strict native eval."""
import argparse
from collections import Counter
import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import statistics
import subprocess
import sys

from recovery import s02_local500 as base
from recovery.s02_nfs500 import ROOT, OUT, STEP0, STEP0_SHA, PYTHON, now, dump, rows, sha, distribution
from recovery.s02_full_stage import LOCAL, IMAGES, MANIFEST_SHA, ensure_local_index

EXP = ROOT/'experiments/nest_clip_v1/nested_detail_500_v1'
RUN = ROOT/'runtime/SAID-nest-clip-v1/nested-detail500-20261006'
PHASE = LOCAL/'formal-nested-detail500-phase-20261006'
CONFIG = EXP/'config.json'
BASELINE = ROOT/'runtime/SAID-nest-clip-v1/s02-local500-20261006/step500'
EDITED_SOURCES = {'train/nested_semantic_data.py','train/train_nested_semantic_mask.py',
                  'model/balanced_hparam_search.py'}
WEIGHTS = dict(F=1.4,Dall=1.4,Ds=.2)
BASELINE_SCORES = {
    'RandomDetail':dict(Score5=70.397054,J_long3=73.838424,J_long=82.765002,Short4=65.235),
    'AllDetail':dict(Score5=70.684209,J_long3=74.267014,J_long=83.375002,Short4=65.310)}


def configure():
    base.RUN,base.PHASE,base.CONFIG = RUN,PHASE,CONFIG


def changed_config(c):
    frozen = json.loads((OUT/'configs/summary02_local500.json').read_text())
    for k,v in frozen.items():
        expected = 'nested_detail' if k=='sampling_mode' else [1.4,1.4,.2] if k=='view_weights' else v
        assert c[k] == expected, ('Frozen config drift',k)
    assert c['inclusion_hierarchy'] == 'detail_chain'


def compare_prefix(actual, expected):
    assert [r['step'] for r in actual] == [1,2,3,4,5]
    frozen = rows(BASELINE/'steps.jsonl')[:5]
    for current,previous in zip(actual,frozen):
        assert current['actual_lrs'] == previous['actual_lrs']
        assert math.isfinite(current['loss']) and current['nonfinite'] == 0
        old = {h['rank']:h for h in previous['rank_health']}
        assert {h['rank'] for h in current['rank_health']} == {0,1,2,3}
        for h in current['rank_health']:
            s,ref = h['sampling'],old[h['rank']]['sampling']
            for k in ('sample_ids','sample_id_sha256','full_view_sha256','fixed_first_reference_stream_sha256'):
                assert s[k] == ref[k], ('Frozen stream drift',k)
            assert s['nested_detail_exact'] and h['gradients_finite']
            assert h['updates'] == current['step'] and h['batch'] == 256
    return dict(passed=True,samples=5120,exact_sample_ids_F_strings_tokens=True,
        exact_Dall_strings_tokens=True,Ds_whole_visible_sentence=True,exact_LR=True,
        cross_run_numeric_comparison=False)


def checkpoint_invariants(current, reference):
    import torch
    assert current['completed_steps'] == 5 and current['scheduler_horizon'] == 4868
    c,old = current['config'],json.loads((BASELINE/'config.json').read_text())
    changed_config(c)
    assert c['start_updates'] == 0 and c['resume'] is None and c['init_sha256'] == STEP0_SHA
    for k in ('component_initialization','data','parameter_counts','horizon','batch_size','world_size','accumulation'):
        assert c[k] == old[k], ('Frozen construction drift',k)
    model = dict(c['runtime_model']); hp = dict(model.pop('search_hparams'))
    assert model.pop('inclusion_hierarchy') == 'detail_chain'
    old_model = dict(old['runtime_model']); old_hp = dict(old_model.pop('search_hparams'))
    assert model == old_model
    assert hp.pop('view_weights') == [1.4,1.4,.2]
    old_hp.pop('view_weights'); assert hp == old_hp
    changed = {k for k in c['code_sha256'] if c['code_sha256'][k] != old['code_sha256'][k]}
    assert changed == EDITED_SOURCES, changed
    assert all(c['code_sha256'][k] == sha(ROOT/k) for k in changed)
    assert current['optimizer']['param_groups'] == reference['optimizer']['param_groups']
    assert current['optimizer']['state'].keys() == reference['optimizer']['state'].keys()
    assert {int(s['step']) for s in current['optimizer']['state'].values()} == {5}
    pairs = [(current['model'],reference['model']),(current['adapter'],reference['adapter'])]
    pairs += [(v,reference['optimizer']['state'][k]) for k,v in current['optimizer']['state'].items()]
    for a,b in pairs:
        assert a.keys() == b.keys()
        for k,v in a.items():
            assert v.shape == b[k].shape and v.dtype == b[k].dtype and torch.isfinite(v).all(), k
    return dict(passed=True,optimizer_steps=[5],horizon=4868,common_step0=True,
        construction_initialization_exact=True,optimizer_groups_order_exact=True,finite_states=True,
        authorized_source_changes=sorted(changed),cross_run_numeric_comparison=False)


def sampling_audit():
    """1000 frozen records, no image decode, no global RNG advancement."""
    import numpy as np
    import torch
    from torch.utils.data import DistributedSampler
    from model import longclip
    from train.nested_semantic_data import sampled_text_views
    from recovery.s02_full_local_data import FullLocalDataset
    dataset = FullLocalDataset(base.INDEX,IMAGES,'nested_detail',0)
    before_py,before_np,before_torch = random.getstate(),np.random.get_state(),torch.get_rng_state().clone()
    evidence = []
    for rank in range(4):
        sampler = DistributedSampler(dataset,num_replicas=4,rank=rank,shuffle=True,seed=0,drop_last=False)
        sampler.set_epoch(0)
        for index in itertools.islice(iter(sampler),250):
            path = dataset.resolved_path(index)
            record = json.loads(dataset._records[dataset._offsets[index]:dataset._offsets[index+1]])
            sid = index+1000
            old = sampled_text_views(record['caption'],'summary_random_detail',0,0,sid)
            new = sampled_text_views(record['caption'],'nested_detail',0,0,sid)
            repeat = sampled_text_views(record['caption'],'nested_detail',0,0,sid)
            assert new['views'][0] == old['views'][0] and torch.equal(new['tokens_f'],old['tokens_f'])
            assert new['valid'] == old['valid'] and new['views'] == repeat['views']
            if new['valid']:
                parts = new['views'][0].split('. '); j = new['detail_indices'][0]
                assert new['views'][1] == '. '.join(parts[1:]) and new['views'][2] == parts[j] and 1 <= j < len(parts)
                assert torch.equal(new['tokens_o'],old['reference_tokens_e'])
                assert torch.equal(new['tokens_e'],longclip.tokenize([parts[j]],truncate=False)[0])
            else:
                assert new['views'][1:] == [None,None] and not new['tokens_o'].any() and not new['tokens_e'].any()
            evidence.append(dict(rank=rank,sample_id=sid,epoch=0,actual_path=str(path),valid=new['valid'],
                strings=dict(zip(('F','Dall','Ds'),new['views'])),
                token_ids={v:new[k].tolist() for v,k in [('F','tokens_f'),('Dall','tokens_o'),('Ds','tokens_e')]},
                sentence_indices=dict(Dall=new['dall_indices'],Ds=new['detail_indices']),
                token_lengths=new['untruncated_lengths']))
    current = np.random.get_state()
    assert before_py == random.getstate() and torch.equal(before_torch,torch.get_rng_state())
    assert current[0] == before_np[0] and np.array_equal(current[1],before_np[1]) and current[2:] == before_np[2:]
    assert len(evidence) == 1000
    raw = RUN/'sampling-audit-1000.json'; dump(raw,evidence)
    audit = dict(passed=True,records=1000,valid_records=sum(r['valid'] for r in evidence),
        F_exact_baseline=True,Dall_all_visible_non_summary_ordered=True,Ds_one_whole_visible_detail=True,
        repeated_seed_epoch_sample_deterministic=True,global_python_numpy_torch_RNG_unchanged=True,
        invalid_fallback_unchanged=True,zero_based_sentence_indices=True,
        containment_note='Ds may equal Dall when only one detail is visible; subset inclusion, no invented fallback',
        sampling_algorithm='SHA256(seed:epoch:sample_id:nested_detail_v1) -> private random.Random.randrange(1,n)',
        raw_evidence=dict(path=str(raw),bytes=raw.stat().st_size,sha256=sha(raw),uploaded=False),examples=evidence[:8])
    dump(EXP/'SAMPLING_AUDIT.json',audit)
    return audit


def classify(percent,metrics,atomic_anomaly=False):
    urban = {r:round(100*metrics['Urban-1k'][r]['R@1'],1) for r in ('I2T','T2I')}
    guard = percent['Score5'] >= 70.48 and percent['J_long3'] >= 74.
    strong_guard = percent['Score5'] >= 70.684209-1e-6 and percent['J_long3'] >= 74.267014-1e-6
    strong = urban['T2I'] >= 88.7 and strong_guard
    positive = urban['T2I'] > 88.2 and guard
    global_positive = strong_guard and percent['Score5'] > 70.684209 and urban['T2I'] <= 88.2
    degraded = percent['Score5'] < 70.397054 and percent['J_long3'] < 73.838424
    status = 'NESTED_DETAIL_STRONG_POSITIVE' if strong else 'NESTED_DETAIL_POSITIVE' if positive else (
        'GLOBAL_POSITIVE_URBAN_INCONCLUSIVE' if global_positive else
        'ATOMIC_DETAIL_OVERCONSTRAINED' if atomic_anomaly and degraded else
        'URBAN_LONG_OR_GLOBAL_TRADEOFF' if urban['T2I'] > 88.2 and not guard else 'NESTED_DETAIL_NEGATIVE')
    return dict(status=status,Urban_R1_percent=urban,positive_guard=guard,strong_guard=strong_guard,
        atomic_anomaly=atomic_anomaly,thresholds=dict(positive_Urban_T2I_gt=88.2,Score5_min=70.48,J_long3_min=74.,
            strong_Urban_T2I_min=88.7,strong_Score5_min=70.684209,strong_J_long3_min=74.267014),
        strong_guard_interpretation='No regression against the stronger AllDetail matched500 baseline; tolerance1e-6pp for rounding',
        automatic_continuation=False,automatic_new_experiments=False)


def diagnostics(steps):
    def views(records):
        result = {}
        for label,prefix in [('F','F'),('Dall','O'),('Ds','E')]:
            means = {k:statistics.fmean(r[prefix+'_'+k] for r in records if prefix+'_'+k in r)
                for k in ('i2t','t2i','keep_ratio','positive_keep_ratio','g_mean','g_variance','g_saturation')}
            means.update(raw_combined_CE=means['i2t']+means['t2i'],
                weighted_CE_contribution=10/3*WEIGHTS[label]*(means['i2t']+means['t2i']))
            result[label] = means
        return result
    last = steps[-50:]
    keys = ('inc','inc_weight','F_Dall_mask_iou','Dall_Ds_mask_iou','Dall_F_hard_violation','Ds_Dall_hard_violation')
    diag = dict(last50_steps=[r['step'] for r in last],last50_views=views(last),
        last50_hierarchy={k:statistics.fmean(r[k] for r in last) for k in keys},
        selected_steps={str(r['step']):dict(views=views([r]),hierarchy={k:r[k] for k in keys})
            for r in steps if r['step'] in (1,100,200,500)},
        weighted_formula='10/3*(1.4*(F_i2t+F_t2i)+1.4*(Dall_i2t+Dall_t2i)+0.2*(Ds_i2t+Ds_t2i))')
    n = 0; sentence_sum = 0; positions = Counter()
    totals = {v:dict(effective=0,content=0) for v in WEIGHTS}; ratios = Counter()
    for row in steps:
        for health in row['rank_health']:
            s = health['sampling']['nested_detail_statistics']; n += s['valid_samples']
            sentence_sum += sum(int(k)*v for k,v in s['Dall_sentence_count_histogram'].items())
            positions.update(s['Ds_sentence_position_histogram'])
            for v in totals:
                totals[v]['effective'] += s['views'][v]['effective_token_sum']
                totals[v]['content'] += s['views'][v]['content_token_sum']
            ratios.update(s['coverage_ratio_sums'])
    diag['sampling'] = dict(valid_records=n,Dall_mean_sentences=sentence_sum/n,
        mean_effective_tokens={v:x['effective']/n for v,x in totals.items()},
        mean_content_tokens={v:x['content']/n for v,x in totals.items()},
        Ds_sentence_position_histogram_1based=dict(sorted(positions.items(),key=lambda x:int(x[0]))),
        mean_per_sample_content_token_coverage={k:v/n for k,v in ratios.items()},
        pooled_content_token_coverage={k:totals[a]['content']/totals[b]['content']
            for k,a,b in [('Dall_F','Dall','F'),('Ds_F','Ds','F'),('Ds_Dall','Ds','Dall')]},
        coverage_definition='EOT-delimited content lengths excluding SOT/EOT, same valid records')
    diag['Ds_CE_ratio_to_Dall'] = diag['last50_views']['Ds']['raw_combined_CE']/diag['last50_views']['Dall']['raw_combined_CE']
    return diag


class Supervisor(base.Supervisor):
    def run(self):
        from recovery.resource_stall_v2 import system_snapshot
        from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_full as pipeline
        ready = json.loads((LOCAL/'full-ready.json').read_text())
        assert ready['status'] == 'LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
        assert ready['manifest']['sha256'] == MANIFEST_SHA
        changed_config(json.loads(CONFIG.read_text()))
        assert sha(STEP0) == STEP0_SHA and not system_snapshot()['memory_events'].get('oom_kill',0)
        hashes = ensure_local_index()
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        forbidden = {'recovery.s02_full_stage','recovery.s02_local500','recovery.s02_local_full','recovery.s02_all_detail500',
            'recovery.local_ssd_stage','recovery.s02_nfs500','train.train_nested_semantic_mask'}
        for p in Path('/proc').iterdir():
            if not p.name.isdigit() or int(p.name) == os.getpid():
                continue
            try:
                args = (p/'cmdline').read_bytes().split(b'\0')
            except OSError:
                continue
            assert not any(a.decode(errors='replace') in forbidden for a in args), 'Concurrent training/copy/audit'
        assert not RUN.exists(), 'Never overwrite or resume a trajectory'
        RUN.mkdir(parents=True); (RUN/'reviewed').mkdir(); PHASE.mkdir(exist_ok=False)
        dump(RUN/'prelaunch-local-path-proof-5000.json',base.path_proof())
        self.audit = sampling_audit()
        sources = ['recovery/nested_detail500.py',str(CONFIG.relative_to(ROOT)),*sorted(EDITED_SOURCES),
            'tools/nest_clip.py','recovery/nested_detail_gradients.py','recovery/s02_local500.py',
            'recovery/local500_policy.py','recovery/s02_full_local_data.py',
            'experiments/nest_clip_v1/armb_summary02_4epoch_v1/reproduction_train_gate.py',
            'experiments/nest_clip_v1/armb_summary02_4epoch_v1/training_phase_timing.py']
        dump(RUN/'launch-provenance.json',dict(started_utc=self.started,supervisor_pid=os.getpid(),
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_sha256={p:sha(ROOT/p) for p in sources},step0_sha256=STEP0_SHA,
            local_image_root=str(IMAGES),local_index=str(base.INDEX),local_index_sha256=hashes,
            NFS_fallback=False,stop_updates=500,horizon=4868,resume=None,initial_resources=system_snapshot(),
            method_changes=dict(views='F/Dall/Ds',weights=[1.4,1.4,.2],inclusion='Dall->F,Ds->Dall only'),
            parent_trial_id_note='Frozen parent trial label retained; actual new objective is explicitly recorded',
            disposable_cache='Ephemeral Docker overlay; persistent NFS originals retained',automatic_retry=False))
        command = [str(ROOT/'.venv/bin/torchrun'),'--standalone','--nnodes=1','--nproc-per-node=4','--max-restarts=0',
            '-m','recovery.nested_detail500','--worker','--config',str(CONFIG),'--init-state',str(STEP0),
            '--index-dir',str(base.INDEX),'--image-root',str(IMAGES),'--output-dir',str(self.train),
            '--run-type','formal','--max-updates','500']
        try:
            self.execute('train500',command,training=True)
            self.acceptance = json.loads((self.train/'acceptance.json').read_text())
            assert self.acceptance['passed'] and all(r['completed_updates']==500 and
                r['max_parameter_difference_from_rank0']==0 for r in self.acceptance['ranks'])
            assert json.loads((RUN/'first-five-gate.json').read_text())['passed']
            self.stream_proof = self.full_stream_proof()
            checkpoint_sha = sha(self.train/'step000500.pt')
            self.execute('gradient500',[str(ROOT/'.venv/bin/torchrun'),'--standalone','--nnodes=1',
                '--nproc-per-node=4','--max-restarts=0','-m','recovery.nested_detail_gradients'])
            assert sha(self.train/'step000500.pt') == checkpoint_sha
            pipeline.EXP,pipeline.RUN = RUN/'reviewed',RUN
            self.result = pipeline.Supervisor.evaluate(self,500)
            p = self.result['scores_percent']; p.update(Score5=p['Score5_R1'],Short4=p['Short4_R1'])
            self.result['status'] = classify(p,self.result['metrics'])['status']
        except Exception as error:
            self.error = type(error).__name__+': '+str(error)
            print(self.error,flush=True)
        finally:
            self.report()
        if self.error:
            raise RuntimeError(self.error)

    def full_stream_proof(self):
        actual,frozen = rows(self.train/'steps.jsonl'),rows(BASELINE/'steps.jsonl')
        assert [r['step'] for r in actual] == list(range(1,501)) and len(frozen)==500
        records = 0
        for a,b in zip(actual,frozen):
            assert a['actual_lrs'] == b['actual_lrs'] and math.isfinite(a['loss']) and a['nonfinite']==0
            old = {h['rank']:h for h in b['rank_health']}
            for h in a['rank_health']:
                s,ref = h['sampling'],old[h['rank']]['sampling']
                for k in ('sample_ids','full_view_sha256','fixed_first_reference_stream_sha256'):
                    assert s[k] == ref[k], (a['step'],h['rank'],k)
                assert s['nested_detail_exact'] and h['gradients_finite'] and h['batch']==256
                records += len(s['sample_ids'])
        assert records == 512000
        return dict(passed=True,records=records,all500_sample_ids_F_strings_tokens_exact=True,
            all500_Dall_ordered_visible_Ds_whole_sentence_exact=True,baseline_steps_path=str(BASELINE/'steps.jsonl'),
            baseline_steps_sha256=sha(BASELINE/'steps.jsonl'),first5_gate='BEFORE_UPDATE6')

    def report(self):
        steps = rows(self.train/'steps.jsonl'); cycles = rows(self.train/'cycle_timing.jsonl')
        phases = {r:rows(PHASE/f'rank{r}.jsonl') for r in range(4)}
        systems = [r['system'] for r in rows(RUN/'resource-telemetry.jsonl') if r['command']=='train500']
        systems += [p['system_after'] for pp in phases.values() for p in pp]
        lookup = {r:{p['step']:p for p in pp} for r,pp in phases.items()}
        waits = [max(lookup[r].get(c['step'],{}).get('data_wait_s',0) for r in range(4)) for c in cycles]
        stats = dict(completed_steps=len(steps),started_at=self.started,finished_at=now(),
            full_cycle_seconds=distribution([c['four_rank_max_seconds'] for c in cycles]),
            steady_steps7_plus_full_cycle_seconds=distribution([c['four_rank_max_seconds'] for c in cycles if c['step']>=7]),
            data_wait_seconds_slowest_rank=distribution(waits),
            rank_data_wait_seconds={str(r):distribution([p['data_wait_s'] for p in pp]) for r,pp in phases.items()},
            steps_gt3s=sum(c['four_rank_max_seconds']>3 for c in cycles),steps_gt10s=sum(c['four_rank_max_seconds']>10 for c in cycles),
            peak_cgroup_memory_bytes=max((s['memory_current'] for s in systems),default=0),
            peak_file_cache_bytes=max((s['file'] for s in systems),default=0),
            peak_anon_bytes=max((s['anon'] for s in systems),default=0),
            oom_kill=max((s['memory_events'].get('oom_kill',0) for s in systems),default=0),
            GPU_peak_allocated_GiB={str(r):max((h['peak_allocated_gib'] for row in steps for h in row['rank_health'] if h['rank']==r),default=0) for r in range(4)},
            stop_reason=self.error,automatic_continuation=False,PSI_scope='Host /proc/pressure, cgroup v1; not per-cgroup PSI')
        train_log = (RUN/'train500.log').read_text(errors='replace') if (RUN/'train500.log').exists() else ''
        stats['true_training_io_error_count'] = len(re.findall(r'Image failure sample=|Input/output error|Missing local sample=',train_log))
        stats['pod_or_supervisor_anomaly_observed'] = bool(self.error and ('supervisor' in self.error.lower() or 'signal' in self.error.lower()))
        for k in ('io_PSI','memory_PSI'):
            stats[k] = {p:distribution([s[k][p]['avg10'] for s in systems if p in s[k]]) for p in ('some','full')}
        paths = [p for f in PHASE.glob('image-paths-*.jsonl') for p in rows(f)]
        stats['local_only_proof'] = dict(count=len(paths),passed=bool(paths) and
            all(Path(p['actual_path']).is_relative_to(IMAGES) and not p['NFS_fallback'] for p in paths))
        stats['local_raw_artifacts'] = [dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),
            time_range_utc=[self.started,now()],uploaded=False) for p in sorted(
            list(RUN.glob('*.log'))+list(RUN.glob('*.jsonl'))+list(self.train.glob('*.jsonl'))+list(PHASE.glob('*.jsonl')))]
        if self.commands:
            first = self.commands[0]
            stats['training_wall_seconds'] = (datetime.datetime.fromisoformat(first['ended_utc'])-
                datetime.datetime.fromisoformat(first['started_utc'])).total_seconds()
        diag = diagnostics(steps) if steps else dict(status='NO_UPDATES')
        gradient_path = EXP/'GRADIENT_SPOTCHECK.json'
        gradients = json.loads(gradient_path.read_text()) if gradient_path.exists() else None
        if gradients and steps:
            ratio = gradients['mean_gradient_norms']['Ds']/gradients['mean_gradient_norms']['Dall']
            diag['Ds_gradient_norm_ratio_to_Dall'] = ratio
            diag['atomic_anomaly_heuristic'] = dict(CE_ratio_to_Dall=diag['Ds_CE_ratio_to_Dall'],
                gradient_norm_ratio_to_Dall=ratio,threshold=2.,
                flagged=diag['Ds_CE_ratio_to_Dall']>2 and ratio>2,
                note='Descriptive >=2x pressure heuristic, not a causal attribution or training stop rule')
        hierarchy = dict(status='INCOMPLETE',inclusion_edges=['Dall->F','Ds->Dall'],detached_child=True,
            Ds_to_F_direct_edge=False,sibling_constraint=False,orthogonality=False,union_loss=False,
            formula='0.5*(relu(Dall.detach()-F).mean(-1)+relu(Ds.detach()-Dall).mean(-1))',
            sparsity='(Omega_F+2*Omega_Dall+2*Omega_Ds)/3',ramp_steps=200,max_weight=1)
        if steps:
            hierarchy.update(last50=diag['last50_hierarchy'],selected_steps={k:v['hierarchy'] for k,v in diag['selected_steps'].items()},
                status='CHAIN_OBJECTIVE_VERIFIED',note='Soft inclusion encourages nesting; hard masks need not have zero violation')
        dump(EXP/'TRAINING_DIAGNOSTICS.json',diag); dump(EXP/'MASK_HIERARCHY_AUDIT.json',hierarchy)
        dump(EXP/'RUNTIME_STATS.json',stats)
        result = self.result or dict(status='INCOMPLETE_HARD_STOP',error=self.error)
        result.update(completed_steps=len(steps),stopped_at_500=len(steps)==500,
            checkpoint_path=str(self.train/'step000500.pt'),local_image_root=str(IMAGES),NFS_fallback=False,
            first_five_gate=json.loads((RUN/'first-five-gate.json').read_text()) if (RUN/'first-five-gate.json').exists() else None,
            stream_proof=getattr(self,'stream_proof',None),acceptance=self.acceptance,
            launch_provenance=json.loads((RUN/'launch-provenance.json').read_text()),commands=self.commands,
            automatic_continuation=False,automatic_new_experiments=False)
        if self.result:
            result['decision'] = classify(result['scores_percent'],result['metrics'],
                diag.get('atomic_anomaly_heuristic',{}).get('flagged',False))
            result['status'] = result['decision']['status']
            result['baseline_scores_percent'] = BASELINE_SCORES
            result['delta_vs_baselines_pp'] = {label:{k:result['scores_percent'][k]-v for k,v in score.items()}
                for label,score in BASELINE_SCORES.items()}
            result['dataset_delta_vs_baselines_pp'] = {}
            for label,path in [('RandomDetail',OUT/'STEP500_RESULTS.json'),('AllDetail',OUT/'ALL_DETAIL500_RESULTS.json')]:
                old = json.loads(path.read_text())
                result['dataset_delta_vs_baselines_pp'][label] = {d:{r:{k:100*(v[r][k]-old['metrics'][d][r][k])
                    for k in ('R@1','R@5','R@10')} for r in ('I2T','T2I')} for d,v in result['metrics'].items()}
        dump(EXP/'RESULTS.json',result)
        lines = ['# F / All Detail / Atomic Detail matched500 experiment','',f'Status: `{result["status"]}`. Completed {len(steps)}/500. No automatic full run or new experiment.',
            '',f'Fresh common step0 SHA256: `{STEP0_SHA}`; horizon4868, global1024, seed0, workers8.',
            f'Local-only image root: `{IMAGES}`. Missing/symlink/escape fails; no NFS fallback. Persistent NFS originals retained.',
            'Visible F packing unchanged. Dall is ordered s2..sn; Ds is one uniformly sampled whole visible detail, using private seed/epoch/sample RNG. Single-detail Ds may equal Dall.',
            'Alignment weights[1.4,1.4,0.2], directional CE summed. Chain detached-child edges only; native200-step inclusion ramp/max1 and original sparsity retained.',
            '', '| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |','|---|---|---|']
        for d,v in result.get('metrics',{}).items():
            vals = [' / '.join(f'{v[r][k]*100:.6f}' for k in ('R@1','R@5','R@10')) for r in ('I2T','T2I')]
            lines.append(f'| {d} | {vals[0]} | {vals[1]} |')
        lines += ['', '| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |','|---|---:|---:|---:|---:|---:|---:|',
            '| S0.2 RandomDetail | 70.397054 | 73.838424 | 82.765002 | 65.235 | 89.800 | 88.200 |',
            '| S0.2 AllDetail | 70.684209 | 74.267014 | 83.375002 | 65.310 | 91.000 | 88.100 |']
        if self.result:
            p = result['scores_percent']; u = result['decision']['Urban_R1_percent']
            lines.append('| Nested Detail | '+' | '.join(f'{p[k]:.6f}' for k in ('Score5','J_long3','J_long','Short4'))+f' | {u["I2T"]:.3f} | {u["T2I"]:.3f} |')
        lines += ['', 'Aggregate delta vs both baselines (pp): `'+json.dumps(result.get('delta_vs_baselines_pp'))+'`.',
            'All dataset/direction/R@k deltas: `RESULTS.json`.',
            'Decision: `'+json.dumps(result.get('decision'))+'`.',
            'Mask hierarchy telemetry: `MASK_HIERARCHY_AUDIT.json`; soft penalty is not an exact hard nesting guarantee.',
            'CE/gate/keep/weighted contributions and sampling: `TRAINING_DIAGNOSTICS.json`; native-backbone gradient norms/cosines: `GRADIENT_SPOTCHECK.json`.',
            'This joint view/weight/hierarchy comparison cannot isolate an independent causal benefit of atomic supervision; no automatic ablation is started.',
            'Strict native inference: normalized image/full-caption embeddings, plain inner product; no mask/gate/detail/rerank/ensemble.',
            'Checkpoint/bare SHA and immutable-evaluation proof: `'+json.dumps(result.get('strict_export'))+'`.',
            'Full raw evidence paths/SHA256/bytes/time ranges: `RUNTIME_STATS.json` and `SAMPLING_AUDIT.json`; retained locally, not uploaded.',
            'Ephemeral Docker overlay cache; NFS source of truth retained. Checkpoints/bare remain under persistent project runtime, excluded from Git.',
            'Git publication uses explicit small-file allowlist and credential/binary/size sanity check; remote HEAD must match final commit.',
            'Error: '+str(self.error)]
        (EXP/'REPORT.md').write_text('\n'.join(lines)+'\n')
        dump(RUN/'supervisor-result.json',dict(status=result['status'],completed_steps=len(steps),error=self.error,finished_at=now()))
        print(json.dumps(dict(status=result['status'],completed_steps=len(steps),scores=result.get('scores_percent'))),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--worker',action='store_true')
    args,remaining = parser.parse_known_args(); configure()
    if args.worker:
        from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_train_gate as gate
        gate.compare_prefix,gate.checkpoint_invariants = compare_prefix,checkpoint_invariants
        sys.argv = [sys.argv[0],*remaining]; base.worker()
    else:
        assert not remaining
        def stop(sig,frame):
            raise RuntimeError('HARD_STOP: supervisor signal '+str(sig))
        signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
        Supervisor().run()


if __name__ == '__main__':
    main()
