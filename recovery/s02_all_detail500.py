"""One AllDetail experiment: fresh common0, local-only, stop500, native eval."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys

from recovery import s02_local500 as base
from recovery.s02_nfs500 import ROOT, OUT, STEP0, STEP0_SHA, PYTHON, now, dump, rows, sha, distribution
from recovery.s02_full_stage import LOCAL, IMAGES, MANIFEST_SHA, ensure_local_index

FAILED_ATTEMPT = ROOT / 'runtime/SAID-nest-clip-v1/s02-all-detail500-20261006'
RUN = ROOT / 'runtime/SAID-nest-clip-v1/s02-all-detail500-20261006-r2'
PHASE = LOCAL / 'formal-all-detail500-phase-20261006-r2'
CONFIG = OUT / 'configs/summary02_all_detail500.json'
BASELINE = ROOT / 'runtime/SAID-nest-clip-v1/s02-local500-20261006/step500'
EDITED_TRAINING_SOURCES = {'train/nested_semantic_data.py', 'train/train_nested_semantic_mask.py'}
HISTORICAL = dict(Score5=70.364367, J_long3=73.726611, J_long=82.765002, Short4=65.321)


def configure():
    base.RUN, base.PHASE, base.CONFIG = RUN, PHASE, CONFIG


def compare_prefix(actual, expected):
    assert [r['step'] for r in actual] == [1, 2, 3, 4, 5]
    frozen = rows(BASELINE / 'steps.jsonl')[:5]
    for current, previous in zip(actual, frozen):
        assert current['actual_lrs'] == previous['actual_lrs']
        assert math.isfinite(current['loss']) and current['nonfinite'] == 0
        old = {h['rank']: h for h in previous['rank_health']}
        assert {h['rank'] for h in current['rank_health']} == {0, 1, 2, 3}
        for h in current['rank_health']:
            s, reference = h['sampling'], old[h['rank']]['sampling']
            for k in ('sample_ids', 'sample_id_sha256', 'full_view_sha256', 'fixed_first_reference_stream_sha256'):
                assert s[k] == reference[k], f'Frozen stream drift: {k}'
            assert s['summary_baseline_exact'] and s['all_detail_selection_complete']
            assert h['gradients_finite'] and h['updates'] == current['step'] and h['batch'] == 256
    return dict(passed=True, samples=5120, exact_sample_ids=True, exact_F_S_strings_tokens=True,
                D_only_change=True, all_detail_ordered_complete=True, exact_LR=True,
                cross_run_numeric_comparison=False, baseline=str(BASELINE))


def checkpoint_invariants(current, reference):
    assert current['completed_steps'] == 5 and current['scheduler_horizon'] == 4868
    c, old = current['config'], json.loads((BASELINE / 'config.json').read_text())
    assert c['start_updates'] == 0 and c['resume'] is None and c['init_sha256'] == STEP0_SHA
    frozen = json.loads((OUT / 'configs/summary02_local500.json').read_text())
    for k, v in frozen.items():
        assert c[k] == ('summary_all_detail' if k == 'sampling_mode' else v), k
    for k in ('component_initialization', 'runtime_model', 'data', 'parameter_counts',
              'horizon', 'batch_size', 'world_size', 'accumulation'):
        assert c[k] == old[k], f'Frozen construction drift: {k}'
    changed = {k for k in c['code_sha256'] if c['code_sha256'][k] != old['code_sha256'][k]}
    assert changed == EDITED_TRAINING_SOURCES
    assert all(c['code_sha256'][k] == sha(ROOT / k) for k in changed)
    assert current['optimizer']['param_groups'] == reference['optimizer']['param_groups']
    assert current['optimizer']['state'].keys() == reference['optimizer']['state'].keys()
    assert {int(s['step']) for s in current['optimizer']['state'].values()} == {5}
    import torch
    pairs = [(current['model'], reference['model']), (current['adapter'], reference['adapter'])]
    pairs += [(v, reference['optimizer']['state'][k]) for k, v in current['optimizer']['state'].items()]
    for a, b in pairs:
        assert a.keys() == b.keys()
        for k, v in a.items():
            assert v.shape == b[k].shape and v.dtype == b[k].dtype and torch.isfinite(v).all(), k
    return dict(passed=True, optimizer_steps=[5], horizon=4868, common_step0=True,
                frozen_config_except_sampling_mode=True, construction_exact=True,
                optimizer_groups_order_exact=True, finite_states=True,
                authorized_source_changes=sorted(changed), cross_run_numeric_comparison=False)


def classify(percent, metrics):
    urban = metrics['Urban-1k']['T2I']['R@1'] * 100
    guard = percent['J_long3'] >= 73.403324
    strong = urban >= 89 and percent['Score5'] >= 70.264367 and guard
    positive = urban >= 88.5 and percent['Score5'] >= 70.164367 and guard
    # Between baseline and the positive threshold: retain the result, no new run.
    status = 'URBAN_STRONG_POSITIVE' if strong else 'URBAN_POSITIVE' if positive else (
        'NEGATIVE' if urban <= 88 or percent['Score5'] < 70.164367 or not guard else 'INCONCLUSIVE')
    return dict(status=status, Urban_T2I_percent=urban, strong_positive=strong, positive=positive,
                thresholds=dict(Urban_T2I_positive=88.5, Urban_T2I_strong=89,
                    Score5_positive=70.164367, Score5_strong=70.264367, J_long3_min=73.403324),
                automatic_continuation=False, automatic_new_experiments=False)


def diagnostics(steps):
    last = steps[-50:]
    result = dict(last50_steps=[r['step'] for r in last], last50_views={})
    for name, prefix, weight in [('F', 'F', 1.4), ('S', 'O', .2), ('D', 'E', 1.4)]:
        i2t = statistics.fmean(r[prefix+'_i2t'] for r in last)
        t2i = statistics.fmean(r[prefix+'_t2i'] for r in last)
        result['last50_views'][name] = dict(CE_I2T=i2t, CE_T2I=t2i, CE_directional_mean=(i2t+t2i)/2,
            weighted_alignment_contribution=10/3*weight*(i2t+t2i),
            keep_ratio=statistics.fmean(r[prefix+'_keep_ratio'] for r in last),
            positive_keep_ratio=statistics.fmean(r[prefix+'_positive_keep_ratio'] for r in last))
    result['weighted_formula'] = '10/sum(weights) * weight * (CE_I2T+CE_T2I); sum(weights)=3'
    for k in ('inc', 'inc_weight', 'oe_iou', 'hard_inclusion_violation', 'loss'):
        result['last50_'+k] = statistics.fmean(r[k] for r in last)
    result['F_D_mask_IoU_diagnostic_steps'] = [{k:v for k,v in r.items() if k in
        ('step', 'E_F_D_positive_mask_iou', 'F_keep_ratio', 'O_keep_ratio', 'E_keep_ratio')}
        for r in steps if 'E_F_D_positive_mask_iou' in r]
    samples = detail_tokens = full_tokens = sentences = ratio_sum = coverage_count = 0
    for row in steps:
        for h in row['rank_health']:
            s = h['sampling']
            d = s['Full_Summary_Detail_token_statistics']['Detail']
            samples += d['samples']; detail_tokens += d['effective_token_sum']
            hist = s['random_detail_sampling']['selected_count_histogram']
            sentences += sum(int(k)*v for k,v in hist.items())
            cov = s['detail_full_token_coverage']
            full_tokens += cov['full_content_token_sum']
            ratio_sum += cov['per_sample_ratio_sum']; coverage_count += cov['samples']
    content_detail = detail_tokens-2*samples
    result.update(valid_detail_samples=samples, D_mean_sentences=sentences/samples,
        D_mean_effective_tokens_including_SOT_EOT=detail_tokens/samples,
        D_mean_content_tokens=content_detail/samples,
        D_F_content_token_coverage_pooled=content_detail/full_tokens,
        D_F_content_token_coverage_mean_per_sample=ratio_sum/coverage_count,
        token_coverage_definition='Effective EOT-delimited token lengths excluding SOT/EOT; same valid samples only',
        F_D_native_text_cosine=dict(computed=False, reason='No existing F-D text-cosine observer; optional metric omitted'))
    return result


class Supervisor(base.Supervisor):
    def run(self):
        from recovery.resource_stall_v2 import system_snapshot
        from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_full as pipeline
        ready = json.loads((LOCAL/'full-ready.json').read_text())
        assert ready['status'] == 'LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
        assert ready['manifest']['sha256'] == MANIFEST_SHA
        cfg = json.loads(CONFIG.read_text()); old = json.loads((OUT/'configs/summary02_local500.json').read_text())
        assert {k for k in cfg if cfg[k] != old[k]} == {'sampling_mode'}
        assert sha(STEP0) == STEP0_SHA
        index_hashes = ensure_local_index()
        assert not system_snapshot()['memory_events'].get('oom_kill', 0)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        for p in Path('/proc').iterdir():
            if not p.name.isdigit() or int(p.name) == os.getpid():
                continue
            try:
                args = (p/'cmdline').read_bytes().split(b'\0')
            except OSError:
                continue
            forbidden = {'recovery.s02_full_stage','recovery.s02_local500','recovery.s02_local_full',
                         'recovery.local_ssd_stage','recovery.s02_nfs500','train.train_nested_semantic_mask'}
            assert not any(a.decode(errors='replace') in forbidden for a in args), 'Concurrent training/copy/audit'
        assert not RUN.exists(), 'Never overwrite or resume this independent experiment'
        RUN.mkdir(parents=True); (RUN/'reviewed').mkdir(); PHASE.mkdir(exist_ok=False)
        dump(RUN/'prelaunch-local-path-proof-5000.json', base.path_proof())
        sources = ['recovery/s02_all_detail500.py','recovery/configs/summary02_all_detail500.json',
                   *sorted(EDITED_TRAINING_SOURCES),'train/random_detail_observer.py',
                   'recovery/s02_local500.py','recovery/local500_policy.py','recovery/s02_full_local_data.py',
                   'experiments/nest_clip_v1/armb_summary02_4epoch_v1/reproduction_train_gate.py',
                   'experiments/nest_clip_v1/armb_summary02_4epoch_v1/training_phase_timing.py']
        dump(RUN/'launch-provenance.json', dict(started_utc=self.started,supervisor_pid=os.getpid(),
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_sha256={p:sha(ROOT/p) for p in sources},step0_sha256=STEP0_SHA,
            local_image_root=str(IMAGES),local_index=str(base.INDEX),local_index_sha256=index_hashes,
            NFS_fallback=False,stop_updates=500,horizon=4868,resume=None,initial_resources=system_snapshot(),
            sole_config_change=dict(sampling_mode=['summary_random_detail','summary_all_detail']),
            D_definition='Ordered all non-summary sentences in unchanged baseline visible F pool',
            disposable_cache='Ephemeral Docker overlay; persistent NFS originals retained',automatic_retry=False))
        command = [str(ROOT/'.venv/bin/torchrun'),'--standalone','--nnodes=1','--nproc-per-node=4',
                   '--max-restarts=0','-m','recovery.s02_all_detail500','--worker','--config',str(CONFIG),
                   '--init-state',str(STEP0),'--index-dir',str(base.INDEX),'--image-root',str(IMAGES),
                   '--output-dir',str(self.train),'--run-type','formal','--max-updates','500']
        try:
            self.execute('train500', command, training=True)
            self.acceptance = json.loads((self.train/'acceptance.json').read_text())
            assert self.acceptance['passed'] and all(r['completed_updates'] == 500 and
                r['max_parameter_difference_from_rank0'] == 0 for r in self.acceptance['ranks'])
            assert json.loads((RUN/'first-five-gate.json').read_text())['passed']
            self.stream_proof = self.full_stream_proof()
            import torch
            state = torch.load(self.train/'step000500.pt',map_location='cpu',weights_only=False)
            assert state['completed_steps'] == state['global_step'] == 500
            assert state['scheduler']['horizon'] == 4868 and len(state['rng_per_rank']) == 4
            assert state['data_cursor'] == dict(next_epoch=0,next_batch=500)
            assert {int(s['step']) for s in state['optimizer']['state'].values()} == {500}
            assert state['trajectory_root'] == str(RUN) and state['config']['resume'] is None
            del state
            pipeline.EXP = RUN/'reviewed'; pipeline.RUN = RUN
            self.result = pipeline.Supervisor.evaluate(self,500)
            percent = self.result['scores_percent']
            percent['Score5'] = percent['Score5_R1']; percent['Short4'] = percent['Short4_R1']
            self.result['urban_gate'] = classify(percent,self.result['metrics'])
            self.result['status'] = self.result['urban_gate']['status']
        except Exception as error:
            self.error = type(error).__name__+': '+str(error)
            print(self.error,flush=True)
        finally:
            self.report()
        if self.error:
            raise RuntimeError(self.error)

    def full_stream_proof(self):
        actual = rows(self.train/'steps.jsonl'); baseline = rows(BASELINE/'steps.jsonl')
        assert len(actual) == len(baseline) == 500
        changed = records = 0
        for a,b in zip(actual,baseline):
            assert a['step'] == b['step'] and a['actual_lrs'] == b['actual_lrs']
            assert math.isfinite(a['loss']) and a['nonfinite'] == 0
            old = {h['rank']:h for h in b['rank_health']}
            for h in a['rank_health']:
                s, ref = h['sampling'], old[h['rank']]['sampling']
                for k in ('sample_ids','full_view_sha256','fixed_first_reference_stream_sha256'):
                    assert s[k] == ref[k], (a['step'],h['rank'],k)
                assert s['summary_baseline_exact'] and s['all_detail_selection_complete']
                assert h['batch'] == 256 and h['gradients_finite']
                changed += s['local_views_sha256'] != ref['local_views_sha256']
                records += len(s['sample_ids'])
        assert records == 512000 and changed > 0
        return dict(passed=True,records_checked=records,all500_sample_ids_F_S_exact=True,
            first5_gate='Before update6',D_only_change=True,changed_rank_batches=changed,
            baseline_steps_sha256=sha(BASELINE/'steps.jsonl'),baseline_path=str(BASELINE),
            summary_proof='Every valid Summary string/token equals unchanged fixed-first reference; same fallback',
            D_proof='Every batch selects ordered range(1,detail_pool_size+1)')

    def report(self):
        steps = rows(self.train/'steps.jsonl'); cycles = rows(self.train/'cycle_timing.jsonl')
        resources = rows(RUN/'resource-telemetry.jsonl')
        phases = {r:rows(PHASE/f'rank{r}.jsonl') for r in range(4)}
        systems = [r['system'] for r in resources] + [p['system_after'] for pp in phases.values() for p in pp]
        stats = dict(completed_steps=len(steps),started_at=self.started,finished_at=now(),
            full_cycle_seconds=distribution([c['four_rank_max_seconds'] for c in cycles]),
            rank_data_wait_seconds={str(r):distribution([p['data_wait_s'] for p in pp]) for r,pp in phases.items()},
            steps_gt3s=sum(c['four_rank_max_seconds']>3 for c in cycles),
            steps_gt10s=sum(c['four_rank_max_seconds']>10 for c in cycles),
            peak_cgroup_memory_bytes=max((s['memory_current'] for s in systems),default=0),
            peak_file_cache_bytes=max((s['file'] for s in systems),default=0),
            oom_kill=max((s['memory_events'].get('oom_kill',0) for s in systems),default=0),
            GPU_peak_allocated_GiB={str(r):max((h['peak_allocated_gib'] for row in steps for h in row['rank_health'] if h['rank']==r),default=0) for r in range(4)},
            stop_reason=self.error,automatic_continuation=False)
        train_log = (RUN/'train500.log').read_text(errors='replace') if (RUN/'train500.log').exists() else ''
        stats['true_training_io_error_count'] = len(re.findall(
            r'Image failure sample=|Input/output error|Missing local sample=', train_log))
        stats['pod_or_supervisor_anomaly_observed'] = bool(self.error and
            ('supervisor' in self.error.lower() or 'signal' in self.error.lower()))
        for k in ('io_PSI','memory_PSI'):
            stats[k] = {p:distribution([s[k][p]['avg10'] for s in systems if p in s[k]]) for p in ('some','full')}
        local_proofs = [p for f in PHASE.glob('image-paths-*.jsonl') for p in rows(f)]
        stats['local_only_proof'] = dict(count=len(local_proofs),passed=bool(local_proofs) and
            all(Path(p['actual_path']).is_relative_to(IMAGES) and not p['NFS_fallback'] for p in local_proofs))
        logs = list(RUN.glob('*.log'))+list(RUN.glob('*.jsonl'))+list(self.train.glob('*.jsonl'))+list(PHASE.glob('*.jsonl'))
        stats['local_raw_artifacts'] = [dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),
            time_range_utc=[self.started,now()],uploaded=False) for p in sorted(logs)]
        diag = diagnostics(steps) if steps else dict(status='NO_UPDATES')
        dump(OUT/'ALL_DETAIL500_DIAGNOSTICS.json',diag)
        dump(OUT/'ALL_DETAIL500_RUNTIME_STATS.json',stats)
        result = self.result or dict(status='INCOMPLETE_HARD_STOP',error=self.error)
        result.update(completed_steps=len(steps),stopped_at_500=len(steps)==500,
            checkpoint_path=str(self.train/'step000500.pt'),local_image_root=str(IMAGES),NFS_fallback=False,
            first_five_gate=json.loads((RUN/'first-five-gate.json').read_text()) if (RUN/'first-five-gate.json').exists() else None,
            stream_proof=getattr(self,'stream_proof',None),launch_provenance=json.loads((RUN/'launch-provenance.json').read_text()),
            acceptance=self.acceptance,automatic_continuation=False,automatic_new_experiments=False)
        result['preserved_implementation_failure'] = dict(
            path=str(FAILED_ATTEMPT),stage='First update completed; failed before writing its diagnostic row',
            reason='Diagnostic torch.equal compared CUDA tokens against CPU reference',
            correction='Compare already-CPU token lists; no model/objective/optimizer changes',
            resumed=False,new_run_fresh_common_step0=True,
            raw_log_sha256=sha(FAILED_ATTEMPT/'train500.log'),
            raw_log_bytes=(FAILED_ATTEMPT/'train500.log').stat().st_size,uploaded=False)
        if self.result:
            percent = result['scores_percent']
            local_baseline = json.loads((OUT/'STEP500_RESULTS.json').read_text())
            result['delta_vs_historical_S02_pp'] = {k:percent[k]-v for k,v in HISTORICAL.items()}
            result['delta_vs_current_local_S02_pp'] = {k:percent[k]-local_baseline['scores_percent'][k] for k in HISTORICAL}
            result['dataset_R1_delta_vs_current_local_S02_pp'] = {d:{r:100*(v[r]['R@1']-local_baseline['metrics'][d][r]['R@1']) for r in ('I2T','T2I')} for d,v in result['metrics'].items()}
            result['historical_Urban_R1_delta_pp'] = {r:100*result['metrics']['Urban-1k'][r]['R@1']-v for r,v in [('I2T',90.1),('T2I',88)]}
            from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics
            _,raw,sources = native_metrics(self.train)
            result['native_evaluation_provenance'] = {d:dict(**{k:v[k] for k in ('protocol','n_images','n_captions','manifest_sha256','checkpoint_sha256','native_only','native_student_only') if k in v},
                result_path=str(sources[d]),result_sha256=sha(sources[d])) for d,v in raw.items()}
        dump(OUT/'ALL_DETAIL500_RESULTS.json',result)
        lines = ['# S=0.2 AllDetail single-variable step500 experiment','',f'Status: `{result["status"]}`. Updates {len(steps)}/500; horizon4868. Fresh common step0; no resume.',
            '', 'Only method change: D uses every ordered non-summary sentence in the same visible F pool as RandomDetail. '
            'F/S strings/tokens, fallback, packing, sampler, optimizer/LR, weights[1.4,0.2,1.4], sparsity/inclusion/ramp and model frozen.',
            f'Local-only root: `{IMAGES}`; miss/symlink/escape fails; no NFS fallback. No staging/audit runs concurrently.',
            f'Common step0 SHA256: `{STEP0_SHA}`.', '',
            '| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |','|---|---|---|']
        for d,v in result.get('metrics',{}).items():
            vals = [' / '.join(f'{v[r][k]*100:.6f}' for k in ('R@1','R@5','R@10')) for r in ('I2T','T2I')]
            lines.append(f'| {d} | {vals[0]} | {vals[1]} |')
        lines += ['', 'Scores (%): `'+json.dumps(result.get('scores_percent'))+'`.',
            'Delta vs historical S0.2 (pp): `'+json.dumps(result.get('delta_vs_historical_S02_pp'))+'`.',
            'Urban delta vs historical (pp): `'+json.dumps(result.get('historical_Urban_R1_delta_pp'))+'`.',
            'Matched current local baseline deltas: `'+json.dumps(result.get('delta_vs_current_local_S02_pp'))+'`.',
            'Decision: `'+json.dumps(result.get('urban_gate'))+'`.',
            'All500 sample/F/S stream proof: `'+json.dumps(result.get('stream_proof'))+'`.',
            'Diagnostics: `ALL_DETAIL500_DIAGNOSTICS.json`; runtime/resources/raw-log inventory: `ALL_DETAIL500_RUNTIME_STATS.json`.',
            'Strict native bare inference: normalized image/full-caption embeddings, plain inner product. No mask/gate/rerank/ensemble.',
            'Checkpoint/bare SHA: `'+json.dumps(result.get('strict_export'))+'`.',
            'Checkpoints/bare, local images, training index and raw logs remain local. Ephemeral overlay cache; persistent NFS originals retained.',
            'No automatic new sampling/weights/full4868 experiment. GitHub publication requires separate small-file review.',
            'Error: '+str(self.error)]
        (OUT/'ALL_DETAIL500_RESULTS.md').write_text('\n'.join(lines)+'\n')
        dump(RUN/'supervisor-result.json',dict(status=result['status'],completed_steps=len(steps),error=self.error,finished_at=now()))
        print(json.dumps(dict(status=result['status'],completed_steps=len(steps),scores=result.get('scores_percent'))),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    args, remaining = parser.parse_known_args()
    configure()
    if args.worker:
        from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_train_gate as gate
        gate.compare_prefix = compare_prefix
        gate.checkpoint_invariants = checkpoint_invariants
        sys.argv = [sys.argv[0],*remaining]
        base.worker()
    else:
        assert not remaining
        def stop(sig, frame):
            raise RuntimeError('HARD_STOP: supervisor signal '+str(sig))
        signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
        Supervisor().run()


if __name__ == '__main__':
    main()
