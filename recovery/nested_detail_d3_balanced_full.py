"""One fresh common0->4868 Nested D3 Balanced trajectory, unchanged native math."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

from recovery import nested_detail_d3_balanced500 as balanced
from recovery import s02_local500 as local
from recovery.s02_nfs500 import ROOT, OUT, STEP0, STEP0_SHA, PYTHON, dump, rows, sha, now
from recovery.s02_full_stage import LOCAL, IMAGES, MANIFEST_SHA, ensure_local_index

EXP=ROOT/'experiments/nest_clip_v1/nested_detail_d3_balanced_full_v1'
RUN=ROOT/'runtime/SAID-nest-clip-v1/nested-detail-d3-balanced-full-20261006'
PHASE=LOCAL/'formal-nested-detail-d3-balanced-full-phase-20261006'
CONFIG=EXP/'config.json'
TRAIN=RUN/'step4868'
REFERENCE_EXP=balanced.EXP
REFERENCE_RUN=balanced.RUN
INDEX=LOCAL/'data_index'
BASELINE=OUT/'FULL_RESULTS.json'
DEBIAS=EXP/'DEBIAS_REFERENCE.json'
CPU_TEST_COUNT=0  # Set from the reviewed test receipt, not an assumed count.


def configure():
    local.RUN,local.PHASE,local.CONFIG=RUN,PHASE,CONFIG


def frozen_config(config):
    expected=json.loads((REFERENCE_EXP/'config.json').read_text())
    assert all(config.get(k)==v for k,v in expected.items()), 'Frozen500 config drift'
    assert expected.keys()<=config.keys()
    assert config['view_weights']==[1.35,1.35,.30]


def checkpoint_invariants(current,reference):
    import torch
    c,old=current['config'],reference['config']
    frozen_config(c)
    assert c['resume'] is None and c['start_updates']==0 and c['init_sha256']==STEP0_SHA
    assert c['max_updates']==4868 and current['completed_steps']==5
    for k in ('component_initialization','data','parameter_counts','horizon','runtime_model','code_sha256',
              'batch_size','world_size','accumulation','seed','sampling_seed','shuffle_seed'):
        assert c[k]==old[k], ('Frozen construction drift',k)
    assert current['optimizer']['param_groups']==reference['optimizer']['param_groups']
    assert current['optimizer']['state'].keys()==reference['optimizer']['state'].keys()
    assert {int(s['step']) for s in current['optimizer']['state'].values()}=={5}
    pairs=[(current['model'],reference['model']),(current['adapter'],reference['adapter'])]
    pairs += [(v,reference['optimizer']['state'][k]) for k,v in current['optimizer']['state'].items()]
    for a,b in pairs:
        assert a.keys()==b.keys()
        for k,v in a.items():
            assert v.shape==b[k].shape and v.dtype==b[k].dtype and torch.isfinite(v).all()
    return dict(passed=True,fresh_common0=True,native_sources_exact=True,initialization_exact=True,
        optimizer_groups_order_exact=True,AdamW_counters=[5],finite_states=True,horizon=4868,
        cross_run_numeric_comparison=False)


def first_five_gate(module,optimizer,config):
    import torch
    import torch.distributed as dist
    from train import train_nested_semantic_mask as trainer
    trainer.save_checkpoint(module,optimizer,config,5,TRAIN)
    difference=trainer.parameter_agreement(module)
    agreement=[None]*4
    dist.all_gather_object(agreement,dict(rank=dist.get_rank(),difference=difference))
    result=None
    if dist.get_rank()==0:
        try:
            actual=rows(TRAIN/'steps.jsonl')
            expected=rows(REFERENCE_RUN/'step500/steps.jsonl')[:5]
            assert [r['step'] for r in actual]==[1,2,3,4,5]
            count=balanced.matched_rows(actual,expected)
            current=torch.load(TRAIN/'step000005.pt',map_location='cpu',weights_only=False)
            old=torch.load(REFERENCE_RUN/'step500/step000005.pt',map_location='cpu',weights_only=False)
            proof=checkpoint_invariants(current,old)
            assert all(r['difference']==0 for r in agreement)
            result=dict(passed=True,gate='BEFORE_UPDATE6',records=count,checkpoint=proof,
                sample_F_Dall_D3_tokens_indices_exact=True,LR_exact=True,rank_agreement=agreement,
                stop_updates=4868,cross_run_numeric_comparison=False)
        except Exception as error:
            result=dict(passed=False,gate='BEFORE_UPDATE6',error=type(error).__name__+': '+str(error))
        dump(RUN/'first-five-gate.json',result)
        print(json.dumps(dict(event='minimal_first_five_gate',**result)),flush=True)
    shared=[result];dist.broadcast_object_list(shared,src=0)
    assert shared[0]['passed'],shared[0].get('error')
    dist.barrier()


def worker():
    """Control/telemetry wrappers; native trainer, model, sampler and loss untouched."""
    import torch.distributed as dist
    from train import train_nested_semantic_mask as trainer
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_train_gate as gate
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import training_phase_timing as timing
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import recovery_train_gate as historical
    configure();gate.RUN=RUN
    supervisor=int(os.environ['SAID_FULL_SUPERVISOR_PID'])
    historical.require_live_supervisor(supervisor)
    context=dict(optimizer=None,first5=False,saved500=False)
    original_optimizer=trainer.build_optimizer
    original_rates=trainer.optimizer_learning_rates
    original_save=trainer.atomic_save
    original_sampling=trainer.sampling_diagnostics
    original_loader=trainer.DataLoader

    def optimizer(module):
        result=original_optimizer(module);assert not result.state
        context['optimizer']=result;return result

    def atomic_save(payload,path):
        if 'model' in payload and 'optimizer' in payload:
            gate.resumable_metadata(payload)
        return original_save(payload,path)

    def sampling(batch):
        result=original_sampling(batch)
        result['D3_selected_sentence_indices_sha256']=balanced.digest_indices(
            batch['sample_id'].tolist(),batch['detail_indices'])
        return result

    def rates(module,completed,horizon):
        historical.require_live_supervisor(supervisor)
        assert horizon==4868 and 0<=completed<4868
        cfg=json.loads((TRAIN/'config.json').read_text()) if completed in (5,500) else None
        if completed==5 and not context['first5']:
            first_five_gate(module,context['optimizer'],cfg);context['first5']=True
        if completed==500 and not context['saved500']:
            started=time.monotonic()
            trainer.save_checkpoint(module,context['optimizer'],cfg,500,TRAIN)
            agreement=trainer.parameter_agreement(module);assert agreement==0
            if dist.get_rank()==0:
                count=balanced.matched_rows(rows(TRAIN/'steps.jsonl'),rows(REFERENCE_RUN/'step500/steps.jsonl'))
                dump(RUN/'first500-stream-proof.json',dict(passed=True,records=count,
                    sample_F_Dall_D3_tokens_indices_exact=True,LR_exact=True,checkpoint_saved=True,
                    parameter_difference=agreement,save_seconds=time.monotonic()-started))
            context['saved500']=True
        return original_rates(module,completed,horizon)

    class HeartbeatIterator(timing.TimedIterator):
        def __next__(self):
            self.recorder.begin()
            stamp=time.monotonic();path=PHASE/f'heartbeat-rank{self.recorder.rank}.json'
            event=dict(rank=self.recorder.rank,step=self.recorder.counter,started_monotonic=stamp,utc=now())
            dump(path,dict(event,state='DATA_WAIT'))
            try:value=next(self.iterator)
            except StopIteration:
                self.recorder.current=None;dump(path,dict(event,state='EXHAUSTED'));raise
            except BaseException:
                dump(path,dict(event,state='DATA_WAIT_FAILED'));raise
            self.recorder.current['data_wait_s']=time.monotonic()-stamp
            dump(path,dict(event,state='ACTIVE_STEP'));return value

    trainer.build_optimizer=optimizer
    trainer.atomic_save=atomic_save
    trainer.optimizer_learning_rates=rates
    trainer.sampling_diagnostics=sampling
    trainer.NestedDataset=local.LoggedLocalDataset
    trainer.DataLoader=lambda *a,**kw:original_loader(*a,**kw,timeout=60)
    timing.TimedIterator=HeartbeatIterator
    recorder=timing.install(str(PHASE),int(os.environ['RANK']))
    try:trainer.main()
    finally:
        recorder.finish()
        if dist.is_initialized():dist.destroy_process_group()
    dump(PHASE/f'heartbeat-rank{os.environ["RANK"]}.json',dict(state='TRAINING_COMPLETE',utc=now()))


def sampling_audit():
    """Offline4000 text/token/indices proofs covering all4 epochs; no image decode."""
    import numpy as np
    import torch
    from torch.utils.data import DistributedSampler
    from train.nested_semantic_data import sampled_text_views
    from model import longclip
    from recovery.s02_full_local_data import FullLocalDataset
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    dataset=FullLocalDataset(INDEX,IMAGES,'nested_detail_d3',0)
    prior=json.loads((REFERENCE_RUN/'sampling-audit-1000.json').read_text())
    evidence=[];baseline_count=0
    for epoch in range(4):
        for rank in range(4):
            sampler=DistributedSampler(dataset,num_replicas=4,rank=rank,seed=0,shuffle=True,drop_last=False)
            sampler.set_epoch(epoch)
            for index in itertools.islice(iter(sampler),250):
                path=dataset.resolved_path(index)
                rec=json.loads(dataset._records[dataset._offsets[index]:dataset._offsets[index+1]])
                sid=index+1000
                new=sampled_text_views(rec['caption'],'nested_detail_d3',0,epoch,sid)
                repeat=sampled_text_views(rec['caption'],'nested_detail_d3',0,epoch,sid)
                assert new['views']==repeat['views'] and new['detail_indices']==repeat['detail_indices']
                assert all(torch.equal(new[k],repeat[k]) for k in ('tokens_f','tokens_o','tokens_e'))
                m=new['detail_pool_size'];chosen=new['detail_indices']
                assert len(chosen)==(min(3,m-1) if m>=2 else m)
                assert chosen==sorted(set(chosen)) and set(chosen)<=set(new['dall_indices']) and 0 not in chosen
                if new['valid']:
                    parts=new['views'][0].split('. ')
                    assert new['views'][1]=='. '.join(parts[1:])
                    assert new['views'][2]=='. '.join(parts[j] for j in chosen)
                    assert torch.equal(new['tokens_e'],longclip.tokenize([new['views'][2]],truncate=False)[0])
                row=dict(rank=rank,sample_id=sid,epoch=epoch,actual_path=str(path),valid=new['valid'],
                    strings=dict(zip(('F','Dall','D3'),new['views'])),
                    token_ids={v:new[k].tolist() for v,k in [('F','tokens_f'),('Dall','tokens_o'),('D3','tokens_e')]},
                    sentence_indices=dict(Dall=new['dall_indices'],D3=chosen),m=m,K_eff=len(chosen),
                    strict_subset=new['valid'] and len(chosen)<m,degenerate=m<=1,
                    token_lengths=new['untruncated_lengths'])
                if epoch==0:
                    assert row==prior[baseline_count];baseline_count+=1
                evidence.append(row)
    after=np.random.get_state()
    assert before[0]==random.getstate() and torch.equal(before[2],torch.get_rng_state())
    assert before[1][0]==after[0] and np.array_equal(before[1][1],after[1]) and before[1][2:]==after[2:]
    raw=RUN/'sampling-audit-4000.json';dump(raw,evidence)
    audit=dict(passed=True,records=len(evidence),epochs=[0,1,2,3],matched500_baseline_records=baseline_count,
        global_python_numpy_torch_RNG_unchanged=True,ordered_without_replacement=True,
        strict_subset_all_m_ge2=True,fallback_unchanged=True,strings_indices_token_ids_verified=True,
        sampling_sources_exact_reference=True,raw_evidence=dict(path=str(raw),bytes=raw.stat().st_size,
            sha256=sha(raw),uploaded=False),examples=[evidence[e*1000] for e in range(4)])
    dump(EXP/'SAMPLING_AUDIT.json',audit);return audit


class Supervisor(local.Supervisor):
    def __init__(self):
        configure();super().__init__();self.train=TRAIN

    def run(self):
        from recovery.resource_stall_v2 import system_snapshot
        from train.train_nested_semantic_mask import code_manifest
        from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_full as pipeline
        frozen_config(json.loads(CONFIG.read_text()))
        old=json.loads((REFERENCE_EXP/'RESULTS.json').read_text())
        assert old['status']=='D3_BALANCED_STRONG_POSITIVE'
        assert json.loads((REFERENCE_EXP/'VALIDATION.json').read_text())['passed']
        assert code_manifest()==json.loads((REFERENCE_RUN/'step500/config.json').read_text())['code_sha256']
        assert sha(STEP0)==STEP0_SHA
        ready=json.loads((LOCAL/'full-ready.json').read_text())
        assert ready['status']=='LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
        assert ready['manifest']['sha256']==MANIFEST_SHA and not system_snapshot()['memory_events'].get('oom_kill',0)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        forbidden={'recovery.s02_full_stage','recovery.s02_local_full','recovery.s02_local500',
            'recovery.local_ssd_stage','recovery.s02_nfs500','train.train_nested_semantic_mask',
            'recovery.nested_detail_d3_balanced500','recovery.nested_detail_d3_equal500'}
        for p in Path('/proc').iterdir():
            if not p.name.isdigit() or int(p.name)==os.getpid():continue
            try:args=(p/'cmdline').read_bytes().split(b'\0')
            except OSError:continue
            assert not any(a.decode(errors='replace') in forbidden for a in args),'Concurrent training/copy/audit'
        assert not RUN.exists() and not PHASE.exists(),'Never overwrite or silently relaunch'
        tests=json.loads((EXP/'CPU_TESTS.json').read_text());assert tests['passed'] and tests['exit_code']==0
        index=ensure_local_index()
        RUN.mkdir(parents=True);(RUN/'reviewed').mkdir();PHASE.mkdir()
        dump(RUN/'prelaunch-local-path-proof-5000.json',local.path_proof())
        self.audit=sampling_audit()
        sources=list(code_manifest())+['recovery/nested_detail_d3_balanced_full.py',
            'recovery/nested_detail_d3_balanced_full_evidence.py','recovery/nested_detail_d3_balanced500.py',
            'recovery/s02_local500.py','recovery/s02_full_local_data.py','recovery/local500_policy.py',
            'experiments/nest_clip_v1/armb_summary02_4epoch_v1/reproduction_train_gate.py',
            'experiments/nest_clip_v1/armb_summary02_4epoch_v1/training_phase_timing.py',
            'tools/nest_clip.py','tools/eval_nest_native.py','tools/eval_urban1k_cls.py',
            'experiments/s0_dualmask_full_v01/evidence/step2000/new_evaluations/eval_extended_real.py',
            str(CONFIG.relative_to(ROOT)),'tests/test_nested_detail_d3_balanced_full.py']
        launch=dict(started_utc=self.started,supervisor_pid=os.getpid(),step0=str(STEP0),step0_sha256=STEP0_SHA,
            resume=None,start_updates=0,stop_updates=4868,horizon=4868,local_image_root=str(IMAGES),NFS_fallback=False,
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
            source_sha256={p:sha(ROOT/p) for p in sources},local_index_sha256=index,initial_resources=system_snapshot(),
            method_changes='Only --max-updates500->4868; input JSON identical to reviewed500 config',
            reference500_checkpoint_used_for_resume=False,CPU_tests=tests,automatic_retry=False,
            persistent_checkpoints=[500,1217,2434,3651,4868],original_epoch_tail_batch_per_rank=180,
            classification_rules=dict(clear_gain_pp=.1,material_decline_pp=.2,Urban_maintained_tolerance_pp=.2,
                strong='Score5/J_long3/UrbanT2I strictly improve;Short4 decline<=.2pp',
                positive='Score5 or J_long3 improves>=.1pp;Urban both maintained within.2pp;global guards-.2pp',
                tradeoff='long or Urban improves but Score5 or Short4 declines>.2pp',
                fallback='FULL_INCONCLUSIVE for outcomes not covered by explicit positive/negative rules'),
            disposable_cache='Ephemeral Docker overlay; NFS source originals retained',pod_identity=__import__('socket').gethostname())
        dump(RUN/'launch-provenance.json',launch)
        command=[str(ROOT/'.venv/bin/torchrun'),'--standalone','--nnodes=1','--nproc-per-node=4','--max-restarts=0',
            '-m','recovery.nested_detail_d3_balanced_full','--worker','--config',str(CONFIG),'--init-state',str(STEP0),
            '--index-dir',str(INDEX),'--image-root',str(IMAGES),'--output-dir',str(TRAIN),
            '--run-type','formal','--max-updates','4868']
        try:
            self.execute('train4868',command,training=True)
            self.acceptance=json.loads((TRAIN/'acceptance.json').read_text())
            assert self.acceptance['passed'] and all(r['completed_updates']==r['updates_this_run']==4868
                and r['max_parameter_difference_from_rank0']==0 for r in self.acceptance['ranks'])
            assert json.loads((RUN/'first-five-gate.json').read_text())['passed']
            assert json.loads((RUN/'first500-stream-proof.json').read_text())['passed']
            assert all((TRAIN/f'step{s:06d}.pt').is_file() for s in (500,1217,2434,3651,4868))
            pipeline.RUN,pipeline.EXP=RUN,RUN/'reviewed'
            self.result=pipeline.Supervisor.evaluate(self,4868)
            p=self.result['scores_percent'];p.update(Score5=p['Score5_R1'],Short4=p['Short4_R1'])
        except Exception as error:
            self.error=type(error).__name__+': '+str(error);print(self.error,flush=True)
        finally:
            dump(RUN/'supervisor-result.json',dict(result=self.result,error=self.error,acceptance=self.acceptance,
                commands=self.commands,started_utc=self.started,ended_utc=now()))
        if self.error:raise RuntimeError(self.error)
        self.execute('review-final',[str(PYTHON),'-m','recovery.nested_detail_d3_balanced_full_evidence'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true')
    args,remaining=parser.parse_known_args()
    if args.worker:
        sys.argv=[sys.argv[0],*remaining];worker()
    else:
        assert not remaining
        def stop(sig,frame):raise RuntimeError('HARD_STOP:supervisor signal '+str(sig))
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        Supervisor().run()


if __name__=='__main__':main()
