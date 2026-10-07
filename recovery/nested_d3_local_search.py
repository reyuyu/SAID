"""Seven isolated common0/local-only500 arms; no combinations or full follow-up."""
import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys

from recovery import s02_local500 as local
from recovery.s02_nfs500 import ROOT,STEP0,STEP0_SHA,PYTHON,dump,rows,sha,now
from recovery.s02_full_stage import LOCAL,IMAGES,MANIFEST_SHA,ensure_local_index

EXP=ROOT/'experiments/nest_clip_v1/nested_d3_local_search500_v1'
RUN_ROOT=ROOT/'runtime/SAID-nest-clip-v1/nested-d3-local-search500-20261007'
ANCHOR_EXP=ROOT/'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1'
ANCHOR_RUN=ROOT/'runtime/SAID-nest-clip-v1/nested-detail-d3-balanced500-20261007'
BRANCH='experiment/nested-d3-local-search500-v1'
ENTRY_MODULE='recovery.nested_d3_local_search'
EXTRA_SOURCES=set()
PHASE_PREFIX='formal-nested-d3-local-search500-20261007-'
ARMS={
    'W20':dict(axis='alignment',weights=[1.40,1.40,.20],r=2.,mode='nested_detail_d3'),
    'W25':dict(axis='alignment',weights=[1.375,1.375,.25],r=2.,mode='nested_detail_d3'),
    'W35':dict(axis='alignment',weights=[1.325,1.325,.35],r=2.,mode='nested_detail_d3'),
    'W40':dict(axis='alignment',weights=[1.30,1.30,.40],r=2.,mode='nested_detail_d3'),
    'S25':dict(axis='sparsity',weights=[1.35,1.35,.30],r=2.5,mode='nested_detail_d3'),
    'S30':dict(axis='sparsity',weights=[1.35,1.35,.30],r=3.,mode='nested_detail_d3'),
    'KR234':dict(axis='granularity',weights=[1.35,1.35,.30],r=2.,mode='nested_detail_kr234'),
}
EDITED={'train/nested_semantic_data.py','train/train_nested_semantic_mask.py','model/balanced_hparam_search.py'}
SHARED_STREAM=('sample_ids','sample_id_sha256','full_view_sha256','summary_view_sha256',
               'fixed_first_reference_stream_sha256')
FULL_STREAM=SHARED_STREAM+('local_views_sha256','split_sha256')
ARM=None
RUN=PHASE=ARM_EXP=CONFIG=None


def sparsity_coefficients(r):
    return [5/(3+r)*v for v in (1.,2.,r)]


def arm_sparsity(arm):
    spec=ARMS[arm]
    return spec.get('sparsity_weights',sparsity_coefficients(spec['r']))


def random_k_arm(arm):
    return ARMS[arm]['mode'] in ('nested_detail_kr234','nested_detail_kr2m1')


def frozen_lowest_reference(arm):
    return not random_k_arm(arm) or ARMS[arm].get('strict_lowest_reference',False)


def experiment_dir(arm):
    return Path(ARMS[arm].get('experiment_dir',EXP/arm))


def arm_config(arm):
    cfg=json.loads((ANCHOR_EXP/'config.json').read_text())
    spec=ARMS[arm]
    cfg['view_weights']=spec['weights']
    cfg['sampling_mode']=spec['mode']
    if spec['axis']=='sparsity':cfg['view_sparsity_weights']=arm_sparsity(arm)
    return cfg


def frozen_config(cfg,arm):
    expected=arm_config(arm)
    assert expected.keys()<=cfg.keys()
    assert all(cfg[k]==v for k,v in expected.items()), 'Frozen arm config drift'
    assert cfg.get('view_sparsity_weights',[1.,2.,2.])==arm_sparsity(arm)
    assert ARMS[arm]['weights'][0]==ARMS[arm]['weights'][1]
    assert math.isclose(sum(ARMS[arm]['weights']),3.,abs_tol=1e-12)


def activate(arm):
    global ARM,RUN,PHASE,ARM_EXP,CONFIG
    assert arm in ARMS
    ARM=arm;RUN=RUN_ROOT/arm;ARM_EXP=experiment_dir(arm);CONFIG=ARM_EXP/'config.json'
    PHASE=LOCAL/(PHASE_PREFIX+arm)
    local.RUN,local.PHASE,local.CONFIG=RUN,PHASE,CONFIG


def indices_digest(ids,choices):
    return hashlib.sha256(json.dumps(dict(sample_ids=ids,choices=choices),
        ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def expected_indices(n,sid,arm,epoch=0):
    from train.nested_semantic_data import (sample_partial_detail_indices,
        sample_random_partial_detail_indices,sample_kr2m1_detail_indices)
    fn={'nested_detail_kr234':sample_random_partial_detail_indices,
        'nested_detail_kr2m1':sample_kr2m1_detail_indices}.get(ARMS[arm]['mode'],sample_partial_detail_indices)
    return fn(n,0,epoch,sid)


def observe_selection(batch):
    from train.nested_semantic_data import sampling_diagnostics
    value=sampling_diagnostics(batch)
    value['lowest_selected_sentence_indices_sha256']=indices_digest(
        batch['sample_id'].tolist(),batch['detail_indices'])
    return value


def matched_stream(actual,reference,arm):
    assert len(actual)==len(reference)
    count=0
    for a,b in zip(actual,reference):
        assert a['step']==b['step'] and a['epoch']==b['epoch']==0
        assert a['actual_lrs']==b['actual_lrs'] and math.isfinite(a['loss']) and a['nonfinite']==0
        old={h['rank']:h for h in b['rank_health']}
        assert {h['rank'] for h in a['rank_health']}==set(old)=={0,1,2,3}
        for h in a['rank_health']:
            s,ref=h['sampling'],old[h['rank']]['sampling']
            keys=FULL_STREAM if frozen_lowest_reference(arm) else SHARED_STREAM
            assert all(s[k]==ref[k] for k in keys), ('Stream drift',arm,a['step'],h['rank'])
            assert s['nested_d3_exact'] and h['gradients_finite'] and h['batch']==256 and h['updates']==a['step']
            choices=[expected_indices(n,sid,arm) for n,sid in zip(s['n'],s['sample_ids'])]
            assert [len(c) for c in choices]==s['K']
            assert s['lowest_selected_sentence_indices_sha256']==indices_digest(s['sample_ids'],choices)
            if frozen_lowest_reference(arm):
                reference_digest=ref.get('lowest_selected_sentence_indices_sha256',ref.get('D3_selected_sentence_indices_sha256'))
                assert reference_digest is not None
                assert s['lowest_selected_sentence_indices_sha256']==reference_digest
                assert s['K']==ref['K'] and s['n']==ref['n']
            count+=len(s['sample_ids'])
    return dict(passed=True,records=count,all_sample_ids_F_Dall_strings_tokens_exact=True,
        all_lowest_strings_tokens_exact_anchor=frozen_lowest_reference(arm),all_selected_indices_replay_verified=True,
        all_LR_exact=True,epoch=0,seed=0,horizon=4868,cross_run_numeric_comparison=False)


def checkpoint_invariants(current,reference):
    import torch
    cfg,old=current['config'],reference['config'];frozen_config(cfg,ARM)
    assert current['completed_steps']==5 and current['scheduler_horizon']==4868
    assert cfg['start_updates']==0 and cfg['resume'] is None and cfg['init_sha256']==STEP0_SHA
    assert cfg['max_updates']==500 and cfg['run_type']=='formal'
    for k in ('component_initialization','data','parameter_counts','horizon','batch_size','world_size',
              'accumulation','seed','sampling_seed','shuffle_seed','workers','optimizer_groups'):
        if k in old:assert cfg[k]==old[k],('Frozen construction drift',k)
    new_model,old_model=copy.deepcopy(cfg['runtime_model']),copy.deepcopy(old['runtime_model'])
    assert new_model.pop('view_sparsity_weights',[1.,2.,2.])==arm_sparsity(ARM)
    assert old_model.pop('view_sparsity_weights',[1.,2.,2.])==[1.,2.,2.]
    assert new_model['search_hparams'].pop('view_weights')==ARMS[ARM]['weights']
    old_model['search_hparams'].pop('view_weights');assert new_model==old_model
    changed={k for k,v in old['code_sha256'].items() if cfg['code_sha256'][k]!=v}
    assert changed==EDITED and all(cfg['code_sha256'][k]==sha(ROOT/k) for k in changed)
    assert current['optimizer']['param_groups']==reference['optimizer']['param_groups']
    assert current['optimizer']['state'].keys()==reference['optimizer']['state'].keys()
    assert {int(s['step']) for s in current['optimizer']['state'].values()}=={5}
    pairs=[(current['model'],reference['model']),(current['adapter'],reference['adapter'])]
    pairs += [(v,reference['optimizer']['state'][k]) for k,v in current['optimizer']['state'].items()]
    for new,old_state in pairs:
        assert new.keys()==old_state.keys()
        for k,v in new.items():assert v.shape==old_state[k].shape and v.dtype==old_state[k].dtype and torch.isfinite(v).all()
    return dict(passed=True,optimizer_steps=[5],fresh_common0=True,initialization_exact=True,
        optimizer_groups_order_exact=True,finite_states=True,authorized_sources=sorted(changed))


def first_five_gate(module,optimizer,output):
    import torch
    import torch.distributed as dist
    from train import train_nested_semantic_mask as trainer
    trainer.save_checkpoint(module,optimizer,json.loads((output/'config.json').read_text()),5,output)
    difference=trainer.parameter_agreement(module);agreement=[None]*4
    dist.all_gather_object(agreement,dict(rank=dist.get_rank(),difference=difference))
    result=None
    if dist.get_rank()==0:
        try:
            actual=rows(output/'steps.jsonl');assert [r['step'] for r in actual]==list(range(1,6))
            proof=matched_stream(actual,rows(ANCHOR_RUN/'step500/steps.jsonl')[:5],ARM)
            current=torch.load(output/'step000005.pt',map_location='cpu',weights_only=False)
            reference=torch.load(ANCHOR_RUN/'step500/step000005.pt',map_location='cpu',weights_only=False)
            proof['checkpoint']=checkpoint_invariants(current,reference)
            assert all(a['difference']==0 for a in agreement)
            result=dict(**proof,gate='BEFORE_UPDATE6',rank_agreement=agreement)
        except Exception as error:result=dict(passed=False,error=type(error).__name__+': '+str(error))
        dump(RUN/'first-five-gate.json',result);print(json.dumps(result),flush=True)
    shared=[result];dist.broadcast_object_list(shared,src=0)
    assert shared[0]['passed'],shared[0].get('error');dist.barrier()


def worker():
    from train import train_nested_semantic_mask as trainer
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_train_gate as gate
    trainer.sampling_diagnostics=observe_selection
    gate.first_five_gate=first_five_gate
    local.worker()


def sampling_audit():
    import numpy as np
    import torch
    from torch.utils.data import DistributedSampler
    from train.nested_semantic_data import sampled_text_views
    from recovery.s02_full_local_data import FullLocalDataset
    dataset=FullLocalDataset(local.INDEX,IMAGES,ARMS[ARM]['mode'],0)
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    previous=json.loads((ANCHOR_RUN/'sampling-audit-1000.json').read_text());evidence=[]
    for rank in range(4):
        sampler=DistributedSampler(dataset,num_replicas=4,rank=rank,seed=0,shuffle=True,drop_last=False)
        sampler.set_epoch(0)
        for index in itertools.islice(iter(sampler),250):
            path=dataset.resolved_path(index)
            record=json.loads(dataset._records[dataset._offsets[index]:dataset._offsets[index+1]])
            sid=index+1000;view=sampled_text_views(record['caption'],ARMS[ARM]['mode'],0,0,sid)
            old=previous[len(evidence)];assert old['sample_id']==sid and old['rank']==rank
            assert view['valid']==old['valid']
            for label,key,pos in [('F','tokens_f',0),('Dall','tokens_o',1),('D3','tokens_e',2)]:
                if label=='D3' and not frozen_lowest_reference(ARM):continue
                old_label='Dk' if label=='D3' and 'Dk' in old['strings'] else label
                assert view['views'][pos]==old['strings'][old_label] and view[key].tolist()==old['token_ids'][old_label]
            assert view['detail_indices']==expected_indices(view['n'],sid,ARM)
            if frozen_lowest_reference(ARM):
                old_indices=old['sentence_indices'].get('D3',old['sentence_indices'].get('lowest'))
                assert view['detail_indices']==old_indices
                if 'K' in old:assert len(view['detail_indices'])==old['K']
            assert view['detail_indices']==sorted(set(view['detail_indices']))
            assert set(view['detail_indices'])<=set(view['dall_indices'])
            label='Dk' if random_k_arm(ARM) else 'D3'
            evidence.append(dict(rank=rank,sample_id=sid,epoch=0,actual_path=str(path),valid=view['valid'],
                strings=dict(zip(('F','Dall',label),view['views'])),
                token_ids={v:view[k].tolist() for v,k in [('F','tokens_f'),('Dall','tokens_o'),(label,'tokens_e')]},
                sentence_indices=dict(Dall=view['dall_indices'],lowest=view['detail_indices']),
                m=view['detail_pool_size'],K=len(view['detail_indices']),token_lengths=view['untruncated_lengths']))
    after=np.random.get_state()
    assert len(evidence)==1000 and random.getstate()==before[0] and torch.equal(torch.get_rng_state(),before[2])
    assert before[1][0]==after[0] and np.array_equal(before[1][1],after[1]) and before[1][2:]==after[2:]
    raw=RUN/'sampling-audit-1000.json';dump(raw,evidence)
    result=dict(passed=True,records=1000,all_sample_ids_F_Dall_tokens_exact_anchor=True,
        all_lowest_text_tokens_indices_exact_anchor=frozen_lowest_reference(ARM),selected_indices_replay_verified=True,
        global_python_numpy_torch_RNG_unchanged=True,fallback_unchanged=True,examples=evidence[:4],
        raw_evidence=dict(path=str(raw),bytes=raw.stat().st_size,sha256=sha(raw),uploaded=False))
    dump(ARM_EXP/'SAMPLING_AUDIT.json',result);return result


class GradientDataset(local.FullLocalDataset):
    def __init__(self,index,images,unused_mode,seed):
        super().__init__(index,images,ARMS[ARM]['mode'],seed)


def gradient():
    from recovery import nested_detail_gradients as protocol
    protocol.RUN,protocol.EXP=RUN,ARM_EXP
    protocol.WEIGHTS=dict(zip(('F','Dall','Ds'),ARMS[ARM]['weights']))
    protocol.FullLocalDataset=GradientDataset
    original=protocol.BalancedSearch
    protocol.BalancedSearch=lambda *a,**kw:original(*a,view_sparsity_weights=arm_sparsity(ARM),**kw)
    protocol.main()
    if os.environ['RANK']=='0':
        path=ARM_EXP/'GRADIENT_SPOTCHECK.json';value=json.loads(path.read_text())
        label='Dk' if random_k_arm(ARM) else 'D3'
        def relabel(x):
            if isinstance(x,dict):return {(label if k=='Ds' else k):relabel(v) for k,v in x.items()}
            if isinstance(x,list):return [relabel(v) for v in x]
            return x
        value=relabel(value);coeff=dict(zip(('F','Dall',label),[10/3*w for w in ARMS[ARM]['weights']]))
        value['actual_gradient_coefficients']=coeff
        weighted={v:coeff[v]*g for v,g in value['mean_gradient_norms'].items()}
        value['weighted_mean_gradient_norms']=weighted
        value['weighted_lowest_Dall_ratio']=weighted[label]/weighted['Dall']
        value['weighted_lowest_F_ratio']=weighted[label]/weighted['F']
        old=json.loads((ANCHOR_EXP/'GRADIENT_SPOTCHECK.json').read_text())
        assert [b['global_sample_ids_sha256'] for b in value['batches']]==[b['global_sample_ids_sha256'] for b in old['batches']]
        value['frozen_anchor_batch_IDs_and_protocol_exact']=True
        dump(path,value)


def source_paths():
    from train.train_nested_semantic_mask import code_manifest
    return sorted(set(code_manifest())|EXTRA_SOURCES|{
        'recovery/nested_d3_local_search.py','recovery/nested_d3_local_search_evidence.py',
        str((EXP/'SEARCH_PLAN.json').relative_to(ROOT)),str((EXP/'CPU_TESTS.json').relative_to(ROOT)),
        'tests/test_nested_d3_local_search.py','recovery/nested_detail_gradients.py',
        'recovery/s02_local500.py','recovery/local500_policy.py','recovery/s02_full_local_data.py',
        'tools/nest_clip.py','tools/eval_nest_native.py',
        'experiments/nest_clip_v1/armb_summary02_4epoch_v1/reproduction_train_gate.py',
        'experiments/nest_clip_v1/armb_summary02_4epoch_v1/training_phase_timing.py',
        'experiments/nest_clip_v1/armb_summary02_4epoch_v1/reproduction_full.py',
        'experiments/s0_dualmask_full_v01/evidence/step2000/new_evaluations/eval_extended_real.py',
        str(CONFIG.relative_to(ROOT))})


class Supervisor(local.Supervisor):
    def run(self):
        from recovery.resource_stall_v2 import system_snapshot
        from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_full as pipeline
        ready=json.loads((LOCAL/'full-ready.json').read_text())
        assert ready['status']=='LOCAL_FULL_TRAINING_DATA_READY' and ready['verification']['passed']
        assert ready['manifest']['sha256']==MANIFEST_SHA
        assert json.loads((ANCHOR_EXP/'VALIDATION.json').read_text())['passed']
        anchor=json.loads((ANCHOR_EXP/'RESULTS.json').read_text())
        assert anchor['completed_steps']==500 and anchor['evaluation_checkpoint_immutable']
        assert sha(ANCHOR_RUN/'step500/step000500.pt')==anchor['checkpoint_sha256']
        tests=json.loads((EXP/'CPU_TESTS.json').read_text());assert tests['passed'] and tests['exit_code']==0
        frozen_config(json.loads(CONFIG.read_text()),ARM)
        assert sha(STEP0)==STEP0_SHA and not system_snapshot()['memory_events'].get('oom_kill',0)
        hashes=ensure_local_index()
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        forbidden={ENTRY_MODULE,'recovery.s02_full_stage','recovery.s02_local500','recovery.s02_local_full',
            'recovery.local_ssd_stage','recovery.s02_nfs500','train.train_nested_semantic_mask',
            'recovery.nested_d3_local_search','recovery.nested_detail_d3_balanced_full'}
        for p in Path('/proc').iterdir():
            if not p.name.isdigit() or int(p.name)==os.getpid():continue
            try:args=(p/'cmdline').read_bytes().split(b'\0')
            except OSError:continue
            assert not any(a.decode(errors='replace') in forbidden for a in args),'Concurrent training/copy/audit'
        assert not RUN.exists() and not PHASE.exists(),'Never silently overwrite or relaunch a trajectory'
        RUN.mkdir(parents=True);(RUN/'reviewed').mkdir();PHASE.mkdir()
        dump(RUN/'prelaunch-local-path-proof-5000.json',local.path_proof())
        audit=sampling_audit()
        launch=dict(started_utc=self.started,supervisor_pid=os.getpid(),step0=str(STEP0),step0_sha256=STEP0_SHA,
            resume=None,start_updates=0,stop_updates=500,horizon=4868,arm=ARM,spec=ARMS[ARM],
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_sha256={p:sha(ROOT/p) for p in source_paths()},CPU_tests=tests,
            local_image_root=str(IMAGES),local_index_sha256=hashes,NFS_fallback=False,
            anchor_reference=dict(result_sha256=sha(ANCHOR_EXP/'RESULTS.json'),checkpoint_sha256=anchor['checkpoint_sha256'],
                actual_steps_path=str(ANCHOR_RUN/'step500/steps.jsonl'),actual_steps_sha256=sha(ANCHOR_RUN/'step500/steps.jsonl'),
                reference_used_for_resume=False),
            config_changes={k:v for k,v in arm_config(ARM).items() if json.loads((ANCHOR_EXP/'config.json').read_text()).get(k)!=v},
            initial_resources=system_snapshot(),automatic_retry=False,automatic_full=False,automatic_combinations=False,
            sampling_audit=audit['raw_evidence'],pod_identity=__import__('socket').gethostname(),
            disposable_cache='Ephemeral Docker overlay; NFS original source retained')
        for p,h in launch['source_sha256'].items():
            assert hashlib.sha256(subprocess.check_output(['git','show',launch['git_head']+':'+p],cwd=ROOT)).hexdigest()==h
        dump(RUN/'launch-provenance.json',launch)
        command=[str(ROOT/'.venv/bin/torchrun'),'--standalone','--nnodes=1','--nproc-per-node=4','--max-restarts=0',
            '-m',ENTRY_MODULE,'--arm',ARM,'--worker','--config',str(CONFIG),
            '--init-state',str(STEP0),'--index-dir',str(local.INDEX),'--image-root',str(IMAGES),
            '--output-dir',str(self.train),'--run-type','formal','--max-updates','500']
        try:
            self.execute('train500',command,training=True)
            self.acceptance=json.loads((self.train/'acceptance.json').read_text())
            assert self.acceptance['passed'] and all(r['completed_updates']==r['updates_this_run']==500 and
                r['max_parameter_difference_from_rank0']==0 for r in self.acceptance['ranks'])
            assert json.loads((RUN/'first-five-gate.json').read_text())['passed']
            proof=matched_stream(rows(self.train/'steps.jsonl'),rows(ANCHOR_RUN/'step500/steps.jsonl'),ARM)
            assert proof['records']==512000;dump(RUN/'full-stream-proof.json',proof)
            self.execute('gradient500',[str(ROOT/'.venv/bin/torchrun'),'--standalone','--nnodes=1',
                '--nproc-per-node=4','--max-restarts=0','-m',ENTRY_MODULE,'--arm',ARM,'--gradient'])
            pipeline.EXP,pipeline.RUN=RUN/'reviewed',RUN
            self.result=pipeline.Supervisor.evaluate(self,500)
            p=self.result['scores_percent'];p.update(Score5=p['Score5_R1'],Short4=p['Short4_R1'])
        except Exception as error:
            self.error=type(error).__name__+': '+str(error);print(self.error,flush=True)
        finally:
            dump(RUN/'supervisor-result.json',dict(result=self.result,error=self.error,acceptance=self.acceptance,
                commands=self.commands,started_utc=self.started,ended_utc=now()))
        if self.error:raise RuntimeError(self.error)
        from recovery.nested_d3_local_search_evidence import review_arm
        review_arm(ARM)


def prepare():
    for arm in ARMS:
        path=experiment_dir(arm)/'config.json';expected=arm_config(arm)
        assert path.exists() and json.loads(path.read_text())==expected,'Arm configs must be reviewed and committed'
    print(json.dumps(dict(prepared=True,arms=list(ARMS),automatic_full=False,automatic_combinations=False)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=list(ARMS));parser.add_argument('--worker',action='store_true')
    parser.add_argument('--gradient',action='store_true');parser.add_argument('--prepare',action='store_true')
    args,remaining=parser.parse_known_args()
    if args.prepare:assert not remaining;prepare();return
    if args.worker or args.gradient:
        activate(args.arm);sys.argv=[sys.argv[0],*remaining]
        worker() if args.worker else gradient();return
    assert not remaining and args.arm is None,'Formal search executes all seven declared arms once'
    def stop(sig,frame):raise RuntimeError('HARD_STOP: supervisor signal '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    assert not RUN_ROOT.exists(),'No automatic relaunch; retain original evidence'
    prepare();completed=[]
    try:
        for arm in ARMS:
            activate(arm)
            dump(EXP/'SEARCH_STATE.json',dict(status='RUNNING',active_arm=arm,completed_arms=completed,updated_utc=now()))
            Supervisor().run();completed.append(arm)
        from recovery.nested_d3_local_search_evidence import summarize
        summarize()
        dump(EXP/'SEARCH_STATE.json',dict(status='ALL_ARMS_REVIEWED',active_arm=None,completed_arms=completed,
            updated_utc=now(),github_sync_pending=True,automatic_full=False,automatic_combinations=False))
    except BaseException as error:
        dump(EXP/'SEARCH_STATE.json',dict(status='STOPPED_WITH_EVIDENCE',active_arm=ARM,completed_arms=completed,
            updated_utc=now(),error=type(error).__name__+': '+str(error),automatic_retry=False))
        raise


if __name__=='__main__':main()
