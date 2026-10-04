"""Two preregistered independent Summary alignment doses; no adaptive continuation."""
import ast
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import statistics
import subprocess
from torch.utils.data import DistributedSampler
from model.balanced_hparam_search import trial_id
from train.nested_semantic_data import file_sha
from experiments.nest_clip_v1.balanced_summary_random_detail_500_v1 import run as base
from experiments.nest_clip_v1.armb_summary04_500_v1 import run as previous

EXP=Path(__file__).resolve().parent
ROOT=EXP.parents[2]
RUN=Path('/root/lk_projects/SAID-nest-clip-v1/armb_summary_dose_500_v1')
BRANCH='codex/nest-balanced-armb-summary-dose-500-v1'
S04=ROOT/'experiments/nest_clip_v1/armb_summary04_500_v1'
SEARCH=ROOT/'experiments/nest_clip_v1/four_arm_text_search_500_v1'
ARMS={'E1':('arm_E1_summary02',[1.4,.2,1.4]),'E2':('arm_E2_summary00',[1.5,0.,1.5])}
STREAM_CHECK=base.stream_check


def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    source=base.load(S04/'RESULTS.json');baseline=base.load(SEARCH/'BASELINE.json')
    assert file_sha(source['checkpoint'])==source['checkpoint_sha256']
    assert file_sha(source['bare_student'])==source['bare_sha256']
    assert file_sha(base.SHARED)==base.INIT_SHA
    assert file_sha(baseline['checkpoint'])==baseline['checkpoint_sha256']
    cfg=base.load(S04/'config.json')
    files=list(base.load(S04/'evidence/preflight.json')['unchanged_production_sources'])
    frozen={}
    for path in files:
        original=subprocess.check_output(['git','show','bbe5daa:'+path],cwd=ROOT)
        actual=(ROOT/path).read_bytes()
        if path=='model/balanced_hparam_search.py':
            oldtree=ast.parse(original);newtree=ast.parse(actual)
            def mathematical_nodes(tree):return [ast.dump(n,include_attributes=False) for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name=='hparams')]
            assert mathematical_nodes(oldtree)==mathematical_nodes(newtree)
        else:assert actual==original,path
        frozen[path]=dict(current_sha256=hashlib.sha256(actual).hexdigest(),reference_sha256=hashlib.sha256(original).hexdigest(),
          identical=actual==original,allowed_validation_only_change=path=='model/balanced_hparam_search.py')
    base.dump(EXP/'BASELINE.json',baseline);base.dump(EXP/'S04_REFERENCE.json',source)
    rows=base.records(Path(source['checkpoint']).parent/'steps.jsonl');assert len(rows)==500
    for rank in range(4):
        sampler=DistributedSampler(range(1245901),num_replicas=4,rank=rank,shuffle=True,seed=0,drop_last=False);sampler.set_epoch(0)
        ids=[i+1000 for i in itertools.islice(iter(sampler),500*256)]
        for step,row in enumerate(rows):
            h=hashlib.sha256(json.dumps(ids[step*256:(step+1)*256],separators=(',',':')).encode()).hexdigest()
            assert h==row['rank_health'][rank]['sampling']['sample_id_sha256']
    for key,(directory,weights) in ARMS.items():
        target=EXP/directory
        for sub in ['commands','evidence','raw']:(target/sub).mkdir(parents=True,exist_ok=True)
        config=dict(cfg,view_weights=weights,experiment_name=f'ArmB-{key}-Summary{weights[1]}-500')
        config['trial_id']=trial_id(config)
        assert sum(weights)==3.
        differences={k:dict(old=cfg[k],new=v) for k,v in config.items() if cfg[k]!=v}
        assert set(differences)=={'view_weights','experiment_name','trial_id'}
        base.dump(target/'config.json',config);base.dump(target/'BASELINE.json',baseline);base.dump(target/'S04_REFERENCE.json',source)
        prior_exp=previous.EXP;previous.EXP=target
        try:sampling=previous.audit_sampling()
        finally:previous.EXP=prior_exp
        proof=dict(passed=True,reference_commit='bbe5daabecff84e393bf0461bb25ceb5b9062f6e',
          only_behavior_change='normalized alignment view weights',weights=weights,config_differences=differences,
          zero_summary_alignment_only=weights[1]==0,Summary_forward_sampling_sparsity_inclusion_retained=True,
          frozen_production_sources=frozen,common_step0_sha256=base.INIT_SHA,all500x4_sampler_IDs_match_S04=True,
          sampling_1000_exact=sampling,scheduler_horizon=4868,stop_updates=500,no_resume=True,no_full_training=True)
        base.dump(target/'evidence/preflight.json',proof)
        base.dump(RUN/directory/'state.json',dict(status='READY',stage='preflight passed',arm=key,branch=BRANCH,started_at=base.now()))
    base.dump(EXP/'PREREGISTRATION.json',dict(arms=ARMS,order=['E1','E2'],only_alignment_weights_changed=True,
      gradient_last50_scope='Read-only replay of last50 training batch inputs at the final step500 checkpoint; not per-update historical gradients.',
      gradient_native_spot_scope='Same first8 seed0 epoch0 global batches as S0.4; no updates.',no_extra_weights=True,no_full_training=True))
    base.dump(RUN/'state.json',dict(status='RUNNING',stage='preflight passed',branch=BRANCH,completed_arms=[],started_at=base.now()))
    print(json.dumps({'preflight':'PASS','arms':ARMS,'forward_unchanged':True}),flush=True)


def bind(key):
    directory,weights=ARMS[key]
    base.EXP=EXP/directory;base.RUN=RUN/directory;base.ROOT=ROOT;base.BRANCH=BRANCH
    source=base.load(S04/'RESULTS.json');source_rows=base.records(Path(source['checkpoint']).parent/'steps.jsonl')
    def match(row,reference):
        STREAM_CHECK(row,reference)
        old=source_rows[row['step']-1]
        for a,b in zip(row['rank_health'],old['rank_health']):
            for field in ['sample_id_sha256','full_view_sha256','local_views_sha256','split_sha256','fixed_first_reference_stream_sha256']:
                assert a['sampling'][field]==b['sampling'][field],(key,row['step'],a['rank'],field)
            assert a['sampling']['random_detail_sampling']['selected_indices_sha256']==b['sampling']['random_detail_sampling']['selected_indices_sha256']
    base.stream_check=match


def diagnostics(key):
    weights=ARMS[key][1];root=base.RUN/'step500';rows=base.records(root/'steps.jsonl')
    def summarize(group):
        metrics={}
        for field in sorted(set().union(*(r.keys() for r in group))):
            values=[r[field] for r in group if isinstance(r.get(field),(int,float)) and not isinstance(r.get(field),bool)]
            if values:metrics[field]=dict(mean=statistics.fmean(values),observations=len(values))
        raw={label:metrics[p+'_i2t']['mean']+metrics[p+'_t2i']['mean'] for label,p in [('F','F'),('S','O'),('D','E')]}
        weighted={label:w*raw[label] for label,w in zip(['F','S','D'],weights)}
        return dict(metrics=metrics,raw_combined_CE=raw,weighted_CE_contribution=weighted,
          weighted_alignment_contributions={label:10/3*v for label,v in weighted.items()},
          weighted_CE_shares={label:v/sum(weighted.values()) for label,v in weighted.items()})
    errors=[]
    for row in rows:
        align=10/3*sum(w*(row[p+'_i2t']+row[p+'_t2i']) for w,p in zip(weights,['F','O','E']))
        expected=align+(row['F_sparse']+2*row['O_sparse']+2*row['E_sparse'])/3+row['inc_weight']*row['inc']
        assert __import__('math').isclose(expected,row['loss'],rel_tol=1e-5,abs_tol=3e-4);errors.append(abs(expected-row['loss']))
    actual=base.load(root/'config.json');old=base.load(S04/'RESULTS.json')['config']
    assert actual['component_initialization']==old['component_initialization'] and actual['parameter_counts']==old['parameter_counts']
    for path,sha in actual['code_sha256'].items():
        if path!='model/balanced_hparam_search.py':assert sha==old['code_sha256'][path],path
        assert sha==file_sha(ROOT/path)
    diag=dict(steps={str(s):summarize([rows[s-1]]) for s in [1,100,200,500]},last50=summarize(rows[-50:]),
      notes='Summary CE continues to be evaluated when its alignment coefficient is zero. Sparsity/inclusion/forward remain unchanged. Last50 CE refers to actual updates451..500; last50 gradient shares are a separate frozen-checkpoint input replay. Gate statistics are scheduled, so last50 gate metrics have one observation.')
    base.dump(base.EXP/'TRAINING_DIAGNOSTICS.json',diag)
    cycles=base.records(root/'cycle_timing.jsonl');normal=[r['four_rank_max_seconds'] for r in cycles if not r['warmup']]
    acceptance=base.load(root/'acceptance.json')
    resource=dict(normal_full_cycle_mean_seconds=statistics.fmean(normal),normal_cycles=len(normal),peak_allocated_gib=max(r['peak_allocated_gib'] for r in acceptance['ranks']),passed=True)
    assert len(rows)==500 and len(normal)==495 and resource['normal_full_cycle_mean_seconds']<=3 and resource['peak_allocated_gib']<=65
    base.dump(base.EXP/'RESOURCE_SUMMARY.json',resource)
    base.dump(base.EXP/'evidence/training-match.json',dict(passed=True,steps=500,ranks=4,F_S_D_tokens_indices_sample_stream_match_S04=True,
      common_initial_components_and_parameter_counts_match=True,forward_retained=True,all500_losses_reconstructed=True,max_loss_error=max(errors)))
    for name in ['steps.jsonl','cycle_timing.jsonl']:
        with (root/name).open('rb') as src,(base.EXP/'evidence'/(name+'.gz')).open('wb') as out:
            with gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0,compresslevel=9) as archive:shutil.copyfileobj(src,archive)
    return resource


def main():
    prepare()
    for key,(directory,weights) in ARMS.items():
        bind(key)
        state=base.load(RUN/'state.json');state.update(active_arm=key,stage=key);base.dump(RUN/'state.json',state)
        base.train('smoke');base.train('formal');resources=diagnostics(key);base.evaluate()
        result=base.load(base.RUN/'result.json');result.update(arm=key,weights=weights,resources=resources)
        result['scores']['Short4_R1']=statistics.fmean(result['metrics'][ds][dr]['R@1'] for ds in ['COCO','Flickr30k-test1k'] for dr in ['I2T','T2I'])
        base.dump(base.EXP/'RESULTS.json',result)
        for scope in ['spot8','last50']:
            base.execute('gradient-'+scope,[base.TORCHRUN,'--standalone','--nnodes=1','--nproc-per-node=4','--max-restarts=0','-m',
              'experiments.nest_clip_v1.armb_summary_dose_500_v1.gradient_check','--arm',key,'--scope',scope],gpu=True)
        result.update(status='COMPLETE');base.dump(base.EXP/'RESULTS.json',result)
        local=base.load(base.RUN/'state.json');local.update(status='COMPLETED',stage='native5 and gradient diagnostics complete');base.dump(base.RUN/'state.json',local)
        state=base.load(RUN/'state.json');state['completed_arms'].append(key);base.dump(RUN/'state.json',state)
    from experiments.nest_clip_v1.armb_summary_dose_500_v1.report import generate
    generate()

if __name__=='__main__':main()
