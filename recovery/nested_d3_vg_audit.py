"""Read-only deterministic successful Anchor V/G audit; no training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,DistributedSampler

from model import longclip
from model.balanced_hparam_search import BalancedSearch,hparams
from model.nested_fusion_mask import pair_logits
from model.nested_joint_vg import vg_ratios,validate_regions
from recovery.s02_nfs500 import ROOT,dump,sha,now
from recovery.s02_local500 import INDEX,IMAGES
from recovery.s02_full_local_data import FullLocalDataset
from train.nested_semantic_data import collate
from train.train_nested_semantic_mask import CappedSampler,seed_all,setup,state_digest

EXP=ROOT/'experiments/nest_clip_v1/nested_d3_joint_vg500_v1'
RAW=ROOT/'runtime/SAID-nest-clip-v1/nested-d3-joint-vg500-20261007.anchor-audit'
ANCHOR_EXP=ROOT/'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1'
CHECKPOINT=ROOT/'runtime/SAID-nest-clip-v1/nested-detail-d3-balanced500-20261007/step500/step000500.pt'
EDGES=('Dall_F','D3_Dall')
KEYS=('V_Dall_F','G_Dall_F','V_D3_Dall','G_D3_Dall','keep_F','keep_Dall','keep_D3',
      'IoU_Dall_F','IoU_D3_Dall','violation_Dall_F','violation_D3_Dall')


def distribution(values):
    a=np.asarray(values,dtype=np.float64)
    assert a.ndim==1 and len(a)>0 and np.isfinite(a).all()
    return dict(count=len(a),mean=float(a.mean()),std=float(a.std(ddof=0)),
        **{name:float(np.quantile(a,q,method='linear')) for name,q in
           [('P10',.1),('P20',.2),('P25',.25),('median',.5),('P75',.75),('P80',.8),('P90',.9)]})


def regions_from_distributions(soft):
    return validate_regions({edge:dict(eps_v=soft[edge]['V']['P80'],gamma_low=soft[edge]['G']['P20'],
                     gamma_high=soft[edge]['G']['P80']) for edge in EDGES})


@torch.no_grad()
def support_vectors(pf,pd,p3):
    values={}
    for name,c,p in [('Dall_F',pd,pf),('D3_Dall',p3,pd)]:
        values['V_'+name],values['G_'+name]=vg_ratios(c,p)
        hc,hp=(c>=.5).float(),(p>=.5).float()
        values['IoU_'+name]=(hc*hp).sum(-1)/((hc+hp)>0).sum(-1).clamp_min(1)
        values['violation_'+name]=(hc>hp).float().mean(-1)
    for name,p in [('F',pf),('Dall',pd),('D3',p3)]:values['keep_'+name]=(p>=.5).float().mean(-1)
    return torch.stack([values[k] for k in KEYS],dim=-1)


def main():
    global RAW
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=Path,default=RAW)
    RAW=parser.parse_args().raw_dir
    started=now();seed_all(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rank,local,world,_=setup();assert world==4
    if rank==0:
        assert not RAW.exists(),'Do not silently overwrite/repeat the frozen audit'
        RAW.mkdir(parents=True)
    dist.barrier()
    reference=json.loads((ANCHOR_EXP/'RESULTS.json').read_text())
    assert json.loads((ANCHOR_EXP/'VALIDATION.json').read_text())['passed']
    checkpoint_sha=sha(CHECKPOINT);assert checkpoint_sha==reference['checkpoint_sha256']
    payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    assert payload['completed_steps']==500 and payload['scheduler_horizon']==4868
    cfg=payload['config']
    assert cfg['view_weights']==[1.35,1.35,.30] and cfg['sampling_mode']=='nested_detail_d3'
    assert cfg.get('view_sparsity_weights',[1.,2.,2.])==[1.,2.,2.]
    assert cfg.get('inclusion_max',1.)==1. and cfg.get('regularizer_mode','independent')=='independent'
    assert reference['evaluation_checkpoint_immutable']
    assert torch.isfinite(torch.tensor(reference['scores_percent']['Score5']))
    clip,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
    clip.load_state_dict(payload['model'],strict=True)
    module=BalancedSearch(clip.float(),arm=cfg['arm'],search_hparams=hparams(cfg),
        inclusion_hierarchy=cfg['inclusion_hierarchy'],fusion=cfg['fusion'],visual=cfg['visual'],
        condition_mode=cfg['condition_mode'],checkpoint_encoders=False,image_chunk=cfg['image_chunk'],
        text_chunk=cfg['text_chunk'],shuffle_seed=cfg['shuffle_seed'],checkpoint_pair_blocks=False)
    module.fusion_branch.load_state_dict(payload['adapter'],strict=True);del payload
    module=module.cuda(local).train().requires_grad_(False)
    before=state_digest(module.state_dict()) if rank==0 else None
    dataset=FullLocalDataset(INDEX,IMAGES,'nested_detail_d3',0)
    sampler=DistributedSampler(dataset,num_replicas=4,rank=rank,shuffle=True,seed=0,drop_last=False)
    sampler.set_epoch(0)
    # Exhaust the bounded cohort loader normally, rather than interrupting
    # prefetched workers after reaching the selected sample count.
    loader=DataLoader(dataset,batch_size=256,sampler=CappedSampler(sampler,17*256),collate_fn=collate,
        num_workers=8,drop_last=False,pin_memory=True,multiprocessing_context='spawn',prefetch_factor=2,
        generator=torch.Generator().manual_seed(0),timeout=60)
    collected=[];identities=[];count=0
    with torch.no_grad():
        for batch_number,batch in enumerate(loader,1):
            images=batch['image'].cuda(non_blocking=True)
            _,visual=module.encode_visual(images)
            probabilities=[]
            for key in ('tokens_f','tokens_o','tokens_e'):
                _,condition=module.encode_view(batch[key].cuda(non_blocking=True))
                probabilities.append(pair_logits(module,visual,condition,paired=True).sigmoid())
            vectors=support_vectors(*probabilities)
            valid=batch['valid'];selected=valid.nonzero().flatten().tolist()
            data=dict(values=vectors[valid.cuda()].cpu().numpy(),
                identities=[dict(sample_id=int(batch['sample_id'][i]),rank=rank,batch=batch_number,
                    position=i,D3_indices=batch['detail_indices'][i]) for i in selected])
            shared=[None]*world;dist.all_gather_object(shared,data)
            for value in shared:
                collected.append(value['values']);identities.extend(value['identities']);count+=len(value['identities'])
            if rank==0:print(json.dumps(dict(audit_batch=batch_number,valid_records=count,training=False)),flush=True)
    assert count>=16384
    assert all(p.grad is None for p in module.parameters())
    assert sha(CHECKPOINT)==checkpoint_sha
    if rank==0:
        after=state_digest(module.state_dict());assert after==before
        values=np.concatenate(collected,axis=0)[:16384];identities=identities[:16384]
        assert len({v['sample_id'] for v in identities})==16384
        assert np.isfinite(values).all()
        raw=RAW/'per_sample_vg.npz';np.savez(raw,values=values,sample_ids=np.array([v['sample_id'] for v in identities]))
        cohort=EXP/'ANCHOR_VG_COHORT.json'
        dump(cohort,dict(count=16384,selection='First16384 valid examples in seed0 epoch0 frozen training stream; batch then rank then within-rank order',
            seed=0,epoch=0,local_only=True,NFS_fallback=False,sample_ids=[v['sample_id'] for v in identities],
            identity_detail_local=dict(path=str(RAW/'cohort_identity_details.json'),uploaded=False)))
        dump(RAW/'cohort_identity_details.json',identities)
        stats={k:distribution(values[:,i]) for i,k in enumerate(KEYS)}
        soft={edge:dict(V=stats['V_'+edge],G=stats['G_'+edge]) for edge in EDGES}
        report=dict(passed=True,checkpoint=str(CHECKPOINT),checkpoint_sha256=checkpoint_sha,
            Anchor_result_sha256=sha(ANCHOR_EXP/'RESULTS.json'),completed_steps=500,records=16384,
            no_parameter_updates=True,state_digest_before=before,state_digest_after=after,checkpoint_unchanged=True,
            forward_mode='Production train mode, gradients disabled; native bf16 encoder and fp32 paired mask probabilities',
            epsilon=1e-6,std_definition='Population std',quantile_definition='Numpy float64 linear interpolation',
            soft_distributions=soft,hard_distributions={k:v for k,v in stats.items() if not k.startswith(('V_','G_'))},
            frozen_regions=regions_from_distributions(soft),band_rule='eps_v=P80(V); gamma_low=P20(G); gamma_high=P80(G), independently per edge',
            retrieval_metrics_not_used_to_select_bands=True,cohort=dict(path=str(cohort),sha256=sha(cohort),count=16384),
            local_raw_artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),uploaded=False,
                time_range_utc=[started,now()]) for p in (raw,RAW/'cohort_identity_details.json')],
            started_utc=started,ended_utc=now(),local_image_root=str(IMAGES),NFS_fallback=False,
            loader_exhausted_normally=True,cohort_candidate_records=17*1024,
            source_sha256={p:sha(ROOT/p) for p in ('recovery/nested_d3_vg_audit.py','model/balanced_hparam_search.py','model/nested_fusion_mask.py','model/nested_joint_vg.py')})
        dump(EXP/'ANCHOR_VG_AUDIT.json',report)
        lines=['# Read-only Anchor V/G audit','',f'Checkpoint SHA256: `{checkpoint_sha}`. Fixed16384 valid training samples, no updates, state/checkpoint immutable, local-only.',
            'Population std; float64 linear percentiles. Thresholds selected only from this audit, before Joint-VG training.',
            '', '| Edge | Quantity | mean | std | P10 | P20 | P25 | median | P75 | P80 | P90 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for edge,stat in soft.items():
            for quantity,s in stat.items():lines.append('| '+edge+' | '+quantity+' | '+' | '.join(f'{s[k]:.9f}' for k in ('mean','std','P10','P20','P25','median','P75','P80','P90'))+' |')
        lines+=['',f'Frozen V/G regions: `{report["frozen_regions"]}`.',f'Hard support distributions: `{report["hard_distributions"]}`.',
            'Sample IDs in ANCHOR_VG_COHORT.json; detailed selected indices and support arrays stay local, with size/SHA/time provenance in JSON.',
            'Audit is observational; Joint-VG starts fresh common0 and never resumes this Anchor.']
        (EXP/'ANCHOR_VG_AUDIT.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps(dict(passed=True,records=16384,bands=report['frozen_regions'])),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
