import argparse,hashlib,json,pathlib,sys,torch
ROOT=pathlib.Path(__file__).resolve().parents[1];REPO=pathlib.Path('/root/lk_projects/SAID-reproduction/full-resume')
sys.path[:0]=[str(REPO),str(REPO/'train')]
from model import longclip
from model.dual_mask_suffix import DualMaskSuffixTrainModule
from train_said_cls_cvssl import state_digest
from train_dual_mask_suffix import load_checkpoint

def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
checkpoint=ROOT/'run/s0_dual_mask_suffix_masked_step003651.pt';bare=ROOT/'exports/bare_student_step3651.pt'
if bare.exists():raise RuntimeError('Refusing to overwrite export')
model,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
module=DualMaskSuffixTrainModule(model,'masked',lambda_suffix=3,lambda_u_sparse=0)
p=load_checkpoint(str(checkpoint),module);cfg=p['config']
for k,v in {'formal_optimizer_updates':3651,'debug_optimizer_updates':0,'world_size':4,'batch_size_per_gpu':256,'epochs':3,'loader_batches':1217,'lr_horizon_steps':3651,'seed':0,'suffix_lambda':3.,'u_sparsity_lambda':0.,'resume_completed_steps':500,'training_sha':'873b43a5bc000528311e22aac01e92045ac898e0'}.items():assert cfg[k]==v,(k,cfg[k],v)
pre=json.loads((ROOT/'validation/preflight.json').read_text())
assert cfg['resumed_from']==pre['parent_checkpoint'] and cfg['init_state']==pre['initial_state']
for k in ['lr','mask_lr','suffix_lr','weight_decay','warmup','image_chunk','text_chunk','amp_dtype']:
 assert cfg['arguments'][k]==pre['configuration'][k],k
assert cfg['arguments']['num_workers']==8
assert all(x['consumed_samples']==806428 for x in cfg['stream_summary'])
assert p['completed_steps']==3651 and p['epoch']==2 and p['step_in_epoch']==1216
assert len(p['clip_state'])==317 and len(p['suffix_gate_state'])==4
assert state_digest(p['clip_state'])==p['provenance']['state_digest']
for group in ['clip_state','suffix_gate_state']:assert all(torch.isfinite(v).all().item() for v in p[group].values())
optimizer_steps={}
for name,opt in p['optimizer_states'].items():
 steps=[int(v['step']) for v in opt['state'].values() if 'step' in v];assert steps and min(steps)==max(steps)==3651
 for entry in opt['state'].values():
  for v in entry.values():
   if torch.is_tensor(v):assert torch.isfinite(v).all().item()
 optimizer_steps[name]={'min':min(steps),'max':max(steps),'count':len(steps)}
rows=[json.loads(x) for x in (ROOT/'run/salu_log.jsonl').read_text().splitlines() if x.strip()]
assert [r['completed_steps'] for r in rows]==list(range(501,3652))
assert (rows[0]['epoch'],rows[0]['step_in_epoch'],rows[-1]['epoch'],rows[-1]['step_in_epoch'])==(0,500,2,1216)

reference=pathlib.Path('/root/lk_projects/SAID-reproduction/runs/full_replica01_3651/salu_log.jsonl')
reference_rows=[json.loads(x) for x in reference.read_text().splitlines() if x.strip()]
assert len(reference_rows)==len(rows)
for actual,expected_row in zip(rows,reference_rows):
 assert actual['completed_steps']==expected_row['completed_steps']
 assert actual['batch_stream_sha256']==expected_row['batch_stream_sha256']
 for a,b in zip(actual['rank_health'],expected_row['rank_health']):
  assert all(a[k]==b[k] for k in ['rank','consumed_samples','stream_sha256'])
(ROOT/'validation/continuation_data_stream.json').write_text(json.dumps({'status':'PASS','reference':str(reference),'compared_updates':len(rows),'rank0_batch_hashes_equal':True,'all_rank_cumulative_stream_hashes_equal':True,'scope':'sample/image IDs, prefix K and caption strings for updates501..3651; excludes pixel bytes and does not reclassify the original trainer as verifying skipped-history hashes'},indent=2)+'\n')

assert all(math_value==math_value and abs(math_value)!=float('inf') for math_value in [r['loss_total'] for r in rows])
parent=pathlib.Path('/root/lk_projects/SAID-tuning-500/runs/suffix3_sparse0/salu_log.jsonl')
old=[json.loads(x) for x in parent.read_text().splitlines() if x.strip()];assert [r['completed_steps'] for r in old]==list(range(1,501))
(ROOT/'reports/training_steps_000001_003651.jsonl').write_text(parent.read_text()+(ROOT/'run/salu_log.jsonl').read_text())
torch.save(p['clip_state'],bare);state=torch.load(bare,map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True)
assert all(torch.equal(v,state[k]) for k,v in p['clip_state'].items())
report={'status':'PASS','checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'completed_steps':3651,'new_updates':3151,'epochs':3,'lambda_suffix':3,'lambda_u_sparse':0,'bare_student':str(bare),'bare_sha256':sha(bare),'tensor_digest':state_digest(state),'tensor_count':317,'roundtrip_equal':True,'optimizer_steps':optimizer_steps,'continuation_rows':len(rows),'first_update':rows[0]['completed_steps'],'final_update':rows[-1]['completed_steps'],'training_config':cfg}
(ROOT/'exports/export.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='training_config'}),flush=True)
