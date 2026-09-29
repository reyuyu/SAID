import argparse,hashlib,json,pathlib,subprocess,sys,torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
REPO=pathlib.Path('/root/lk_projects/SAID-reproduction/full-resume')
PARENT=pathlib.Path('/root/lk_projects/SAID-tuning-500/runs/suffix3_sparse0/s0_dual_mask_suffix_masked_step000500.pt')
INIT=pathlib.Path('/root/lk_projects/SAID-reproduction/checkpoints/common_init_reconstructed.pt')
EXPECTED_PARENT='4e905bf1160f640da70de6045729cf27bf6f8ca67834198f2e029e030ec5ed21'
EXPECTED_INIT='6e9cb606eec915b5b8613b7c81dccbd73dcb6e928a3ded49ac21c0505c454483'
EXPECTED_CODE='e92232b78511c53612a4eec9ebc7fea1302f60c397eba53c81fba0b9c12ff630'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()=='873b43a5bc000528311e22aac01e92045ac898e0'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
assert sha(PARENT)==EXPECTED_PARENT and sha(INIT)==EXPECTED_INIT
assert sha(REPO/'model/dual_mask_suffix.py')==EXPECTED_CODE
sys.path[:0]=[str(REPO),str(REPO/'train')]
from model import longclip
from model.dual_mask_suffix import DualMaskSuffixTrainModule
import train_dual_mask_suffix as trainer
from train_said_cls_cvssl import state_digest
from tools.data.full_data_gate import require_full_data
DATA=pathlib.Path('/root/lk_projects/SAID-assets/training/ShareGPT4V')
audit=require_full_data('/root/lk_projects/SAID-assets/training/full_audit/sharegpt4v_full_audit.json',DATA/'share-captioner_coco_lcs_sam_1246k_1107.json',DATA)
model,_=longclip.load_from_clip('ViT-B/16',device='cpu',args=argparse.Namespace())
module=DualMaskSuffixTrainModule(model,'masked',lambda_suffix=3,lambda_u_sparse=0)
args=argparse.Namespace(lr=1e-6,mask_lr=1e-3,suffix_lr=1e-4,weight_decay=.01)
opts=trainer.build_optimizers(module,args)
p=trainer.load_checkpoint(str(PARENT),module,opts);c=p['config']
assert p['completed_steps']==c['formal_optimizer_updates']==500
required={'world_size':4,'batch_size_per_gpu':256,'epochs':3,'lr_horizon_steps':3651,'loader_batches':1217,'seed':0,'suffix_lambda':3.,'u_sparsity_lambda':0.,'init_state':str(INIT),'total_len':1000}
for k,v in required.items():assert c[k]==v,(k,c[k],v)
assert c['arguments']['num_workers']==8 and c['arguments']['amp_dtype']=='bf16'
expected={'suffix_mode':'masked','suffix_lambda':3.,'u_sparsity_lambda':0.,'batch_size_per_gpu':256,'seed':0,'epochs':3,'total_len':1000,'lr':1e-6,'mask_lr':1e-3,'suffix_lr':1e-4,'weight_decay':.01,'warmup':200,'image_chunk':16,'text_chunk':32,'amp_dtype':'bf16'}
assert trainer.validate_resume(c,c['arguments'],expected,EXPECTED_CODE,EXPECTED_CODE)==[]
assert trainer.resume_position_problem(p['epoch'],p['step_in_epoch'],1217,500) is None
assert trainer.resume_skip(0,499,1217,500) and not trainer.resume_skip(0,500,1217,500)
assert state_digest(module.clip.state_dict())==p['provenance']['state_digest']
for key,val in p['clip_state'].items():assert torch.equal(val,module.clip.state_dict()[key]) and torch.isfinite(val).all()
for key,val in p['suffix_gate_state'].items():assert torch.equal(val,module.suffix_gate.state_dict()[key]) and torch.isfinite(val).all()
steps={}
for name,opt in zip(['clip','mask','suffix'],opts):
 original=p['optimizer_states'][name];actual=opt.state_dict();assert actual['param_groups']==original['param_groups']
 for ident,fields in original['state'].items():
  for key,value in fields.items():
   if torch.is_tensor(value):assert torch.equal(value,actual['state'][ident][key]) and torch.isfinite(value).all()
   else:assert value==actual['state'][ident][key]
 values=[int(x['step']) for x in actual['state'].values() if 'step' in x];assert values and min(values)==max(values)==500
 steps[name]={'min':min(values),'max':max(values),'count':len(values)}
rows=[json.loads(s) for s in PARENT.with_name('salu_log.jsonl').read_text().splitlines() if s.strip()]
assert [x['completed_steps'] for x in rows]==list(range(1,501))
for old,new in zip(c['stream_summary'],rows[-1]['rank_health']):
 assert all(old[k]==new[k] for k in ['rank','consumed_samples','stream_sha256'])
report={'status':'PASS','parent_checkpoint':str(PARENT),'parent_sha256':EXPECTED_PARENT,'initial_state':str(INIT),'initial_state_sha256':EXPECTED_INIT,'fixed_training_sha':'873b43a5bc000528311e22aac01e92045ac898e0','objective_code_sha256':EXPECTED_CODE,'clip_state_digest':p['provenance']['state_digest'],'optimizer_steps_restored':steps,'completed_steps':500,'next_update':501,'next_epoch':0,'next_batch':500,'target_updates':3651,'additional_updates':3151,'horizon':3651,'configuration':expected,'all_restored_tensors_equal':True,'data_gate':'PASS','records':audit['records'],'parent_log_rows':len(rows),'continuation_note':'Uses the same fixed Full continuation implementation as the prior Full replica: consumed batches are replayed/skipped and only the new continuation stream is hashed. No claim of an independent content-level replay assertion.'}
(ROOT/'validation/preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
