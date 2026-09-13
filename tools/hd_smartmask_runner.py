"""HDSM v0.1 staged runner with real exit codes and an exclusive run lock."""
import argparse, json, os, subprocess, sys, time

REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY='/root/miniconda3/envs/said-smartclip/bin/python'
RUN='/root/SAID-hd-smartmask-v01/runs_salu/hdsm_v01'
INIT='/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt'

def write_status(path, **values):
    current={}
    if os.path.exists(path):
        try: current=json.load(open(path))
        except Exception: pass
    current.update(values); current['updated_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    tmp=path+'.tmp'; json.dump(current,open(tmp,'w'),indent=2,sort_keys=True); os.replace(tmp,path)

def run_stage(name, cmd, status):
    started=time.time(); write_status(status, stage=name, **{name+'_exit':None})
    env=os.environ.copy(); env.update({'PYTHONPATH':'/root/SAID-hd-smartmask-v01','SHARE4V_DATA_ROOT':'/root/datasets/ShareGPT4V','SHARE4V_JSON':'share-captioner_coco_lcs_sam_1246k_1107.json','COCO_DATA_ROOT':'/root/datasets/coco','NCCL_SOCKET_IFNAME':'lo','GLOO_SOCKET_IFNAME':'lo'})
    log=os.path.join(RUN,name+'.log')
    with open(log,'w',encoding='utf-8') as handle:
        rc=subprocess.run(cmd,cwd='/root/SAID-hd-smartmask-v01',env=env,stdout=handle,stderr=subprocess.STDOUT).returncode
    write_status(status, **{name+'_exit':rc, name+'_seconds':time.time()-started})
    return rc

def main():
    global RUN
    ap=argparse.ArgumentParser(); ap.add_argument('--max-steps',type=int,default=500); ap.add_argument('--batch-size',type=int,default=256); ap.add_argument('--num-workers',type=int,default=8); ap.add_argument('--run-dir',default=RUN); a=ap.parse_args()
    RUN=a.run_dir
    os.makedirs(RUN,exist_ok=True); status=os.path.join(RUN,'status.json'); lock=os.path.join(RUN,'.lock')
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.write(fd,str(os.getpid()).encode()); os.close(fd)
    except FileExistsError:
        raise SystemExit('HDSM run lock already exists')
    try:
        write_status(status, arm='HDSM_V01',objective='clip_highdim_smartmask_fourterm',target_steps=a.max_steps,stage='preflight',formal_training_exit=None)
        gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
        rows=[x for x in gpu.stdout.splitlines() if x.strip()]
        if gpu.returncode!=0 or len(rows)<4: write_status(status,stage='preflight',preflight_exit=1,preflight_output=gpu.stderr); return 1
        write_status(status,gpu_count=len(rows),preflight_exit=0)
        train=['/root/miniconda3/envs/said-smartclip/bin/python','-m','torch.distributed.run','--standalone','--nproc_per_node=4','--max_restarts=0','--role=','--tee','3','train/train_hd_smartmask.py','--init-state',INIT,'--output-dir',RUN,'--max-steps',str(a.max_steps),'--batch-size',str(a.batch_size),'--num-workers',str(a.num_workers)]
        rc=run_stage('formal_training',train,status)
        if rc: return rc
        ckpt=os.path.join(RUN,f'hdsm_HDSM_V01_step{a.max_steps:06d}.pt');
        if not os.path.exists(ckpt): write_status(status,stage='verify_checkpoint',verify_checkpoint_exit=1); return 1
        rc=run_stage('verify_checkpoint',[PY,'-c',"import torch,sys; p=torch.load(sys.argv[1],map_location='cpu',weights_only=False); req={'clip_state','latent_encoder_state','decoder_state','gate_state','optimizer_state','scheduler_state','completed_steps','data_cursor','rng_states','config','provenance','gate_config'}; assert req<=set(p) and int(p['completed_steps'])==int(sys.argv[2]); print('CHECKPOINT_ROUNDTRIP_OK')",ckpt,str(a.max_steps)],status)
        if rc: return rc
        student=os.path.join(RUN,f'hdsm_HDSM_V01_student_step{a.max_steps:06d}.pt')
        rc=run_stage('export_student',[PY,'tools/diag/export_hdsm_student.py','--checkpoint',ckpt,'--out',student,'--expect-steps',str(a.max_steps)],status)
        if rc: return rc
        coco=os.path.join(RUN,'coco_canonical.json')
        rc=run_stage('coco_canonical',[PY,'tools/phase30a_fixed_cohort_eval.py','--label','HDSM_V01','--gap_anti_temperature','1.0','--canonical','--canonical_only','--canonical_tags',f'final','--checkpoints',f'final:{student}','--output',coco,'--coco','--coco_root','/root/datasets/coco','--image_batch_size','64'],status)
        if rc: return rc
        urban=os.path.join(RUN,'urban1k.json')
        rc=run_stage('urban1k',[PY,'tools/eval_urban1k_cls.py','--checkpoint',student,'--label',f'HDSM_V01@{a.max_steps}','--expect-steps',str(a.max_steps),'--out',urban],status)
        if rc: return rc
        write_status(status,stage='complete',formal_training_exit=0,overall_exit=0); return 0
    finally:
        try: os.unlink(lock)
        except FileNotFoundError: pass

if __name__=='__main__': raise SystemExit(main())
