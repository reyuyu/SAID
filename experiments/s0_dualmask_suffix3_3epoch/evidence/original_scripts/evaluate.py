import concurrent.futures,datetime,hashlib,json,os,pathlib,queue,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
DOCS=pathlib.Path('/root/lk_projects/SAID-publish-suffix3');MODEL_REPO=pathlib.Path('/root/lk_projects/SAID-reproduction/full-resume');DATA=pathlib.Path('/root/lk_projects/SAID-assets')
NATIVE=DOCS/'experiments/s0_dualmask_masked_3epoch/tools/native_export_eval.py'
EXT=DOCS/'experiments/s0_dualmask_full_v01/evidence/step2000/new_evaluations/eval_extended_real.py'
PROTOCOLS={'flickr_test1k':('flickr30k_test1k.jsonl','flickr30k/images','113dbc616ca66db9400107ed4b97b56d33adae3c18608225902d97ee943600dc',1000,5000),'docci':('docci_test.jsonl','docci/images','e852a96b4efb9fa6585fd70b2686cb4e456409b1144c4a0e3c7bd8a24a36cb11',5000,5000),'dci':('dci_full.jsonl','dci/images','14530fb8bf3c7b4a75bb451412d4562c548f9ca1bdbe1fdbf39d6ee30fb4de24',7805,7805),'long_dci':('long_dci_reconstructed.jsonl','dci/images','8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b',7602,7602)}
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
export=json.loads((ROOT/'exports/export.json').read_text());assert export['status']=='PASS' and export['completed_steps']==3651 and export['lambda_suffix']==3 and export['lambda_u_sparse']==0
bare=pathlib.Path(export['bare_student']);assert sha(bare)==export['bare_sha256']
assert sha(EXT)=='5942fe84cb1befe68019d6344d7ae9fede8d453d34c9283cfafe1796ee660319'
assert sha(NATIVE)=='5dfeed501f9ab5da8f0fe0e274e508c9d0e288443f31e55d12151180d7db6765'
assert not pathlib.Path('/root/SAID-gap-completion').exists(),'Legacy absolute import path now exists; verify imports before evaluation'
for manifest,images,digest,ni,nt in PROTOCOLS.values():assert sha(DATA/'retrieval_benchmarks/manifests'/manifest)==digest
frozen=json.loads((DOCS/'experiments/s0_dualmask_masked_3epoch/manifests/frozen_evaluators.json').read_text())
for path,item in frozen.items():assert sha(DOCS/'experiments/s0_dualmask_masked_3epoch/reference_eval'/pathlib.Path(path).name)==item['sha256']
q=queue.Queue()
for gpu in range(4):q.put(gpu)
completed=[]
def one(protocol):
 gpu=q.get()
 try:
  out=ROOT/'evaluation'/protocol;out.mkdir(exist_ok=True);result=out/(protocol+'.json')
  if result.exists():raise RuntimeError('Refusing to overwrite result '+str(result))
  if protocol in ['coco','urban']:
   cmd=[sys.executable,str(NATIVE),'--repo',str(MODEL_REPO),'--action',protocol,'--checkpoint',str(bare),'--device','cuda:0','--report',str(result)]
   cmd+=['--coco-root',str(DATA/'evaluation/coco')] if protocol=='coco' else ['--urban-root',str(DATA/'evaluation/Urban1k/Urban1k')]
  else:
   manifest,images,_,_,_=PROTOCOLS[protocol]
   cmd=[sys.executable,str(EXT),'--checkpoint',str(bare),'--device','cuda:0','--batch-size','64','--output-dir',str(out),f'{protocol}:{DATA/"retrieval_benchmarks/manifests"/manifest}:{DATA/"retrieval_benchmarks"/images}']
  (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(DOCS),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
  start=time.time()
  with (out/'console.log').open('w') as f:r=subprocess.run(cmd,cwd=DOCS,env=env,stdout=f,stderr=subprocess.STDOUT)
  (out/'exitcode').write_text(str(r.returncode)+'\n')
  if r.returncode:raise RuntimeError(f'{protocol} failed; see {out}')
  raw=json.loads(result.read_text());assert raw['checkpoint_sha256']==export['bare_sha256']
  if protocol in PROTOCOLS:
   _,_,digest,ni,nt=PROTOCOLS[protocol];assert raw['manifest_sha256']==digest and (raw['n_images'],raw['n_captions'])==(ni,nt)
  elif protocol=='coco':assert raw['n_images']==5000 and raw['n_texts']==25000 and raw['similarity_chunk']==512 and raw['annotation_sha256']=='afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9'
  else:assert raw['dataset']['pair_caption_sha256']=='3b0b2a3b743ed6011fccc72f8a9777bec35b44cadf2483f707fc5ef317da996c'
  return {'protocol':protocol,'exit_code':0,'elapsed_seconds':time.time()-start,'result':str(result),'physical_gpu':gpu}
 finally:q.put(gpu)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for f in concurrent.futures.as_completed([pool.submit(one,p) for p in ['coco','urban',*PROTOCOLS]]):
  completed.append(f.result());(ROOT/'reports/evaluation_progress.json').write_text(json.dumps({'completed':completed,'total':6},indent=2)+'\n');print(json.dumps(completed[-1]),flush=True)
(ROOT/'reports/evaluation_complete.json').write_text(json.dumps({'status':'PASS','evaluations':6,'checkpoint_sha256':export['bare_sha256'],'frozen_evaluators':frozen,'extended_evaluator_sha256':sha(EXT),'completed':completed},indent=2)+'\n')
