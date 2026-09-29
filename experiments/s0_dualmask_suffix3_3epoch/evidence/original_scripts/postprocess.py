import datetime,fcntl,hashlib,json,os,pathlib,subprocess,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];STATUS=ROOT/'status.json'
lock=(ROOT/'postprocess.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def save(phase,**extra):
 previous=json.loads(STATUS.read_text()) if STATUS.exists() else {}
 previous.update(phase=phase,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),postprocess_pid=os.getpid(),**extra)
 temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(previous,indent=2)+'\n');temp.replace(STATUS)
 print(json.dumps({'phase':phase,**extra}),flush=True)
def execute(script,cpu=False):
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 if cpu:env['CUDA_VISIBLE_DEVICES']=''
 with (ROOT/'logs'/(script+'.log')).open('w') as f:r=subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 (ROOT/'logs'/(script+'.exitcode')).write_text(str(r.returncode)+'\n')
 if r.returncode:raise RuntimeError(f'{script} failed with exit code {r.returncode}')
try:
 save('TRAINING')
 while not (ROOT/'run/exitcode').exists():
  pid=int((ROOT/'train_launcher.pid').read_text())
  if not pathlib.Path('/proc',str(pid)).exists():raise RuntimeError('Training launcher disappeared without exit record')
  time.sleep(15)
 code=int((ROOT/'run/exitcode').read_text())
 if code:raise RuntimeError(f'Training failed with exit code {code}')
 save('VERIFYING_AND_EXPORTING')
 execute('export_final.py',cpu=True)
 save('EVALUATING',completed_steps=3651)
 execute('evaluate.py')
 save('COMPARING',completed_steps=3651)
 execute('compare.py',cpu=True)
 save('COMPLETE',completed_steps=3651,epochs=3,report=str(ROOT/'reports/FINAL_REPORT.md'),bare_student=str(ROOT/'exports/bare_student_step3651.pt'))
except Exception as e:
 save('FAILED',error=str(e),traceback=traceback.format_exc());raise
