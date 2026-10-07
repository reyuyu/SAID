"""Repair torchrun audit argument parsing; preserve completed E1 and run E2 once."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from recovery import hns_final_two500 as final
from recovery.s02_nfs500 import ROOT,dump,sha,now

SCRIPT='recovery/hns_final_two500_continue.py'
TEST='tests/test_hns_final_two500_launcher.py'
ENTRY='recovery.hns_final_two500_continue'
LOG=final.RUN_ROOT.parent/(final.RUN_ROOT.name+'.continuation.log')
IDENTITY=final.RUN_ROOT.parent/(final.RUN_ROOT.name+'.continuation.json')
ORIGINAL_TORCHRUN=final.old.torchrun


def safe_torchrun(module,*args):
    if module!='recovery.hns_gradient_audit':return ORIGINAL_TORCHRUN(module,*args)
    # Keep --run out of torchrun's top-level argparse option scan. Interpreter
    # code is one structured argv element, never a shell-interpolated command.
    code='import sys;from recovery import hns_gradient_audit as audit;sys.argv='+repr(['audit',*map(str,args)])+';audit.main()'
    return [str(ROOT/'.venv/bin/torchrun'),'--standalone','--nnodes=1','--nproc-per-node=4',
        '--max-restarts=0','--no-python',str(ROOT/'.venv/bin/python'),'-c',code]


def configure(arm):
    final.configure(arm)
    final.old.torchrun=safe_torchrun
    final.search.EXTRA_SOURCES.update({SCRIPT,TEST,str((final.search.EXP/'LAUNCHER_TESTS.json').relative_to(ROOT))})
    from recovery import check_stage500_publish as guard
    guard.ALLOWED.update({SCRIPT,TEST})
    for a in final.ARMS.values():
        p=Path(a['experiment_dir'])
        guard.ALLOWED.update(str((p/n).relative_to(ROOT)) for n in ('AUDIT_LAUNCH_RECOVERY.json','LAUNCHER_TESTS.json'))
    final.publication.publication_paths=lambda:final.report_paths(arm)+[
        final.search.EXP/'LAUNCHER_TESTS.json']+([
        ROOT/SCRIPT,ROOT/TEST,final.search.EXP/'AUDIT_LAUNCH_RECOVERY.json'] if arm=='HNS-Weak' else [])


def continue_E1():
    arm='HNS-Weak';configure(arm);exp=final.search.EXP;run=final.search.RUN
    assert final.read(exp/'VALIDATION.json')['passed']
    assert final.read(run/'full-stream-proof.json')['records']==512000
    result=final.read(exp/'RESULTS.json')
    assert result['completed_steps']==500 and result['evaluation_checkpoint_immutable']
    assert sha(run/'step500/step000500.pt')==result['checkpoint_sha256']
    commands=final.read(run/'commands.json')
    failed=[c for c in commands if c['name']=='HNS-gradient-audit500']
    assert len(failed)==1 and failed[0]['returncode']==2
    assert 'ambiguous option: --run' in Path(failed[0]['raw_log']).read_text()
    assert not (exp/'GRADIENT_AUDIT.json').exists(),'No silent readonly audit retry'
    saved=final.read(run/'supervisor-result.json')
    for p,h in final.read(run/'launch-provenance.json')['source_sha256'].items():assert sha(ROOT/p)==h
    proof=dict(training_restarted=False,formal_updates=500,training_and_evaluation_already_passed=True,
        reason='torchrun argparse abbreviated --run to --run-path/--run_path; no audit worker launched',
        original_failed_command=failed[0],all_original_launch_sources_unchanged=True,
        launcher_source_sha256=sha(ROOT/SCRIPT),started_utc=now())
    final.state('READONLY_AUDIT_LAUNCH_RECOVERY',arm)
    supervisor=final.Supervisor()
    supervisor.commands=commands;supervisor.started=saved['started_utc']
    supervisor.result=saved['result'];supervisor.acceptance=saved['acceptance']
    supervisor.execute('HNS-gradient-audit500-recovery',safe_torchrun('recovery.hns_gradient_audit',
        '--run',run,'--experiment',exp))
    dump(run/'supervisor-result.json',dict(result=supervisor.result,error=None,acceptance=supervisor.acceptance,
        commands=supervisor.commands,started_utc=supervisor.started,ended_utc=now()))
    assert final.read(exp/'GRADIENT_AUDIT.json')['passed']
    proof.update(passed=True,ended_utc=now(),recovery_command=supervisor.commands[-1],
        checkpoint_sha256_after=sha(run/'step500/step000500.pt'))
    assert proof['checkpoint_sha256_after']==result['checkpoint_sha256']
    dump(exp/'AUDIT_LAUNCH_RECOVERY.json',proof)
    final.summarize_arm(arm)
    stats=final.read(exp/'RUNTIME_STATS.json');log=Path(supervisor.commands[-1]['raw_log'])
    stats['additional_local_artifacts'].append(dict(path=str(log),bytes=log.stat().st_size,
        sha256=sha(log),uploaded=False,time_range_utc=[supervisor.commands[-1]['started_utc'],supervisor.commands[-1]['ended_utc']]))
    stats['audit_launcher_recovery']='AUDIT_LAUNCH_RECOVERY.json; no retraining or reevaluation'
    dump(exp/'RUNTIME_STATS.json',stats)
    final.state('PUBLISHING',arm);receipt=final.publication.publish()
    final.state('COMPLETED_AND_SYNCED',arm,github=receipt)
    return receipt


def queue():
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    def stop(sig,frame):raise RuntimeError('HARD_STOP continuation signal '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (final.RUN_ROOT.parent/(final.RUN_ROOT.name+'.runner.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        arm='HNS-Weak'
        try:
            receipt=continue_E1()
            assert receipt['passed'] and receipt['remote_HEAD_matches_local']
            assert final.git('rev-parse','origin/'+receipt['branch'])==receipt['commit']==final.git('rev-parse','HEAD')
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            arm='HNS-InnerFocus';configure(arm)
            subprocess.run(['git','switch','-c',final.ARMS[arm]['branch']],cwd=ROOT,check=True)
            final.state('CORRECTNESS_AND_PREFLIGHT',arm)
            subprocess.run(ORIGINAL_TORCHRUN('recovery.hns_ddp_correctness','--beta',1,2,
                '--output',final.search.EXP/'DDP_CORRECTNESS.json'),cwd=ROOT,check=True,env=dict(os.environ,OMP_NUM_THREADS='1'))
            final.matched_preflight(arm);configure(arm)
            assert not final.git('diff','--cached','--name-only')
            paths=[final.search.EXP/n for n in (*final.STATIC,'LAUNCHER_TESTS.json')]
            subprocess.run(['git','status','--short'],cwd=ROOT,check=True)
            subprocess.run(['git','add','--',*[str(p.relative_to(ROOT)) for p in paths]],cwd=ROOT,check=True)
            from recovery.check_stage500_publish import inspect
            assert inspect()['passed']
            subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
            subprocess.run(['git','commit','-m','Verify InnerFocus frozen real1000, DDP and safe audit launcher'],cwd=ROOT,check=True)
            final.push_verify(final.ARMS[arm]['branch'])
            final.state('SMOKE5_RUNNING',arm);final.Supervisor().run();final.summarize_arm(arm);final.combined()
            final.state('PUBLISHING',arm);receipt=final.publication.publish()
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            final.state('BOTH_COMPLETED_AND_SYNCED_GPU_IDLE',arm,completed_arms=list(final.ARMS),GPU_idle=True,github=receipt)
        except BaseException as error:
            final.state('STOPPED_WITH_EVIDENCE',arm,error=type(error).__name__+': '+str(error),automatic_retry=False)
            raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--detach',action='store_true');args=parser.parse_args()
    if args.detach:
        assert not IDENTITY.exists()
        assert final.git('branch','--show-current')==final.ARMS['HNS-Weak']['branch']
        for a in final.ARMS.values():assert final.read(Path(a['experiment_dir'])/'LAUNCHER_TESTS.json')['passed']
        with LOG.open('ab',buffering=0) as handle:
            p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m',ENTRY],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,
                close_fds=True,env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
        value=dict(pid=p.pid,session=p.pid,main_log=str(LOG),entry=ENTRY,start_new_session=True,
            stdin='/dev/null',training_resume=False,queue=['E1 readonly audit/report/publish','E2 fresh common0 smoke5/formal500/eval/audit/publish'],started_utc=now())
        dump(IDENTITY,value);print(json.dumps(value),flush=True)
    else:queue()


if __name__=='__main__':main()
