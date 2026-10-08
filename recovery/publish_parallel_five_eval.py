"""Publish an explicit small engineering artifact list; never weights or logs."""
import subprocess

from recovery import check_stage500_publish as review
from recovery import validate_parallel_five_eval as validation
from tools import eval_five_parallel as ev

ROOT=ev.ROOT
FILES=(
    'tools/eval_five_parallel.py', 'recovery/validate_parallel_five_eval.py',
    'recovery/publish_parallel_five_eval.py', 'tests/test_eval_five_parallel.py',
    'tests/test_parallel_eval_integration.py',
    'experiments/nest_clip_v1/armb_summary02_4epoch_v1/reproduction_full.py',
    'experiments/nest_clip_v1/balanced_hparam_search_v1/search.py',
)
REPORTS=('README.md','REPORT.md','STATE.json','SERIAL_RUN.json','PARALLEL_RUN.json',
         'COMPARISON.json','RUNTIME_COMPARISON.json','INTEGRATION.json','CPU_TESTS.json')


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def main():
    assert git('branch','--show-current')==validation.BRANCH
    assert not git('diff','--cached','--name-only'),'Unrelated staged changes'
    names=list(FILES)+['engineering/parallel_five_eval_v1/'+p for p in REPORTS]
    names += ['engineering/parallel_five_eval_v1/results/'+mode+'/'+ds+'.json'
              for mode in ('serial','parallel') for ds in ('COCO','Urban-1k','Flickr30k-test1k','DOCCI','Long-DCI')]
    assert all((ROOT/p).is_file() for p in names)
    subprocess.run(['git','add','--',*names],cwd=ROOT,check=True)
    # Reuse the project's existing binary/credential/size checker, scoped to
    # this task's explicit allowlist in this process only.
    review.ALLOWED=set(names)
    checked=review.inspect();assert checked['passed']
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    if git('diff','--cached','--name-only'):
        subprocess.run(['git','commit','-m','Validate and integrate unchanged native five-eval parallel scheduling'],cwd=ROOT,check=True)
    head=git('rev-parse','HEAD');branch=validation.BRANCH
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],cwd=ROOT,check=True,timeout=120)
    subprocess.run(['git','fetch','origin','refs/heads/'+branch+':refs/remotes/origin/'+branch],cwd=ROOT,check=True,timeout=120)
    remote=git('rev-parse','origin/'+branch)
    assert remote==head==git('rev-parse','FETCH_HEAD')
    ev.save(validation.EXP/'GITHUB_RECEIPT.json',dict(branch=branch,commit=head,remote_HEAD=remote,
        remote_HEAD_matches_local=True,push_success=True,fetch_success=True,checked_utc=ev.utc(),publication_check=checked))
    print(head)


if __name__=='__main__':main()
