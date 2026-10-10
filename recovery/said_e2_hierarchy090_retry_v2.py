"""One authorized retry, isolated from the failed run; no scientific changes."""
from pathlib import Path
import os
import subprocess

from recovery import said_e2_hierarchy090_full4868 as run
from recovery import hierarchy090_retry_preflight as checks
from recovery.s02_nfs500 import dump, sha, now

BRANCH='experiment/said-e2-hierarchy090-full4868-retry-v2'
EXP=run.ROOT/'experiments/nest_clip_v1/said_e2_hierarchy090_full4868_retry_v2'
RUN=run.PROJECT/'runtime/SAID-nest-clip-v1/said-e2-hierarchy090-full4868-retry-v2'
ENTRY='recovery.said_e2_hierarchy090_retry_v2'
CODE=(checks.CONTROLLER,'tests/test_hierarchy090_retry.py',
    'recovery/said_e2_hierarchy090_retry_v2.py','recovery/hierarchy090_retry_preflight.py',
    'tests/test_said_e2_hierarchy090_full4868.py')
ORIGINAL_PREPARE=run.prepare
ORIGINAL_REPORT=run.report


def phase(target):
    assert target==4868
    return run.protocol.local.IMAGES.parent/'formal-said-e2-hierarchy090-full4868-retry-v2'


def prepare():
    assert not EXP.exists() and not RUN.exists() and not phase(4868).exists(), 'Refuse second attempt or overwrite'
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/experiment/said-e2-hierarchy090-full4868-v1'],cwd=run.ROOT,text=True).split()[0]
    assert remote==checks.FAILED_COMMIT
    tests=[str(run.PROJECT/'.venv/bin/python'),'-m','pytest','-q',
        'tests/test_hierarchy090_retry.py','tests/test_said_e2_hierarchy090_full4868.py',
        'tests/test_said_e2_hierarchy090.py','tests/test_hns_s12_2434_validation.py']
    check=subprocess.run(tests,cwd=run.ROOT,capture_output=True,text=True,env=dict(os.environ,OMP_NUM_THREADS='4'))
    assert check.returncode==0,check.stdout+check.stderr
    receipt=checks.provenance()
    ORIGINAL_PREPARE()
    receipt['retry_wrapper_sha256']=sha(Path(__file__))
    dump(EXP/'REPAIR_AND_RETRY_PROOF.json',receipt)
    dump(EXP/'REPAIR_CPU_TESTS.json',dict(passed=True,command=tests,returncode=0,summary=check.stdout+check.stderr,utc=now()))
    plan=run.protocol.common.read(EXP/'PLAN.json')
    plan.update(controlled_retry_number=1,failed_attempt_commit=checks.FAILED_COMMIT,
        fresh_runtime=True,restore_update=500,reexecute501=True,skip502=False)
    dump(EXP/'PLAN.json',plan)


def report(target,supervisor,result):
    ORIGINAL_REPORT(target,supervisor,result)
    preserved=checks.immutable_snapshot(run.protocol.common.read(EXP/'REPAIR_AND_RETRY_PROOF.json')['failure_evidence_snapshot'])
    dump(EXP/'FAILURE_EVIDENCE_IMMUTABLE.json',preserved)


def main():
    run.BRANCH=BRANCH;run.EXP=EXP;run.RUN=RUN;run.ENTRY=ENTRY;run.CODE=CODE
    run.phase=phase;run.prepare=prepare;run.report=report
    run.main()


if __name__=='__main__':main()
