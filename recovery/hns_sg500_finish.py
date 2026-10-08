"""Continue ONLY immutable post500 audit/report/publication after torchrun CLI rejection."""
import json
import signal
import sys

from recovery import hns_sg500 as sg
from recovery.s02_nfs500 import dump, sha, now


def main():
    sg.configure();sg.search.activate('HNS-SG')
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    run=sg.search.RUN;exp=sg.EXP
    saved=sg.read(run/'supervisor-result.json')
    assert saved['error'] is None and saved['acceptance']['passed']
    assert sg.read(exp/'VALIDATION.json')['passed'] and sg.read(exp/'RESULTS.json')['completed_steps']==500
    checkpoint=run/'step500/step000500.pt';identity=sha(checkpoint)
    assert identity==sg.read(exp/'RESULTS.json')['checkpoint_sha256']
    commands=sg.read(run/'commands.json')
    assert commands[-1]['name']=='HNS-SG-matched-gradient-audit500' and commands[-1]['returncode']==2
    failure=__import__('pathlib').Path(commands[-1]['raw_log'])
    assert 'ambiguous option: --run' in failure.read_text()
    assert not (exp/'GRADIENT_AUDIT.json').exists(),'Do not repeat an accepted audit'
    launch=sg.read(run/'launch-provenance.json')
    assert all(sha(sg.ROOT/p)==h for p,h in launch['source_sha256'].items())
    supervisor=sg.local.Supervisor();supervisor.started=saved['started_utc'];supervisor.commands=commands
    sg.runner.state('POST500_MATCHED_AUDIT_RECOVERY','HNS-SG',training_complete=True,no_training_resume=True)
    # -- before module stops torchrun from interpreting the module's --run as
    # its own abbreviated --run-path. The audited code and weights are unchanged.
    command=[str(sg.ROOT/'.venv/bin/torchrun'),'--standalone','--nnodes=1','--nproc-per-node=4',
        '--max-restarts=0','-m','--','recovery.hns_sg_gradient_audit',
        '--run',str(run),'--experiment',str(exp)]
    try:
        supervisor.execute('HNS-SG-matched-gradient-audit500-cli-fixed',command)
        assert sg.read(exp/'GRADIENT_AUDIT.json')['passed'] and sha(checkpoint)==identity
        saved.update(commands=supervisor.commands,ended_utc=now());dump(run/'supervisor-result.json',saved)
        recovery=dict(passed=True,only_post500_audit_reexecuted=True,training_updates_reexecuted=0,
            checkpoint_sha256_before=identity,checkpoint_sha256_after=sha(checkpoint),
            failure_reason='torchrun argparse abbreviation collision --run/--run-path; audit module not entered',
            remedy='Insert -- before module; original launch sources and trained weights unchanged',
            preserved_failed_log=dict(path=str(failure),bytes=failure.stat().st_size,sha256=sha(failure),uploaded=False),
            actual_audit_command=command,reviewed_utc=now())
        dump(exp/'POST500_RECOVERY.json',recovery)
        sg.runner.augment_diagnostics('HNS-SG');sg.summarize()
        stats=sg.read(exp/'RUNTIME_STATS.json')
        log=run/'HNS-SG-matched-gradient-audit500-cli-fixed.log'
        stats['additional_local_artifacts'].append(dict(path=str(log),bytes=log.stat().st_size,
            sha256=sha(log),uploaded=False,time_range_utc=[supervisor.commands[-1]['started_utc'],supervisor.commands[-1]['ended_utc']]))
        dump(exp/'RUNTIME_STATS.json',stats)
        for name in ('REPORT.md','SEARCH_SUMMARY.md'):
            with (exp/name).open('a') as out:
                out.write('\nPost500 audit CLI recovery only: torchrun needed -- before module. Original audit rejection and corrected exact command are archived in POST500_RECOVERY.json/COMMANDS.json. No training/evaluation restart.\n')
        validation=sg.read(exp/'VALIDATION.json')
        validation.update(matched_SG_noSG_gradient_audit_passed=True,
            same_graph_forward_exact=True,child_hierarchy_gradient_exact_zero=True,parent_gradient_exact=True)
        dump(exp/'VALIDATION.json',validation)
        original=sg.publication_paths
        sg.runner.publication_paths=lambda:original()+[exp/'POST500_RECOVERY.json']
        from recovery import check_stage500_publish as check
        check.ALLOWED.update({'recovery/hns_sg500_finish.py',str((exp/'POST500_RECOVERY.json').relative_to(sg.ROOT))})
        sg.runner.state('PUBLISHING',completed=['HNS-SG'])
        receipt=sg.runner.publish()
        sg.runner.state('COMPLETED_AND_SYNCED',completed=['HNS-SG'],github=receipt)
    except BaseException as error:
        sg.runner.state('STOPPED_WITH_EVIDENCE','HNS-SG',error=type(error).__name__+': '+str(error),
            automatic_retry=False,training_complete=True,no_training_resume=True)
        raise


if __name__=='__main__':main()
