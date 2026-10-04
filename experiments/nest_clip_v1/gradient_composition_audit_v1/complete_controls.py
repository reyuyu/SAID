"""Complete precision-only diagnostics after preserving the strict-repeat failure."""
from pathlib import Path
import shutil
from experiments.nest_clip_v1.gradient_composition_audit_v1.controller import command
from experiments.nest_clip_v1.gradient_composition_audit_v1.run_audit import EXP,RUN,stages,load,dump
from train.nested_semantic_data import file_sha

def main():
    for stage in ['0','R','B']:assert load(RUN/('stage_'+stage)/'result.json')['status']=='COMPLETE'
    shutil.copy2(RUN/'execution/fp32-control-0.console.txt',EXP/'evidence/failed-FP32-bitwise-repeat.txt')
    command('fp32-control-0-v2',['--stage','0','--batches','1','--pilot','--label','fp32_v2','--fp32-control'])
    for stage in ['R','B']:command('fp32-control-'+stage,['--stage',stage,'--batches','1','--pilot','--label','fp32','--fp32-control'])
    provenance=load(EXP/'CHECKPOINT_PROVENANCE.json')
    for stage,record in stages().items():
        sha=file_sha(record['checkpoint']);assert sha==provenance[stage]['sha256_before']
        provenance[stage].update(sha256_after=sha,unchanged=True)
    dump(EXP/'CHECKPOINT_PROVENANCE.json',provenance)
    from experiments.nest_clip_v1.gradient_composition_audit_v1.report import generate
    generate()

if __name__=='__main__':main()
