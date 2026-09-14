"""Seed a NEW output folder from the original checkpoint and inherited log, without tensor edits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

BUNDLE=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):
            h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--step',required=True,type=int,choices=[500,1000])
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    manifest=json.loads((BUNDLE/'manifests/assets.json').read_text())
    expected=manifest['masked'+str(args.step)]['sha256']
    if sha(args.checkpoint)!=expected:
        raise SystemExit('not the original recorded checkpoint: SHA256 mismatch')
    lines=(BUNDLE/'evidence/training_steps_000001_001000.jsonl').read_text().splitlines()[:args.step]
    if [json.loads(line)['completed_steps'] for line in lines]!=list(range(1,args.step+1)):
        raise SystemExit('inherited log is not consecutive')
    args.output.mkdir(parents=True,exist_ok=False)
    checkpoint=args.output/f's0_dual_mask_suffix_masked_step{args.step:06d}.pt'
    shutil.copyfile(args.checkpoint,checkpoint)
    assert sha(checkpoint)==expected
    (args.output/'salu_log.jsonl').write_text('\n'.join(lines)+'\n')
    (args.output/'migration.json').write_text(json.dumps({'source':str(args.checkpoint),
        'step':args.step,'checkpoint_sha256':expected,'checkpoint_content_modified':False,
        'note':'Use the original config init_state absolute path when resuming migrated original checkpoints.'},indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'inherited_updates':args.step,'checkpoint':str(checkpoint)}))


if __name__=='__main__':
    main()
