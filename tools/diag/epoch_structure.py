"""Report the epoch structure of a run: how many optimizer steps one epoch actually contains."""
import json
import os
import sys

path = sys.argv[1]
rows = [json.loads(line) for line in open(os.path.join(path, 'salu_log.jsonl')) if line.strip()]
summary = json.load(open(os.path.join(path, 'run_summary.json')))
config = json.load(open(os.path.join(path, 'config.json')))

print('records %d' % len(rows))
print('steps_per_epoch field = %s' % summary.get('steps_per_epoch'))
print('loader_batches (config) = %s' % config.get('loader_batches'))
for row in rows[:2] + rows[-2:]:
    print('   completed=%s epoch=%s step_in_epoch=%s'
          % (row['completed_steps'], row['epoch'], row['step_in_epoch']))
print('max step_in_epoch seen = %d' % max(row['step_in_epoch'] for row in rows))
print('max completed seen     = %d' % max(row['completed_steps'] for row in rows))
print('epochs seen = %s' % sorted({row['epoch'] for row in rows}))
print()
print('A step_in_epoch that keeps growing past steps_per_epoch would mean the epoch counter does '
      'not advance; a reset back to 0 means one epoch really is steps_per_epoch optimizer steps.')
