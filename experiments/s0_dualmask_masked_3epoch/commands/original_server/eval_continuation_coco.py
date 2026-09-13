"""Call the frozen COCO canonical evaluator, with a strict bare-student load."""
import hashlib
import json
import sys
import time
import argparse
from pathlib import Path
sys.path[:0] = ['/root/SAID-gap-completion', '/root/SAID-gap-completion/eval/retrieval']
from tools.eval_urban1k_cls import load_student, sha256_of
from coco_retrieval import evaluate_coco

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
parser.add_argument('--step', type=int, required=True)
args = parser.parse_args()
run = Path(args.run)
checkpoint = run/f'bare_student_step{args.step}.pt'
started = time.perf_counter()
model, preprocess, meta = load_student(str(checkpoint), 'ViT-B/16', 'cuda:0')
annotation = Path('/root/datasets/coco/annotations/captions_val2017.json')
ann = json.loads(annotation.read_text())
assert len(ann['images']) == 5000
metrics = evaluate_coco(model, preprocess, root='/root/datasets/coco/val2017',
    ann_file=str(annotation), batch_size=64, similarity_chunk=512, device='cuda:0',
    image_representation='legacy_cls')
report = {'protocol': 'frozen-coco-canonical-native-cls', 'n_images': 5000, 'n_texts': 25000,
    'similarity_chunk': 512, 'image_representation': 'normalize(student.encode_image)',
    'text_representation': 'normalize(student.encode_text)', 'metrics': metrics,
    'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256_of(checkpoint),
    'evaluator_sha256': sha256_of('/root/SAID-gap-completion/eval/retrieval/coco_retrieval.py'),
    'annotation_sha256': sha256_of(annotation), 'strict_load': meta, 'elapsed_seconds': time.perf_counter()-started}
(run/f'coco_canonical_step{args.step}.json').write_text(json.dumps(report, indent=2, sort_keys=True))
print(json.dumps(report, indent=2, sort_keys=True))
