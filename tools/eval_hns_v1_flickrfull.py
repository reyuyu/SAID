"""Single-model HNS-v1@3651 entrypoint; all inference/retrieval uses frozen worker.

Only the worker's fixed model identity and source receipt are rebound, within
this dedicated process. No function, ranking rule or tensor operation changes.
"""
import argparse
import json
from pathlib import Path
from tools import eval_hyfl_native as native
from tools.retrieval_bounded import sha

MODEL_SHA = 'a60150575f7e8f7b41effb730c0f9d866757954833a240813d4695d9a1cfcb44'
CHECKPOINT = '/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/epoch-curve-eval-v1/HNS_v1/step3651/student_step3651.pt'
E2_SHA = '50634512e226e79d526e269ba1a6ca75d7248f47e75417537f1605ca71ac2a0e'
ENTRY = 'tools/eval_hns_v1_flickrfull.py'
FROZEN = {
    'model/longclip.py': 'cb34aa8b672fd4999e03db9d3b7293a8133aa8b55f0ad612db75f422b720ffb0',
    'model/model_longclip.py': '4ff824f026ed75d4070ef4889130ac14466f72b86bc3eb8575ee7ea8388ff4e9',
    'model/simple_tokenizer.py': 'd1b09f10ed3d1e2c343619cb98cbfacb92363dd8607526a43aa7fe2d60d15bad',
    'model/bpe_simple_vocab_16e6.txt.gz': '924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a',
    'tools/eval_hyfl_native.py': '09c634564209c9b7e71545243e3537638095af94b6bd6eb5e28c21f1969f2cab',
    'tools/retrieval_bounded.py': 'be043a1fe98f79936c31062f37ce1406332475a5d17c5c15c5e57fc79918b088',
    'tools/eval_resource_queue.py': '14d58eb9d4deb8a1861d7b67c65044c6bdf440198cdde408bb731fc4880ab8c4',
}


def bind():
    for name, expected in FROZEN.items():
        if sha(native.ROOT / name) != expected:
            raise ValueError('frozen source changed: ' + name)
    if native.MODEL_SHA not in (E2_SHA, MODEL_SHA):
        raise ValueError('unexpected worker identity')
    native.MODEL_SHA = MODEL_SHA
    if ENTRY not in native.SOURCES:
        native.SOURCES = native.SOURCES + (ENTRY,)


def validate_job(job, checkpoint):
    if Path(checkpoint).resolve() != Path(CHECKPOINT).resolve():
        raise ValueError('only the specified HNS-v1@3651 file is authorized')
    fixed = {
        'name': 'Flickr30k-Full',
        'manifest': '/root/said_hyfl_data_completion_v1/full_protocol/FULL_MANIFEST.jsonl',
        'manifest_sha256': 'bfed72a299e4d8cbfef817c13c836cd5aa189e18dd2fd29b632f5808e8dd12bc',
        'image_root': '/root/said_hyfl_data_completion_v1/flickr_images',
        'image_content_sha256': 'e4fef70dd3f73034970b1ec0e20d81a4b421d1871e350e3266e41b744d3ab159',
        'protocol': 'Flickr30k-Full-official-token-31783x158915',
        'protocol_status': 'VERIFIED_SOURCE_PROTOCOL',
        'image_batch': 64, 'text_batch': 64, 'query_chunk': 256, 'gallery_chunk': 4096,
        'n_images': 31783, 'n_captions': 158915,
    }
    for key, expected in fixed.items():
        if job.get(key) != expected:
            raise ValueError('fixed Full Flickr setting mismatch: ' + key)
    if job.get('normalization', 'gpu') != 'gpu' or 'legacy_expected' in job:
        raise ValueError('not the frozen Full inference/ranking path')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True); p.add_argument('--cache-dir', required=True)
    p.add_argument('--device', required=True)
    a = p.parse_args()
    job = json.loads(Path(a.job).read_text())
    validate_job(job, a.checkpoint)
    bind()
    native.run(job, a.checkpoint, a.output, a.cache_dir, a.device)


if __name__ == '__main__':
    main()
