"""One-variable read-only cuDNN flag diagnostic on the observed DOCCI batch."""
import argparse
import json
from pathlib import Path
import torch
from tools.eval_hyfl_native import Images, load_model, rows_and_images
from tools.retrieval_bounded import atomic_json, sha


@torch.no_grad()
def run(checkpoint, evidence):
    evidence = Path(evidence); torch.set_num_threads(4)
    job = json.loads((evidence / 'evaluation/DOCCI/attempt1/JOB.json').read_text())
    rows, images = rows_and_images(job['manifest']); start = 2496
    model, pre = load_model(checkpoint, 'cuda:0')
    loader = torch.utils.data.DataLoader(Images(images[start:start + 64], job['image_root'], pre),
                                        batch_size=64, num_workers=4)
    x = next(iter(loader)).to('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    a = torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1).cpu()
    torch.backends.cudnn.allow_tf32 = False
    b = torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1).cpu()
    old = torch.load(evidence / 'docci_numeric_diagnosis/FROZEN_LEGACY_FEATURES.pt', weights_only=True)['images'][start:start + 64]
    p = evidence / f'evaluation/cache/DOCCI/image_{start:08d}.pt'
    meta = json.loads(p.with_suffix('.json').read_text()); assert sha(p) == meta['sha256']
    new = torch.load(p, weights_only=True)
    error_true = float((a - old).abs().max()); error_false = float((b - new).abs().max())
    result = {'batch_start': start, 'batch_size': 64, 'observed_query_index': 2516,
              'only_variable': 'torch.backends.cudnn.allow_tf32 (True versus False)',
              'same_input_parameters_preprocessing_fp32_outputs': True,
              'true_matches_frozen_legacy_max_abs': error_true,
              'false_matches_new_formal_max_abs': error_false,
              'true_false_embedding_max_abs': float((a - b).abs().max()),
              'query_embedding_max_abs': float((a[2516 - start] - b[2516 - start]).abs().max()),
              'numerical_effect_supported': max(error_true, error_false) <= 5e-6,
              'checkpoint_sha256': sha(checkpoint), 'production_code_modified': False}
    atomic_json(evidence / 'docci_numeric_diagnosis/TF32_ONE_VARIABLE_PROOF.json', result, exclusive=True)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', required=True); p.add_argument('--evidence', required=True)
    a = p.parse_args(); run(a.checkpoint, a.evidence)
