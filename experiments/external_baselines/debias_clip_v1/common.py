"""Pinned official implementation and strict released-checkpoint loading."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

OFFICIAL = Path('/root/lk_projects/DeBias-CLIP-official')
COMMIT = '18a06c98bcb50018e22c3febca2aefc8c4a12e0d'
EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[2]
RUN = Path('/root/lk_projects/SAID-nest-clip-v1/debias_clip_official_v1')
DATA = RUN / 'official-data'
ASSETS = Path('/root/lk_projects/SAID-assets')
BENCH = ASSETS / 'retrieval_benchmarks'
CHECKPOINT = ASSETS / 'external_baselines/debias_clip_v1/checkpoints/debias_vitb_3e.pt'
LONG_SHA = '8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str)+'\n')


def official_import(name):
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=OFFICIAL, text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=OFFICIAL, text=True).strip()
    if str(OFFICIAL) not in sys.path:
        sys.path.insert(0, str(OFFICIAL))
    return importlib.import_module(name)


def official_runtime():
    import torch
    # The untouched main.py sets these explicitly for CUDA evaluation.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_num_threads(8)


def reconstruct(checkpoint=CHECKPOINT, device='cuda:0', precision='amp'):
    import torch
    factory = official_import('local_clip.factory')
    internal = json.loads((RUN / 'checkpoint-internal.json').read_text())
    assert sha256(checkpoint) == internal['sha256']
    assert internal['epoch'] == 3 and internal['state_dict_key_count'] == 303
    assert internal['shapes']['visual.conv1.weight'] == [768, 3, 16, 16]
    assert internal['shapes']['text.positional_embedding'] == [248, 512]
    assert internal['shapes']['text.positional_embedding_res'] == [248, 512]
    raw_payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    raw_state = {k.removeprefix('module.'): v for k, v in raw_payload['state_dict'].items()}
    del raw_payload
    loads = []
    original = torch.nn.Module.load_state_dict
    def strict_load(module, state, strict=True, *a, **kw):
        assert strict is True, 'No non-strict checkpoint load is permitted'
        result = original(module, state, strict=strict, *a, **kw)
        actual = module.state_dict()
        equal = all(torch.equal(actual[k].cpu(), v.cpu()) for k, v in state.items())
        assert not result.missing_keys and not result.unexpected_keys and equal
        loads.append({'module': type(module).__name__, 'tensor_count': len(state),
                      'missing_keys': result.missing_keys, 'unexpected_keys': result.unexpected_keys,
                      'all_loaded_tensors_equal_effective_official_state': equal})
        return result
    official_runtime()
    with patch.object(torch.nn.Module, 'load_state_dict', strict_load):
        model, _, transform = factory.create_model_and_transforms(
            'ViT-B-16-longclip', str(checkpoint), precision=precision, device='cpu',
            force_custom_clip=True, force_quick_gelu=True, output_dict=True,
            longclip_keep_length=20, longclip_pca_dim=32)
    assert len(loads) == 1 and loads[0]['tensor_count'] == 303
    loaded_state = model.state_dict()
    raw_mismatches = [{'key': k, 'max_abs_difference': float((loaded_state[k].cpu()-v.cpu()).abs().max())}
                      for k, v in raw_state.items() if not torch.equal(loaded_state[k].cpu(), v.cpu())]
    del loaded_state, raw_state
    assert model.context_length == 248 and model.longclip and model.longclip_pca_dim == 32
    model.to(device).eval()
    tokenizer = factory.get_tokenizer('ViT-B-16-longclip', context_length=None, is_siglip='none')
    assert tokenizer.context_length == 248
    metadata = {**internal, 'official_commit': COMMIT, 'strict_load': loads,
        'construction': {'model': 'ViT-B-16-longclip', 'force_custom_clip': True,
            'force_quick_gelu': True, 'longclip_keep_length': 20, 'longclip_pca_dim': 32,
            'precision': precision},
        'tokenizer': type(tokenizer).__module__+'.'+type(tokenizer).__name__,
        'image_preprocess': repr(transform), 'preprocess_cfg': model.visual.preprocess_cfg,
        'raw_checkpoint_tensor_mismatches_after_official_loading': raw_mismatches,
        'all_loaded_tensors_equal_raw_checkpoint': not raw_mismatches,
        'official_source_modified': False, 'no_training': True,
        'configuration_provenance': 'No args in checkpoint; actual shapes plus pinned official test-script defaults. Training flags recorded separately from saved metadata.'}
    return model, tokenizer, transform, metadata


class DeBiasCLIPNativeAdapter:
    """Call only official independent encoders, with official L2 normalization."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def encode_image_native(self, images):
        return self.model.encode_image(images, normalize=True, output_tokens=False)['image_features']

    def encode_text_native(self, texts):
        import torch
        tokens = texts if torch.is_tensor(texts) else self.tokenizer(texts)
        tokens = tokens.to(next(self.model.parameters()).device)
        return self.model.encode_text(tokens, normalize=True, output_tokens=False)['text_features']
