"""SAID-CLS-CVSSL v0.1 data layer: two views of one image, one caption stream.

The caption stream is the reference one, untouched: ``share4v_train_dataset``'s
``random.randint(1, num_sentences)`` prefix draw, executed in the same order, so the C_S stream of
a CVSSL run is bit-identical to a SmartCLIP run with the same seed (they are audited by digest).

View a is the *reference* preprocessing (openai-clip ``_transform(224)``): Resize(224, BICUBIC) ->
CenterCrop(224) -> RGB -> ToTensor -> CLIP Normalize. It is rebuilt here with the same torchvision
ops (and a unit test proves it is tensor-identical to ``clip._transform(224)``), because the
dataset must not load a second CLIP model just to obtain a transform.

View b is ``shared_content_weak_v1``: the **same** resize/center-crop content window, then a light
down-up resample (224 -> uniform integer in [192, 224] -> 224, bilinear) and a light Gaussian blur
(kernel 3, sigma ~ U[0.1, 1.0]), then the **same** normalisation. No independent crop, no colour
jitter, no grayscale, no hue/saturation, no flip, no rotation, no erasing, no CutMix.

The view RNG is stateless and derived from ``(seed, epoch, sample_id, view)``: it can never shift
the caption stream, and it is reproducible across resumes. The random-mask RNG of the R0 arm is a
third, separate stream.
"""
import hashlib
import json
import os
import random
from typing import Dict, Tuple

import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_SIZE = 224
VIEW_B_MIN_SIZE = 192
VIEW_B_BLUR_KERNEL = 3
VIEW_B_BLUR_SIGMA = (0.1, 1.0)
VIEW_B_RESAMPLE_INTERPOLATION = 'bilinear'

data4v_root = os.environ.get('SHARE4V_DATA_ROOT', '../datasets/ShareGPT4V/')
json_name = os.environ.get('SHARE4V_JSON', 'share-captioner_coco_lcs_sam_1246k_1107.json')
image_root = os.environ.get('SHARE4V_DATA_ROOT', '../datasets/ShareGPT4V/')


def reference_view_a_transform(n_px: int = IMAGE_SIZE):
    """The reference SmartCLIP preprocessing, rebuilt with the identical torchvision ops.

    ``clip._transform(n_px)`` is ``Compose([Resize(n_px, BICUBIC), CenterCrop(n_px),
    _convert_image_to_rgb, ToTensor(), Normalize(mean, std)])``; ``_convert_image_to_rgb`` is
    ``image.convert('RGB')`` and the dataset already hands over an RGB image, so the tensor is
    identical (verified in ``tests/test_said_cls_cvssl.py``).
    """
    return T.Compose([T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
                      T.CenterCrop(n_px),
                      T.ToTensor(),
                      T.Normalize(CLIP_MEAN, CLIP_STD)])


def reference_window_transform(n_px: int = IMAGE_SIZE):
    """The shared content window: exactly the Resize + CenterCrop part of view a."""
    return T.Compose([T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
                      T.CenterCrop(n_px)])


def stateless_seed(*parts) -> int:
    """``hash(seed, epoch, sample_id, view_id)`` as a reproducible 63-bit integer."""
    key = ':'.join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:8], 'big') % (2 ** 63)


def image_id_from_path(relative_path: str) -> int:
    """Stable 63-bit id of the source image (duplicate detection across the global batch)."""
    return int.from_bytes(hashlib.sha256(relative_path.encode('utf-8')).digest()[:8],
                          'big') % (2 ** 63)


def shared_content_weak_v1(window: Image.Image, generator: torch.Generator,
                           size: int = IMAGE_SIZE) -> Tuple[Image.Image, Dict]:
    """View b: same content window, light resample + light blur. Returns ``(pil, parameters)``."""
    resample_size = int(torch.randint(VIEW_B_MIN_SIZE, size + 1, (1,), generator=generator).item())
    down = T.Resize(resample_size,
                    interpolation=T.InterpolationMode.BILINEAR)(window)
    up = T.Resize(size, interpolation=T.InterpolationMode.BILINEAR)(down)
    sigma = float(torch.empty(1).uniform_(VIEW_B_BLUR_SIGMA[0], VIEW_B_BLUR_SIGMA[1],
                                          generator=generator).item())
    view_b = T.GaussianBlur(kernel_size=VIEW_B_BLUR_KERNEL, sigma=sigma)(up)
    parameters = {'resample_size': resample_size,
                  'resample_interpolation': VIEW_B_RESAMPLE_INTERPOLATION,
                  'blur_kernel': VIEW_B_BLUR_KERNEL,
                  'blur_sigma': sigma}
    return view_b, parameters


class Share4VCvsslDataset(data.Dataset):
    """``(I_a, I_b, C_S)`` with the reference caption stream and the reference json slicing."""

    def __init__(self, root: str = data4v_root, json_file: str = json_name,
                 img_root: str = image_root, seed: int = 0, total_len: int = 1000,
                 strict_manifest=None, view_a=None, window=None, augment_view_b: bool = True):
        self.root = root
        self.image_root = img_root
        self.seed = int(seed)
        self.epoch = 0
        self.total_len = int(total_len)
        self.augment_view_b = bool(augment_view_b)
        strict_manifest = strict_manifest or os.environ.get('SHARE4V_FULL_AUDIT')
        if strict_manifest:
            from tools.data.full_data_gate import require_full_data
            require_full_data(strict_manifest, os.path.join(root, json_file), img_root)
        with open(os.path.join(root, json_file), 'r', encoding='utf8') as handle:
            self.json_data = json.load(handle)[self.total_len:]
        self.view_a = view_a or reference_view_a_transform()
        self.window = window or reference_window_transform()
        print('Share4VCvsslDataset loaded, total length:', len(self.json_data), flush=True)

    def __len__(self):
        return len(self.json_data)

    def set_epoch(self, epoch: int):
        """Pass the real epoch into the workers (they are re-created when persistent_workers is
        False, which is the default here, so the value is genuinely visible in the worker)."""
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> Dict:
        record = self.json_data[index]
        caption = record['conversations'][1]['value'].replace('\n', ' ')
        num_sentences = len(caption.split('. '))
        image_name = os.path.join(self.image_root, record['image'])
        image = Image.open(image_name).convert('RGB')

        window = self.window(image)
        image_a = self.view_a(image)

        parameters = {}
        if self.augment_view_b:
            generator = torch.Generator().manual_seed(
                stateless_seed(self.seed, self.epoch, index + self.total_len, 'view_b'))
            window_b, parameters = shared_content_weak_v1(window, generator)
        else:
            window_b = window
        image_b = T.Compose([T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])(window_b)

        # reference caption draw (same call, same order, same RNG stream as SmartCLIP)
        prefix_k = random.randint(1, num_sentences)
        caption_said = '. '.join(caption.split('. ')[:prefix_k])

        return {
            'image_a': image_a,
            'image_b': image_b,
            'caption_said': caption_said,
            'image_id': image_id_from_path(record['image']),
            'sample_id': int(index) + self.total_len,
            'prefix_k': int(prefix_k),
            'num_sentences': int(num_sentences),
            'view_b_resample_size': int(parameters.get('resample_size', IMAGE_SIZE)),
            'view_b_blur_sigma': float(parameters.get('blur_sigma', 0.0)),
        }


def cvssl_collate(samples) -> Dict[str, torch.Tensor]:
    """Default-collate equivalent that keeps the scalar metadata as tensors."""
    batch = {
        'image_a': torch.stack([sample['image_a'] for sample in samples], dim=0),
        'image_b': torch.stack([sample['image_b'] for sample in samples], dim=0),
        'caption_said': [sample['caption_said'] for sample in samples],
        'image_id': torch.tensor([sample['image_id'] for sample in samples], dtype=torch.long),
        'sample_id': torch.tensor([sample['sample_id'] for sample in samples], dtype=torch.long),
        'prefix_k': torch.tensor([sample['prefix_k'] for sample in samples], dtype=torch.long),
        'num_sentences': torch.tensor([sample['num_sentences'] for sample in samples],
                                      dtype=torch.long),
        'view_b_resample_size': torch.tensor([sample['view_b_resample_size'] for sample in samples],
                                             dtype=torch.long),
        'view_b_blur_sigma': torch.tensor([sample['view_b_blur_sigma'] for sample in samples]),
    }
    return batch
