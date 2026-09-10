import hashlib
import json
import cv2
from PIL import Image
import clip

import torch
import torch.utils.data as data
import os
import numpy as np
import random
from torchvision.utils import save_image

data4v_root = os.environ.get('SHARE4V_DATA_ROOT', '../datasets/ShareGPT4V/')
json_name = os.environ.get('SHARE4V_JSON', 'share-captioner_coco_lcs_sam_1246k_1107.json')
image_root = os.environ.get('SHARE4V_DATA_ROOT', '../datasets/ShareGPT4V/')


def caption_sentences(caption):
    """Sentence list under the legacy SmartCLIP convention (``'. '`` split)."""
    return caption.replace('\n', ' ').split('. ')


def sample_unsaid_index(index, num_sentences, prefix_k, seed=0):
    """Stateless deterministic 1-based suffix index ``J`` in ``[K+1, N]`` (``None`` if K == N).

    Phase 2.9A must not consume the worker RNG for the suffix choice: the legacy prefix
    sampling (``random.randint(1, N)``) has to stay bit-identical, and any extra draw
    would shift every later sample's prefix. The suffix index is therefore derived from
    ``sha256(seed:index:K)`` instead of a random draw.
    """
    if num_sentences < 1 or prefix_k >= num_sentences:
        return None
    key = '%d:%d:%d' % (int(seed), int(index), int(prefix_k))
    value = int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:8], 'big')
    return int(prefix_k) + 1 + value % (int(num_sentences) - int(prefix_k))


class share4v_train_dataset(data.Dataset):
    def __init__(self, data4v_root=data4v_root, json_name=json_name, image_root=image_root,
                 strict_manifest=None, preprocess=None, caption_views=False, suffix_seed=0):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 1000
        self.caption_views = bool(caption_views)
        self.suffix_seed = int(suffix_seed)
        strict_manifest = strict_manifest or os.environ.get('SHARE4V_FULL_AUDIT')
        if strict_manifest:
            from tools.data.full_data_gate import require_full_data
            require_full_data(strict_manifest, os.path.join(data4v_root, json_name), image_root)
        with open(os.path.join(data4v_root, json_name), 'r', encoding='utf8') as fp:
            self.json_data = json.load(fp)[self.total_len:]
        if preprocess is None:
            _, self.preprocess = clip.load("ViT-L/14")
            del _
        else:
            self.preprocess = preprocess
        print('share4v_train_dataset loaded, total length:', len(self.json_data))

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        num_sentences = len(caption.split(". "))
        image_name = os.path.join(self.image_root, self.json_data[index]['image'])
        image = Image.open(image_name).convert('RGB')
        image_tensor = self.preprocess(image)
        # legacy prefix draw: unchanged call and unchanged RNG stream
        prefix_k = random.randint(1, num_sentences)
        use_caption = '. '.join(caption.split(". ")[:prefix_k])
        if not self.caption_views:
            return image_tensor, use_caption
        sentences = caption_sentences(caption)
        unsaid_index = sample_unsaid_index(index, num_sentences, prefix_k, self.suffix_seed)
        return {
            'image': image_tensor,
            'caption_full': caption,
            'caption_said': use_caption,
            'caption_unsaid': sentences[unsaid_index - 1] if unsaid_index else '',
            'has_unsaid': unsaid_index is not None,
            'num_sentences': int(num_sentences),
            'prefix_k': int(prefix_k),
            'unsaid_sentence_index': int(unsaid_index) if unsaid_index else 0,
            'index': int(index),
        }


class share4v_val_dataset(data.Dataset):
    def __init__(self, data4v_root=data4v_root, json_name=json_name, image_root=image_root):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 1000
        with open(os.path.join(data4v_root, json_name), 'r', encoding='utf8') as fp:
            self.json_data = json.load(fp)[:self.total_len]
        _, self.preprocess = clip.load("ViT-L/14")
        del _

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        image_name = os.path.join(self.image_root, self.json_data[index]['image'])
        image = Image.open(image_name)
        image_tensor = self.preprocess(image)
        return image_tensor, caption



if __name__ == '__main__':
    dataset = share4v_val_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    for i, (image_tensor, caption) in enumerate(dataloader):
        print(i, caption)
        save_image(image_tensor, 'sharegpt4v_{i}.jpg')
        break
