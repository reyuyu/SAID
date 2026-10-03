"""Fixed SAID reference Urban predictions for the requested query error audit."""
import json

from PIL import Image
import torch

from tools.eval_urban1k_cls import load_student
from tools.urban1k_retrieval import image_caption_pairs, read_captions, _recall
from model import longclip
from .common import ASSETS, RUN, dump, sha256

CHECKPOINT = '/root/lk_projects/SAID-nest-clip-v1/three_followup_v1/four_epoch/trials/6d44ae8d5c34438e672287b09f3e6930af01473a4aa4905cd90c02c259b2a4a8/step4868/student_step4868.pt'
SHA = 'f36438934947ed7f1523fe87e877a0e62dd807dc8a7c89c751eac537e0ab64bb'


@torch.no_grad()
def main():
    assert sha256(CHECKPOINT) == SHA
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, transform, meta = load_student(CHECKPOINT, 'ViT-B/16', 'cuda:1')
    pairs = image_caption_pairs(str(ASSETS/'evaluation/Urban1k/Urban1k'))
    captions = read_captions(pairs)
    tokens = longclip.tokenize(captions, truncate=True).to('cuda:1')
    tx = model.encode_text(tokens)
    tx = tx / tx.norm(dim=-1, keepdim=True)
    ims = []
    for start in range(0, len(pairs), 64):
        inputs = torch.stack([transform(Image.open(p).convert('RGB')) for p, _ in pairs[start:start+64]])
        ims.append(model.encode_image(inputs.to('cuda:1')))
    im = torch.cat(ims)
    im = im / im.norm(dim=-1, keepdim=True)
    sim_i = im @ tx.T
    sim_t = tx @ im.T
    metrics = {'I2T': _recall(sim_i), 'T2I': _recall(sim_t)}
    assert abs(metrics['I2T']['R1']-.921) < 1e-6
    assert abs(metrics['T2I']['R1']-.911) < 1e-6
    torch.save({'image_basenames': [p.rsplit('/', 1)[-1] for p, _ in pairs],
                'similarity_i2t': sim_i.cpu(), 'similarity_t2i': sim_t.cpu()},
               RUN/'said-urban-predictions.pt')
    result = {'checkpoint': CHECKPOINT, 'checkpoint_sha256': SHA, 'strict_loading': meta,
              'metrics': metrics, 'matches_frozen_said_reference': True,
              'purpose': 'Requested query-level Urban error/overlap analysis; no training or model selection'}
    dump(RUN/'said-urban-predictions.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
