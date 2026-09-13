"""One-off strict export/verification of the actual completed masked formal run."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path('/root/SAID-s0-dualmask-clean-v01')
sys.path[:0] = [str(REPO), str(REPO/'train')]
from model import longclip
from model.dual_mask_suffix import DualMaskSuffixTrainModule, build_suffix_gate, pairwise_masked_scores
from model.said_cls_cvssl import said_mask_from_hidden
from train_dual_mask_suffix import load_checkpoint, export_bare_student

EXPECTED_SHA = '11af80b344c623b27b93069f9be526970c9c950c'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    global RUN
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--step', type=int, required=True)
    args = parser.parse_args()
    RUN = Path(args.run)
    step = args.step
    exit_file = 'resume.exitcode' if step == 1000 else 'resume1000.exitcode'
    assert (RUN/exit_file).read_text().strip() == '0'
    rows = [json.loads(line) for line in (RUN/'salu_log.jsonl').read_text().splitlines()]
    assert [r['completed_steps'] for r in rows] == list(range(1, step + 1))
    assert all(r['formal_optimizer_updates'] == r['completed_steps'] and r['debug_optimizer_updates'] == 0 for r in rows)
    checkpoint = RUN/f's0_dual_mask_suffix_masked_step{step:06d}.pt'
    clip, preprocess = longclip.load_from_clip('ViT-B/16', device='cpu', args=argparse.Namespace())
    module = DualMaskSuffixTrainModule(clip, 'masked')
    payload = load_checkpoint(str(checkpoint), module)
    assert payload['completed_steps'] == step
    assert payload['provenance']['git_head'] == EXPECTED_SHA
    assert payload['config']['formal_optimizer_updates'] == step
    assert payload['config']['init_state'] == '/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt'
    optimizer_steps = {}
    for name, state in payload['optimizer_states'].items():
        values = [int(s['step']) for s in state['state'].values() if 'step' in s]
        assert values and min(values) == max(values) == step, (name, min(values), max(values))
        optimizer_steps[name] = {'min': min(values), 'max': max(values), 'parameters_with_state': len(values)}
    bare = RUN/f'bare_student_step{step}.pt'
    export_bare_student(str(checkpoint), str(bare))
    student, _ = longclip.load_from_clip('ViT-B/16', device='cpu', args=argparse.Namespace())
    student.load_state_dict(torch.load(bare, map_location='cpu', weights_only=False), strict=True)
    gate = build_suffix_gate()
    gate.load_state_dict(payload['suffix_gate_state'], strict=True)
    for key, value in module.suffix_gate.state_dict().items():
        torch.testing.assert_close(value, gate.state_dict()[key], atol=0, rtol=0)
    urban = Path('/root/datasets/Urban1k/Urban1k')
    image_paths = sorted((urban/'image').glob('*.jpg'))[:2]
    captions = [(urban/'caption'/(p.stem+'.txt')).read_text().splitlines()[0] for p in image_paths]
    images = torch.stack([preprocess(Image.open(p).convert('RGB')) for p in image_paths]).cuda()
    tokens = longclip.tokenize(captions, truncate=True).cuda()
    module.cuda().eval(); student.cuda().eval(); gate.cuda().eval()
    with torch.no_grad():
        g1, g2 = module.clip.encode_image(images), student.encode_image(images)
        t1, h1 = module.clip.encode_text(tokens, return_full=True)
        t2, h2 = student.encode_text(tokens, return_full=True)
        for a, b in ((g1, g2), (t1, t2)):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        m1, _, _ = said_mask_from_hidden(module.clip.mask_net, h1)
        m2, _, _ = said_mask_from_hidden(student.mask_net, h2)
        q1, _ = pairwise_masked_scores(g1, m1, t1, module.suffix_gate, image_chunk=1, text_chunk=2)
        q2, _ = pairwise_masked_scores(g2, m2, t2, gate, image_chunk=1, text_chunk=2)
        torch.testing.assert_close(q1, q2, atol=1e-5, rtol=1e-5)
    errors = {'native_image_max_abs': float((F.normalize(g1, dim=-1)-F.normalize(g2, dim=-1)).abs().max()),
              'native_text_max_abs': float((F.normalize(t1, dim=-1)-F.normalize(t2, dim=-1)).abs().max()),
              'suffix_score_max_abs': float((q1-q2).abs().max())}
    report = {'status': 'passed', 'training_sha': EXPECTED_SHA, 'actual_updates': len(rows),
        'strict_clip': True, 'strict_gate': True, 'strict_bare_student': True,
        'optimizer_steps': optimizer_steps, 'roundtrip_errors': errors,
        'checkpoint': str(checkpoint), 'bare_student': str(bare),
        'checkpoint_sha256': sha256(checkpoint), 'bare_student_sha256': sha256(bare),
        'initial_checkpoint_sha256': sha256(RUN/'s0_dual_mask_suffix_masked_step000000.pt'),
        'common_init_sha256': sha256(payload['config']['init_state'])}
    (RUN/f'export_report_step{step}.json').write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
