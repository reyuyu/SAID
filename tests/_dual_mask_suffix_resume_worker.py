"""Actual training main loop: uninterrupted versus mid-epoch stop/replay/continue."""
import argparse
import json
import random
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

REPO = Path(__file__).resolve().parents[1]
if not (REPO/'train/train_dual_mask_suffix.py').is_file():
    REPO = Path('/root/SAID-s0-dualmask-clean-v01')
sys.path[:0] = [str(REPO), str(REPO/'train'), str(REPO/'tests')]
from _dual_mask_suffix_ddp_worker import ToyClip
from model.dual_mask_suffix import DualMaskSuffixTrainModule as ProductionModule
import train_dual_mask_suffix as train


class Samples(Dataset):
    def __len__(self):
        return 24

    def __getitem__(self, index):
        k = random.randint(1, 5)
        image = torch.randn(1, 6, generator=torch.Generator().manual_seed(index+190))
        return dict(image_a=image, image_b=image, caption_said=str(k), caption_full=str(k),
                    suffix_text=str(6+k), suffix_valid=k<5, image_id=index, sample_id=index+1000,
                    prefix_k=k, num_sentences=5, view_b_resample_size=224, view_b_blur_sigma=0.)


def make_args(run, stop, resume=None):
    return argparse.Namespace(suffix_mode='masked', max_steps=stop, output_dir=str(run),
        base_model='B16', batch_size=4, epochs=3, lr=1e-6, mask_lr=1e-3, suffix_lr=1e-4,
        weight_decay=1e-2, warmup=2, seed=0, init_state='tiny-common-init', image_chunk=2,
        text_chunk=3, num_workers=2, total_len=1000, amp_dtype='fp32', save_every=3,
        resume=resume, run_type='formal')


def run_main(args):
    def load(*unused, **kwargs):
        return ToyClip(), None
    def factory(clip, **kwargs):
        return ProductionModule(clip, feature_dim=4, **kwargs)
    def tokenize(texts, **kwargs):
        return torch.tensor([[int(t)]*5 for t in texts])
    with patch.object(train, '_args', return_value=args), \
         patch.object(train.longclip, 'load_from_clip', side_effect=load), \
         patch.object(train.longclip, 'tokenize', side_effect=tokenize), \
         patch.object(train, 'load_init_state', return_value=None), \
         patch.object(train, 'DualMaskSuffixTrainModule', side_effect=factory), \
         patch.object(train, 'DualMaskSuffixDataset', side_effect=lambda **kwargs: Samples()):
        assert train.main() == 0


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='masked-resume-main-') as temp:
        root=Path(temp)
        run_main(make_args(root/'full', 18))
        run_main(make_args(root/'split', 9))
        checkpoint = root/'split/s0_dual_mask_suffix_masked_step000009.pt'
        payload = torch.load(checkpoint, weights_only=False)
        assert (payload['epoch'],payload['step_in_epoch']) == (1,2)
        # Emulate the actual old 500-step final checkpoint's one-ahead cursor.
        payload['step_in_epoch'] += 1
        torch.save(payload, checkpoint)
        run_main(make_args(root/'split', 18, str(checkpoint)))
        a=torch.load(root/'full/s0_dual_mask_suffix_masked_step000018.pt',weights_only=False)
        b=torch.load(root/'split/s0_dual_mask_suffix_masked_step000018.pt',weights_only=False)
        assert (b['epoch'],b['step_in_epoch']) == (2,5)
        assert a['config']['stream_summary'] == b['config']['stream_summary']
        histories = [[json.loads(line) for line in (root/part/'salu_log.jsonl').read_text().splitlines()]
                     for part in ('full','split')]
        assert [r['completed_steps'] for r in histories[1]] == list(range(1,19))
        for x,y in zip(*histories):
            for key in ('epoch','step_in_epoch','lr','mask_lr','suffix_lr'):
                assert x[key] == y[key], (key,x[key],y[key])
        errors = {}
        for group in ('clip_state','suffix_gate_state'):
            errors[group] = 0.
            for key,value in a[group].items():
                ref=b[group][key]
                torch.testing.assert_close(value,ref,atol=1e-7,rtol=1e-6)
                errors[group]=max(errors[group],float((value-ref).abs().max()))
        errors['optimizer_states']=0.
        for group, state in a['optimizer_states'].items():
            assert state['param_groups'] == b['optimizer_states'][group]['param_groups']
            for key,fields in state['state'].items():
                for name,value in fields.items():
                    ref=b['optimizer_states'][group]['state'][key][name]
                    torch.testing.assert_close(value,ref,atol=1e-7,rtol=1e-6)
                    errors['optimizer_states']=max(errors['optimizer_states'],float((value-ref).abs().max()))
        print(json.dumps(dict(status='within_tolerance',actual_main_loop=True,world_size=1,
            backend='gloo',full_updates=18,split_after=9,epochs=3,
            legacy_one_ahead_cursor_tested=True,worker_prefix_stream_replayed=True,
            cumulative_stream_digest_equal=True,lr_and_batch_positions_preserved=True,
            max_abs_errors=errors,atol=1e-7,rtol=1e-6,torch_version=torch.__version__),indent=2))


if __name__=='__main__':
    main()
