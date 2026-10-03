"""Compare the predeclared official AMP and separate FP32 Urban diagnostic."""
from pathlib import Path

import torch

from .common import RUN, dump


def main():
    a = torch.load(RUN/'urban-amp/urban-embeddings.pt', weights_only=False)
    b = torch.load(RUN/'urban-fp32/urban-embeddings.pt', weights_only=False)
    rows_a, rows_b = a['raw_rows'], b['raw_rows']
    assert rows_a == rows_b
    names = [Path(r['image']).name for r in rows_a]
    sa, sb = a['official_scaled_scores'], b['official_scaled_scores']
    result = {'same_raw_image_caption_rows_and_order': True,
        'same_checkpoint_and_official_loader': True, 'source_patch_applied': False,
        'primary_published_reproduction_precision': 'amp',
        'separate_adapter_diagnostic_precision': 'fp32',
        'image_feature_max_abs': float((a['image_features']-b['image_features']).abs().max()),
        'text_feature_max_abs': float((a['text_features']-b['text_features']).abs().max()),
        'scaled_score_max_abs': float((sa-sb).abs().max()), 'directions': {}}
    for direction, x, y in [('I2T', sa, sb), ('T2I', sa.T, sb.T)]:
        pa, pb = x.argmax(1), y.argmax(1)
        changed = torch.where(pa != pb)[0].tolist()
        truth = torch.arange(1000)
        result['directions'][direction] = {
            'amp_correct': int((pa == truth).sum()), 'fp32_correct': int((pb == truth).sum()),
            'top1_changed_count': len(changed),
            'changes': [{'query_id': names[i], 'amp_top1': names[int(pa[i])],
                         'fp32_top1': names[int(pb[i])],
                         'amp_correct': int(pa[i]) == i, 'fp32_correct': int(pb[i]) == i} for i in changed]}
    dump(RUN/'urban-precision-audit.json', result)
    print({d: {k: v for k, v in r.items() if k != 'changes'} for d, r in result['directions'].items()})


if __name__ == '__main__':
    main()
