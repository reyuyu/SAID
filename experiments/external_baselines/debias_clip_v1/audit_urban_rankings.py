"""Compare frozen native predictions with the official FP32 Urban ranking."""
import json
from pathlib import Path

import torch

from .common import RUN, dump


def main():
    native = torch.load(RUN/'unified/urban-embeddings.pt', weights_only=True)
    official = torch.load(RUN/'urban-fp32/urban-embeddings.pt', weights_only=False)
    indexes = {Path(r['image']).name: i for i, r in enumerate(official['raw_rows'])}
    order = [indexes[n] for n in native['image_basenames']]
    scaled = official['official_scaled_scores'][order][:, order]
    scale = json.loads((RUN/'urban-fp32/Urban1k.json').read_text())['logit_scale_exp']
    result = {'candidate_order_restored_to_frozen_said': True,
              'unscaled_score_max_abs': float((native['similarity_i2t']-scaled/scale).abs().max()),
              'directions': {}}
    for direction, key, sim in [('I2T', 'similarity_i2t', scaled), ('T2I', 'similarity_t2i', scaled.T)]:
        # Exactly the official metric's per-row 1-D descending argsort.
        top = torch.stack([row.argsort(descending=True)[:10] for row in sim])
        frozen = native[key+'_top10']
        first = native[key+'_top1']
        changed = torch.where(top[:, 0] != first)[0].tolist()
        result['directions'][direction] = {'top1_query_mismatches': len(changed),
            'top1_mismatch_ids': [native['image_basenames'][i] for i in changed],
            'top10_exact_order_mismatch_queries': int((top != frozen).any(dim=1).sum()),
            'mean_top10_set_overlap': sum(len(set(a.tolist()) & set(b.tolist()))/10 for a, b in zip(top, frozen))/1000}
    dump(RUN/'urban-native-ranking-consistency.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
