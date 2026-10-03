"""Read saved scores in their original evaluator environment; save exact GPU top-k."""
import argparse
import json

import torch

from .common import RUN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['said', 'debias'], required=True)
    args = ap.parse_args()
    path = (RUN/'said-urban-predictions.pt' if args.model == 'said' else RUN/'unified/urban-embeddings.pt')
    reference = ([921, 911] if args.model == 'said' else [933, 931])
    x = torch.load(path, weights_only=True)
    counts = {}
    for key, expected in zip(['similarity_i2t', 'similarity_t2i'], reference):
        score = x[key].to('cuda:2')
        for k in (1, 5, 10):
            top = score.topk(k, dim=1).indices.cpu()
            x[key+f'_top{k}'] = top.squeeze(1) if k == 1 else top
        correct = int((x[key+'_top1'] == torch.arange(1000)).sum())
        assert correct == expected, (args.model, key, correct, expected)
        counts[key] = correct
    torch.save(x, path)
    print(json.dumps({'model': args.model, 'torch': torch.__version__, 'correct_counts': counts,
                      'new_model_inference': False}), flush=True)


if __name__ == '__main__':
    main()
