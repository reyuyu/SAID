"""Model-free caption length/248-token digest audit of every actual manifest."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch
from model import longclip
from tools.retrieval_bounded import atomic_json, sha


def run(plan, output):
    torch.set_num_threads(4); results = {}
    for job in json.loads(Path(plan).read_text())['jobs']:
        rows = [json.loads(s) for s in Path(job['manifest']).read_text().splitlines()]
        words = sorted(len(r['caption'].split()) for r in rows)
        pre = [len(longclip._tokenizer.encode(r['caption'])) + 2 for r in rows]
        h = hashlib.sha256()
        for start in range(0, len(rows), 128):
            tokens = longclip.tokenize([r['caption'] for r in rows[start:start + 128]], truncate=True)
            h.update(tokens.numpy().tobytes())
        counts = Counter(r['caption'] for r in rows)
        results[job['name']] = {'manifest_sha256': sha(job['manifest']), 'captions': len(rows),
                                'word_count': {'min': words[0], 'median': words[len(words)//2],
                                               'max': words[-1], 'mean': sum(words)/len(words)},
                                'pre_truncation_tokens_including_sos_eos': {
                                    'min': min(pre), 'max': max(pre), 'mean': sum(pre)/len(pre)},
                                'captions_truncated_to_248': sum(n > 248 for n in pre),
                                'token_stream_sha256': h.hexdigest(),
                                'duplicate_raw_caption_occurrences': sum(n - 1 for n in counts.values()),
                                'rows_filtered': 0}
    atomic_json(output, results, exclusive=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--plan', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); run(a.plan, a.output)
