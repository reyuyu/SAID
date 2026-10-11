"""Static public-evidence investigation, without evaluation or author contact."""
import argparse
import json
from pathlib import Path
import statistics
from model import longclip
from tools.retrieval_bounded import atomic_json, sha, digest


def run(evidence, said_manifest, output):
    e = Path(evidence)
    commits = json.loads((e / 'hyfl_commits.json').read_text())
    tree = json.loads((e / 'hyfl_tree.json').read_text())['tree']
    initial = json.loads((e / 'hyfl_initial_tree.json').read_text())['tree']
    old = {r['path']: r['sha'] for r in initial}
    eval_files = [r['path'] for r in tree if r['type'] == 'blob' and r['path'].startswith('eval/')]
    changes = {}
    for name in ['hyfl_change_76ba.json', 'hyfl_change_882a.json', 'hyfl_change_2503.json']:
        data = json.loads((e / name).read_text())
        changes[data['sha']] = [r['filename'] for r in data['files']]
    goal = json.loads((e / 'GOAL_DCI_test.json').read_text())
    said = [json.loads(s) for s in Path(said_manifest).read_text().splitlines()]
    source = {Path(r['image_path']).name: r['caption'] for r in said}
    names = [Path(r['filename']).name for r in goal]
    if len(set(names)) != len(names): raise ValueError('duplicate GOAL filename')
    common = sorted(set(names) & set(source))
    gm = {Path(r['filename']).name: r['caption'] for r in goal}
    exact = sum(gm[i] == source[i] for i in common)
    stripped = sum(gm[i].strip() == source[i].strip() for i in common)
    equal_tokens = 0
    for start in range(0, len(common), 64):
        batch = common[start:start + 64]
        a = longclip.tokenize([gm[i] for i in batch], truncate=True)
        b = longclip.tokenize([source[i] for i in batch], truncate=True)
        equal_tokens += int((a == b).all(dim=1).sum())
    # Raw tokenizer lengths are auxiliary: no inference and no protocol selection.
    def lengths(texts):
        return [len(longclip._tokenizer.encode(t)) for t in texts]
    said_lengths = lengths(source.values()); goal_lengths = lengths(gm.values())
    proof = {
        'status': 'PROTOCOL_UNVERIFIED',
        'reason': 'No HyFL-author DCI_test.json or precise generation rule found in inspected public sources.',
        'hyfl_public_commits': [r['sha'] for r in commits],
        'hyfl_eval_files': eval_files,
        'eval_blob_identical_initial_and_latest': all(old.get(r['path']) == r['sha'] for r in tree
                                                     if r['path'] in eval_files),
        'middle_commit_changed_paths': changes,
        'direct_annotation_files_in_tree': [r['path'] for r in tree if r['path'].lower().endswith(('.json', '.csv', '.tsv'))],
        'loader_evidence': {'file': 'eval/retrieval/unified_eval.py', 'class': 'DCIDataset',
                            'read': 'DCI_test.json', 'fields': ['filename', 'caption'], 'sort': 'filename',
                            'caption_generation_rule': None},
        'readme_ambiguity': 'Calls dci Long-DCI; paper Table1 has separate DCI and Long-DCI columns.',
        'goal_candidate': {'repo': 'PerceptualAI-Lab/GOAL',
                           'commit': json.loads((e / 'goal_commit.json').read_text())['sha'],
                           'json_sha256': sha(e / 'GOAL_DCI_test.json'),
                           'records': len(goal), 'said_records': len(said), 'common_images': len(common),
                           'missing_from_goal_count': len(set(source) - set(names)),
                           'goal_not_in_said_count': len(set(names) - set(source)),
                           'matched_image_raw_caption_equal': exact,
                           'matched_image_strip_caption_equal': stripped,
                           'matched_image_token_equal': equal_tokens,
                           'candidate_set_status': 'PROTOCOL_MISMATCH' if set(names) != set(source) else 'MATCH',
                           'used_for_evaluation': False},
        'token_lengths_without_sot_eot': {
            'said_mean': statistics.mean(said_lengths), 'said_max': max(said_lengths),
            'said_over_246_count': sum(x > 246 for x in said_lengths),
            'goal_mean': statistics.mean(goal_lengths),
            'limitation': 'Descriptive only; averages and shared subset do not identify HyFL inputs.'},
        'saids_caption_rule': '(short_caption + space + extra_caption).strip(), 7805, filename sort',
        'author_contact_sent': False,
        'evidence_files': {p.name: sha(p) for p in sorted(e.iterdir())
                           if p.is_file() and p.suffix in {'.json', '.py', '.md'}
                           and not p.name.startswith(('LONG_', 'DCI_INVESTIGATION'))},
    }
    proof['said_manifest_sha256'] = sha(said_manifest)
    atomic_json(output, proof, exclusive=True)
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', required=True); p.add_argument('--said-manifest', required=True)
    p.add_argument('--output', required=True); a = p.parse_args()
    run(a.evidence, a.said_manifest, a.output)
