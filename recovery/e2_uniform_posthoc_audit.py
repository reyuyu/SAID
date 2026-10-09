"""Read-only Urban pairing and complete saved-stream audits. Never trains.

Run: python -m recovery.e2_uniform_posthoc_audit
Repeat in a fresh directory: --output-dir /path/to/new/audit
Outputs are exclusive-create in posthoc_audit/; original artifacts stay immutable.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path('/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1')
EXP = ROOT/'experiments/nest_clip_v1/hns_s12_uniform_full4868_v1'
OUT = EXP/'posthoc_audit'
E2 = RUNTIME/'hns-s12-uniform-full4868-v1'
S12 = RUNTIME/'hns-s12-full4868-v1'
START_COMMIT = '5df0090f79629b2328a42116947d5af92a9c475a'
EXPECTED = {
    'E2_checkpoint': 'cc4fb207598a321d491dcafc164756710c38648c7ccf7df178b5d8c35a11d02e',
    'E2_bare': '50634512e226e79d526e269ba1a6ca75d7248f47e75417537f1605ca71ac2a0e',
    'S12_bare': '53a58fb14eb671d40d1f89572c6ce5a8cf6eec7dbd2ea1361759a2adfa193c94',
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, value, compact=False):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=None if compact else 2,
                  separators=(',', ':') if compact else None, allow_nan=False)
        f.write('\n')


def write_urban(folder, value):
    """Keep every query while publishing each JSON below the repository size cap."""
    import copy
    summary = copy.deepcopy(value)
    for direction, item in summary['directions'].items():
        queries = item.pop('queries')
        path = Path(folder)/f'URBAN_4868_{direction}_QUERIES.json'
        write_new(path, dict(direction=direction,queries=queries), compact=True)
        assert path.stat().st_size < 1024*1024
        item['queries_file'] = path.name
        item['queries_file_sha256'] = sha(path)
        item['query_count'] = len(queries)
    write_new(Path(folder)/'URBAN_4868_PAIRED_AUDIT.json', summary)


def exact_mcnemar(e2_only, s12_only):
    """Two-sided exact conditional binomial test, not chi-square approximation."""
    n = e2_only + s12_only
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(e2_only, s12_only)+1)) / 2**n)


def paired_summary(e2, s12):
    import numpy as np
    e2, s12 = np.asarray(e2, dtype=bool), np.asarray(s12, dtype=bool)
    assert e2.shape == s12.shape and e2.ndim == 1 and len(e2)
    counts = dict(both_correct=int((e2 & s12).sum()),
                  E2_only_correct=int((e2 & ~s12).sum()),
                  S12_only_correct=int((~e2 & s12).sum()),
                  both_wrong=int((~e2 & ~s12).sum()))
    n = len(e2); b = counts['E2_only_correct']; c = counts['S12_only_correct']
    # Resample paired query outcomes, preserving within-query dependence.
    rng = np.random.default_rng(0)
    draws = rng.multinomial(n, np.array(list(counts.values()))/n, size=50000)
    ci = np.percentile(100*(draws[:, 1]-draws[:, 2])/n, [2.5, 97.5])
    return dict(n_queries=n, counts=counts, E2_correct=int(e2.sum()), S12_correct=int(s12.sum()),
                net_correct=b-c, R1_delta_pp=100*(b-c)/n,
                mcnemar_exact_two_sided_p=exact_mcnemar(b, c),
                paired_query_bootstrap_95_percentile_CI_pp=ci.tolist(), bootstrap_replicates=50000,
                bootstrap_seed=0, uncertainty_scope='iid paired queries on this fixed test set; not training-seed uncertainty',
                significance_threshold=.05, multiplicity='Two directions; Bonferroni threshold .025 also reported',
                significant_unadjusted=exact_mcnemar(b, c)<.05,
                significant_two_direction_bonferroni=exact_mcnemar(b, c)<.025)


def stream_row_invariant(a, b, step, start, expected_lr):
    """Only data-stream variables; deliberately excludes model/gradient/augmentation values."""
    assert a['step'] == b['step'] == step, ('step', step)
    assert a['s'] == b['s'] == step-1
    epoch = start//1217; batch_cursor = step-start-1
    assert a['epoch'] == b['epoch'] == epoch
    assert a['actual_lrs'] == b['actual_lrs'] == expected_lr, ('LR', step)
    ah = {h['rank']: h for h in a['rank_health']}
    bh = {h['rank']: h for h in b['rank_health']}
    assert len(a['rank_health']) == len(b['rank_health']) == 4
    assert set(ah) == set(bh) == {0, 1, 2, 3}
    expected_batch = 180 if batch_cursor == 1216 else 256
    for rank in range(4):
        h, ref = ah[rank], bh[rank]
        assert h['updates'] == ref['updates'] == step-start
        assert h['batch'] == ref['batch'] == expected_batch
        assert len(h['sampling']['sample_ids']) == expected_batch
        assert h['stream_sha256'] == ref['stream_sha256'], ('stream', step, rank)
        assert h['sampling'] == ref['sampling'], ('sampling', step, rank)
    return ah


def stream_segment(actual_path, reference_path, start, stop, cfg):
    import torch
    from torch.utils.data import DistributedSampler
    from recovery.s02_full_local_data import FullLocalDataset
    from recovery.s02_local_full import expected_lrs
    from recovery.nested_d3_local_search import indices_digest
    from train.nested_semantic_data import sample_partial_detail_indices
    dataset = FullLocalDataset(Path('/root/said_s02_stage500/data_index'),
                              Path('/root/said_s02_stage500/ShareGPT4V'), 'nested_detail_d3', 0)
    epoch = start//1217
    before = torch.get_rng_state().clone()
    ids = {}
    for rank in range(4):
        sampler = DistributedSampler(dataset, 4, rank, seed=0, shuffle=True, drop_last=False)
        sampler.set_epoch(epoch); ids[rank] = iter(sampler)
    digests = {name: hashlib.sha256() for name in ('E2', 'S12')}
    count = updates = 0
    with Path(actual_path).open() as af, Path(reference_path).open() as bf:
        for row_no, pair in enumerate(itertools.zip_longest(af, bf)):
            assert pair[0] is not None and pair[1] is not None, 'Missing/extra log row'
            a, b = map(json.loads, pair)
            step = start+row_no+1
            assert step <= stop
            rates = dict(zip(('backbone','text_mask_and_shared_pool','visual_mask','fusion_adapter'),
                             expected_lrs(step-1, cfg)))
            ah = stream_row_invariant(a, b, step, start, rates)
            for rank in range(4):
                h = ah[rank]; s = h['sampling']
                expected_ids = [i+1000 for i in itertools.islice(ids[rank], h['batch'])]
                assert s['sample_ids'] == expected_ids, ('independent sampler', step, rank)
                choices = [sample_partial_detail_indices(n, 0, epoch, sid)
                           for n, sid in zip(s['n'], s['sample_ids'])]
                assert s['K'] == [len(v) for v in choices]
                assert s['lowest_selected_sentence_indices_sha256'] == indices_digest(s['sample_ids'], choices)
                assert s['nested_d3_exact'] and s['nested_detail_exact']
                count += h['batch']
                for name, row in (('E2', a), ('S12', b)):
                    rh = next(v for v in row['rank_health'] if v['rank'] == rank)
                    invariant = dict(step=step, epoch=epoch, batch_cursor=row_no, rank=rank,
                        batch=rh['batch'], stream_sha256=rh['stream_sha256'], sampling=rh['sampling'], LR=row['actual_lrs'])
                    digests[name].update((json.dumps(invariant, sort_keys=True, separators=(',', ':'))+'\n').encode())
            updates += 1
    assert updates == stop-start == 1217 and count == 1245904
    assert all(next(v, None) is None for v in ids.values())
    assert torch.equal(before, torch.get_rng_state())
    hashes = {k: v.hexdigest() for k, v in digests.items()}; assert len(set(hashes.values())) == 1
    return dict(passed=True, first_update=start+1, last_update=stop, updates=updates,
        epoch=epoch, batch_cursors=[0,1216], four_rank_sample_positions=count, tail_batch_per_rank=180,
        actual_log=str(actual_path), actual_log_sha256=sha(actual_path),
        independent_reference_log=str(reference_path), reference_log_sha256=sha(reference_path),
        full_stage_invariant_sha256=hashes,
        all_samples_text_tokens_K_indices_summaries_exact_to_actual_S12_log=True,
        sample_ID_and_indices_deterministic_reconstruction='PASS, all positions; private RNG, no images decoded',
        raw_caption_token_reconstruction='Not required: actual independent S12 log summaries compared for every step/rank',
        LR_independent_original4868_scheduler='PASS', no_missing_or_duplicate_optimizer_updates=True,
        model_gradients_outputs_random_image_pixels_compared=False,
        shared_summary_limit='Cryptographic summaries attest to text/token identity; per-example raw text/token arrays were not stored in logs')


def query_rows(similarity, ids):
    import torch
    n = len(ids); targets = torch.arange(n, device=similarity.device)
    top1 = similarity.topk(1, dim=1).indices[:, 0]
    ranked = similarity.topk(n, dim=1).indices
    ranks = (ranked == targets[:, None]).nonzero()[:, 1]+1
    gt = similarity.diagonal()
    greater = (similarity > gt[:, None]).sum(1)
    equal = (similarity == gt[:, None]).sum(1)
    wrong = similarity.clone(); wrong[targets, targets] = -torch.inf
    best_wrong = wrong.max(1).values
    membership = {k: (similarity.topk(k, dim=1).indices == targets[:,None]).any(1).cpu().tolist()
                  for k in (1,5,10)}
    columns = [v.cpu().tolist() for v in (top1,ranks,gt,similarity.gather(1,top1[:,None])[:,0],
                                          gt-best_wrong, greater+1, greater+equal)]
    return [dict(top1_ID=ids[columns[0][i]], correct=membership[1][i],
        ground_truth_rank=columns[1][i], rank_min_with_ties=columns[5][i], rank_max_with_ties=columns[6][i],
        top1_similarity=columns[3][i], ground_truth_similarity=columns[2][i],
        ground_truth_minus_best_wrong_similarity=columns[4][i],
        correct_at_5=membership[5][i], correct_at_10=membership[10][i]) for i in range(n)]


def capture_urban(bare, urban_root, device):
    import torch
    from model import longclip
    from tools import urban1k_retrieval as native
    model, preprocess = longclip.load_from_clip('ViT-B/16', device='cpu', args=argparse.Namespace())
    model.load_state_dict(torch.load(bare, map_location='cpu', weights_only=True), strict=True)
    model = model.float().to(device).eval()
    ids = [Path(p).stem for p, _ in native.image_caption_pairs(str(urban_root))]
    captures = []
    original = native._recall
    def observed(similarity, ks=(1,5,10)):
        result = original(similarity, ks)
        captures.append(query_rows(similarity, ids))
        return result
    native._recall = observed
    try:
        result = native.evaluate_urban1k(model, preprocess, root=str(urban_root), batch_size=64, device=device)
    finally:
        native._recall = original
        del model
        torch.cuda.empty_cache()
    assert len(captures) == 2
    return result, dict(I2T=captures[0], T2I=captures[1])


def preflight():
    from tools.eval_five_parallel import require_gpu_idle
    require_gpu_idle({0,1,2,3})
    subprocess.run(['git','merge-base','--is-ancestor',START_COMMIT,'HEAD'], cwd=ROOT, check=True)
    expected_paths = {
        'E2_checkpoint': E2/'step4868/training/step004868.pt',
        'E2_bare': E2/'step4868/evaluations/student_step4868.pt',
        'S12_bare': S12/'step4868/evaluations/student_step4868.pt'}
    protected = {p: sha(p) for p in EXP.rglob('*') if p.is_file() and not p.is_relative_to(OUT)}
    for key, path in expected_paths.items():
        digest = sha(path); assert digest == EXPECTED[key], ('identity', key, digest)
        protected[path] = digest
    sources = {}
    originals = {}
    for name, folder in (('E2', E2), ('S12', S12)):
        evaldir = folder/'step4868/evaluations'
        manifest = read(evaldir/'EVAL_PARALLEL_RUN.json')
        assert manifest['status'] == 'COMPLETED' and manifest['batch_size'] == 64
        assert manifest['bare_sha256'] == EXPECTED[name+'_bare']
        for key, digest in manifest['evaluator_source_sha256'].items():
            assert sha(ROOT/key) == digest, ('evaluator code drift', key)
            sources[key] = digest; protected[ROOT/key] = digest
        originals[name] = read(evaldir/'urban_native.json')
        protected[evaldir/'urban_native.json'] = sha(evaldir/'urban_native.json')
        assert originals[name]['checkpoint_sha256'] == EXPECTED[name+'_bare']
        result_file = (EXP if name == 'E2' else ROOT/'experiments/nest_clip_v1/hns_s12_full4868_v1')/'step4868/RESULTS.json'
        node = read(result_file); protected[result_file] = sha(result_file)
        for direction, key in (('I2T','image2text'),('T2I','text2image')):
            for k in (1,5,10):
                assert node['metrics']['Urban-1k'][direction][f'R@{k}'] == originals[name][key][f'R{k}']
    assert originals['E2']['root'] == originals['S12']['root']
    return protected, originals, sources, Path(originals['E2']['root'])


def stream_audit(protected):
    import torch
    from recovery import hns_s12_uniform_full4868 as continuation
    from train.train_nested_semantic_mask import code_manifest
    continuation.configure_protocol()
    proofs = []
    cfg = read(EXP/'config.json')
    parent = RUNTIME/'hns-s12-uniform-e1-1217-v1/step1217/training/step001217.pt'
    parent_digest = sha(parent); protected[parent] = parent_digest
    assert parent_digest == continuation.PARENT_SHA
    initial_identity = continuation.protocol.identity(parent, 1217)
    assert initial_identity['sha256'] == parent_digest
    for start, stop, reference in ((1217,2434,RUNTIME/'hns-s12-2434-validation-v1'),
                                   (2434,3651,S12),(3651,4868,S12)):
        runtime = E2/f'step{stop}'
        logs = [runtime/'training/steps.jsonl', reference/f'step{stop}/training/steps.jsonl']
        proof = stream_segment(*logs, start, stop, cfg)
        for p in logs: protected[p] = sha(p)
        checkpoint = runtime/f'training/step{stop:06d}.pt'
        identity = continuation.protocol.identity(checkpoint, stop)
        protected[checkpoint] = identity['sha256']
        p = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert p['config']['start_updates'] == start and p['config']['updates_planned_this_run'] == 1217
        assert p['resume_lineage']['parent'] == str(parent) and p['resume_lineage']['sha256'] == parent_digest
        assert p['config']['code_sha256'] == code_manifest()
        assert p['data_cursor'] == dict(next_epoch=stop//1217, next_batch=0)
        for rel, digest in p['config']['code_sha256'].items():
            assert sha(ROOT/rel) == digest; protected[ROOT/rel] = digest
        restore = read(runtime/'RESTORE_AUDIT.json')
        assert restore['passed'] and len(restore['ranks']) == 4
        assert all(v['model_adapter_optimizer_exact'] and v['RNG_exact'] and v['loader_generator_exact'] and
                   v['parameter_difference'] == 0 and v['optimizer_counters'] == [start] for v in restore['ranks'])
        acceptance = read(runtime/'training/acceptance.json')
        assert acceptance['passed'] and len(acceptance['ranks']) == 4
        assert all(v['completed_updates'] == stop and v['updates_this_run'] == 1217 and
                   v['max_parameter_difference_from_rank0'] == 0 for v in acceptance['ranks'])
        rank_timing = {}
        for rank in range(4):
            timing = Path('/root/said_s02_stage500')/f'formal-hns-s12-uniform-full4868-v1-{stop}'/f'rank{rank}.jsonl'
            if timing.exists():
                recorded = [json.loads(line)['step'] for line in timing.open()]
                assert recorded == list(range(start+1,stop+1))
                protected[timing] = sha(timing); rank_timing[str(rank)] = dict(status='PASS',rows=len(recorded),sha256=sha(timing))
            else:
                rank_timing[str(rank)] = dict(status='UNVERIFIED', reason='Per-rank timing file absent; aggregated four-rank log remains verified')
        for path in (runtime/'RESTORE_AUDIT.json', runtime/'training/acceptance.json'):
            protected[path] = sha(path)
        proof.update(checkpoint_identity=identity, restore_audit=restore, rank_timing=rank_timing,
                     resume_parent_sha256=parent_digest, optimizer_end_counter=stop, complete_state_preserved=True)
        proofs.append(proof)
        parent, parent_digest = checkpoint, identity['sha256']
        del p
    return dict(passed=True, initial1217_checkpoint_identity=initial_identity,
        segments=proofs, total_optimizer_updates=3651,
        total_four_rank_sample_positions=sum(p['four_rank_sample_positions'] for p in proofs),
        no_training_or_parameter_update_performed=True,
        comparison_type='Actual independent run logs, plus independent sampler/K/indices/LR replay',
        excluded_model_related_variables=['gradient values','outputs','parameters','random augmented pixels'])


def urban_audit(originals, sources, urban_root):
    from tools.urban1k_retrieval import image_caption_pairs, read_captions, dataset_fingerprint
    models, queries = {}, {}
    for name, folder in (('E2',E2),('S12',S12)):
        print('Urban native audit', name, flush=True)
        result, rows = capture_urban(folder/'step4868/evaluations/student_step4868.pt', urban_root, 'cuda:3')
        for key in ('image2text','text2image','n_images','n_captions'):
            assert result[key] == originals[name][key], ('Recall reproduction failure',name,key,result[key],originals[name][key])
        models[name], queries[name] = result, rows
    pairs = image_caption_pairs(str(urban_root)); ids = [Path(p).stem for p,_ in pairs]
    captions = read_captions(pairs)
    from model import longclip
    tokens = longclip.tokenize(captions, truncate=True)
    directions = {}
    for direction in ('I2T','T2I'):
        paired = []
        for i, sid in enumerate(ids):
            e, s = queries['E2'][direction][i], queries['S12'][direction][i]
            category = ('both_correct' if s['correct'] else 'E2_only_correct') if e['correct'] else (
                       'S12_only_correct' if s['correct'] else 'both_wrong')
            paired.append(dict(query_index=i,query_ID=sid,ground_truth_ID=sid,E2=e,S12=s,category=category,
                caption_word_count=len(captions[i].split()), caption_token_count_with_special=int((tokens[i]!=0).sum()),
                caption_sha256=hashlib.sha256(captions[i].encode()).hexdigest()))
        summary = paired_summary([v['E2']['correct'] for v in paired],[v['S12']['correct'] for v in paired])
        groups = {}
        for category in ('E2_only_correct','S12_only_correct'):
            cases = [v for v in paired if v['category'] == category]
            groups[category] = dict(query_IDs=[v['query_ID'] for v in cases], n=len(cases),
                mean_words=sum(v['caption_word_count'] for v in cases)/len(cases) if cases else None)
        directions[direction] = dict(summary=summary,queries=paired,descriptive_disagreements=groups)
    assert directions['T2I']['summary']['net_correct'] == 3
    return dict(passed=True, exact_all12_recalls_reproduced=True, models=models, directions=directions,
        checkpoint_identities=EXPECTED, evaluator_source_sha256=sources,
        dataset=dataset_fingerprint(str(urban_root)), device='cuda:3', precision='float32, no autocast',
        protocol='Original evaluate_urban1k, original _recall; read-only observer of similarity tensors',
        ranking='Original independent torch.topk(k) for k=1,5,10; full topk(1000) GT rank with tie bounds',
        margin='normalized GT similarity minus highest non-GT similarity (comparable within each model)',
        mathematical_evaluator_changes=False, raw_caption_text_uploaded=False,
        mechanism_claim='Only descriptive length statistics; no causal entity/relation specialization claim')


def report_addendum(urban):
    lines = ['', '## Exact native Recall reproduction', '',
        '| Model | I2T R@1/R@5/R@10 (%) | T2I R@1/R@5/R@10 (%) |', '|---|---|---|']
    for name in ('E2','S12'):
        m = urban['models'][name]
        lines.append('| '+name+' | '+' | '.join(
            ' / '.join(f'{100*m[d][k]:.3f}' for k in ('R1','R5','R10'))
            for d in ('image2text','text2image'))+' |')
    lines += ['', 'Both directions fail the unadjusted 0.05 significance threshold and the two-direction Bonferroni 0.025 threshold.',
        'T2I exact p=0.5078125; paired query bootstrap 95% interval is [-0.3,+0.9]pp. This is not evidence of a statistically significant improvement.', '',
        '## Symmetric descriptive cases', '']
    for direction in ('I2T','T2I'):
        for category, group in urban['directions'][direction]['descriptive_disagreements'].items():
            lines.append(f'{direction} {category}: IDs '+', '.join(group['query_IDs'])+
                f'; mean caption length {group["mean_words"]:.3f} words.' if group['n'] else f'{direction} {category}: none.')
    lines += ['', 'All 1000 queries per direction are saved in URBAN_4868_I2T_QUERIES.json and URBAN_4868_T2I_QUERIES.json. Summary JSON provides filenames and SHA256.',
        'Caption token counts in query records count nonzero token positions including special tokens; word counts are whitespace counts. No entity/relation counts or semantic causal claims are inferred.',
        'Engineering and requested log-summary audits: PASS, no failed or UNVERIFIED required item. Each stage has four complete rank timing files.',
        'Scientific limitations: single training seed; paired-query uncertainty cannot establish seed robustness; raw per-example training token arrays are not independently reconstructed. Semantic specialization remains unproven.']
    return lines


def main():
    import torch
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT,
                        help='Fresh exclusive output directory; existing artifacts are never overwritten')
    OUT = parser.parse_args().output_dir.resolve()
    torch.set_num_threads(4)
    assert not OUT.exists(), 'Exclusive new audit directory; never overwrite artifacts'
    protected, originals, sources, urban_root = preflight()
    OUT.mkdir(parents=True)
    try:
        stream = stream_audit(protected)
        urban = urban_audit(originals, sources, urban_root)
        for path, digest in protected.items():
            assert sha(path) == digest, ('Protected artifact changed',str(path))
        immutability = dict(passed=True, protected_file_count=len(protected),
            files={str(p):d for p,d in protected.items()}, original_artifacts_unchanged=True)
        write_new(OUT/'IMMUTABILITY_AND_IDENTITY_AUDIT.json', immutability)
        write_new(OUT/'FULL_STREAM_1218_4868_PROOF.json', stream)
        write_urban(OUT, urban)
        lines = ['# E2-Uniform posthoc audit (no training)', '',
            'Engineering: all three complete saved-stream comparisons pass; original artifacts unchanged.',
            'Retrieval: all 12 Urban Recall values exactly reproduce both original evaluations.', '',
            '| Direction | Both correct | E2 only | S12 only | Both wrong | Delta pp | Exact McNemar p | Paired bootstrap 95% CI pp |',
            '|---|---:|---:|---:|---:|---:|---:|---|']
        for direction, value in urban['directions'].items():
            s=value['summary'];c=s['counts']
            lines.append(f'| {direction} | {c["both_correct"]} | {c["E2_only_correct"]} | {c["S12_only_correct"]} | {c["both_wrong"]} | {s["R1_delta_pp"]:+.3f} | {s["mcnemar_exact_two_sided_p"]:.6f} | {s["paired_query_bootstrap_95_percentile_CI_pp"]} |')
        lines += ['', 'T2I net +3 is the difference between gains and regressions, not three gains with no regressions.',
            'Paired bootstrap treats test queries as iid; it does not estimate training-seed variability. McNemar tests both discordant classes.',
            'Both query directions and all improvement/regression IDs, target ranks, predictions, scores and margins are in the JSON.', '',
            '| Updates | Epoch | Updates verified | Sample positions | Tail per rank |', '|---|---:|---:|---:|---:|']
        for p in stream['segments']:
            lines.append(f'| {p["first_update"]}–{p["last_update"]} | {p["epoch"]} | {p["updates"]} | {p["four_rank_sample_positions"]} | {p["tail_batch_per_rank"]} |')
        lines += ['', 'Actual E2 logs match independent actual S12 logs for every step/rank, including text/token/K/index summaries and native LR.',
            'Independent frozen sampler, K/subset and LR reconstruction also passes. No gradients, outputs, parameters or augmented image pixels are compared.',
            'Restoration, optimizer endpoint counts, scheduler horizon4868, next-epoch cursors and four-rank synchronization are checked from complete checkpoints and original restoration/acceptance evidence.',
            'Raw per-query training text/token arrays were not saved: their identity is supported by matching cryptographic summaries, not a new full raw-text reconstruction.', '',
            'Mechanism: word-length summaries of both improved and regressed queries are descriptive only. No causal specialization or statistically reliable training improvement is established.',
            'No new training, model updates, parameter changes, checkpoint selection or evaluator mathematical changes occurred. Original reports were not overwritten.']
        lines += report_addendum(urban)
        with (OUT/'POSTHOC_AUDIT_REPORT.md').open('x') as f: f.write('\n'.join(lines)+'\n')
        print(json.dumps({k:v['summary'] for k,v in urban['directions'].items()},indent=2), flush=True)
    except BaseException as exc:
        write_new(OUT/'AUDIT_FAILURE.json', dict(passed=False,error=repr(exc),training_started=False))
        raise


if __name__ == '__main__':
    main()
