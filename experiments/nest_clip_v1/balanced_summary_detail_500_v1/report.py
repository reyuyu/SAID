"""Freeze all500 sampling/resource/recall evidence without further training."""
import json
from pathlib import Path
import shutil
import statistics

import numpy as np

from .run import EXP, ROOT, RUN, load, dump, records, stream_check, now

NAMES = ['COCO','Urban-1k','Flickr30k-test1k','DOCCI','Long-DCI']


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,r))+' |' for r in rows]])


def token_summary(selected):
    result = {}
    for view in ['Full','Summary','Detail']:
        samples = total = before = truncated = 0
        for row in selected:
            for health in row['rank_health']:
                stats = health['sampling']['Full_Summary_Detail_token_statistics'][view]
                samples += stats['samples']; total += stats['effective_token_sum']
                before += stats['before_truncation_token_sum']; truncated += stats['truncated_count']
        result[view] = {'samples': samples, 'effective_tokens_mean': total/samples if samples else None,
                       'before_truncation_tokens_mean': before/samples if samples else None,
                       'truncation_fraction': truncated/samples if samples else None}
    result['valid_fraction'] = sum(sum(h['valid'] for h in r['rank_health']) for r in selected)/(len(selected)*1024)
    return result


def main():
    r = load(RUN/'result.json'); b = r['baseline']; root = RUN/'step500'
    rows = records(root/'steps.jsonl'); old = records(Path(b['root'])/'steps.jsonl')
    assert len(rows) == 500 and [x['step'] for x in rows] == list(range(1,501))
    for a, ref in zip(rows, old):
        stream_check(a, ref)
        assert a['nonfinite'] == 0 and all(h['gradients_finite'] for h in a['rank_health'])
        assert a['F_candidates'] == 1024
    assert r['config']['horizon'] == 4868 and r['config']['start_updates'] == 0
    assert r['config']['resume'] is None and r['strict_export']['optimizer_steps'] == [500]
    assert r['strict_export']['image_max_abs'] == r['strict_export']['text_max_abs'] == 0
    delta_scores = {k:(v-b['scores'][k])*100 for k,v in r['scores'].items()}
    delta = {n:{d:(r['metrics'][n][d]['R@1']-b['metrics'][n][d]['R@1'])*100
                for d in ('I2T','T2I')} for n in NAMES}
    means = {n:statistics.fmean(delta[n].values()) for n in NAMES}
    # Stated reporting threshold, not a training gate or parameter selector.
    clear_tradeoff_pp = .2
    if delta_scores['Score5_R1']>0 and all(v>=-clear_tradeoff_pp for v in means.values()):
        conclusion='POSITIVE'
        statement='Summary–Detail decomposition is worth full4868-step confirmation.'
    elif means['Urban-1k']>0 and (delta_scores['Score5_R1']<=0 or means['Long-DCI'] < -clear_tradeoff_pp):
        conclusion='MIXED'
        statement='Summary–Detail improves summary-bias-sensitive retrieval but trades off broader retrieval.'
    else:
        conclusion='NEGATIVE'
        statement='Fixed Summary–Detail decomposition does not improve the matched RandomK baseline at500.'
    cycles = records(root/'cycle_timing.jsonl')
    normal = [x['four_rank_max_seconds'] for x in cycles if not x['warmup']]
    all_times = [x['four_rank_max_seconds'] for x in cycles]
    accept = load(root/'acceptance.json')
    resource = {'passed': statistics.fmean(normal)<=3 and all(v['peak_allocated_gib']<=65 for v in accept['ranks']),
        'normal_updates': len(normal), 'warmup_updates': 5, 'mean_seconds':statistics.fmean(normal),
        'median_seconds':statistics.median(normal), 'p95_seconds':float(np.quantile(normal,.95)),
        'max_seconds':max(normal), 'all500_mean_seconds_including_warmup':statistics.fmean(all_times),
        'peak_allocated_gib_max':max(v['peak_allocated_gib'] for v in accept['ranks']),
        'peak_reserved_gib_max':max(v['peak_reserved_gib'] for v in accept['ranks']),
        'ranks':accept['ranks'], 'scope':'Slowest rank complete cycle including real DataLoader wait, transfer, update, communication and ordinary logging; checkpoint writes separate.'}
    assert resource['passed']
    diagnostics = {}
    for step in [1,100,200,500]:
        row = rows[step-1]
        diagnostics[str(step)] = {'sampling':token_summary([row]),
            'loss_mask_gate':{k:v for k,v in row.items() if k not in ['rank_health','duplicate_image_ids']},
            'rank_fingerprints':[{k:h['sampling'][k] for k in ['sample_id_sha256','full_view_sha256','fixed_first_reference_stream_sha256']}
                                 for h in row['rank_health']]}
    last = rows[-50:]
    last50 = {'sampling':token_summary(last),
        'loss_mask_gate_mean':{k:statistics.fmean(row[k] for row in last if k in row)
                              for k,v in rows[-1].items() if isinstance(v,(float,int))}}
    r.update(status='COMPLETE',conclusion=conclusion,statement=statement,
        delta_direction='SummaryDetail minus matched RandomK old-R',delta_scores_pp=delta_scores,
        delta_R1_pp=delta,dataset_mean_delta_pp=means,
        reporting_significant_dataset_mean_drop_pp=clear_tradeoff_pp,
        resources=resource,sampling_diagnostics=diagnostics,last50=last50,
        baseline_retrained=False,from_common_step0=True,formal_start_updates=0,
        all500x4_sample_and_full_streams_match=True,
        F_1000_equivalence=load(EXP/'evidence/F_1000_EQUIVALENCE.json'),
        sampling_audit=load(EXP/'SAMPLING_AUDIT.json'),
        preflight=load(EXP/'evidence/preflight.json'),code_base='14653c92c6da9d552a2b624ab169eaaa275cdde8',
        final_checkpoint_budget=500,full4868_run_started=False,completed_at=now())
    dump(EXP/'RESULTS.json',r)
    dump(EXP/'evidence/resource-summary.json',resource)
    dump(EXP/'evidence/step-diagnostics.json',diagnostics)
    dump(EXP/'evidence/last50.json',last50)
    dump(EXP/'evidence/all500-streams.json',{'passed':True,'updates':500,'ranks':4,
        'sample_ids_F_raw_F_tokens_and_fixed_reference_streams_equal':True,
        'image_RNG_supporting_gate':'tests/test_summary_detail.py:test_dataset_image_rng_and_sample_id_match',
        'local_views_expected_to_differ':True})
    for k in ['config.json','acceptance.json','export-check.json','checkpoint_timing.jsonl']:
        shutil.copyfile(root/k,EXP/'evidence'/k)
    for p in (RUN/'execution').glob('*.console.txt'):
        lines=p.read_text(errors='replace').replace('\r','\n').splitlines()
        lines=[line.rstrip() for line in lines if not ('it/s]' in line or 's/it]' in line)]
        (EXP/'evidence'/(p.stem+'.summary.txt')).write_text('\n'.join(lines)+'\n')
    main_table = table(['Model','Score5_R1 %','J_long3 %','J_long %'],
        [['Matched RandomK old-R@500',*[f'{b["scores"][k]*100:.6f}' for k in r['scores']]],
         ['Summary–Detail@500',*[f'{r["scores"][k]*100:.6f}' for k in r['scores']]],
         ['Delta pp',*[f'{delta_scores[k]:+.6f}' for k in r['scores']]]])
    direction_table = table(['Dataset','Direction','RandomK baseline %','Summary–Detail %','Delta pp'],
        [[n,d,f'{b["metrics"][n][d]["R@1"]*100:.6f}',f'{r["metrics"][n][d]["R@1"]*100:.6f}',f'{delta[n][d]:+.6f}']
         for n in NAMES for d in ('I2T','T2I')])
    recall_table = table(['Model','Dataset','Direction','R@1 %','R@5 %','R@10 %'],
        [[label,n,d,*[f'{m[n][d]["R@"+str(k)]*100:.6f}' for k in (1,5,10)]]
         for label,m in [('Matched RandomK',b['metrics']),('Summary–Detail',r['metrics'])]
         for n in NAMES for d in ('I2T','T2I')])
    audit = r['sampling_audit']
    text = f'''# Full–Summary–Detail: strict single-variable500-update result

**{conclusion}.** {statement}
Only500 optimizer updates were trained, from the common step0 after an independent5-step smoke;
scheduler horizon4868 throughout. No full4868 training, mixture, extra loss or inference trick follows.

## Main matched comparison

{main_table}

{direction_table}

All aggregates use unrounded fractions. Delta is **Summary–Detail minus matched RandomK old-R**.
The reporting rule treats a dataset mean-R1 drop beyond0.2pp as a clear tradeoff; it is not a
training/parameter-selection gate. This is one seed and an early500-update experiment.

## Sampling definition and full-corpus audit

Full raw string/tokens remain bitwise-equivalent to the baseline's longest visible complete-sentence prefix.
Summary is the first raw cleaned segment; Detail is every remaining raw segment in original order,
compactly retokenized with the existing248-token tokenizer. No random K/subset/drop/shuffle/prefix PAD runs.
For raw n<2, local captions are absent, valid=false and only Full contributes. The unchanged batch
interface uses collation PAD sentinels for disabled local views, not a copied summary as Detail.

Complete train corpus:1245901 rows, original skip1000 preserved. Average raw segments
{audit['sentence_count']['mean']:.6f}, median{audit['sentence_count']['median']:.1f}.
Fixed seed0/epoch0/sample-ID100k audit: old R covers
{audit['old_R_non_summary_detail_coverage_mean']*100:.6f}% of raw non-summary BPE content;
new Detail covers100% before truncation and
{audit['new_D_effective_detail_coverage_after_truncation_mean']*100:.6f}% after truncation.
Detail truncation fraction:{audit['detail_truncated_fraction_valid']*100:.6f}% of valid complete-corpus samples.

The Full-preservation gate is1000/1000 raw strings and1000/1000 token tensors. The sentence parser
is unchanged. Raw Detail extends beyond the already-packed Full in
{audit['detail_extends_visible_F_fraction_subset']*100:.6f}% of eligible audit cases;
raw-n validity differs in{audit['validity_difference_count_subset']} of100k samples from old visible-n validity.
This follows the requested all-raw-detail construction and is explicit: Detail⊆raw caption does
not guarantee Detail⊆post-budget Full tokens. Inclusion mathematics remains unchanged.
DATA_SAMPLING_AUDIT.md/SAMPLING_AUDIT.json contain all requested quantiles and coverage definitions.

## Correctness, matching and unchanged mathematics

61 sampling/RandomK/model/hyperparameter/scheduler regression tests passed before training.
No model source changed relative to14653c92c6da9d552a2b624ab169eaaa275cdde8.
Optimizer/scheduler/RNG helper ASTs are unchanged; no View-Relation/RMask/SentenceDrop code is added.
Balanced architecture, independent visual blocks, shared pool, Hard-ST pair masks, F/S/D symmetric CE,
10/3 alignment, (ΩF+2ΩS+2ΩD)/3 sparsity and200-update inclusion ramp are unchanged.

All500 updates ×4 ranks match baseline sample-ID/Full/reference-stream SHA256 values, checked during
the run. A nonuniform-image RNG test verifies identical image augmentation after the same RNG state.
Shared step0 SHA256:`{r['preflight']['common_step0_sha256']}`; optimizer initially empty.
The baseline full/bare hashes were verified and its metrics reused without retraining.
Formal training starts at0 with resume=None, not from smoke or a trained checkpoint.

## Resources and diagnostics

4×A10080GB,256/rank,full1024 image candidates, accumulation1,workers8, encoder checkpoint ON,
pair checkpoint OFF,image_chunk128,text_chunk128; BF16 encoders and unchanged FP32 masks/scores/losses.
Slowest-rank normal-cycle mean{resource['mean_seconds']:.6f}s, median{resource['median_seconds']:.6f}s,
P95{resource['p95_seconds']:.6f}s,max{resource['max_seconds']:.6f}s; all500 mean including startup
{resource['all500_mean_seconds_including_warmup']:.6f}s.
Max peak allocated{resource['peak_allocated_gib_max']:.6f}GiB, reserved{resource['peak_reserved_gib_max']:.6f}GiB.
The ≤3s mean/≤65GiB resource gate passed. Warmup5 and checkpoint writes are recorded separately.
Loss/gradients are finite and final cross-rank parameter difference0. Step1/100/200/500 and last50
record actual F/S/D lengths, validity/truncation, mask keep/IoU and original loss/gate diagnostics.
Runtime O/E field names are translated to Summary/Detail only in reports; their mathematical roles do not change.

## Strict native export and complete Recall

Full checkpoint:`{r['checkpoint']}`; SHA256:`{r['checkpoint_sha256']}`.
Bare student:`{r['bare_student']}`; SHA256:`{r['bare_sha256']}`.
Strict reconstruction/export matches training native image/text embeddings with max_abs0;
all optimizer state steps are500. Native inference uses full-caption text and unmasked native image
embeddings only. No Summary/Detail, training masks/gates, reranking or ensemble enter evaluation.

Frozen COCO5000/25000,Urban1000/1000,Flickr test1K1000/5000,DOCCI5000/5000,
Long-DCI reconstructed7602/7602 (SHA8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b).
No DCI Full or alternative split runs.

{recall_table}

Config and exact commands are in config.json/commands/. Evidence includes CPU correctness tests,
preflight, resource/stream matching, sampling and strict export; raw/ retains every native Recall.
Large checkpoints, raw data, caches and indexes remain on the evaluation server.
Full4868 confirmation is a future explicit decision, not started by this experiment.
'''
    (EXP/'SUMMARY_DETAIL_500_REPORT.md').write_text(text)
    (EXP/'README.md').write_text(f'''# Summary–Detail@500: {conclusion}

{statement}
All500 updates and five frozen native retrieval datasets completed. Matched baseline is reused;
common step0 and horizon4868 are preserved. No full4868 run is started.

Read [SUMMARY_DETAIL_500_REPORT.md](SUMMARY_DETAIL_500_REPORT.md) and [RESULTS.json](RESULTS.json)
for every directional R1/R5/R10, deltas, checkpoint hashes, matching and resource evidence.
[DATA_SAMPLING_AUDIT.md](DATA_SAMPLING_AUDIT.md) distinguishes raw captions from packed Full,
and records complete-corpus quantiles and actual Detail truncation/coverage.
Code/logs use tokens_o/e as unchanged internal slots; method/report names are Full/Summary/Detail.
Commands are structured arrays in commands/. Source base14653c9, branch
codex/nest-balanced-summary-detail-500-v1. Large weights/data/caches are outside Git.
''')
    state=load(RUN/'state.json');state.update(status='EVALUATED_REPORT_READY',stage='awaiting Git synchronization',
        conclusion=conclusion,scores=r['scores'],completed_at=now(),full4868_run_started=False)
    dump(RUN/'state.json',state)
    print(json.dumps({'conclusion':conclusion,'scores_percent':{k:v*100 for k,v in r['scores'].items()},
                      'delta_scores_pp':delta_scores,'resources':resource['mean_seconds']}),flush=True)


if __name__=='__main__':
    main()
