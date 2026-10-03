"""Three-way sampling result and evidence-based redundancy/T2I interpretation."""
import json
import shutil
import statistics
from pathlib import Path

import numpy as np

from .run import EXP,RUN,load,dump,records,stream_check,now
from experiments.nest_clip_v1.balanced_summary_detail_500_v1.report import token_summary,table

NAMES=['COCO','Urban-1k','Flickr30k-test1k','DOCCI','Long-DCI']


def main():
    r=load(RUN/'result.json');a=r['baseline'];b=load(EXP/'ALL_DETAIL_BASELINE.json')
    root=RUN/'step500';rows=records(root/'steps.jsonl');old=records(Path(a['root'])/'steps.jsonl')
    assert len(rows)==500 and [x['step'] for x in rows]==list(range(1,501))
    for x,y in zip(rows,old):
        stream_check(x,y)
        assert x['nonfinite']==0 and all(h['gradients_finite'] for h in x['rank_health'])
        for h in x['rank_health']:
            assert h['sampling']['random_detail_sampling']['all_detail_m_ge3_count']==0
    assert r['config']['horizon']==4868 and r['config']['start_updates']==0 and r['config']['resume'] is None
    assert r['strict_export']['passed'] and r['strict_export']['optimizer_steps']==[500]
    assert r['strict_export']['image_max_abs']==r['strict_export']['text_max_abs']==0
    da={k:(v-a['scores'][k])*100 for k,v in r['scores'].items()}
    db={k:(v-b['scores'][k])*100 for k,v in r['scores'].items()}
    dira={n:{d:(r['metrics'][n][d]['R@1']-a['metrics'][n][d]['R@1'])*100 for d in ['I2T','T2I']} for n in NAMES}
    dirb={n:{d:(r['metrics'][n][d]['R@1']-b['metrics'][n][d]['R@1'])*100 for d in ['I2T','T2I']} for n in NAMES}
    longmean=statistics.fmean(dira['Long-DCI'].values())
    if da['Score5_R1']>0 and longmean>=-.2:
        conclusion='POSITIVE';statement='Random partial detail sampling improves over both exhaustive detail and the original RandomK baseline; full4868 confirmation is worth considering.'
    elif db['Score5_R1']>=.30 and da['Score5_R1']<=0:
        conclusion='RESCUE-ONLY';statement='Random partial detail sampling repairs much of the exhaustive-detail degradation but does not yet outperform the original RandomK design.'
    elif dirb['Urban-1k']['T2I']>0 or dirb['Long-DCI']['T2I']>0 or db['Score5_R1']>0:
        conclusion='MIXED';statement='Partial detail sampling addresses only part of the Summary–Detail failure.'
    else:
        conclusion='NEGATIVE';statement='The Summary-first decomposition itself, rather than exhaustive detail coverage alone, is likely incompatible with the current objective at500.'
    cycles=records(root/'cycle_timing.jsonl');times=[x['four_rank_max_seconds'] for x in cycles if not x['warmup']]
    accept=load(root/'acceptance.json')
    resource={'passed':statistics.fmean(times)<=3 and all(x['peak_allocated_gib']<=65 for x in accept['ranks']),
        'mean_seconds':statistics.fmean(times),'median_seconds':statistics.median(times),
        'p95_seconds':float(np.quantile(times,.95)),'max_seconds':max(times),
        'peak_allocated_gib_max':max(x['peak_allocated_gib'] for x in accept['ranks']),
        'peak_reserved_gib_max':max(x['peak_reserved_gib'] for x in accept['ranks']),
        'normal_updates':len(times),'warmup_updates':5,'ranks':accept['ranks']}
    assert resource['passed']
    diagnostics={str(step):{'sampling':token_summary([rows[step-1]]),
        'loss_mask_gate':{k:v for k,v in rows[step-1].items() if k not in ['rank_health','duplicate_image_ids']},
        'KD_per_rank':[h['sampling']['random_detail_sampling'] for h in rows[step-1]['rank_health']]}
        for step in [1,100,200,500]}
    last=rows[-50:]
    last50={'sampling':token_summary(last),
        'loss_mask_gate_mean':{k:statistics.fmean(x[k] for x in last if k in x) for k,v in rows[-1].items() if isinstance(v,(float,int))},
        'loss_mask_gate_observation_counts':{k:sum(k in x for x in last) for k,v in rows[-1].items() if isinstance(v,(float,int))}}
    # F/D IoU is sampled at fixed diagnostic steps, not mislabeled as50 measurements.
    redundancy=load(EXP/'TEXT_REDUNDANCY_AUDIT.json');ambiguity=load(EXP/'T2I_AMBIGUITY_AUDIT.json')
    assert set(redundancy['stages'])==set(ambiguity['stages'])=={'step0','step500'}
    assert redundancy['stages']['step0']['provenance']['sample_id_sha256']==redundancy['stages']['step500']['provenance']['sample_id_sha256']
    r.update(status='COMPLETE',conclusion=conclusion,statement=statement,all_detail_baseline=b,
        delta_scores_pp_vs_RandomK=da,delta_scores_pp_vs_AllDetail=db,
        delta_R1_pp_vs_RandomK=dira,delta_R1_pp_vs_AllDetail=dirb,resources=resource,
        sampling_diagnostics=diagnostics,last50=last50,sampling_audit=load(EXP/'SAMPLING_AUDIT.json'),
        text_redundancy=redundancy,t2i_ambiguity=ambiguity,
        corrected_prior_diagnostics=load(EXP/'evidence/baseline-diagnostic-correction.json'),
        baseline_retrained=False,all_detail_baseline_retrained=False,all_2000_F_stream_hashes_equal=True,
        full4868_run_started=False,completed_at=now())
    dump(EXP/'RESULTS.json',r)
    for filename,data in [('resource-summary.json',resource),('step-diagnostics.json',diagnostics),('last50.json',last50)]:
        dump(EXP/'evidence'/filename,data)
    for file in ['config.json','acceptance.json','export-check.json','checkpoint_timing.jsonl']:
        shutil.copyfile(root/file,EXP/'evidence'/file)
    for p in (RUN/'execution').glob('*.console.txt'):
        lines=p.read_text(errors='replace').replace('\r','\n').splitlines()
        (EXP/'evidence'/(p.stem+'.summary.txt')).write_text('\n'.join(x.rstrip() for x in lines if not ('it/s]' in x or 's/it]' in x))+'\n')
    main_table=table(['Model','Score5_R1 %','J_long3 %','J_long %'],
        [[label,*[f'{x["scores"][k]*100:.6f}' for k in r['scores']]]
         for label,x in [('RandomK',a),('Summary + All Detail',b),('Summary + Random Detail',r)]]+
        [['Delta versus RandomK pp',*[f'{da[k]:+.6f}' for k in r['scores']]],
         ['Delta versus AllDetail pp',*[f'{db[k]:+.6f}' for k in r['scores']]]])
    directions=table(['Dataset','Dir','RandomK %','All-Detail %','Random-Detail %','Δ versus RandomK pp','Δ versus AllDetail pp'],
        [[n,d,*[f'{x["metrics"][n][d]["R@1"]*100:.6f}' for x in [a,b,r]],f'{dira[n][d]:+.6f}',f'{dirb[n][d]:+.6f}'] for n in NAMES for d in ['I2T','T2I']])
    ce_rows=[]
    baseline_logs=records(Path(a['root'])/'steps.jsonl')[-50:]
    for view,prefix in [('Full','F'),('Summary/P','O'),('Detail/R','E')]:
        for direction,key in [('I2T','i2t'),('T2I','t2i')]:
            field=prefix+'_'+key
            ce_rows.append([view,direction,f'{statistics.fmean(x[field] for x in baseline_logs):.6f}',
                f'{b["last50"]["loss_mask_gate_mean"][field]:.6f}',f'{last50["loss_mask_gate_mean"][field]:.6f}'])
    ce_table=table(['View','CE direction','RandomK last50','AllDetail actual last50','RandomDetail last50'],ce_rows)
    red_rows=[]
    for stage,data in redundancy['stages'].items():
        for name,v in data['metrics'].items():red_rows.append([stage,name,f'{v["mean"]:.6f}',f'{v["median"]:.6f}'])
    red_table=table(['Same model stage','Text pair','Mean cosine','Median cosine'],red_rows)
    amb_rows=[]
    for stage,data in ambiguity['stages'].items():
        for name,v in data['metrics'].items():
            amb_rows.append([stage,name,*[f'{v[k]["mean"]:.6f}' for k in ['nearest_off_diagonal_cosine','top5_nearest_negative_cosine','mean_off_diagonal_cosine']]])
    amb_table=table(['Stage','View','Nearest off-diagonal','Top5 negative mean','Mean off-diagonal'],amb_rows)
    mask_rows=[]
    for name in ['F_keep_ratio','O_keep_ratio','E_keep_ratio','oe_iou','hard_inclusion_violation','inc']:
        mask_rows.append([name,f'{statistics.fmean(x[name] for x in baseline_logs):.6f}',
            f'{b["last50"]["loss_mask_gate_mean"][name]:.6f}',f'{last50["loss_mask_gate_mean"][name]:.6f}'])
    mask_table=table(['Last50 read-only metric','RandomK','AllDetail','RandomDetail'],mask_rows)
    recalls=table(['Model','Dataset','Dir','R@1 %','R@5 %','R@10 %'],
        [[label,n,d,*[f'{x["metrics"][n][d]["R@"+str(k)]*100:.6f}' for k in [1,5,10]]]
         for label,x in [('RandomK',a),('AllDetail',b),('RandomDetail',r)] for n in NAMES for d in ['I2T','T2I']])
    sa=r['sampling_audit'];post=redundancy['stages']['step500']['metrics']
    text=f'''# Summary + Random Detail Subset: strict500-update result

**{conclusion}.** {statement}
Exactly500 optimizer updates from common step0 after separate smoke5, horizon4868.
No baseline retraining, full4868 confirmation or new sampling/weight/loss experiment follows.

## Three-way primary comparison

{main_table}

{directions}

All scores/deltas use unrounded fractions. RandomK is the primary baseline; All-Detail is the
secondary failure-repair comparison. POSITIVE requires beating RandomK with no Long-DCI mean drop beyond0.2pp;
RESCUE-ONLY requires ≥0.30pp over AllDetail without beating RandomK. Remaining partial improvements are
MIXED; no improvement versus AllDetail is NEGATIVE. These are reporting rules, not training gates.

## Visible-only bounded sampling

Full raw strings/tokens preserve the formal baseline packing exactly. Summary is the first visible
sentence. D_pool is only the remaining complete visible sentences. K is uniform2..m-1 for m>=3;
m2 selects1 of2, m1 uses the sole detail, m0 Full-only. Independent SHA256(seed,epoch,sample_id,domain)
RNG chooses without replacement; selected indices are sorted. No shuffle, raw trailing sentence,
offset PAD, extra loss or changed candidate denominator runs. Old three modes remain reproducible.

Fixed100k simulation: Random Detail averages{sa['views']['Random_Detail']['tokens']['mean']:.6f}
effective tokens versus AllDetail{sa['views']['All_Detail_raw_previous']['tokens']['mean']:.6f};
mean K={sa['KD']['mean']:.6f}, median{sa['KD']['median']}, K1 fallback{sa['K1_fallback_fraction']*100:.6f}%.
K=m count at m>=3 is0. Mean visible detail token coverage
{sa['visible_detail_token_coverage']['mean']*100:.6f}%. All100k Full tensors/raw strings match;
all500×4 training sample-ID/Full/reference digests match the baseline, monitored live.

## Q1: Was All-Detail failure mainly excessive similarity to Full?

Before training, same common-step0 F/AllDetail mean cosine
{redundancy['stages']['step0']['metrics']['F_vs_All_Detail_raw']['mean']:.6f};
F/RandomDetail{redundancy['stages']['step0']['metrics']['F_vs_Random_Detail']['mean']:.6f}.
On the same final500 model and same fixed1024 IDs, F/AllDetail
{post['F_vs_All_Detail_raw']['mean']:.6f}, F/RandomDetail{post['F_vs_Random_Detail']['mean']:.6f}.
Thus construction/embedding redundancy is measured separately from its retrieval utility.
The actual retrieval result and gate/mask/CE evidence above determine whether reduced redundancy
repairs the failure; cosine reduction alone is not causal proof that proximity caused the old degradation.
This run also removes old raw-tail budget violations; the all-visible-detail diagnostic control
distinguishes text-construction effects but is not a separate trained factorial control.
One seed and500 updates cannot prove a unique failure mechanism.
The actual last50 S-D mask IoU changes from
{b['last50']['loss_mask_gate_mean']['oe_iou']:.6f} (AllDetail model) to
{last50['loss_mask_gate_mean']['oe_iou']:.6f} (RandomDetail model); masks do not become more separated
merely because text cosine decreases. The RandomDetail model's fixed step500 F-D IoU is
{last50['loss_mask_gate_mean']['E_F_D_positive_mask_iou']:.6f} (one read-only observation).
These are diagnostics, not additional constraints or an isolated causal training control.

{red_table}

## Q2: Does the subset recover T2I and per-view supervision?

Urban T2I deltas versus RandomK/AllDetail:{dira['Urban-1k']['T2I']:+.6f}/{dirb['Urban-1k']['T2I']:+.6f}pp.
Long-DCI T2I deltas:{dira['Long-DCI']['T2I']:+.6f}/{dirb['Long-DCI']['T2I']:+.6f}pp.
Actual last50 S_t2i/D_t2i:{last50['loss_mask_gate_mean']['O_t2i']:.6f}/
{last50['loss_mask_gate_mean']['E_t2i']:.6f}; actual AllDetail counterparts
{b['last50']['loss_mask_gate_mean']['O_t2i']:.6f}/{b['last50']['loss_mask_gate_mean']['E_t2i']:.6f}.

The task prompt's approximate prior S/D CE and IoU do not match saved AllDetail artifacts.
Actual prior S_t2i0.749354,D_t2i0.126268,S-D IoU0.830646. Original logs are authoritative;
baseline-diagnostic-correction.json records the discrepancy. Lower CE of an almost-Full view can
reflect an easier/less-partial task; it is not alone evidence of more useful supervision.

{ce_table}

## Ambiguity and mask diagnostics

Each view's nearest/top5/mean off-diagonal text cosine uses one fixed1024 set and the same model/tokenizer
within each stage. It measures semantic similarity, not confirmed false negatives or a modified loss.
No nearest neighbor changes training candidates. All-visible control is diagnostic-only.

{amb_table}

{mask_table}

Step1/100/200/500 and last50 record F/S/D lengths, CE directions, keep/gate stats,
S-D IoU, inclusion and hard violations. F-D IoU is read-only at the four existing diagnostic steps;
its last50 observation count is explicitly1 (step500), not50 measurements. Model source/math stays
byte-identical; detached observer tests show bitwise-equivalent loss, every gradient and actualAdamW update.

## Initialization, resources and strict export

Common initializer SHA54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6;
same architecture, four optimizer groups, best coefficients, seeds/batch/scheduler and F RNG streams.
Formal starts0, resume=None; smoke weights are not used. Loss/gradients finite and rank parameters agree.
4×A10080GB,256/rank,1024 full candidates, encoder checkpoint ON,pair OFF,128 image/text chunks.
Normal complete-cycle mean{resource['mean_seconds']:.6f}s,median{resource['median_seconds']:.6f}s,
P95{resource['p95_seconds']:.6f}s,max{resource['max_seconds']:.6f}s.
Peak allocated/reserved{resource['peak_allocated_gib_max']:.6f}/{resource['peak_reserved_gib_max']:.6f}GiB.
≤3s mean/≤65GiB gate passed; checkpoint writes and first5 warmup recorded separately.
Strict export has optimizer steps[500], image/text native max_abs0.
Full checkpoint:{r['checkpoint']}; SHA256`{r['checkpoint_sha256']}`.
Bare student:{r['bare_student']}; SHA256`{r['bare_sha256']}`.

## Complete frozen native recalls

Inference is normalized native image/full-caption text dot product only. No masks/gates/S/D,
PCA/reranking/ensemble/augmentation. COCO5000/25000,Urban1000,Flickr test1K1000/5000,
DOCCI5000,Long-DCI reconstructed7602 with immutable8890a2be... manifest. No DCI Full.

{recalls}

Config, structured commands, sampling/semantic audits, raw JSON, correctness and resource/export evidence
are in this directory. Large weights/data/embedding dumps stay outside Git. Completed at{r['completed_at']}.
'''
    (EXP/'SUMMARY_RANDOM_DETAIL_500_REPORT.md').write_text(text)
    (EXP/'README.md').write_text(f'''# Summary + Random Detail@500: {conclusion}

{statement}
All500 updates and five frozen native evaluations complete; no full4868 run is started.
Read [SUMMARY_RANDOM_DETAIL_500_REPORT.md](SUMMARY_RANDOM_DETAIL_500_REPORT.md) and RESULTS.json
for three-way deltas, Q1/Q2, all recalls, corrected prior CE/IoU, resource and strict-export evidence.
SAMPLING_AUDIT.json records fixed100k; TEXT_REDUNDANCY_AUDIT.json/T2I_AMBIGUITY_AUDIT.json
use identical1024 IDs and one model for all views at each step0/500 stage.
New mode summary_random_detail uses only visible Full, independent bounded-K RNG and ordered subsets.
Code branch codex/nest-balanced-summary-random-detail-500-v1. Large artifacts remain on server.
''')
    state=load(RUN/'state.json');state.update(status='EVALUATED_REPORT_READY',stage='awaiting Git synchronization',
        conclusion=conclusion,scores=r['scores'],completed_at=r['completed_at'],full4868_run_started=False)
    dump(RUN/'state.json',state)
    print(json.dumps({'conclusion':conclusion,'scores_percent':{k:v*100 for k,v in r['scores'].items()},
                      'delta_RandomK_pp':da,'delta_AllDetail_pp':db}),flush=True)


if __name__=='__main__':main()
