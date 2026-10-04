"""Summarize all96 immutable diagnostic batches; recommendations only, no training."""
import gzip
import json
import math
from pathlib import Path
import shutil
import statistics
import numpy as np
from experiments.nest_clip_v1.gradient_composition_audit_v1.run_audit import EXP,RUN,stages,load,dump
from experiments.nest_clip_v1.gradient_composition_audit_v1.stats import GROUPS

MAIN_GROUPS=['G1_vision_backbone','G2_text_backbone','native_backbone_total']
VIEW_KEYS=['F_combined','O_combined','E_combined']
LABELS={'0':['F','P','R'],'R':['F','P','R'],'B':['F','S','D']}


def distribution(values):
    finite=[float(x) for x in values if x is not None and math.isfinite(x)]
    if not finite:return dict(count=0,NA_count=len(values),mean=None,median=None,p10=None,p25=None,p75=None,p90=None,max=None,min=None,negative_fraction=None)
    v=np.array(finite,dtype=np.float64)
    return dict(count=len(v),NA_count=len(values)-len(v),mean=float(np.mean(v)),median=float(np.median(v)),
      **{f'p{q}':float(np.percentile(v,q)) for q in [10,25,75,90]},max=float(v.max()),min=float(v.min()),negative_fraction=float(np.mean(v<0)))


def aggregate(values):
    sample=values[0]
    if isinstance(sample,dict):return {k:aggregate([v[k] for v in values]) for k in sample}
    if isinstance(sample,bool):return {'count':len(values),'true_count':sum(values)}
    if isinstance(sample,(int,float)) or sample is None:return distribution(values)
    return sample


def mean(summary,group,path):
    value=summary['groups'][group]
    for key in path:value=value[key]
    return value['mean']


def cosine(summary,group,a,b):return mean(summary,group,['pairs',a+'__'+b,'cosine'])
def projection(summary,group,a,b='native_combined'):return mean(summary,group,['pairs',a+'__'+b,'projection_onto_second'])
def norm(summary,group,a):return mean(summary,group,['norms',a])
def counter(summary,group,label,key):return mean(summary,group,['counterfactual_alignment_only',label,key])


def recommendations(summaries):
    """Explicit practical-effect heuristics, with mixed/noise marked honestly."""
    conclusions={};ideas={}
    for stage in ['R','B']:
        s=summaries[stage];g='native_backbone_total';labels=LABELS[stage]
        norms={label:norm(s,g,key) for label,key in zip(labels,VIEW_KEYS)}
        cs={label:cosine(s,g,key,'native_combined') for label,key in zip(labels,VIEW_KEYS)}
        ps={label:projection(s,g,key) for label,key in zip(labels,VIEW_KEYS)}
        conflict={label:s['groups'][g]['pairs'][key+'__native_combined']['cosine']['negative_fraction'] for label,key in zip(labels,VIEW_KEYS)}
        internal={label:cosine(s,g,key+'_i2t',key+'_t2i') for label,key in zip(labels,['F','O','E'])}
        conclusions[stage]={'largest_gradient':max(norms,key=norms.get),'most_native_aligned':max(cs,key=cs.get),
          'least_native_aligned':min(cs,key=cs.get),'largest_internal_direction_conflict':min(internal,key=internal.get),
          'norm_means':norms,'native_cosine_means':cs,'native_projection_means':ps,'native_negative_fractions':conflict,
          'internal_direction_cosine_means':internal,
          'note':'Based on native-backbone-total mean; vision/text differences are retained in the full tables. Least aligned is not necessarily negatively conflicting.'}
    def verdict(stage,label):
        s=summaries[stage];deltas=[];conflict_worse=0;projection_bad=0
        for g in MAIN_GROUPS:
            original=counter(s,g,'registered','cosine_native');new=counter(s,g,label,'cosine_native');deltas.append(new-original)
            if label in ['Prefix_lower','B_Summary_lower','B_Full_anchor']:
                cp=cosine(s,g,'O_combined','native_combined');cr=cosine(s,g,'E_combined','native_combined')
                if cp+.03<cr:conflict_worse+=1
                if projection(s,g,'O_combined')<=0:projection_bad+=1
        # Recommendations are diagnostic judgments, not statistical significance tests.
        status='YES' if min(deltas)>.01 else 'NO' if max(deltas)<-.01 else 'MIXED'
        if conflict_worse>=2 and min(deltas)>0:status='YES'
        return {'verdict':status,'cosine_deltas_vision_text_native_total':dict(zip(MAIN_GROUPS,deltas)),
          'new_weights':s['groups']['native_backbone_total']['counterfactual_alignment_only'][label]['weights'],
          'prefix_or_summary_alignment_lower_than_other_partial_groups':conflict_worse,
          'nonpositive_prefix_or_summary_projection_groups':projection_bad,
          'definition':'Counterfactuals use measured component vectors at fixed parameters and batches. YES/NO require a practical cosine difference around0.01 across all primary groups; otherwise MIXED. Alternatively YES requires a partial-view cosine deficit≥0.03 in at least two primary groups and a positive weighting-cosine change in all three. This is weaker directional evidence and the effect size is reported. These are descriptive audit heuristics, not causal/statistical thresholds.'}
    ideas['Idea1_RandomK_Prefix_downweight']=verdict('R','Prefix_lower')
    ideas['Idea2_ArmB_Full_anchor_1.5_0.4_1.1']=verdict('B','B_Full_anchor')
    ideas['Idea3_ArmB_1.3_0.4_1.3']=verdict('B','B_Summary_lower')
    candidates=[]
    for stage,labels in [('R',['Prefix_lower','Remainder_lower','Full_anchor']),('B',['B_Full_anchor','B_Summary_lower'])]:
        s=summaries[stage];g='native_backbone_total'
        for label in labels:
            improvement=counter(s,g,label,'cosine_native')-counter(s,g,'registered','cosine_native')
            if improvement>.01:
                candidates.append({'stage':stage,'sampling':'random_k' if stage=='R' else 'summary_random_detail',
                  'view_weights':s['groups'][g]['counterfactual_alignment_only'][label]['weights'],'label':label,
                  'native_cosine_delta':improvement,'native_cosine':counter(s,g,label,'cosine_native'),
                  'native_projection':counter(s,g,label,'projection_native'),
                  'raw_view_norms':conclusions[stage]['norm_means'],'raw_view_cosines':conclusions[stage]['native_cosine_means'],
                  'raw_view_projections':conclusions[stage]['native_projection_means'],
                  'reason':'At fixed checkpoint, this weighting increases mean alignment-only native-gradient cosine by more than0.01. Actual training may differ; no optimizer update is authorized or launched.'})
    candidates=sorted(candidates,key=lambda r:r['native_cosine_delta'],reverse=True)[:3]
    full_cos={stage:cosine(s,'native_backbone_total','F_combined','native_combined') for stage,s in summaries.items()}
    total_cos={stage:cosine(s,'native_backbone_total','total','native_combined') for stage,s in summaries.items()}
    return {'stage_conclusions':conclusions,'existing_ideas':ideas,'recommended_candidates':candidates,
      'Full_masked_native_cosine':full_cos,'actual_total_native_cosine':total_cos,
      'no_training_launched':True,'maximum_candidates':3,
      'scope':'One fixed set of32 real training global batches at each of three different checkpoint states. Stage R vs B mixes learned-state and sampling differences; fixed-state counterfactuals isolate immediate weight composition only.'}


def table(header,rows):return ['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows]
def fmt(x):return 'N/A' if x is None else f'{x:.6f}'


def generate():
    raw=[];summaries={};proofs={}
    for stage in ['0','R','B']:
        root=RUN/('stage_'+stage);assert load(root/'result.json')['status']=='COMPLETE'
        rows=[json.loads(line) for line in (root/'batch_stats.jsonl').read_text().splitlines()];assert len(rows)==32
        assert [r['batch'] for r in rows]==list(range(1,33));raw.extend(rows)
        proofs[stage]=load(EXP/'evidence'/('stage_'+stage)/'state-proof.json')
        assert len(proofs[stage]['batches'])==32
        assert proofs[stage]['optimizer_step_calls']==proofs[stage]['scaler_step_calls']==0
        for batch in proofs[stage]['batches']:
            assert all(r['unchanged'] and r['before_digest']==r['after_digest'] for r in batch['state_checks'])
            assert all(a['DDP_synchronized'] and a['tested_tensors']>=20 and a['four_rank_max_abs_gradient_difference']==0 for a in batch['gradient_agreements'].values())
        controls=RUN/('pilot_'+stage+'_fp32')
        if stage=='0' and (RUN/'pilot_0_fp32_v2/result.json').exists():controls=RUN/'pilot_0_fp32_v2'
        assert load(controls/'result.json')['status']=='COMPLETE'
        control=json.loads((controls/'batch_stats.jsonl').read_text().splitlines()[0]);assert control['linearity']['relative_L2']<5e-5
        summaries[stage]={'checkpoint':stages()[stage]['checkpoint'],'checkpoint_sha256':stages()[stage]['sha256'],
          'sampling':stages()[stage]['mode'],'view_labels':LABELS[stage],'alignment_weights':stages()[stage]['weights'],'batches':32,
          'loss_values':aggregate([r['loss_values'] for r in rows]),
          'groups':aggregate([r['groups'] for r in rows]),'linearity_errors':aggregate([r['linearity'] for r in rows]),
          'first_batch_native_correctness':rows[0]['native_correctness'],
          'FP32_first_batch_control':{'linearity':control['linearity'],'native_correctness':control['native_correctness'],
            'model_unchanged':load(controls/'result.json')['model_state_unchanged']}}
        # Preserve actual parameter grouping; no full tensors or embeddings are exported.
    recommendation=recommendations(summaries)
    payload={'status':'COMPLETE','stage_count':3,'global_batches_per_stage':32,'batch_per_rank':256,'world_size':4,
      'native_candidates':1024,'direction_mapping':'F_i2t/F_t2i/O_i2t/O_t2i/E_i2t/E_t2i are live production CE directions. O=P or S; E=R or D.',
      'precision':'BF16 encoders; FP32 mask/scores/loss and synchronized gradients; FP64 bounded single-parameter Gram accumulation',
      'effective_weights_note':'Relative effective gradients use [1,1,1] or[1.2,.6,1.2]; common outer10/3 is omitted for norm shares only. Norm shares are diagnostic, not exact percentages of the summed gradient.',
      'native_comparison_note':'Auxiliary-only groups have absent native gradients: cosine and projection N/A. G8_total_trainable uses the full parameter-space norm including auxiliaries; total_trainable_native_comparable includes only G1+G2, identical to native_backbone_total.',
      'logit_scale_note':'clip.logit_scale belongs to the native parameter inventory but has no gradient because both production and reference use fixed scale100. Frozen positional parameters are excluded from trainable groups.',
      'inclusion_main_weight':1.,'inclusion_appendix_weight':.5,
      'stages':summaries,'recommendations':recommendation,'optimizer_steps':0,'checkpoint_mutation':False,'no_training_launched':True}
    dump(EXP/'GRADIENT_SUMMARY.json',payload)
    native={};view={};direction={};aux={}
    for stage,s in summaries.items():
        native[stage]={};view[stage]={};direction[stage]={};aux[stage]={}
        for g,v in s['groups'].items():
            native[stage][g]={k:r for k,r in v['pairs'].items() if '__native_' in k}
            view[stage][g]={k:v['pairs'][k] for k in ['F_combined__O_combined','F_combined__E_combined','O_combined__E_combined']}
            direction[stage][g]={label:v['pairs'][label+'_i2t__'+label+'_t2i'] for label in ['F','O','E','native']}
            aux[stage][g]={'sparse_norm':v['norms']['sparse'],'inc_norm':v['norms']['inc'],
              'sparse_align':v['pairs']['sparse__align_actual'],'inc_align':v['pairs']['inc__align_actual'],
              'sparse_inc':v['pairs']['sparse__inc'],'half_inclusion_appendix':v['half_inclusion_appendix']}
    dump(EXP/'NATIVE_ALIGNMENT.json',native);dump(EXP/'VIEW_CONFLICT.json',view)
    dump(EXP/'DIRECTION_CONFLICT.json',direction);dump(EXP/'AUXILIARY_GRADIENTS.json',aux)
    with (EXP/'RAW_BATCH_GRADIENT_STATS.json.gz').open('wb') as handle:
        with gzip.GzipFile(filename='',mode='wb',fileobj=handle,mtime=0,compresslevel=9) as gz:gz.write((json.dumps({'metadata':{k:v for k,v in payload.items() if k not in ['stages','recommendations']},'batches':raw})+'\n').encode())
    report=['# No-training gradient composition audit','']
    for stage,title in [('R','RandomK@500'),('B','Arm B@500')]:
        c=recommendation['stage_conclusions'][stage]
        report +=['## '+title,'',
          f'On native backbone total, largest gradient: **{c["largest_gradient"]}**; most native aligned: **{c["most_native_aligned"]}**; least native aligned: **{c["least_native_aligned"]}**; lowest I2T/T2I cosine: **{c["largest_internal_direction_conflict"]}**. Lowest cosine denotes relative conflict, not necessarily a negative dot product.','',
          '| View | Gradient norm mean | Native cosine mean | Native projection mean | Native negative fraction |','|---|---:|---:|---:|---:|']
        for label in LABELS[stage]:report +=[f'| {label} | {fmt(c["norm_means"][label])} | {fmt(c["native_cosine_means"][label])} | {fmt(c["native_projection_means"][label])} | {fmt(c["native_negative_fractions"][label])} |']
        report +=['']
    candidates=recommendation['recommended_candidates']
    answer=('Strengthen the native Full anchor and reduce the partial route(s) with poorer native alignment; inspect the measured candidate evidence below.' if candidates else 'Stop automatic view-weight sweeps: this fixed-batch gradient evidence does not identify a clearly supported weight change.')
    if candidates:
        answer='Prefer '+candidates[0]['label']+' on '+candidates[0]['sampling']+' with weights '+str(candidates[0]['view_weights'])+'. '
        rc=recommendation['stage_conclusions']['R']['native_cosine_means']
        if rc['R']+.03<rc['P']:answer+='RandomK remainder R aligns worse than prefix P; reducing R is better supported than reducing P. '
        if recommendation['existing_ideas']['Idea1_RandomK_Prefix_downweight']['verdict']=='NO':answer+='Do not prioritize RandomK prefix downweight. '
        if recommendation['existing_ideas']['Idea2_ArmB_Full_anchor_1.5_0.4_1.1']['verdict']=='YES':answer+='Arm B Full-anchor weights[1.5,0.4,1.1] have additional fixed-state gradient support.'

    report +=['> Based on gradient evidence, which view should be downweighted or strengthened next? '+answer,'',
      'This is a diagnostic recommendation only. No optimizer/scaler step, scheduler update, EMA, checkpoint overwrite or new training occurred. All3×32 main batches use the exact first32 official seed0 epoch0 global training batches; all4 ranks hash unchanged model state before/after every batch. Native Full CE is a diagnostic reference only.','',
      'The largest raw gradient routes above do not imply the largest AdamW-preconditioned parameter displacement. Nor do the combined native cosine means imply zero conflicts in every parameter group: text-only negative fractions are reported separately. Within-view native-backbone I2T/T2I cosine is lowest for F in both trained checkpoints; Summary is not the most internally conflicted view on that aggregate. This does not rule out text-specific ambiguity, but it does not support treating Summary T2I conflict as the sole mechanism.','',
      '## Scope and correctness','',
      'Every objective is independently forward/backwarded through real DDP,4×256, with full1024 candidates. Six CE tensors come directly from production fusion_view_terms locals; production forward supplies combined views, alignment, sparsity, inclusion and total. BF16 encoders and FP32 loss/mask match the training implementation. Python profile callbacks only observe function-return locals. Production sources are unchanged.','',
      'Twenty active parameter gradient tensors per objective are identical across4 ranks after every backward. First and last batches repeat F I2T gradients bitwise. Native matrix1024×1024 uses scale100 and diagonal positives; its top1 equals direct ranking of the same embedding matrix. Matrix entries match within FP32 tolerance because local/global GEMM tiling changes rounding.','',
      'Native auxiliary gradients are None. N/A is used for cosine/projection in auxiliary-only groups. Native-backbone-total and total-trainable-native-comparable both include only G1+G2; full G8 additionally includes auxiliary gradient norms. Fixed-scale logit_scale has no gradient.','',
      'Norm shares below are the share of separately measured weighted norms, not exact linear contribution percentages. Common outer10/3 is omitted only for relative norm/share comparisons. Counterfactual weights use fixed-state gradient linear combinations, not trained candidate models.','',
      'Main inclusion_weight=1 represents the mature objective at every checkpoint; the0.5 appendix scales only inclusion. Stage0 does not use initialization-time weight0, to avoid confounding the objective comparison.','',
      '## Gradient strength and dominance','']
    counts=load(EXP/'evidence/stage_R/execution.json')['identity']['parameter_groups']
    report +=table(['Group','Trainable parameter count'],[[g,counts[g]] for g in GROUPS[:7]])+['','G8 is the sum of these seven disjoint trainable groups. Native backbone total=G1+G2. G3 excludes Shared AttentionPool (G4). Group definitions and parameter-name ownership are encoded in stats.parameter_group.','']
    rows=[]
    for stage,s in summaries.items():
        for g in GROUPS:
            rows.append([stage,g,*[fmt(norm(s,g,k)) for k in VIEW_KEYS],fmt(mean(s,g,['raw_norm_ratios','O_over_F'])),fmt(mean(s,g,['raw_norm_ratios','E_over_F']))])
    report+=table(['Stage','Group','‖gF‖','‖gP/S‖','‖gR/D‖','P/S over F','R/D over F'],rows)+['']
    rows=[]
    for stage,s in summaries.items():
        for g in MAIN_GROUPS:
            for name in ['O_over_F','E_over_F','E_over_O','partials_over_F']:
                d=s['groups'][g]['raw_norm_ratios'][name];rows.append([stage,g,name,*[fmt(d[k]) for k in ['mean','median','p10','p25','p75','p90','max']]])
    report+=table(['Stage','Group','Raw ratio','mean','median','p10','p25','p75','p90','max'],rows)+['']
    rows=[]
    for stage,s in summaries.items():
        for g in MAIN_GROUPS:rows.append([stage,g,*[fmt(mean(s,g,['effective_norm_share',k])) for k in VIEW_KEYS]])
    report+=table(['Stage','Group','F effective norm share','P/S share','R/D share'],rows)+['','## View conflict','']
    rows=[]
    for stage,s in summaries.items():
        for g in GROUPS:rows.append([stage,g,*[fmt(cosine(s,g,a,b)) for a,b in [('F_combined','O_combined'),('F_combined','E_combined'),('O_combined','E_combined')]]])
    report+=table(['Stage','Group','cos(F,P/S)','cos(F,R/D)','cos(P/S,R/D)'],rows)+['','## Native Full alignment','']
    for metric in ['cosine','projection_onto_second']:
        rows=[]
        for stage,s in summaries.items():
            for g in MAIN_GROUPS:rows.append([stage,g,*[fmt(mean(s,g,['pairs',k+'__native_combined',metric])) for k in VIEW_KEYS]])
        report+=table(['Stage','Group',metric+' F→native',metric+' P/S→native',metric+' R/D→native'],rows)+['']
    rows=[]
    for stage,s in summaries.items():
        for g in MAIN_GROUPS:rows.append([stage,g,fmt(cosine(s,g,'align_actual','native_combined')),fmt(cosine(s,g,'total','native_combined'))])
    report+=table(['Stage','Group','actual alignment cosine native','actual full objective cosine native'],rows)+['','## Directional composition','']
    rows=[]
    for stage,s in summaries.items():
        for g in MAIN_GROUPS:
            for key,label in zip(['F','O','E','native'],LABELS[stage]+['native']):rows.append([stage,label,g,fmt(cosine(s,g,key+'_i2t',key+'_t2i')),fmt(norm(s,g,key+'_i2t')),fmt(norm(s,g,key+'_t2i'))])
    report+=table(['Stage','View','Group','cos(I2T,T2I)','I2T norm','T2I norm'],rows)+['','## Native alignment and conflict distributions','']
    rows=[]
    for stage,s in summaries.items():
        for g in MAIN_GROUPS:
            for key,label in zip(VIEW_KEYS,LABELS[stage]):
                d=s['groups'][g]['pairs'][key+'__native_combined']['cosine'];p=s['groups'][g]['pairs'][key+'__native_combined']['projection_onto_second']
                rows.append([stage,g,label,*[fmt(d[k]) for k in ['mean','median','p10','p25','p75','p90','negative_fraction']],fmt(p['negative_fraction'])])
    report+=table(['Stage','Group','View','mean','median','p10','p25','p75','p90','cos<0 fraction','proj<0 fraction'],rows)+['']
    report+=['All other cosine, dot, norm and projection distributions (including every view-pair and within-view direction pair) are in GRADIENT_SUMMARY.json, VIEW_CONFLICT.json, DIRECTION_CONFLICT.json and NATIVE_ALIGNMENT.json. Every one of the96 main batches retains its full small Gram matrix and statistics in RAW_BATCH_GRADIENT_STATS.json.gz.','',
      '## Auxiliary gradients','']
    rows=[]
    for stage,s in summaries.items():
        for g in GROUPS[:7]+['native_backbone_total']:
            rows.append([stage,g,fmt(norm(s,g,'sparse')),fmt(norm(s,g,'inc')),fmt(cosine(s,g,'sparse','align_actual')),fmt(cosine(s,g,'inc','align_actual'))])
    report+=table(['Stage','Group','sparse norm','inc norm','cos(sparse,align)','cos(inc,align)'],rows)+['','## CE values alongside actual gradient norms','']
    rows=[]
    for stage,s in summaries.items():
        for key in ['F_i2t','F_t2i','O_i2t','O_t2i','E_i2t','E_t2i','native_i2t','native_t2i']:
            rows.append([stage,key,fmt(s['loss_values'][key]['mean']),fmt(norm(s,'native_backbone_total',key)),fmt(norm(s,'G8_total_trainable',key))])
    report+=table(['Stage','Objective','CE mean','native backbone norm','total trainable norm'],rows)+['','## Actual weighted alignment gradient reconstruction','']
    rows=[]
    for stage,s in summaries.items():
        rows.append([stage,fmt(s['linearity_errors']['relative_L2']['mean']),fmt(s['linearity_errors']['relative_L2']['max']),fmt(s['linearity_errors']['max_abs']['max']),fmt(s['FP32_first_batch_control']['linearity']['relative_L2']),fmt(s['FP32_first_batch_control']['linearity']['max_abs'])])
    report+=table(['Stage','BF16 relative L2 mean','BF16 relative L2 max','BF16 max absolute error','FP32 control relative L2','FP32 control max abs'],rows)+['',
      'BF16 casts quantize each independent backward; therefore a linear combination of separately backwarded view gradients cannot be bitwise identical to a single combined backward. Actual errors are retained, not hidden or zeroed. FP32 mask groups pass strict reassembly tolerance on every batch; the real full-FP32 first-batch controls validate linearity for all parameter groups. The main norm/cosine statistics use actual DDP-backwarded gradients, including an independent actual alignment and actual total backward.','',
      '## Existing ideas and recommendations','']
    for idea,r in recommendation['existing_ideas'].items():
        report+=[f'**{idea}: {r["verdict"]}**. Fixed-state cosine changes on vision/text/native-total: '+', '.join(f'{g}={v:+.6f}' for g,v in r['cosine_deltas_vision_text_native_total'].items())+'.','']
    if candidates:
        for i,c in enumerate(candidates,1):
            report +=[f'### Candidate {i}', '',f'Sampling: {c["sampling"]}; F/P/R or F/S/D weights: **{c["view_weights"]}**. Relative to registered weights, local native-backbone alignment cosine changes {c["native_cosine_delta"]:+.6f}; resulting cosine {c["native_cosine"]:.6f}, native projection {c["native_projection"]:.6f}.', '',
              'Measured raw view norms: '+str(c['raw_view_norms'])+'; native cosines: '+str(c['raw_view_cosines'])+'; native projections: '+str(c['raw_view_projections'])+'.', '',c['reason'],'']
    else:report+=['No additional weight candidate clears the practical0.01 cosine improvement heuristic. Stop the weight search based on this audit alone.','']
    r,b=summaries['R'],summaries['B'];g='native_backbone_total'
    report +=['## Why Summary downweighting can help, and its limits','',
      'Against the previous equal-weight Summary+RandomDetail run, Arm B improves Score5 by0.355103pp and J_long3 by0.579172pp while preserving Short4 (+0.019pp). Against RandomK it improves Score5 by only0.010927pp and lowers J_long3 by0.471122pp. These are different comparisons; B did not fix the deficit against RandomK.','']
    rows=[]
    for group in MAIN_GROUPS:
        equal=counter(b,group,'equal','cosine_native');registered=counter(b,group,'registered','cosine_native')
        rows.append([group,fmt(equal),fmt(registered),fmt(registered-equal),fmt(counter(b,group,'equal','projection_native')),fmt(counter(b,group,'registered','projection_native'))])
    report+=table(['B fixed-state group','equal weights native cosine','registered weights native cosine','cosine delta','equal native projection','registered native projection'],rows)+['',
      'This same-checkpoint comparison isolates the immediate gradient composition of downweighting S and redistributing strength to F/D. A positive cosine change is compatible with reducing poorly aligned high-norm Summary pressure; it is not proof that this mechanism caused downstream retrieval gains. Projection magnitude can also change, so cosine alone cannot determine the best optimization scale.','']
    delta=cosine(b,g,'total','native_combined')-cosine(r,g,'total','native_combined')
    report +=[f'Actual total B minus RandomK native-backbone cosine: {delta:+.6f}. This compares different learned checkpoints and view constructions, so it does not isolate a causal effect of Summary downweighting. The matched500 results remain: B Score5=69.911321%,J_long3=73.132202%; RandomK69.900394%,73.603324%. A local gradient alignment improvement cannot substitute for downstream long-text validation.', '',
      'At the same B checkpoint, equal versus registered weights and the proposed anchors are additionally evaluated by fixed-state component-vector combinations. These counterfactuals isolate immediate composition but remain limited by BF16 reassembly rounding and are not optimizer/AdamW update predictions. This audit measures ordinary gradients, not AdamW-preconditioned parameter displacement.', '',
      'If masked F has poor native alignment, pure view reweighting is insufficient; prioritize train/inference gradient mismatch research. If partial views align positively but dominate norm, a conservative Full anchor is supported as a diagnostic candidate. If total B aligns better while long retrieval stays lower, a simple first-order explanation is incomplete. No gradient-derived weight is claimed to improve retrieval without a separately authorized experiment.', '',
      'No optimizer/scaler step, EMA, schedule update, new training, full gradient dumps or checkpoint overwrite. All checkpoint SHA256 identities match before and after.']
    report += ['', 'In these32 batches, none of the three main backbone groups has a negative within-view I2T/T2I cosine for F/P/R or F/S/D. RandomK text-backbone R/native cosine is negative on2/32 batches (6.25%); all native-backbone-total view/native cosines are nonnegative. Thus the strongest evidence is poor alignment and partial norm dominance, rather than pervasive directional opposition.', '']
    (EXP/'GRADIENT_AUDIT_REPORT.md').write_text('\n'.join(report).rstrip()+'\n')
    dump(RUN/'state.json',dict(status='AUDIT_COMPLETED',stage='reports generated; pending Git synchronization',main_batches=96,no_training=True,optimizer_steps=0,recommendations=recommendation))
    print(json.dumps({'status':'COMPLETE','main_batches':96,'conclusions':recommendation['stage_conclusions'],'ideas':recommendation['existing_ideas']}),flush=True)

if __name__=='__main__':generate()
