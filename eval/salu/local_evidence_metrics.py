"""Numpy-only locality metrics with geometry shared across candidates."""
import itertools
import numpy as np
from .grounding_metrics import patch_coverage, contains, union_iou, distribution
from .semantic_grounding_eval import switching_groups


def rank_values(x):
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = counts.cumsum()
    return (ends - (counts + 1) / 2)[inverse]


def correlation(a, b, ranks=False):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if ranks: a, b = rank_values(a), rank_values(b)
    if a.std() == 0 or b.std() == 0: return None
    return float(np.corrcoef(a, b)[0, 1])


def orientation_margin(logits, coverage):
    """Coverage-weighted mean true logit minus 180-degree opposite GT logit."""
    l, w = np.asarray(logits), np.asarray(coverage)
    flipped = w.reshape(-1,14,14)[:,::-1,::-1].reshape(w.shape)
    return np.sum(l*(w-flipped),axis=-1)/np.maximum(w.sum(-1),1e-12)


def geometry(records):
    phrases=[{**p,'image_id':im['id']} for im in records for p in im['phrases']]
    ids=[p['id'] for p in phrases]; lookup={k:i for i,k in enumerate(ids)}
    coverage=[]; centres=[]; image_groups=[]; offset=0
    for im in records:
        entities={k:b for k,b in im['entities'].items() if b}
        names=list(entities)
        cov=np.stack([patch_coverage(entities[k]).ravel() for k in names])
        cent=np.array([[contains(entities[k],(c+.5)*16,(r+.5)*16)
                        for r in range(14) for c in range(14)] for k in names])
        distractor=[]
        for p in im['phrases']:
            coverage.append(cov[names.index(p['entity_id'])]);centres.append(cent[names.index(p['entity_id'])])
            distractor.append([k!=p['entity_id'] and union_iou(p['boxes'],entities[k])<.1 for k in names])
        end=offset+len(im['phrases'])
        image_groups.append((offset,end,cov,np.array(distractor,dtype=bool)))
        offset=end
    groups=switching_groups(records)
    pairs=[(lookup[a],lookup[b]) for g in groups for a,b in itertools.combinations(g['phrase_ids'],2)]
    directed=np.asarray([(a,b) for a,b in pairs for a,b in [(a,b),(b,a)]],dtype=int).reshape(-1,2)
    return {'phrases':phrases,'ids':ids,'coverage':np.stack(coverage),'centres':np.stack(centres),
            'images':image_groups,'pairs':directed,'groups':groups}


def softmax(logits,tau):
    z=np.asarray(logits,dtype=np.float64)/tau
    z-=z.max(-1,keepdims=True);a=np.exp(z)
    return a/a.sum(-1,keepdims=True)


def core(att,g):
    a=np.asarray(att,dtype=np.float64).reshape(-1,196)
    w=g['coverage']; mass=np.sum(a*w,1);area=w.mean(1)
    peak=a.argmax(1);point=g['centres'][np.arange(len(a)),peak]
    margin=np.full(len(a),np.nan);target_rate=np.full(len(a),np.nan)
    for start,end,cov,mask in g['images']:
        if end==start:continue
        other=a[start:end]@cov.T
        other=np.where(mask,other,-np.inf).max(1)
        valid=np.isfinite(other)
        margin[start:end][valid]=mass[start:end][valid]-other[valid]
        target_rate[start:end][valid]=(margin[start:end][valid]>0).astype(float)
    q,d=g['pairs'].T
    shuffled_mass=np.sum(a[q]*w[d],1)
    excess=mass[q]-shuffled_mass
    point_excess=point[q].astype(float)-g['centres'][d,peak[q]].astype(float)
    sw=excess.reshape(-1,2).mean(1)
    summary={k:distribution(v.tolist()) for k,v in {
        'pointing':point.astype(float),'gt_mass':mass,'gt_area_fraction':area,'mass_gain':mass-area,
        'target_gt_distractor':target_rate[np.isfinite(target_rate)],'localization_margin':margin[np.isfinite(margin)],
        'switch_margin':sw,'switch_positive_fraction':(sw>0).astype(float),'semantic_mass_excess':excess,
        'semantic_pointing_excess':point_excess,'pair_true_mass':mass[q],'pair_shuffled_mass':shuffled_mass,
        'pair_true_pointing':point[q].astype(float),'pair_shuffled_pointing':g['centres'][d,peak[q]].astype(float)}.items()}
    summary.update({'phrases':len(a),'pair_query_count':len(q),'pairs':len(q)//2})
    per_query=np.full(len(a),np.nan)
    sums=np.bincount(q,weights=excess,minlength=len(a));counts=np.bincount(q,minlength=len(a));valid=counts>0
    per_query[valid]=sums[valid]/counts[valid]
    return summary,{'pointing':point,'gt_mass':mass,'gt_area_fraction':area,'mass_gain':mass-area,
                     'localization_margin':margin,'semantic_mass_excess':per_query,'peak_index':peak}


def evaluate(logits,g,tau=.07):
    att=softmax(logits,tau)
    summary,rows=core(att,g)
    rotated=att.reshape(-1,14,14)[:,::-1,::-1].reshape(-1,196)
    rot,_=core(rotated,g)
    summary['rotate180']={k:rot[k] for k in ['mass_gain','semantic_mass_excess','semantic_pointing_excess','pointing']}
    for out,key in [('orientation_semantic_gap','semantic_mass_excess'),('orientation_mass_gain_gap','mass_gain')]:
        summary[out]=summary[key]['mean']-rot[key]['mean']
    om=orientation_margin(logits,g['coverage'])
    summary['local_semantic_orientation_margin']=distribution(om.tolist())
    summary['orientation_positive_fraction']=float(np.mean(om>0));rows['orientation_margin']=om
    mean=att.mean(0);gt=g['coverage'].mean(0)
    summary['prior_pearson']=correlation(mean,gt);summary['prior_spearman']=correlation(mean,gt,True)
    return summary,rows,att


PARETO_KEYS=('pointing','mass_gain','semantic_mass_excess','localization_margin','switch_margin','target_gt_distractor')


def pareto_front(summary):
    keys=list(summary)
    values=np.asarray([[summary[k][metric]['mean'] for metric in PARETO_KEYS]+
                       [summary[k]['orientation_semantic_gap']] for k in keys])
    return [key for i,key in enumerate(keys) if not any(np.all(values[j]>=values[i]) and np.any(values[j]>values[i])
                                                        for j in range(len(keys)) if j!=i)]
