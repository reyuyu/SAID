"""Bounded per-parameter FP64 Gram accumulation; never flatten the whole model."""
import math
import torch

GROUPS=['G1_vision_backbone','G2_text_backbone','G3_text_mask_blocks','G4_shared_attention_pool',
        'G5_visual_mask_blocks','G6_visual_adapter','G7_balanced_gate','G8_total_trainable',
        'native_backbone_total','total_trainable_native_comparable']


def parameter_group(name):
    if name.startswith('clip.visual.'):return GROUPS[0]
    if name.startswith('clip.mask_net.resblocks.'):return GROUPS[2]
    if name.startswith('clip.mask_net.attn_pool.'):return GROUPS[3]
    if name.startswith('fusion_branch.visual_blocks.'):return GROUPS[4]
    if name.startswith('fusion_branch.visual_adapter.'):return GROUPS[5]
    if name.startswith('fusion_branch.gate.'):return GROUPS[6]
    if name.startswith('clip.'):return GROUPS[1]
    raise ValueError(name)


def gram_matrices(named,gradients,objectives,chunk=131072):
    size=len(objectives);device=next(iter(named.values())).device
    grams={g:torch.zeros((size,size),device=device,dtype=torch.float64) for g in GROUPS[:7]}
    support={g:[False]*size for g in GROUPS[:7]}
    for name,p in named.items():
        group=parameter_group(name)
        # Flatten only one parameter at a time; use small bounded tiles.
        active=[(i,gradients[key].get(name)) for i,key in enumerate(objectives) if gradients[key].get(name) is not None]
        if not active:continue
        for i,_ in active:support[group][i]=True
        indices=torch.tensor([i for i,_ in active],device=device)
        for start in range(0,p.numel(),chunk):
            tile=torch.stack([v.reshape(-1)[start:start+chunk] for _,v in active]).double()
            block=tile@tile.T
            grams[group][indices[:,None],indices[None,:]]+=block
    grams[GROUPS[7]]=sum(grams[g] for g in GROUPS[:7])
    grams[GROUPS[8]]=grams[GROUPS[0]]+grams[GROUPS[1]]
    grams[GROUPS[9]]=grams[GROUPS[8]]
    support[GROUPS[7]]=[any(support[g][i] for g in GROUPS[:7]) for i in range(size)]
    support[GROUPS[8]]=[support[GROUPS[0]][i] or support[GROUPS[1]][i] for i in range(size)]
    support[GROUPS[9]]=support[GROUPS[8]]
    return {g:matrix.cpu().tolist() for g,matrix in grams.items()},support


def derive(grams,support,objectives,weights):
    indexes={name:i for i,name in enumerate(objectives)}
    result={}
    for group,G in grams.items():
        def norm(name):return math.sqrt(max(0,G[indexes[name]][indexes[name]]))
        def pair(a,b):
            ia,ib=indexes[a],indexes[b];dot=G[ia][ib];na,nb=norm(a),norm(b)
            return {'dot':dot,'cosine':dot/(na*nb) if na>0 and nb>0 else None,
              'projection_onto_second':dot/(nb*nb) if nb>0 else None,
              'norm_a':na,'norm_b':nb,'reason_NA':None if na>0 and nb>0 else 'zero/absent gradient; cosine undefined'}
        view=['F_combined','O_combined','E_combined']
        pairs={}
        for a,b in [('F_combined','O_combined'),('F_combined','E_combined'),('O_combined','E_combined')]:pairs[a+'__'+b]=pair(a,b)
        for label in ['F','O','E','native']:
            pairs[label+'_i2t__'+label+'_t2i']=pair(label+'_i2t',label+'_t2i')
        for label in ['F','O','E']:
            pairs[label+'_combined__native_combined']=pair(label+'_combined','native_combined')
            for direction in ['i2t','t2i']:pairs[label+'_'+direction+'__native_'+direction]=pair(label+'_'+direction,'native_'+direction)
        for name in ['align_actual','total','sparse','inc']:
            pairs[name+'__native_combined']=pair(name,'native_combined')
        for name in ['sparse','inc']:pairs[name+'__align_actual']=pair(name,'align_actual')
        pairs['sparse__inc']=pair('sparse','inc')
        norms={name:norm(name) for name in objectives}
        n=[norm(name) for name in view];eff=[w*v for w,v in zip(weights,n)];denom=sum(eff)
        ratio=lambda a,b:a/b if b>0 else None
        # Half-inclusion appendix and counterfactual weighting use measured component vectors.
        half_i=indexes['inc'];total_i=indexes['total'];native_i=indexes['native_combined']
        half_norm2=G[total_i][total_i]-.5*G[total_i][half_i]-.5*G[half_i][total_i]+.25*G[half_i][half_i]
        half_dot=G[total_i][native_i]-.5*G[half_i][native_i]
        nn=norm('native_combined');hn=math.sqrt(max(0,half_norm2))
        candidates={}
        for label,values in [('equal',[1.,1.,1.]),('registered',weights),('Full_anchor',[1.5,.75,.75]),('B_Full_anchor',[1.5,.4,1.1]),('B_Summary_lower',[1.3,.4,1.3]),('Prefix_lower',[1.2,.6,1.2]),('Remainder_lower',[1.2,1.2,.6])]:
            coeff=[(10/3)*x for x in values];ids=[indexes[x] for x in view]
            dot=sum(c*G[i][native_i] for c,i in zip(coeff,ids));norm2=sum(a*b*G[i][j] for a,i in zip(coeff,ids) for b,j in zip(coeff,ids));cn=math.sqrt(max(0,norm2))
            candidates[label]={'weights':values,'alignment_norm':cn,'cosine_native':dot/(cn*nn) if cn>0 and nn>0 else None,'projection_native':dot/(nn*nn) if nn>0 else None}
        result[group]={'norms':norms,'gradient_present':{name:support[group][i] for i,name in enumerate(objectives)},'pairs':pairs,
          'raw_norm_ratios':{'O_over_F':ratio(n[1],n[0]),'E_over_F':ratio(n[2],n[0]),'E_over_O':ratio(n[2],n[1]),'partials_over_F':ratio(n[1]+n[2],n[0])},
          'effective_norms':dict(zip(view,eff)),'effective_norm_share':dict(zip(view,[v/denom if denom>0 else None for v in eff])),
          'half_inclusion_appendix':{'inc_norm':.5*norm('inc'),'total_norm':hn,'total_cosine_native':half_dot/(hn*nn) if hn>0 and nn>0 else None},
          'counterfactual_alignment_only':candidates}
    return result


def linearity_errors(named,gradients,weights):
    max_abs=0.;diff2=torch.zeros((),device=next(iter(named.values())).device,dtype=torch.float64);actual2=diff2.clone()
    group={g:{'diff2':0.,'actual2':0.,'max_abs':0.} for g in GROUPS[:7]}
    with torch.no_grad():
        for name,p in named.items():
            actual=gradients['align_actual'].get(name)
            pieces=[gradients[key].get(name) for key in ['F_combined','O_combined','E_combined']]
            if actual is None and all(v is None for v in pieces):continue
            expected=torch.zeros_like(p)
            for w,v in zip(weights,pieces):
                if v is not None:expected.add_(v,alpha=w)
            expected.mul_(10/3)
            a=torch.zeros_like(p) if actual is None else actual;difference=a-expected
            mx=float(difference.abs().max());d=float(difference.double().square().sum());n=float(a.double().square().sum())
            max_abs=max(max_abs,mx);diff2+=d;actual2+=n
            g=group[parameter_group(name)];g['diff2']+=d;g['actual2']+=n;g['max_abs']=max(g['max_abs'],mx)
    return {'max_abs':max_abs,'relative_L2':math.sqrt(float(diff2)/max(float(actual2),1e-300)),
      'groups':{g:{'max_abs':v['max_abs'],'relative_L2':math.sqrt(v['diff2']/max(v['actual2'],1e-300))} for g,v in group.items()},
      'precision_note':'BF16 encoder backward is quantized; independent view backward and combined backward can differ by BF16 rounding. FP32 mask groups are expected within FP32 tolerance. Full FP32 control is checked separately.'}
