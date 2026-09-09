"""Artifact-only comparison of caption-conditioned evidence maps."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from data import ArtifactError
from grounding_data import require, load_crop, annotated_image
from local_evidence_page import attention, phrase_metrics, picture


@st.cache_data(show_spinner=False)
def bundle(root):
    root=Path(root)
    manifest=json.loads(require(root/'manifest.json').read_text())
    if manifest['status']!='complete':raise ArtifactError('Local-Evidence Router evaluation is not complete')
    return manifest,json.loads(require(root/'summary.json').read_text()),json.loads(require(root/'per_phrase.json').read_text())


@st.cache_data(show_spinner=False)
def ranking(root,key):
    with np.load(require(Path(root)/(key+'_geometry.npz'))) as data:
        return {k:data[k] for k in data.files}


def main():
    root=st.sidebar.text_input('local router artifact root',os.environ.get('LOCAL_ROUTER_ROOT','outputs/local_evidence_router/small'))
    try:render(root)
    except (ArtifactError,ValueError,KeyError) as exc:st.error(str(exc))


def render(root):
    manifest,summary,phrases=bundle(root)
    st.title('Local-Evidence Router')
    st.caption('Caption-conditioned evidence maps | primary identity coordinates | tau=0.07')
    tag=st.sidebar.selectbox('Checkpoint',manifest['tags'],index=len(manifest['tags'])-1)
    image_id=st.sidebar.selectbox('Image',manifest['image_ids'])
    choices=[k for k in manifest['phrase_ids'] if phrases[k]['image_id']==image_id]
    ident=st.sidebar.selectbox('Query phrase',choices,format_func=lambda k:phrases[k]['phrase']+' ['+k+']')
    index=manifest['phrase_ids'].index(ident);p=phrases[ident];image=load_crop(root,image_id)
    keys=[arm+'_'+tag+'_'+mode for arm in ['residual','attention_delta'] for mode in ['direct','router']]
    maps={k:attention(root,k,ident) for k in keys};scale=max(a.max() for a in maps.values())
    st.write('Query:',p['phrase']);st.caption(p['sentence'])
    columns=st.columns(5)
    columns[0].image(annotated_image(image,p['boxes']),caption='Model input + GT boxes')
    labels=['Residual direct','Residual Router','Attention-delta direct','Attention-delta Router']
    for column,key,label in zip(columns[1:],keys,labels):
        column.image(picture(image,p['boxes'],maps[key],scale),caption=label)
    st.caption('All four evidence maps use one shared color scale. These maps are token weights, not segmentation masks.')
    rows=[]
    for key,label in zip(keys,labels):
        m=phrase_metrics(root,key)
        rows.append({'map':label,**{q:float(m[q][index]) if np.isfinite(m[q][index]) else None
                    for q in ['pointing','mass_gain','semantic_mass_excess','switch_margin']}})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    st.caption('Pair metrics are N/A outside the fixed switching subset. Per-phrase switch averages symmetric pair contributions.')
    st.subheader('Router versus direct patch ranking')
    rows=[]
    for arm in ['residual','attention_delta']:
        r=ranking(root,arm+'_'+tag)
        rows.append({'arm':arm,**{k:float(r[k][index]) for k in ['spearman','top1','top5','top10','top20']}})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    st.subheader('Small grounding trajectory')
    rows=[]
    for key,m in summary.items():
        rows.append({'candidate':key,**{q:m[q]['mean'] for q in ['pointing','mass_gain','semantic_mass_excess','switch_margin','target_gt_distractor']},'orientation semantic gap':m['orientation_semantic_gap']})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    st.caption('rotate180 appears only in aggregate orientation diagnostics; displayed maps retain identity orientation.')
