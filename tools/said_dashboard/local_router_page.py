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
    manifest=json.loads(require(root/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='complete':raise ArtifactError('Local-Evidence Router evaluation is not complete')
    return manifest,json.loads(require(root/'summary.json').read_text(encoding='utf-8')),json.loads(require(root/'per_phrase.json').read_text(encoding='utf-8'))


@st.cache_data(show_spinner=False)
def ranking(root,key):
    with np.load(require(Path(root)/(key+'_geometry.npz'))) as data:
        return {k:data[k] for k in data.files}


def main():
    root=st.sidebar.text_input('历史局部 Router 目录',os.environ.get('LOCAL_ROUTER_ROOT','outputs/local_evidence_router/small'))
    try:render(root)
    except (ArtifactError,ValueError,KeyError) as exc:st.error('历史诊断文件不可用：' + str(exc))


def render(root):
    manifest,summary,phrases=bundle(root)
    st.title('历史：局部证据 Router')
    st.caption('历史消融：文本条件化证据图；主要结果使用原始方向；温度 0.07')
    tag=st.sidebar.selectbox('模型阶段',manifest['tags'],index=len(manifest['tags'])-1)
    image_id=st.sidebar.selectbox('图像',manifest['image_ids'])
    choices=[k for k in manifest['phrase_ids'] if phrases[k]['image_id']==image_id]
    ident=st.sidebar.selectbox('查询短语',choices,format_func=lambda k:phrases[k]['phrase']+' ['+k+']')
    index=manifest['phrase_ids'].index(ident);p=phrases[ident];image=load_crop(root,image_id)
    keys=[arm+'_'+tag+'_'+mode for arm in ['residual','attention_delta'] for mode in ['direct','router']]
    maps={k:attention(root,k,ident) for k in keys};scale=max(a.max() for a in maps.values())
    st.write('查询短语：',p['phrase']);st.caption(p['sentence'])
    columns=st.columns(5)
    columns[0].image(annotated_image(image,p['boxes']),caption='模型输入与标注框')
    labels=['残差直接匹配','残差 Router','注意力增量直接匹配','注意力增量 Router']
    for column,key,label in zip(columns[1:],keys,labels):
        column.image(picture(image,p['boxes'],maps[key],scale),caption=label)
    st.caption('四幅证据图共享颜色范围，显示 token 权重，不表示分割掩码。')
    rows=[]
    for key,label in zip(keys,labels):
        m=phrase_metrics(root,key)
        rows.append({'map':label,**{q:float(m[q][index]) if np.isfinite(m[q][index]) else None
                    for q in ['pointing','mass_gain','semantic_mass_excess','switch_margin']}})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    st.caption('固定切换子集外的配对指标显示不适用，逐短语切换指标取对称配对贡献均值。')
    st.subheader('Router 与直接匹配的 patch 排名对比')
    rows=[]
    for arm in ['residual','attention_delta']:
        r=ranking(root,arm+'_'+tag)
        rows.append({'arm':arm,**{k:float(r[k][index]) for k in ['spearman','top1','top5','top10','top20']}})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    st.subheader('历史小样本定位轨迹')
    rows=[]
    for key,m in summary.items():
        rows.append({'candidate':key,**{q:m[q]['mean'] for q in ['pointing','mass_gain','semantic_mass_excess','switch_margin','target_gt_distractor']},'orientation semantic gap':m['orientation_semantic_gap']})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    st.caption('旋转 180° 仅出现在汇总方向诊断，展示的证据图保持原始方向。')
