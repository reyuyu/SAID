"""Read-only local-evidence page; all inference and metrics are precomputed."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from data import ArtifactError
from grounding_data import annotated_image,load_crop,require

@st.cache_data(show_spinner=False)
def bundle(root):
 root=Path(root)
 values=[json.loads(require(root/name).read_text()) for name in ['manifest.json','per_candidate_metrics.json','per_phrase.json','temperature_summary.json']]
 if values[0]['status']!='complete':raise ArtifactError('Local evidence run is not complete')
 return values

@st.cache_data(show_spinner=False)
def attention(root,key,ident):
 a=np.load(require(Path(root)/'attention'/key/(ident+'.npy')))
 if a.shape!=(14,14) or not np.isfinite(a).all() or not np.isclose(a.sum(),1,atol=1e-6):raise ArtifactError('Invalid local evidence attention')
 return a

@st.cache_data(show_spinner=False)
def phrase_metrics(root,key):
 with np.load(require(Path(root)/'per_phrase_candidate'/(key+'.npz'))) as data:return {k:data[k] for k in data.files}

def key_of(b,l,s):return f'{b}_block{l:02d}_{s}'

def picture(image,boxes,a,scale):
 r,c=np.unravel_index(a.argmax(),a.shape)
 return annotated_image(image,boxes,a,scale,[(c+.5)*16,(r+.5)*16])

def main():
 root=st.sidebar.text_input('local evidence artifact root',os.environ.get('LOCAL_EVIDENCE_ROOT','outputs/local_semantic_evidence/small'))
 try:render(root)
 except (ArtifactError,ValueError,KeyError) as exc:st.error(str(exc))

def render(root):
 manifest,metrics,phrases,temperature=bundle(root)
 st.title('Local Semantic Evidence')
 st.caption(f"Frozen direct phrase attention | {manifest['num_images']} images / {manifest['num_phrases']} phrases | primary tau=0.07")
 st.caption('Intermediate and branch-only features reuse final ln_post + proj diagnostically. Text alignment is evaluated, not assumed.')
 backbone=st.sidebar.selectbox('Backbone',sorted({x['backbone'] for x in metrics.values()}))
 layer=st.sidebar.selectbox('Layer',sorted({x['layer'] for x in metrics.values() if x['backbone']==backbone}))
 stages=[s for s in ['residual','after_attention','attention_delta','mlp_delta'] if key_of(backbone,layer,s) in metrics]
 stage=st.sidebar.selectbox('Stage',stages,index=stages.index('attention_delta') if 'attention_delta' in stages else 0)
 key=key_of(backbone,layer,stage);baseline=key_of(backbone,11,'residual')
 image_id=st.sidebar.selectbox('Sample',manifest['image_ids'])
 choices=[k for k in manifest['phrase_ids'] if phrases[k]['image_id']==image_id]
 ident=st.sidebar.selectbox('Phrase',choices,format_func=lambda k:phrases[k]['phrase']+' ['+k+']')
 p=phrases[ident];idx=manifest['phrase_ids'].index(ident);image=load_crop(root,image_id)
 a=attention(root,key,ident);base=attention(root,baseline,ident);scale=max(a.max(),base.max())
 tabs=st.tabs(['Sample comparison','Layer × Stage','Metric evolution','Ranking and temperature','Spatial priors'])
 with tabs[0]:
  st.write('Query:',p['phrase']);st.caption('Sentence context: '+p['sentence'])
  cols=st.columns(4)
  cols[0].image(annotated_image(image,p['boxes']),caption='Model input + GT')
  cols[1].image(picture(image,p['boxes'],base,scale),caption='Current final residual')
  cols[2].image(picture(image,p['boxes'],a,scale),caption='Selected candidate')
  diff=np.abs(a-base);cols[3].image(annotated_image(image,p['boxes'],diff,float(diff.max())),caption='Absolute difference (own scale)')
  rows=[]
  for k in [baseline,key]:
   m=phrase_metrics(root,k)
   rows.append({'candidate':k,**{name:None if not np.isfinite(m[name][idx]) else float(m[name][idx]) for name in ['pointing','gt_mass','mass_gain','semantic_mass_excess','orientation_margin']}})
  st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
  st.caption('Per-phrase semantic excess averages available within-image pair controls; N/A means this phrase is outside the deterministic pair subset.')
  st.subheader('Orientation diagnostic')
  st.warning('rotate180: diagnostic only — NOT corrected attention')
  cols=st.columns(2);cols[0].image(picture(image,p['boxes'],a,float(a.max())),caption='Identity (primary)')
  cols[1].image(picture(image,p['boxes'],np.rot90(a,2),float(a.max())),caption='rotate180 diagnostic only')
 with tabs[1]:
  order=['after_attention','attention_delta','residual','mlp_delta']
  available=[key_of(backbone,l,s) for l in [3,6,9,11] for s in order if key_of(backbone,l,s) in metrics]
  matrix={k:attention(root,k,ident) for k in available};maximum=max(x.max() for x in matrix.values())
  for l in [3,6,9,11]:
   st.write('Block',l);cols=st.columns(4)
   for col,s in zip(cols,order):
    k=key_of(backbone,l,s)
    if k in matrix:col.image(picture(image,p['boxes'],matrix[k],maximum),caption=s)
    else:col.caption('Not selected for this run')
  st.caption('One shared color scale across all available cells. Block indices are zero-based.')
 with tabs[2]:
  for metric in ['mass_gain','semantic_mass_excess','switch_margin','orientation_semantic_gap']:
   rows=[]
   for k,m in metrics.items():
    if m['backbone']==backbone:rows.append({'layer':m['layer'],'stage':m['stage'],'value':m[metric] if metric=='orientation_semantic_gap' else m[metric]['mean']})
   st.write(metric);st.line_chart(pd.DataFrame(rows).pivot(index='layer',columns='stage',values='value'))
 with tabs[3]:
  rows=[]
  for k,m in metrics.items():
   rows.append({'candidate':k,'Pareto front':k in manifest['pareto_front'],**{q:m[q]['mean'] for q in ['pointing','mass_gain','semantic_mass_excess','target_gt_distractor','localization_margin','switch_margin']},'orientation gap':m['orientation_semantic_gap']})
  st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
  st.caption('No weighted total score. Semantic mass excess and mean symmetric switch margin are algebraically equal on the same pair subset.')
  st.subheader('Temperature robustness (primary stays 0.07)')
  st.dataframe(pd.DataFrame([{'tau':float(t),**{q:v['mean'] for q,v in vals.items()}} for t,vals in temperature[key].items()]),hide_index=True,width='stretch')
 with tabs[4]:
  st.write({'random patch-center pointing':manifest['random_patch_pointing'],'mean GT area fraction':manifest['mean_gt_area_fraction']})
  cols=st.columns(3)
  from data import heatmap_rgb
  paths=[Path(root)/'mean_gt_coverage.npy',Path(root)/'spatial_priors'/('mean_attention_'+key+'.npy'),Path(root)/'spatial_priors'/('peak_hist_'+key+'.npy')]
  for col,path,title in zip(cols,paths,['Mean GT coverage','Mean attention','Peak histogram']):col.image(heatmap_rgb(np.load(require(path))),caption=title+' (own scale)')
  st.write({'Pearson':metrics[key]['prior_pearson'],'Spearman':metrics[key]['prior_spearman']})
