import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/said_dashboard'))

@pytest.fixture
def local_fixture(tmp_path):
 from eval.salu.local_evidence_metrics import geometry,evaluate,pareto_front
 from eval.salu.semantic_grounding_eval import write_json
 phrases=[{'id':str(i),'entity_id':str(i),'phrase':phrase,'sentence':'A dog and a person','categories':['animals' if i==0 else 'people'],'boxes':[box]} for i,(phrase,box) in enumerate([('a dog',[0,0,32,32]),('a person',[192,192,224,224])])]
 g=geometry([{'id':'i','entities':{p['entity_id']:p['boxes'] for p in phrases},'phrases':phrases}])
 summary={};temp={}
 for layer in [3,6,9,11]:
  for stage in ['after_attention','attention_delta','residual','mlp_delta']:
   key=f'initial_block{layer:02d}_{stage}';logits=np.zeros((2,196));logits[0,0]=1;logits[1,-1]=1
   s,rows,a=evaluate(logits,g);s.update(backbone='initial',layer=layer,stage=stage);summary[key]=s
   folder=tmp_path/'attention'/key;folder.mkdir(parents=True)
   for ident,x in zip(['0','1'],a):np.save(folder/(ident+'.npy'),x.reshape(14,14))
   (tmp_path/'per_phrase_candidate').mkdir(exist_ok=True);np.savez_compressed(tmp_path/'per_phrase_candidate'/(key+'.npz'),**rows)
   (tmp_path/'spatial_priors').mkdir(exist_ok=True)
   np.save(tmp_path/'spatial_priors'/('mean_attention_'+key+'.npy'),a.mean(0).reshape(14,14))
   np.save(tmp_path/'spatial_priors'/('peak_hist_'+key+'.npy'),a.mean(0).reshape(14,14))
   temp[key]={'0.07':{k:s[k] for k in ['mass_gain','semantic_mass_excess','switch_margin','target_gt_distractor']}}
 (tmp_path/'images').mkdir();Image.new('RGB',(224,224),'white').save(tmp_path/'images/i.png')
 np.save(tmp_path/'mean_gt_coverage.npy',g['coverage'].mean(0).reshape(14,14))
 write_json(tmp_path/'manifest.json',{'status':'complete','num_images':1,'num_phrases':2,'image_ids':['i'],'phrase_ids':['0','1'],'pareto_front':pareto_front(summary),'random_patch_pointing':.02,'mean_gt_area_fraction':.02})
 write_json(tmp_path/'per_candidate_metrics.json',summary);write_json(tmp_path/'temperature_summary.json',temp);write_json(tmp_path/'per_phrase.json',{p['id']:p for p in g['phrases']})
 return tmp_path

def test_local_page_selectors_and_all_views(local_fixture,monkeypatch):
 from streamlit.testing.v1 import AppTest
 monkeypatch.setenv('LOCAL_EVIDENCE_ROOT',str(local_fixture))
 app=AppTest.from_file(str(ROOT/'tools/said_dashboard/app.py')).run()
 app.sidebar.radio[0].set_value('Local Semantic Evidence').run()
 assert not app.exception and not app.error
 assert len(app.tabs)==5
 for label,value in [('层',11),('特征阶段','attention_delta'),('短语','1')]:
  next(s for s in app.selectbox if s.label==label).set_value(value).run()
  assert not app.exception and not app.error
 assert any('不代表修正后的注意力' in w.value for w in app.warning)

def test_local_page_missing_artifact_is_controlled(local_fixture,monkeypatch):
 from streamlit.testing.v1 import AppTest
 monkeypatch.setenv('LOCAL_EVIDENCE_ROOT',str(local_fixture))
 (local_fixture/'images/i.png').unlink()
 app=AppTest.from_file(str(ROOT/'tools/said_dashboard/app.py')).run()
 app.sidebar.radio[0].set_value('Local Semantic Evidence').run()
 assert not app.exception and any('missing grounding artifact' in x.value for x in app.error)
