import sys
from pathlib import Path
import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/said_dashboard'))


def test_four_way_page_and_missing_image(tmp_path,monkeypatch):
    from eval.salu.local_router_eval import save_variant,patch_geometry
    from eval.salu.local_evidence_metrics import geometry
    from eval.salu.semantic_grounding_eval import write_json
    phrases=[{'id':str(i),'entity_id':str(i),'phrase':str(i),'sentence':'two entities','boxes':[box]}
             for i,box in enumerate([[0,0,16,16],[208,208,224,224]])]
    g=geometry([{'id':'image','entities':{p['entity_id']:p['boxes'] for p in phrases},'phrases':phrases}])
    summary={};logits=np.zeros((2,196));logits[0,0]=1;logits[1,-1]=1
    for arm in ['residual','attention_delta']:
        for tag in ['initial','final']:
            for mode in ['direct','router']:
                key=arm+'_'+tag+'_'+mode
                summary[key],_=save_variant(tmp_path,key,logits,g)
            np.savez_compressed(tmp_path/(arm+'_'+tag+'_geometry.npz'),**patch_geometry(logits,logits))
    write_json(tmp_path/'manifest.json',{'status':'complete','tags':['initial','final'],'phrase_ids':['0','1'],'image_ids':['image']})
    write_json(tmp_path/'summary.json',summary)
    write_json(tmp_path/'per_phrase.json',{p['id']:p for p in g['phrases']})
    (tmp_path/'images').mkdir();Image.new('RGB',(224,224),'white').save(tmp_path/'images/image.png')
    monkeypatch.setenv('LOCAL_ROUTER_ROOT',str(tmp_path))
    app=AppTest.from_file(str(ROOT/'tools/said_dashboard/app.py')).run()
    app.sidebar.radio[0].set_value('Local-Evidence Router').run()
    assert not app.error and not app.exception
    assert len(app.dataframe)==3
    next(s for s in app.selectbox if s.label=='Query phrase').set_value('1').run()
    next(s for s in app.selectbox if s.label=='Checkpoint').set_value('initial').run()
    assert not app.error and not app.exception
    (tmp_path/'images/image.png').unlink()
    # The crop loader is intentionally uncached, so disappearance is actionable.
    app.run()
    assert not app.exception and any('missing grounding artifact' in e.value for e in app.error)
