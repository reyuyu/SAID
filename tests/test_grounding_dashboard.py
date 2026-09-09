import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/said_dashboard'))
import grounding_data as gd
from data import ArtifactError
from eval.salu.grounding_metrics import phrase_metrics, summarize


@pytest.fixture()
def grounding_root(tmp_path):
    root = tmp_path/'grounding'
    (root/'images').mkdir(parents=True)
    Image.new('RGB', (224, 224), 'white').save(root/'images/i.png')
    entities = {'1': [[0, 0, 32, 32]], '2': [[192, 192, 224, 224]]}
    phrases = {}
    for i, text in enumerate(['a dog', 'a person']):
        key = 'p%d' % i
        phrases[key] = {'id': key, 'entity_id': str(i+1), 'image_id': 'i', 'phrase': text,
                        'sentence': 'A person and a dog.', 'categories': ['animals' if i == 0 else 'people'],
                        'boxes': entities[str(i+1)], 'metrics': {}, 'visible_area_retention': 1}
        for variant in gd.LABELS:
            a = np.full((14, 14), .1/196)
            a[0, 0] += .9
            folder = root/'attention'/variant
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder/(key+'.npy'), a)
            phrases[key]['metrics'][variant] = phrase_metrics(a, phrases[key]['boxes'],
                                                             {k: v for k, v in entities.items() if k != str(i+1)})
    manifest = {'status': 'complete', 'dataset': 'fixture', 'split': 'val', 'num_images': 1,
                'num_phrases': 2, 'precision': 'fp32', 'variants': list(gd.LABELS),
                'switching_groups': [{'image_id': 'i', 'phrase_ids': list(phrases)}]}
    summary = {v: summarize([p['metrics'][v] for p in phrases.values()], []) for v in gd.LABELS}
    for name, value in [('manifest', manifest), ('summary', summary), ('per_phrase', phrases)]:
        (root/(name+'.json')).write_text(json.dumps(value))
    return root


def test_grounding_bundle_selector_maps_and_metrics(grounding_root):
    m, summary, phrases = gd.load_bundle(grounding_root)
    assert len(m['variants']) == 4
    assert [p['id'] for p in gd.select_phrases(phrases, 'phase22_router', 'Wrong pointing')] == ['p1']
    assert [p['id'] for p in gd.select_phrases(phrases, 'phase22_router', 'Highest successes')] == ['p0']
    assert gd.load_map(grounding_root, 'initial_direct', 'p0').shape == (14, 14)
    assert gd.switching_metrics(grounding_root, 'phase22_router', phrases['p0'], phrases['p1'])['switch_margin'] == 0
    image = gd.annotated_image(gd.load_crop(grounding_root, 'i'), phrases['p0']['boxes'], peak_xy=[8, 8])
    assert image.getpixel((0, 0)) == (0, 255, 0)
    assert image.getpixel((8, 8)) == (255, 0, 0)


def test_missing_grounding_artifact(grounding_root):
    with pytest.raises(ArtifactError, match='missing grounding artifact'):
        gd.load_map(grounding_root, 'phase22_router', 'missing')


def test_grounding_page_phrase_switch_and_filter(grounding_root, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('SEMANTIC_GROUNDING_ROOT', str(grounding_root))
    app = AppTest.from_file(str(ROOT/'tools/said_dashboard/app.py')).run()
    app.sidebar.radio[0].set_value('Semantic Grounding Audit').run()
    assert not app.exception and not app.error
    assert any(s.value == 'Same-image phrase switching' for s in app.subheader)
    def select(label, value):
        next(s for s in app.selectbox if s.label == label).set_value(value).run()
        assert not app.exception and not app.error
    select('target phrase', 'p1')
    select('Phrase A', 'p1')
    select('switching model', 'phase22_direct')
    select('case filter', 'Wrong pointing')
    assert next(s for s in app.selectbox if s.label == 'target phrase').value == 'p1'


def test_grounding_page_missing_image_is_actionable(grounding_root, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('SEMANTIC_GROUNDING_ROOT', str(grounding_root))
    (grounding_root/'images/i.png').unlink()
    app = AppTest.from_file(str(ROOT/'tools/said_dashboard/app.py')).run()
    app.sidebar.radio[0].set_value('Semantic Grounding Audit').run()
    assert not app.exception
    assert any('missing grounding artifact' in e.value for e in app.error)
