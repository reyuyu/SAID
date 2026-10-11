import json
from pathlib import Path

ROOT=Path(__file__).parents[1]
IDENTITY=json.loads((ROOT/'docs/model_release/hns_v1_b16_epoch3/MODEL_IDENTITY.json').read_text())
def test_identity_is_epoch3_horizon():
    assert IDENTITY['selected_step']==3651
    assert IDENTITY['scheduler_horizon']==4868
    assert IDENTITY['text_context_length']==248
    assert IDENTITY['embedding_dimension']==512
def test_no_credentials_or_server_paths_in_release_docs():
    for p in (ROOT/'docs/model_release/hns_v1_b16_epoch3').rglob('*'):
        if p.is_file():
            text=p.read_text(errors='ignore')
            assert '/opt/data/private' not in text
            assert 'MODELSCOPE_API_TOKEN=' not in text
def test_scripts_are_real_entries():
    for name in ('infer_hns_v1_b16.py','download_hns_v1_b16.py','verify_hns_v1_b16.py','eval_hns_v1_b16.py'):
        p=ROOT/'scripts'/name; assert p.exists() and 'argparse' in p.read_text()
