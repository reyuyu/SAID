import io
import json
import tarfile
from pathlib import Path

from PIL import Image

from tools.data.probe_hf_sa1b_shard import canonical_key, probe_tar, required_sam_ids


def test_canonical_key_variants():
    assert canonical_key('./sa_3631.jpg') == 'sa_3631'
    assert canonical_key('sa_3631') == 'sa_3631'


def test_probe_intersection_duplicate_and_decode(tmp_path):
    path = tmp_path / 'probe.tar'
    with tarfile.open(path, 'w') as tar:
        for name in ('./sa_1.jpg', 'sa_1.png', 'sa_2.jpg', 'caption.txt'):
            if name.endswith(('.jpg', '.png')):
                image = Image.new('RGB', (512, 512), 'red'); stream = io.BytesIO(); image.save(stream, format='PNG')
                data = stream.getvalue()
            else: data = b'caption'
            info = tarfile.TarInfo(name); info.size = len(data); tar.addfile(info, io.BytesIO(data))
    result = probe_tar(path, {'sa_1', 'sa_2', 'sa_3'})
    assert result['member_count'] == 4
    assert result['image_member_count'] == 3
    assert result['intersection_count'] == 2
    assert result['duplicate_key_count'] == 1
    assert result['decode_success'] == 2
    assert result['resolution_summary']['possible_resized_source'] is True


def test_required_ids(tmp_path):
    p = tmp_path / 'data.json'
    p.write_text(json.dumps([{'image': 'sam/images/sa_1.jpg'}, {'image': 'coco/x.jpg'}]))
    assert required_sam_ids(p) == {'sa_1'}
