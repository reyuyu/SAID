"""Build official layout from raw datasets; never use SAID manifests for Phase A."""
import ast
import json
from pathlib import Path
import shutil

from .common import ASSETS, BENCH, DATA, OFFICIAL, RUN, digest, dump, sha256


def link(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        assert target.resolve() == source.resolve()
    elif not target.exists():
        target.symlink_to(source, target_is_directory=source.is_dir())


def path_only_script(path, overrides):
    tree = ast.parse(path.read_text())
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in overrides:
                node.value = ast.Constant(str(overrides[name]))
                found.add(name)
    assert found == set(overrides)
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), {})


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    link(ASSETS/'evaluation/coco/annotations', DATA/'coco/annotations')
    link(ASSETS/'evaluation/coco/val2017', DATA/'coco/images/val2017')
    urban = ASSETS/'evaluation/Urban1k/Urban1k'
    link(urban/'image', DATA/'urban1k/image')
    link(urban/'caption', DATA/'urban1k/caption')
    path_only_script(OFFICIAL/'data_parsers/parse_urban1k_captions.py',
        {'caption_dir': DATA/'urban1k/caption', 'file_out': DATA/'urban1k/captions.json'})
    link(BENCH/'docci/images', DATA/'docci/images')
    link(BENCH/'downloads/docci_descriptions.jsonlines', DATA/'docci/docci_descriptions.jsonlines')
    dcibase = BENCH/'dci/annotations/densely_captioned_images'
    link(dcibase/'splits.json', DATA/'dci/densely_captioned_images/splits.json')
    link(dcibase/'annotations', DATA/'dci/densely_captioned_images/annotations')
    link(BENCH/'dci/images', DATA/'dci/densely_captioned_images/photos')
    # The documented upstream Flickr file is a list with official singular
    # 'caption' fields, unlike the repository's helper output 'captions'.
    payload = json.loads((RUN/'dataset_flickr32k.json').read_text())
    print(json.dumps({'flickr_annotation_root_type': type(payload).__name__,
                      'first_keys': list(payload[0]) if isinstance(payload, list) else list(payload)}), flush=True)
    rows = payload if isinstance(payload, list) else payload['images']
    by_split = {'train': [], 'val': [], 'test': []}
    for row in rows:
        split = row['split']
        assert split in by_split
        # Preserve every raw string and original order; adapt only field names
        # when the documented raw file uses the standard Karpathy schema.
        image = row.get('image', row.get('filename'))
        captions = row.get('caption', row.get('captions'))
        if captions is None:
            captions = [s['raw'] for s in row['sentences']]
        assert isinstance(captions, list) and all(isinstance(c, str) for c in captions)
        by_split[split].append({'image': image, 'caption': captions})
    assert {k: len(v) for k, v in by_split.items()} == {'train': 29000, 'val': 1014, 'test': 1000}
    destination = DATA/'flickr30k-images'
    destination.mkdir(exist_ok=True)
    for split, rows in by_split.items():
        dump(destination/f'flickr30k_{split}.json', rows)
    full = ASSETS/'external_baselines/debias_clip_v1/data/flickr30k-full'
    photos = [p for p in full.rglob('*.jpg')
              if '__MACOSX' not in p.parts and not p.name.startswith('._')]
    if photos:
        assert len(photos) == 31783 or len(photos) >= 31014
        for photo in photos:
            link(photo, destination/photo.name)
    record = {'phase_a_uses_said_manifest': False,
        'urban_helper_source_sha256': sha256(OFFICIAL/'data_parsers/parse_urban1k_captions.py'),
        'urban_helper_changes': 'In-memory replacements of the two hardcoded path assignments only; official file unchanged.',
        'flickr_raw_annotation_sha256': sha256(RUN/'dataset_flickr32k.json'),
        'flickr_raw_expected_LFS_sha256': '64c85af0e9d187091cb88890e56b3f640ac48386196a930d3ff8433d73165135',
        'flickr_annotation_schema': 'Singular caption list required by official read_flickr_pairs; raw text/order/splits preserved.',
        'flickr_split_counts': {k: len(v) for k, v in by_split.items()},
        'flickr_image_files_available': len(photos), 'data_root': str(DATA)}
    dump(RUN/'data-preparation.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
