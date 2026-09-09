"""Read official Flickr30K Entities phrase mentions and XML boxes for evaluation.

Source format: https://github.com/BryanPlummer/flickr30k_entities
No data is downloaded at import or loaded into training.
"""
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image

from .grounding_metrics import boxes_array, clip_geometry, transform_boxes, union_area

PHRASE = re.compile(r'\[/EN#([^/\s]+)/([^\s]+)\s+([^\]]+)\]')


def read_sentences(path):
    result = []
    for sentence_index, line in enumerate(Path(path).read_text().splitlines()):
        matches = list(PHRASE.finditer(line))
        sentence = PHRASE.sub(lambda m: m.group(3), line)
        if '[' in sentence or ']' in sentence:
            raise ValueError('malformed phrase markup: %s line %d' % (path, sentence_index+1))
        for phrase_index, match in enumerate(matches):
            result.append({'sentence_index': sentence_index, 'phrase_index': phrase_index,
                           'sentence': sentence, 'phrase': match.group(3),
                           'entity_id': match.group(1), 'categories': match.group(2).split('/')})
    return result


def read_annotation(path):
    xml = ET.parse(path).getroot()
    size = xml.find('size')
    if size is None:
        raise ValueError('annotation has no image size: %s' % path)
    width, height = int(size.findtext('width')), int(size.findtext('height'))
    entities, flags = {}, {}
    for obj in xml.findall('object'):
        ids = [n.text for n in obj.findall('name')]
        box = obj.find('bndbox')
        if box is None:
            for entity in ids:
                flags[entity] = 'scene' if obj.findtext('scene') == '1' else 'no_box'
            continue
        # PASCAL VOC: 1-based inclusive -> continuous zero-based half-open.
        b = [float(box.findtext(k)) for k in ['xmin', 'ymin', 'xmax', 'ymax']]
        b[0] -= 1
        b[1] -= 1
        boxes_array([b])
        for entity in ids:
            entities.setdefault(entity, []).append(b)
    return width, height, entities, flags


def load_split(image_root=None, entities_root=None, split='val', max_images=None):
    image_root = image_root or os.environ.get('FLICKR30K_ROOT')
    entities_root = entities_root or os.environ.get('FLICKR30K_ENTITIES_ROOT')
    if not image_root or not entities_root:
        raise ValueError('Set FLICKR30K_ROOT (directory of JPGs) and FLICKR30K_ENTITIES_ROOT '
                         '(official split files, Annotations/ and Sentences/); '
                         'see docs/phase23_grounding_audit.md')
    images, ann = Path(image_root), Path(entities_root)
    unrelated = set()
    if (ann/'UNRELATED_CAPTIONS').exists():
        for line in (ann/'UNRELATED_CAPTIONS').read_text().splitlines():
            if line.strip() and not line.startswith('#'):
                image_id, number = line.split()
                unrelated.add((image_id, int(number)-1))
    ids = (ann/('%s.txt' % split)).read_text().split()
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate image ID in official split')
    if max_images is not None:
        if max_images <= 0:
            raise ValueError('max_images must be positive')
        ids = ids[:max_images]
    exclusions, records = Counter(), []
    for image_id in ids:
        image_path = images/(image_id+'.jpg')
        if not image_path.is_file():
            raise FileNotFoundError('Flickr30K image missing: %s (no silent image skipping)' % image_path)
        width, height, original, flags = read_annotation(ann/'Annotations'/(image_id+'.xml'))
        with Image.open(image_path) as image:
            if image.size != (width, height):
                raise ValueError('image/annotation size mismatch: %s' % image_id)
        geometry = clip_geometry(width, height)
        visible = {key: transform_boxes(b, geometry).tolist() for key, b in original.items()}
        mentions = []
        for phrase in read_sentences(ann/'Sentences'/(image_id+'.txt')):
            if (image_id, phrase['sentence_index']) in unrelated:
                exclusions['official_unrelated_caption_mentions'] += 1
                continue
            entity = phrase['entity_id']
            if entity not in original:
                exclusions[flags.get(entity, 'missing_box_or_nonvisual')] += 1
                continue
            if not visible[entity]:
                exclusions['fully_outside_center_crop'] += 1
                continue
            phrase['id'] = '%s_s%d_p%d' % (image_id, phrase['sentence_index'], phrase['phrase_index'])
            phrase['boxes'] = visible[entity]
            phrase['original_boxes'] = original[entity]
            rw, rh = geometry['resized_size']
            original_area_scaled = union_area(original[entity])*(rw/width)*(rh/height)
            phrase['visible_area_retention'] = union_area(visible[entity])/original_area_scaled
            mentions.append(phrase)
        records.append({'id': image_id, 'path': str(image_path), 'geometry': geometry,
                        'entities': visible, 'phrases': mentions})
    return records, dict(exclusions)
