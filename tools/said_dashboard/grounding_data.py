"""Read-only semantic grounding artifacts and CPU rendering; no model loading."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from eval.salu.grounding_metrics import checked_attention, patch_coverage, phrase_switch
from data import ArtifactError, overlay_rgb

LABELS = {'initial_direct': 'Initial OpenAI CLIP / direct',
          'phase21_router': 'Phase 2.1 / Said Router',
          'phase22_router': 'Phase 2.2 / Said Router',
          'phase22_direct': 'Phase 2.2 backbone / direct'}


def require(path):
    path = Path(path)
    if not path.is_file():
        raise ArtifactError('missing grounding artifact: %s' % path)
    return path


def load_bundle(root):
    root = Path(root)
    def read(name):
        return json.loads(require(root/name).read_text())
    manifest, summary, phrases = read('manifest.json'), read('summary.json'), read('per_phrase.json')
    if manifest.get('status') != 'complete':
        raise ArtifactError('Grounding audit is not complete; select a completed artifact root')
    if not phrases or set(manifest['variants']) != set(LABELS):
        raise ArtifactError('Grounding manifest is empty or lacks the four model variants')
    return manifest, summary, phrases


def load_map(root, variant, phrase_id):
    a = checked_attention(np.load(require(Path(root)/'attention'/variant/(phrase_id+'.npy'))))
    if a.shape != (14, 14):
        raise ValueError('grounding attention must be 14x14')
    return a


def load_crop(root, image_id):
    with Image.open(require(Path(root)/'images'/(image_id+'.png'))) as image:
        return image.convert('RGB')


def select_phrases(phrases, variant, view='All', sort_by='mass_gain', descending=False):
    rows = list(phrases.values())
    if view == 'Correct pointing':
        rows = [p for p in rows if p['metrics'][variant]['pointing_correct']]
    elif view == 'Wrong pointing':
        rows = [p for p in rows if not p['metrics'][variant]['pointing_correct']]
    elif view == 'Negative localization margin':
        rows = [p for p in rows if p['metrics'][variant]['localization_margin'] is not None
                and p['metrics'][variant]['localization_margin'] < 0]
    elif view == 'Highest failures':
        rows = [p for p in rows if not p['metrics'][variant]['pointing_correct']]
        descending = False
    elif view == 'Highest successes':
        rows = [p for p in rows if p['metrics'][variant]['pointing_correct']]
        descending = True
    elif view != 'All':
        raise ValueError('unknown failure-browser filter')
    available = [p for p in rows if p['metrics'][variant][sort_by] is not None]
    missing = [p for p in rows if p['metrics'][variant][sort_by] is None]
    available.sort(key=lambda p: (p['metrics'][variant][sort_by], p['id']), reverse=descending)
    rows = available+missing
    return rows[:50] if view in ['Highest failures', 'Highest successes'] else rows


def annotated_image(image, boxes, attention=None, scale_max=None, peak_xy=None, other_boxes=()):
    out = image.copy() if attention is None else Image.fromarray(overlay_rgb(image, attention, scale_max))
    draw = ImageDraw.Draw(out)
    for box in other_boxes:
        draw.rectangle(tuple(box), outline='cyan', width=2)
    for box in boxes:
        draw.rectangle(tuple(box), outline='lime', width=2)
    if peak_xy is not None:
        x, y = peak_xy
        draw.line((x-5, y, x+5, y), fill='red', width=2)
        draw.line((x, y-5, x, y+5), fill='red', width=2)
    return out


def switching_metrics(root, variant, phrase_a, phrase_b):
    return phrase_switch(load_map(root, variant, phrase_a['id']), load_map(root, variant, phrase_b['id']),
                         patch_coverage(phrase_a['boxes']), patch_coverage(phrase_b['boxes']))
