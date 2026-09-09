"""Build checkpoint-comparison figures for the Said attention (PIL only, no training).

Reads the diagnostics artifacts produced by ``said_grounding_diagnostics.py`` and
writes, for a few fixed samples:

  * ``comparison_sampleXX.png``: rows = checkpoints, columns = original image,
    own-caption Said map, shuffled-caption Said map. All maps of the same column
    share one colour scale, so intensity changes across checkpoints are visible.
  * ``metrics_evolution.png``: entropy / effective patch count / caption-shuffle
    JS divergence / positive-alignment margin across checkpoints.

Figures go to ``outputs/`` and are never committed.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval.salu.visualize_said import colorize, upsample  # noqa: E402

FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
]


def get_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def load_map(path, grid):
    return np.load(path).reshape(grid, grid)


def overlay(original, attn2d, scale_max, tile=224):
    norm = np.clip(attn2d / max(scale_max, 1e-8), 0.0, 1.0)
    heat = colorize(upsample(norm, tile))
    return (0.5 * original.astype(np.float32) + 0.5 * heat.astype(np.float32)).astype(np.uint8)


def build_sample_figure(args, sample, tags):
    grid = args.grid
    tile = args.tile
    header_h = 34
    label_w = 150
    img_dir = os.path.join(args.diagnostics_dir, 'heatmaps')
    image_path = os.path.join(img_dir, tags[0], 'sample%02d_image.png' % sample)
    original = np.asarray(Image.open(image_path).convert('RGB'))

    own_maps = {t: load_map(os.path.join(img_dir, t, 'sample%02d_own_attn14.npy' % sample), grid) for t in tags}
    shuf_maps = {t: load_map(os.path.join(img_dir, t, 'sample%02d_shuffled_attn14.npy' % sample), grid) for t in tags}
    own_scale = max(m.max() for m in own_maps.values())
    shuf_scale = max(m.max() for m in shuf_maps.values())

    width = label_w + 3 * tile
    height = header_h + len(tags) * tile
    canvas = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font_h = get_font(18)
    font_l = get_font(16)

    draw.text((label_w + tile // 3, 8), 'original', fill=(0, 0, 0), font=font_h)
    draw.text((label_w + tile + 12, 8), 'Said map (own caption)', fill=(0, 0, 0), font=font_h)
    draw.text((label_w + 2 * tile + 12, 8), 'Said map (shuffled caption)', fill=(0, 0, 0), font=font_h)

    for row, tag in enumerate(tags):
        y = header_h + row * tile
        draw.text((8, y + tile // 2 - 10), tag, fill=(0, 0, 0), font=font_l)
        canvas.paste(Image.fromarray(original), (label_w, y))
        canvas.paste(Image.fromarray(overlay(original, own_maps[tag], own_scale, tile)), (label_w + tile, y))
        canvas.paste(Image.fromarray(overlay(original, shuf_maps[tag], shuf_scale, tile)), (label_w + 2 * tile, y))
        draw.rectangle([label_w, y, width - 1, y + tile - 1], outline=(200, 200, 200))

    out_path = os.path.join(args.output_dir, 'comparison_sample%02d.png' % sample)
    canvas.save(out_path)
    return out_path, own_scale, shuf_scale


def line_panel(draw, box, title, series, tags, colour=(30, 90, 200)):
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], outline=(180, 180, 180))
    font = get_font(15)
    draw.text((x0 + 8, y0 + 6), title, fill=(0, 0, 0), font=font)
    if not series:
        return
    lo, hi = min(series), max(series)
    if hi - lo < 1e-12:
        hi = lo + 1.0
    pad_l, pad_r, pad_t, pad_b = 52, 14, 34, 26
    px0, py0, px1, py1 = x0 + pad_l, y0 + pad_t, x1 - pad_r, y1 - pad_b
    draw.line([px0, py1, px1, py1], fill=(120, 120, 120))
    draw.line([px0, py0, px0, py1], fill=(120, 120, 120))
    n = len(series)
    pts = []
    for i, v in enumerate(series):
        x = px0 + (px1 - px0) * (i / max(1, n - 1))
        y = py1 - (py1 - py0) * ((v - lo) / (hi - lo))
        pts.append((x, y))
    if len(pts) > 1:
        draw.line(pts, fill=colour, width=2)
    for i, (x, y) in enumerate(pts):
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=colour)
        draw.text((x - 18, py1 + 6), tags[i], fill=(60, 60, 60), font=get_font(12))
    draw.text((x0 + 6, py0 - 2), '%.4g' % hi, fill=(90, 90, 90), font=get_font(12))
    draw.text((x0 + 6, py1 - 12), '%.4g' % lo, fill=(90, 90, 90), font=get_font(12))


def build_metrics_figure(args, tags):
    data = {}
    for tag in tags:
        path = os.path.join(args.diagnostics_dir, 'diagnostics_%s.json' % tag)
        data[tag] = json.load(open(path))

    series = {
        'Said attention entropy': [data[t]['said_attention']['mean_entropy'] for t in tags],
        'effective patch count': [data[t]['said_attention']['mean_effective_patch_count'] for t in tags],
        'caption-shuffle JS divergence': [data[t]['caption_shuffle']['mean_js_divergence'] for t in tags],
        'positive alignment margin': [data[t]['positive_alignment_margin']['mean_margin'] for t in tags],
        'short vs long caption JS': [data[t]['short_vs_long_caption']['mean_js_divergence'] for t in tags],
        'z_s cosine (caption shuffle)': [data[t]['caption_shuffle']['mean_zs_cosine'] for t in tags],
    }
    panel_w, panel_h, cols = 420, 240, 3
    rows = (len(series) + cols - 1) // cols
    canvas = Image.new('RGB', (cols * panel_w + 20, rows * panel_h + 60), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), 'Said grounding diagnostics across checkpoints (fixed 64-image set)',
              fill=(0, 0, 0), font=get_font(22))
    for i, (title, values) in enumerate(series.items()):
        r, c = divmod(i, cols)
        box = (10 + c * panel_w, 50 + r * panel_h, 10 + c * panel_w + panel_w - 12, 50 + r * panel_h + panel_h - 12)
        line_panel(draw, box, title, values, tags)
    out_path = os.path.join(args.output_dir, 'metrics_evolution.png')
    canvas.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Said attention checkpoint comparison figures')
    parser.add_argument('--diagnostics_dir', default='outputs/salu_grounding')
    parser.add_argument('--tags', default='initial,step100,step200,step400,final')
    parser.add_argument('--samples', default='0,1,2,3')
    parser.add_argument('--grid', type=int, default=14)
    parser.add_argument('--tile', type=int, default=224)
    parser.add_argument('--output_dir', default='outputs/salu_grounding/comparison')
    args = parser.parse_args()

    tags = [t.strip() for t in args.tags.split(',') if t.strip()]
    samples = [int(s) for s in args.samples.split(',') if s.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    for sample in samples:
        path, own_scale, shuf_scale = build_sample_figure(args, sample, tags)
        print('WROTE %s (own_scale=%.5f shuffled_scale=%.5f)' % (path, own_scale, shuf_scale), flush=True)
    metrics_path = build_metrics_figure(args, tags)
    print('WROTE %s' % metrics_path, flush=True)
    print('COMPARE_RESULT PASS')


if __name__ == '__main__':
    main()
