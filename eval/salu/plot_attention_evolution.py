"""Plot the evolution of the Said attention heatmaps across checkpoints (fixed cases).

Produces, for a few fixed diagnostic images:

  * ``evolution_own_sampleXX.png``: the original image plus the own-caption Said
    attention at every checkpoint in one row (pure heatmaps, shared colour scale).
  * ``evolution_grid_sampleXX.png``: 4 caption variants (own / shuffled / short /
    long) x every checkpoint as a pure heatmap grid, again on one shared scale,
    with a colour bar in raw attention units.

All tiles of a figure share a single colour scale (max over every variant and
checkpoint of that sample), so sharpening across checkpoints is directly visible.
Figures go to ``outputs/`` and are never committed.
"""
import argparse
import json
import os
import sys
import textwrap

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
]
VARIANTS = ['own', 'shuffled', 'short', 'long']
VARIANT_LABEL = {
    'own': 'own caption',
    'shuffled': 'shuffled caption',
    'short': 'short caption (1st sentence)',
    'long': 'long caption (full)',
}


def get_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def load_attn(path, grid):
    return np.load(path).reshape(grid, grid)


def heat_tile(attn2d, scale_max, tile):
    norm = np.clip(attn2d / max(scale_max, 1e-12), 0.0, 1.0)
    return Image.fromarray(colorize(upsample(norm, tile)))


def draw_colorbar(canvas, draw, x, y0, height, scale_max, width=26):
    for i in range(height):
        value = 1.0 - i / max(1, height - 1)
        colour = colorize(np.array([[value]], dtype=np.float32))[0, 0]
        draw.line([x, y0 + i, x + width, y0 + i], fill=tuple(int(c) for c in colour))
    draw.rectangle([x, y0, x + width, y0 + height], outline=(120, 120, 120))
    font = get_font(13)
    draw.text((x + width + 6, y0 - 2), '%.4f' % scale_max, fill=(60, 60, 60), font=font)
    draw.text((x + width + 6, y0 + height - 14), '0', fill=(60, 60, 60), font=font)


def figure_own(args, sample, tags, caption):
    tile = args.tile
    label_w = 96
    header_h = 40
    text_h = 54
    width = label_w + len(tags) * tile
    height = header_h + text_h + tile
    canvas = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font_t = get_font(19)
    font_s = get_font(14)

    draw.text((10, 8), 'sample %02d  own-caption Said attention evolution' % sample,
              fill=(0, 0, 0), font=font_t)
    wrapped = textwrap.wrap(caption, width=110)[:2]
    for k, line in enumerate(wrapped):
        draw.text((10, header_h + 4 + k * 18), line, fill=(90, 90, 90), font=font_s)

    scale_max = max(
        load_attn(os.path.join(args.diagnostics_dir, 'heatmaps', t, 'sample%02d_own_attn14.npy' % sample), args.grid).max()
        for t in tags
    )
    y = header_h + text_h
    for col, tag in enumerate(tags):
        attn = load_attn(os.path.join(args.diagnostics_dir, 'heatmaps', tag, 'sample%02d_own_attn14.npy' % sample), args.grid)
        canvas.paste(heat_tile(attn, scale_max, tile), (label_w + col * tile, y))
        draw.rectangle([label_w + col * tile, y, label_w + (col + 1) * tile - 1, y + tile - 1], outline=(200, 200, 200))
        draw.text((label_w + col * tile + 6, y + tile + 2), tag, fill=(40, 40, 40), font=font_s)
    draw.text((10, y + tile // 2), 'own\ncaption', fill=(40, 40, 40), font=font_s)
    out = os.path.join(args.output_dir, 'evolution_own_sample%02d.png' % sample)
    canvas.save(out)
    return out, scale_max


def figure_grid(args, sample, tags, caption):
    tile = args.tile
    label_w = 230
    header_h = 40
    text_h = 54
    bar_w = 70
    width = label_w + len(tags) * tile + bar_w
    height = header_h + text_h + len(VARIANTS) * tile + 24
    canvas = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font_t = get_font(19)
    font_s = get_font(14)

    draw.text((10, 8), 'sample %02d  Said attention: caption variants x checkpoints (pure heatmaps, shared scale)' % sample,
              fill=(0, 0, 0), font=font_t)
    wrapped = textwrap.wrap(caption, width=150)[:2]
    for k, line in enumerate(wrapped):
        draw.text((10, header_h + 4 + k * 18), line, fill=(90, 90, 90), font=font_s)

    scale_max = max(
        load_attn(os.path.join(args.diagnostics_dir, 'heatmaps', t, 'sample%02d_%s_attn14.npy' % (sample, v)), args.grid).max()
        for t in tags for v in VARIANTS
    )

    y0 = header_h + text_h
    for col, tag in enumerate(tags):
        draw.text((label_w + col * tile + 6, y0 - 22), tag, fill=(40, 40, 40), font=font_s)
    for row, variant in enumerate(VARIANTS):
        y = y0 + row * tile
        for k, line in enumerate(textwrap.wrap(VARIANT_LABEL[variant], width=26)[:2]):
            draw.text((10, y + tile // 2 - 16 + k * 18), line, fill=(40, 40, 40), font=font_s)
        for col, tag in enumerate(tags):
            attn = load_attn(
                os.path.join(args.diagnostics_dir, 'heatmaps', tag, 'sample%02d_%s_attn14.npy' % (sample, variant)),
                args.grid,
            )
            canvas.paste(heat_tile(attn, scale_max, tile), (label_w + col * tile, y))
            draw.rectangle([label_w + col * tile, y, label_w + (col + 1) * tile - 1, y + tile - 1], outline=(200, 200, 200))
    draw_colorbar(canvas, draw, label_w + len(tags) * tile + 16, y0, len(VARIANTS) * tile, scale_max)
    out = os.path.join(args.output_dir, 'evolution_grid_sample%02d.png' % sample)
    canvas.save(out)
    return out, scale_max


def main():
    parser = argparse.ArgumentParser(description='Said attention heatmap evolution figures')
    parser.add_argument('--diagnostics_dir', default='outputs/salu_grounding')
    parser.add_argument('--tags', default='initial,step100,step200,step400,final')
    parser.add_argument('--samples', default='0,1,2,3')
    parser.add_argument('--grid', type=int, default=14)
    parser.add_argument('--tile', type=int, default=224)
    parser.add_argument('--output_dir', default='outputs/salu_grounding/evolution')
    args = parser.parse_args()

    tags = [t.strip() for t in args.tags.split(',') if t.strip()]
    samples = [int(s) for s in args.samples.split(',') if s.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    caps = {}
    caps_path = os.path.join(args.diagnostics_dir, 'diagnostic_captions.json')
    if os.path.exists(caps_path):
        with open(caps_path) as fp:
            caps = {int(k): v for k, v in json.load(fp).items()}

    for sample in samples:
        caption = caps.get(sample, '')
        p1, s1 = figure_own(args, sample, tags, caption)
        print('WROTE %s (scale_max=%.5f)' % (p1, s1), flush=True)
        p2, s2 = figure_grid(args, sample, tags, caption)
        print('WROTE %s (scale_max=%.5f)' % (p2, s2), flush=True)
    print('EVOLUTION_FIGURES PASS')


if __name__ == '__main__':
    main()
