#!/usr/bin/env python3
"""Construct reproducible random caption views without loading a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path

import numpy as np

from model import longclip

CONTEXT = 248
CONTENT_LIMIT = 246
SEEDS = (0, 1, 2, 3, 4)
ABBREVIATIONS = {"e.g", "i.e", "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def token_ids(text: str) -> list[int]:
    return list(longclip._tokenizer.encode(text))


def first_sentence(text: str) -> tuple[str, dict]:
    candidates, skipped = [], []
    for match in re.finditer(r'''[.!?]+["”’)\]]*(?=\s|$)''', text):
        punct = match.group(0)[0]
        remainder = text[match.end():].lstrip()
        if punct == ".":
            before = text[:match.start()].rstrip()
            word = re.split(r"\s+", before)[-1].lower().strip("(\"'[]{}")
            if word in ABBREVIATIONS and remainder:
                skipped.append({"position": match.start(), "reason": "known_abbreviation", "word": word}); continue
            if re.fullmatch(r"(?:[a-z]\.)+[a-z]", word) and remainder:
                skipped.append({"position": match.start(), "reason": "initialism", "word": word}); continue
            if re.fullmatch(r"[a-z]", word) and remainder and remainder[0].isupper():
                skipped.append({"position": match.start(), "reason": "single_initial", "word": word}); continue
        if punct in ("!", "?") and remainder and remainder[0].islower():
            skipped.append({"position": match.start(), "reason": "internal_punctuation_before_lowercase", "punctuation": punct}); continue
        candidates.append(match)
    if not candidates:
        return text.strip(), {"boundary_found": False, "fallback_full_caption": True, "boundary": None, "skipped_candidates": skipped}
    m = candidates[0]
    return text[:m.end()].strip(), {"boundary_found": True, "fallback_full_caption": False, "boundary": m.group(0), "char_end": m.end(), "skipped_candidates": skipped}


def effective_mother(caption: str) -> dict:
    """Keep complete whitespace words while respecting the 246 BPE budget."""
    matches = list(re.finditer(r"\S+", caption))
    if not matches:
        return {"text": "", "tokens": [], "units": [], "token_count": 0}
    last_end = 0
    unit_lengths = []
    for m in matches:
        unit_len = len(token_ids(m.group(0)))
        if sum(unit_lengths) + unit_len > CONTENT_LIMIT:
            break
        unit_lengths.append(unit_len)
        last_end = m.end()
    if not last_end:
        # Preserve at least one complete word when the first word is unusually long.
        last_end = matches[0].end()
    text = caption[:last_end].strip()
    units = [{"index": i, "text": m.group(0), "start": m.start(), "end": m.end(), "bpe_len": unit_lengths[i]} for i, m in enumerate(re.finditer(r"\S+", text))]
    exact_tokens = token_ids(text)
    return {"text": text, "tokens": exact_tokens, "units": units, "token_count": len(exact_tokens)}


def join_units(units: list[dict], indices: list[int]) -> str:
    if not indices:
        return ""
    indices = sorted(indices)
    return " ".join(units[i]["text"] for i in indices)


def measure_view(text: str, mother: dict, first_text: str, first_end: int, selected_indices: list[int] | None = None) -> dict:
    ids = token_ids(text)
    first_units = [u for u in mother["units"] if u["end"] <= first_end]
    selected_indices = selected_indices if selected_indices is not None else [u["index"] for u in mother["units"] if u["text"] in text.split()]
    selected_set = set(selected_indices)
    first_bpe = sum(u["bpe_len"] for u in first_units if u["index"] in selected_set)
    first_overlap = first_bpe / max(1, len(ids))
    first_token_count = len(token_ids(first_text))
    return {"text": text, "bpe_tokens": len(ids), "retained_fraction": len(ids) / max(1, mother["token_count"]), "first_sentence_bpe_tokens": first_token_count, "first_sentence_word_overlap_fraction": first_overlap}


def distributed(units: list[dict], mother: dict, target: int, rng: random.Random, first_text: str, first_end: int) -> dict:
    if not units or target <= 0:
        return {"text": "", "bpe_tokens": 0, "valid": False, "reason": "empty"}
    order = list(range(len(units))); rng.shuffle(order)
    selected: list[int] = []
    selected_len = 0
    for idx in order:
        candidate_len = selected_len + units[idx]["bpe_len"]
        if candidate_len <= target or selected_len < max(1, int(target * .78)):
            selected.append(idx)
            selected_len += units[idx]["bpe_len"]
        if selected_len >= target:
            break
    if not selected:
        selected = [order[0]]
    text = join_units(units, selected)
    return {"strategy": "random_distributed", "target_bpe_tokens": target, "selected_unit_indices": sorted(selected), "valid": True, **measure_view(text, mother, first_text, first_end, selected)}


def contiguous(units: list[dict], mother: dict, target: int, rng: random.Random, first_text: str, first_end: int, eligible: list[int] | None = None) -> dict:
    eligible = list(range(len(units))) if eligible is None else list(eligible)
    if not eligible or target <= 0:
        return {"strategy": "random_contiguous_crop", "target_bpe_tokens": target, "valid": False, "reason": "insufficient_units"}
    candidates = []
    for start_pos, start in enumerate(eligible):
        selected = []
        selected_len = 0
        for idx in eligible[start_pos:]:
            selected.append(idx)
            selected_len += units[idx]["bpe_len"]
            if selected_len >= target:
                candidates.append((abs(selected_len - target), selected[:]))
                break
    if not candidates:
        return {"strategy": "random_contiguous_crop", "target_bpe_tokens": target, "valid": False, "reason": "insufficient_bpe_content"}
    best_distance = min(x[0] for x in candidates)
    near = [x[1] for x in candidates if x[0] <= best_distance + max(2, int(target * .10))]
    selected = list(rng.choice(near))
    text = join_units(units, selected)
    return {"strategy": "random_contiguous_crop", "target_bpe_tokens": target, "selected_unit_indices": sorted(selected), "valid": True, **measure_view(text, mother, first_text, first_end, selected)}


def nofirst(mother: dict, first_text: str, first_end: int, seed: int) -> dict:
    target = len(token_ids(first_text))
    eligible = [u for u in mother["units"] if u["start"] >= first_end]
    rng = random.Random(seed + 100003)
    result = contiguous(mother["units"], mother, target, rng, first_text, first_end, [u["index"] for u in eligible])
    result["strategy"] = "nofirst_matched_contiguous"
    result["target_bpe_tokens"] = target
    result["excludes_first_sentence"] = True
    return result


def build_sample(row: dict) -> dict:
    caption = row["caption"]
    mother = effective_mother(caption)
    first_text, first_meta = first_sentence(caption)
    # Clip the first-sentence boundary to the word-safe mother.
    first_end = len(first_text)
    if first_meta.get("boundary_found"):
        first_end = min(first_meta.get("char_end", len(first_text)), len(mother["text"]))
    else:
        first_end = len(mother["text"]) + 1
    target50 = max(1, int(round(mother["token_count"] * .50)))
    target10 = max(1, int(round(mother["token_count"] * .10)))
    out = {"index": None, "image_id": row["image_id"], "image_path": row["image_path"], "caption": caption, "mother": {"text": mother["text"], "bpe_tokens": mother["token_count"], "unit_count": len(mother["units"])}, "first_sentence": {"text": first_text, "bpe_tokens": len(token_ids(first_text)), "meta": first_meta}, "views": {}}
    for seed in SEEDS:
        rng = random.Random(seed)
        for frac, target in (("random50", target50), ("random10", target10)):
            out["views"][f"random_distributed_{frac}_seed{seed}"] = distributed(mother["units"], mother, target, rng, first_text, first_end) | {"seed": seed, "view": frac}
            out["views"][f"random_contiguous_{frac}_seed{seed}"] = contiguous(mother["units"], mother, target, random.Random(seed + (17 if frac == "random50" else 31)), first_text, first_end) | {"seed": seed, "view": frac}
        out["views"][f"nofirst_matched_seed{seed}"] = nofirst(mother, first_text, first_end, seed) | {"seed": seed, "view": "nofirst_matched"}
    return out


def self_test() -> None:
    row = {"image_id": "x", "image_path": "x.jpg", "caption": "First short sentence. The second part has many words and details about a blue object."}
    item = build_sample(row)
    assert item["mother"]["bpe_tokens"] <= 246
    assert all(item["views"][f"random_distributed_random50_seed{s}"]["valid"] for s in SEEDS)
    assert all(item["views"][f"random_contiguous_random10_seed{s}"]["valid"] for s in SEEDS)
    assert all(item["views"][f"nofirst_matched_seed{s}"]["excludes_first_sentence"] for s in SEEDS)
    short = build_sample({"image_id": "y", "image_path": "y.jpg", "caption": "No boundary here"})
    assert not short["views"]["nofirst_matched_seed0"]["valid"]
    print("Caption sampling CPU regression checks: PASS")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--samples", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--self-test", action="store_true"); args = ap.parse_args()
    if args.self_test:
        self_test(); return
    rows = json.loads(args.samples.read_text())
    out = []
    for i, row in enumerate(rows):
        item = build_sample(row); item["index"] = i; out.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"dataset": args.dataset, "seed_list": list(SEEDS), "sample_count": len(out), "source_samples": str(args.samples), "source_samples_sha256": sha256(args.samples), "records": out}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"dataset": args.dataset, "sample_count": len(out), "output": str(args.output)}, indent=2))


if __name__ == "__main__": main()
