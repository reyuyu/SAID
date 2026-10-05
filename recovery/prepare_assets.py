"""Extract verified archives and reuse the original frozen manifest builders."""

import argparse
import os
from pathlib import Path
import sys

from audit import ASSETS, EVIDENCE, PROTOCOLS, ROOT, digest, dump
from fetch_assets import DOWNLOADS, SPECS, extract


def available(name):
    return (DOWNLOADS / SPECS[name][0]).is_file()


def publish(filename, builder):
    destination = ASSETS / "retrieval_benchmarks/manifests" / filename
    candidate = destination.with_suffix(destination.suffix + ".candidate")
    expected = next(value[4] for value in PROTOCOLS.values() if value[0] == filename)
    observed, rows = builder(candidate)
    if observed != expected:
        raise RuntimeError(f"Frozen manifest mismatch for {filename}; candidate retained, no replacement published")
    candidate.replace(destination)
    return dict(sha256=observed, rows=rows, historical_sha_matches=True)


def bind_images(source, manifest, destination):
    import json

    source = Path(source)
    destination = Path(destination)
    files = {}
    for path in source.rglob("*"):
        if path.is_file():
            files.setdefault(path.name, []).append(path)
    rows = [json.loads(line) for line in Path(manifest).open() if line.strip()]
    needed = sorted({row["image_path"] for row in rows})
    for relative in needed:
        target = destination / relative
        if Path(relative).is_absolute() or not target.parent.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Manifest path escape")
        candidates = files.get(Path(relative).name, [])
        if len(candidates) != 1:
            raise ValueError(f"Image filename is missing or ambiguous: {relative}, matches={len(candidates)}")
        original = candidates[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.resolve() != original.resolve() and digest(target) != digest(original):
                raise ValueError(f"Existing image differs from verified archive: {target}")
        else:
            target.symlink_to(os.path.relpath(original, target.parent))
    return dict(mapped_images=len(needed), pixels="Original archive bytes; relative symlinks only; no re-encoding")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", action="store_true", help="Also extract available COCO/LLaVA training archives; no training")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from tools.prepare_retrieval_benchmarks import parse_docci, parse_flickr, reconstruct_long_dci

    bench = ASSETS / "retrieval_benchmarks"
    evidence = dict(manifests={}, images={}, missing_archives=[])
    for name in ("coco_annotations", "coco_val2017", "urban1k"):
        if available(name):
            destination = ASSETS / ("evaluation/Urban1k" if name == "urban1k" else "evaluation/coco")
            extract(name, destination)
        else:
            evidence["missing_archives"].append(name)
    if available("flickr_images") and available("flickr_captions"):
        extract("flickr_images", bench / "flickr30k/archive")
        evidence["manifests"]["flickr30k_test1k.jsonl"] = publish("flickr30k_test1k.jsonl",
            lambda target: parse_flickr(DOWNLOADS / "test_1k_flickr.csv", target, test1k=True))
        image_root = bench / "flickr30k/images"
        if not image_root.exists():
            image_root.symlink_to("archive/images_flickr_1k_test", target_is_directory=True)
    if available("docci_captions"):
        evidence["manifests"]["docci_test.jsonl"] = publish("docci_test.jsonl",
            lambda target: parse_docci(DOWNLOADS / "docci_descriptions.jsonlines", target))
        if available("docci_images"):
            extract("docci_images", bench / "docci/image_archive")
            evidence["images"]["docci"] = bind_images(bench / "docci/image_archive",
                bench / "manifests/docci_test.jsonl", bench / "docci/images")
        else:
            evidence["missing_archives"].append("docci_images")
    if available("dci_annotations"):
        extract("dci_annotations", bench / "dci/source")
        directory = bench / "dci/source/densely_captioned_images/annotations"
        evidence["manifests"]["long_dci_reconstructed.jsonl"] = publish("long_dci_reconstructed.jsonl",
            lambda target: reconstruct_long_dci(directory, target))
        if available("dci_images_1") and available("dci_images_2"):
            for name in ("dci_images_1", "dci_images_2"):
                extract(name, bench / "dci/image_archives")
            evidence["images"]["long_dci"] = bind_images(bench / "dci/image_archives",
                bench / "manifests/long_dci_reconstructed.jsonl", bench / "dci/images")
        else:
            evidence["missing_archives"].extend(name for name in ("dci_images_1", "dci_images_2") if not available(name))
    if args.training:
        for name, folder in [("coco_train2017", "coco"), ("llava_images", "llava/llava_pretrain")]:
            if available(name):
                extract(name, ASSETS / "training/ShareGPT4V" / folder)
            else:
                evidence["missing_archives"].append(name)
    dump(EVIDENCE / "asset-preparation.json", evidence)
    print(evidence)


if __name__ == "__main__":
    main()
