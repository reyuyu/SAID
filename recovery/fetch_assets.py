"""Resume public asset downloads; never start a trainer or bypass gated access."""

import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import time

from audit import ASSETS, EVIDENCE, ROOT, digest, dump


DOWNLOADS = ASSETS / "downloads"
SPECS = {
    "urban1k": ("Urban1k.zip", "https://hf-mirror.com/datasets/BeichenZhang/Urban1k/resolve/main/Urban1k.zip", "08e42b3fada77abf7f890a087ed6e9f9fbba1dc143f9f09801ed83187ae617b8"),
    "flickr_images": ("images_flickr_1k_test.zip", "https://hf-mirror.com/datasets/nlphuji/flickr_1k_test_image_text_retrieval/resolve/main/images_flickr_1k_test.zip", "b0fa9970cc1680a9334818b9da151a62024772b72b961e1454eb6e9866dd7cee"),
    "flickr_captions": ("test_1k_flickr.csv", "https://hf-mirror.com/datasets/nlphuji/flickr_1k_test_image_text_retrieval/resolve/main/test_1k_flickr.csv", "a427bd870759196b9af09ba9cbd3f0a49c6e6a50c7eafcd9001fdd9f81219105"),
    "docci_captions": ("docci_descriptions.jsonlines", "https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines", None),
    "docci_images": ("docci_images.zip", "https://hf-mirror.com/datasets/qihoo360/DOCCI-CN/resolve/main/images.zip", "6a4f4f0ecc74e454702ffe6a24d5a32cf826fbeddc0aaf13cced1052d5befe72"),
    "dci_annotations": ("dci.tar.gz", "https://dl.fbaipublicfiles.com/densely_captioned_images/dci.tar.gz", "d865c244150168d3f25daaad0bf5b70b2123cf3e83ca7b0e207d5b26943c5fc7"),
    "dci_images_1": ("dci_images_1.zip", "https://hf-mirror.com/datasets/qihoo360/DCI-CN/resolve/main/images/images_1.zip", "9e3b6bbe78747c74a12192d9b54895e187f0a37b2e2d230df583260e791e7dee"),
    "dci_images_2": ("dci_images_2.zip", "https://hf-mirror.com/datasets/qihoo360/DCI-CN/resolve/main/images/images_2.zip", "1d686645bd7886ed9e7f5c9ea1b3441b9513dd30d6203a11c89edb18cd940186"),
    "coco_val2017": ("val2017.zip", "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip", None),
    "coco_annotations": ("annotations_trainval2017.zip", "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip", None),
    "coco_train2017": ("train2017.zip", "https://s3.amazonaws.com/images.cocodataset.org/zips/train2017.zip", None),
    "llava_images": ("llava_pretrain_images.zip", "https://hf-mirror.com/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip", None),
}


def fetch(name):
    filename, url, expected = SPECS[name]
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    destination = DOWNLOADS / filename
    record = dict(asset=name, url=url, destination=str(destination), expected_sha256=expected, completed=False)
    try:
        if not destination.is_file():
            temporary = destination.with_name(destination.name + ".part")
            with (EVIDENCE / ("download-" + name + ".log")).open("a") as log:
                for attempt in range(3):
                    result = subprocess.run(["curl", "-4", "--fail", "--location", "--continue-at", "-",
                        "--connect-timeout", "15", "--max-time", "3600", "--retry", "0",
                        "--retry-delay", "2", url, "--output", str(temporary)], stdout=log, stderr=log)
                    if result.returncode == 0:
                        break
                    if result.returncode not in (18, 28, 56, 92) or attempt == 2:
                        raise RuntimeError(f"curl exit {result.returncode}; partial download preserved")
                    time.sleep(2)
            observed = digest(temporary)
            if expected and observed != expected:
                raise ValueError(f"SHA256 mismatch: expected {expected}, observed {observed}; partial preserved")
            temporary.replace(destination)
        observed = digest(destination)
        if expected and observed != expected:
            raise ValueError("Existing asset SHA256 mismatch; file left untouched")
        historical = name in ("flickr_images", "docci_images", "dci_annotations", "dci_images_1", "dci_images_2")
        record.update(completed=True, observed_sha256=observed, size_bytes=destination.stat().st_size,
                      source_sha256_verified=bool(expected), historical_archive_identity_verified=bool(expected) and historical)
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error))
    dump(EVIDENCE / ("download-" + name + ".json"), record)
    print(json.dumps(record), flush=True)
    return record["completed"]


def extract(name, destination):
    import sys

    sys.path.insert(0, str(ROOT))
    from tools.prepare_retrieval_benchmarks import extract_safe

    filename, _, expected = SPECS[name]
    archive = DOWNLOADS / filename
    if expected and digest(archive) != expected:
        raise ValueError("Archive identity mismatch")
    extract_safe(archive, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", nargs="+", choices=sorted(SPECS))
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error("jobs must be between 1 and 4")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        results = list(executor.map(fetch, args.assets))
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
