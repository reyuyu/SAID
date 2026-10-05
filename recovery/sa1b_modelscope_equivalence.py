"""Compare source JPEGs against normal-shard originals using the native transform."""

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import sys

from PIL import Image
import torch

import sa1b_recovery as recovery


FOLDER = recovery.EVIDENCE / "modelscope-sa1b"
FORMAL = recovery.ROOT / "worktrees/randomk"
ALLOWED = {"BITWISE_EQUIVALENT", "TRAINING_INPUT_EQUIVALENT"}


def byte_sha(content):
    return hashlib.sha256(content).hexdigest()


def tensor_sha(tensor):
    return byte_sha(tensor.detach().cpu().contiguous().numpy().tobytes())


def compare(row, source, preprocess):
    result = dict(image_id=row["image_id"], basename=row["basename"], original_shard=row["shard"],
        local_path=row["local_path"], source_url=row["source_url"], source_field=row["source_field"],
        parquet_path=row["parquet_path"], parquet_sha256=row["parquet_sha256"],
        source_repository=row["repo"], source_revision=row["revision"],
        classification="COMPARISON_UNAVAILABLE", passed=False)
    if not source.get("downloaded"):
        result.update(reason="Source image could not be downloaded", http_status=source.get("http_status"))
        return result
    original = Path(row["local_path"]).read_bytes()
    downloaded = Path(source["source_path"]).read_bytes()
    original_sha, downloaded_sha = byte_sha(original), byte_sha(downloaded)
    if original_sha != row["sha256"]:
        raise ValueError("Local JPEG no longer matches verified original tar inventory")
    if downloaded_sha != source["sha256"]:
        raise ValueError("Source JPEG no longer matches its download proof")
    source_basename = Path(row["source_url"]).name
    expected_basename = f"sa_{row['image_id']}.jpg"
    id_equal = row["basename"] == expected_basename and source_basename == expected_basename
    with Image.open(io.BytesIO(original)) as reference, Image.open(io.BytesIO(downloaded)) as candidate:
        if reference.format != "JPEG" or candidate.format != "JPEG":
            raise ValueError("Only original encoded JPEG image objects are allowed")
        reference_rgb = reference.convert("RGB")
        candidate_rgb = candidate.convert("RGB")
        reference_rgb.load()
        candidate_rgb.load()
    reference_pixels = reference_rgb.tobytes()
    candidate_pixels = candidate_rgb.tobytes()
    reference_tensor = preprocess(reference_rgb)
    candidate_tensor = preprocess(candidate_rgb)
    dimensions_equal = reference_rgb.size == candidate_rgb.size and min(reference_rgb.size) > 0
    encoded_equal = original == downloaded
    pixels_equal = dimensions_equal and reference_pixels == candidate_pixels
    tensors_equal = torch.equal(reference_tensor, candidate_tensor) and tensor_sha(reference_tensor) == tensor_sha(candidate_tensor)
    valid = id_equal and dimensions_equal and pixels_equal and tensors_equal
    classification = "BITWISE_EQUIVALENT" if valid and encoded_equal else "TRAINING_INPUT_EQUIVALENT" if valid else "NOT_EQUIVALENT"
    result.update(classification=classification, passed=classification in ALLOWED,
        id_basename_equal=id_equal, original_width=reference_rgb.width, original_height=reference_rgb.height,
        source_width=candidate_rgb.width, source_height=candidate_rgb.height,
        dimensions_equal=dimensions_equal, encoded_jpeg_sha256_original=original_sha,
        encoded_jpeg_sha256_source=downloaded_sha, encoded_bytes_exact_equal=encoded_equal,
        rgb_pixels_sha256_original=byte_sha(reference_pixels), rgb_pixels_sha256_source=byte_sha(candidate_pixels),
        rgb_pixels_exact_equal=pixels_equal, preprocess_tensor_sha256_original=tensor_sha(reference_tensor),
        preprocess_tensor_sha256_source=tensor_sha(candidate_tensor), preprocess_tensor_exact_equal=tensors_equal,
        preprocess_tensor_shape=list(reference_tensor.shape), preprocess_tensor_dtype=str(reference_tensor.dtype),
        jpeg_decode_passed=True)
    return result


def native_preprocess():
    sys.path.insert(0, str(FORMAL))
    from train import said_cvssl_data

    module = Path(said_cvssl_data.__file__).resolve()
    if module != (FORMAL / "train/said_cvssl_data.py").resolve():
        raise ValueError("Image transform was not imported from the canonical pinned training code")
    return said_cvssl_data.reference_view_a_transform(), dict(factory="train.said_cvssl_data.reference_view_a_transform",
        source_file=str(module), source_sha256=recovery.digest(module),
        native_dataset_source_sha256=recovery.digest(FORMAL / "train/nested_semantic_data.py"),
        canonical_git_revision=__import__("subprocess").check_output(["git", "-C", str(FORMAL), "rev-parse", "HEAD"], text=True).strip(),
        loader_rgb_conversion="PIL.Image.open(path).convert('RGB')", device="CPU")


def summary(results, requested, preprocess_provenance):
    classifications = Counter(row["classification"] for row in results)
    distinct = len({row["image_id"] for row in results})
    passed = requested >= 200 and len(results) == requested and distinct == requested and all(row["classification"] in ALLOWED and row["passed"] for row in results)
    return dict(checked_utc=recovery.now(), passed=passed, replacement_authorized=passed,
        required_samples=200, requested=requested, compared=len(results), classifications=dict(classifications),
        preprocess_provenance=preprocess_provenance, training_index_sha256=recovery.RECORDS_SHA,
        image_ids_distinct=distinct,
        original_shard_coverage=dict(Counter(row["original_shard"] for row in results)), samples=results,
        missing_image_downloads_started=False, smoke_started=False, formal_training_started=False)


def report(audit):
    lines = ["# ModelScope SA-1B image equivalence audit", "", f"Checked UTC: {audit['checked_utc']}.",
        "Repository: Tongyi-DataEngine/SA1B-Paired-Captions-Images.",
        "Pinned revision: 4859a29a7aa78102d3364852755c7818e7583366.",
        "Only the opensource_url full-image field is used; matched_local_urls cropped images are excluded.",
        "Only source-indexed IDs already installed from checksum-/tar-verified normal SA-1B shards are references.",
        "Local encoded bytes are independently rehashed against their original tar extraction inventories.",
        "Only parquet footer/projected URL-column HTTP ranges were fetched, not complete parquet files or the full dataset.",
        "", "## Exact comparisons", "", f"Requested distinct IDs: {audit['requested']}.",
        f"Compared: {audit['compared']}; distinct: {audit['image_ids_distinct']}.",
        f"Classification counts: {json.dumps(audit['classifications'], ensure_ascii=False)}.",
        f"Replacement gate passed: {audit['passed']}.",
        f"Normal-shard coverage: {json.dumps(audit['original_shard_coverage'], ensure_ascii=False)}.",
        "Dimensions, ID/basename, encoded-JPEG SHA256, RGB-pixel SHA256/byte equality and native-preprocess tensor SHA256/torch.equal are recorded per image.",
        "", "## Actual training preprocessing", "",
        f"Factory: {audit['preprocess_provenance']['factory']}.",
        f"Canonical source SHA256: {audit['preprocess_provenance']['source_sha256']}.",
        f"Canonical git revision: {audit['preprocess_provenance']['canonical_git_revision']}.",
        "Preprocessing is imported from the existing canonical training module, never reimplemented or substituted.",
        "", "## Safety decision", "",
        "Replacement is authorized only if all200 distinct references are BITWISE_EQUIVALENT or TRAINING_INPUT_EQUIVALENT with no dimension/ID errors.",
        "Any pixel/tensor/dimension/ID mismatch closes the gate; no required missing JPEG may be installed from this source after a failed audit.",
        "No missing-ID batch is downloaded by this audit. Existing HF/ODL recovery is not stopped or modified.",
        "Per-image results: evidence/modelscope-sa1b/equivalence-audit.json.",
        "Source row-group and download provenance: evidence/modelscope-sa1b/audit-samples.json, audit-downloads.json, parquet-scans.json.", ""]
    bad = next((row for row in audit["samples"] if row["classification"] == "NOT_EQUIVALENT"), None)
    if bad:
        lines.extend(["## First incompatible example", "", f"Image: {bad['basename']}.",
            f"Original dimensions: {bad['original_width']}x{bad['original_height']}; source: {bad['source_width']}x{bad['source_height']}.",
            f"Original JPEG SHA256: {bad['encoded_jpeg_sha256_original']}.",
            f"Source JPEG SHA256: {bad['encoded_jpeg_sha256_source']}.",
            f"RGB pixels exact equal: {bad['rgb_pixels_exact_equal']}; native training tensor exact equal: {bad['preprocess_tensor_exact_equal']}.", ""])
    (recovery.RECOVERY / "MODELSCOPE_SA1B_EQUIVALENCE_AUDIT.md").write_text("\n".join(lines))


def main():
    torch.set_num_threads(1)
    preprocess, provenance = native_preprocess()
    manifest = recovery.load(FOLDER / "audit-samples.json")
    downloads = recovery.load(FOLDER / "audit-downloads.json")
    sources = {row["image_id"]: row for row in downloads["samples"]}
    if not manifest["complete"] or manifest["selected_count"] < 200:
        raise ValueError("Fewer than200 independently indexed reference IDs")
    results = []
    for row in manifest["samples"]:
        try:
            result = compare(row, sources[row["image_id"]], preprocess)
        except (OSError, ValueError, KeyError) as error:
            result = dict(image_id=row["image_id"], basename=row["basename"], original_shard=row["shard"],
                classification="COMPARISON_ERROR", passed=False, error_type=type(error).__name__)
        results.append(result)
        if len(results) % 10 == 0:
            recovery.atomic_json(FOLDER / "equivalence-progress.json", dict(compared=len(results),
                classifications=dict(Counter(item["classification"] for item in results))))
    audit = summary(results, manifest["selected_count"], provenance)
    recovery.atomic_json(FOLDER / "equivalence-audit.json", audit)
    recovery.atomic_json(recovery.RECOVERY / "MODELSCOPE_SA1B_EQUIVALENCE.json", audit)
    report(audit)
    print(json.dumps({key:audit[key] for key in ["passed", "compared", "classifications", "original_shard_coverage"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
