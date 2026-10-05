"""Audit original COCO directory names against verified extraction and frozen paths."""

import json
import os
from pathlib import PurePosixPath
import zipfile

from audit import EVIDENCE, load
from hf_training_recovery import audit_zip_structure, publish_family_completeness, target_root_images, worker_path
from sa1b_recovery import atomic_json, now, requirements


COUNT = 118287
SHA = "69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929"
RESULT = EVIDENCE / "coco-independent-index-audit.json"


def validate_extraction(progress):
    if not (progress.get("sha256_passed") and progress.get("actual_sha256") == SHA
            and progress.get("actual_size_bytes") == 19336861798 and progress.get("zip_integrity_passed")
            and progress.get("extracted_images") == COUNT and progress.get("archive_images") == COUNT
            and progress.get("phase") in {"index_check", "complete"}):
        raise ValueError("COCO original archive has not completed verified safe extraction")


def compare_path_sets(physical, required, archive):
    if archive != required:
        raise ValueError("Original COCO archive differs from the frozen training index")
    if physical != required:
        raise ValueError(f"COCO physical directory differs: missing={len(required-physical)}, extra={len(physical-required)}")


def main():
    record = dict(started_utc=now(), passed=False, phase="verified_extraction_gate")
    atomic_json(RESULT, record)
    try:
        progress = load(worker_path("coco"))
        validate_extraction(progress)
        frozen = requirements()["coco"]
        if len(frozen) != COUNT or any(str(PurePosixPath(path).parent) != "coco/train2017" for path in frozen):
            raise ValueError("Frozen COCO training paths differ from original train2017 protocol")
        required_names = {PurePosixPath(path).name for path in frozen}
        with zipfile.ZipFile(progress["archive"]) as archive:
            members = audit_zip_structure(archive, "coco", COUNT)
            archive_names = {PurePosixPath(member.filename).name for member in members}
        record.update(phase="physical_directory_name_set", required_images=COUNT)
        atomic_json(RESULT, record)
        with os.scandir(target_root_images("coco")) as entries:
            physical_names = {entry.name for entry in entries}
        compare_path_sets(physical_names, required_names, archive_names)
        record.update(passed=True, phase="complete", finished_utc=now(),
            physical_image_count=len(physical_names), recovered_required_images=len(required_names),
            index_missing=0, original_sha256=progress["actual_sha256"], zip_integrity=progress["zip_integrity"],
            extraction_worker_evidence=str(worker_path("coco")),
            method="Exact physical directory entry set = verified original ZIP file set = frozen index; regular original files proven by all successful atomic extraction writes",
            decode_executed=False, per_file_stat_scan_replaces_decode=False)
        atomic_json(RESULT, record)
        publish_family_completeness("coco", dict(progress, completed=True, physical_image_count=COUNT,
            recovered_required_images=COUNT, index_missing=0, independent_audit_evidence=str(RESULT)))
        print(json.dumps(record), flush=True)
    except Exception as error:
        record.update(passed=False, phase="failed", error=str(error), finished_utc=now())
        atomic_json(RESULT, record)
        raise


if __name__ == "__main__":
    main()
