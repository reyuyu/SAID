import pytest

import coco_index_audit as audit


def test_requires_actual_archive_identity_and_completed_extraction():
    progress = dict(sha256_passed=True, actual_sha256=audit.SHA, actual_size_bytes=19336861798,
                    zip_integrity_passed=True, extracted_images=audit.COUNT, archive_images=audit.COUNT,
                    phase="index_check")
    audit.validate_extraction(progress)
    for invalid in [dict(progress, extracted_images=98000), dict(progress, zip_integrity_passed=False),
                    dict(progress, actual_sha256="a"*64), dict(progress, phase="extracting")]:
        with pytest.raises(ValueError, match="verified safe extraction"):
            audit.validate_extraction(invalid)


def test_exact_physical_archive_and_frozen_index_sets_required():
    required = {"000000000009.jpg", "000000000025.jpg"}
    audit.compare_path_sets(set(required), required, set(required))
    with pytest.raises(ValueError, match="frozen training index"):
        audit.compare_path_sets(required, required, {"substitute.jpg"})
    with pytest.raises(ValueError, match="missing=1"):
        audit.compare_path_sets({"000000000009.jpg"}, required, required)
    with pytest.raises(ValueError, match="extra=1"):
        audit.compare_path_sets(required | {"untracked.jpg"}, required, required)
