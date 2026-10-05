"""Audit the documented Meta source without touching active recovery queues."""

from concurrent.futures import ThreadPoolExecutor
import base64
import hashlib
import json
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

import sa1b_recovery as recovery


SHARDS = (14, 16, 17)
FOLDER = recovery.EVIDENCE / "meta-official-shards"
SMARTCLIP_COMMIT = "729c5a6acdaa0d51797c094095f5868cc5c4c7f8"
SMARTCLIP_URL = f"https://raw.githubusercontent.com/Mid-Push/SmartCLIP/{SMARTCLIP_COMMIT}/train/train.md"
SAM_README_URL = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/README.md"
DOWNLOAD_URL = "https://ai.meta.com/datasets/segment-anything-downloads/"
OFFICIAL_ROUTES = (DOWNLOAD_URL, "https://ai.meta.com/datasets/segment-anything/",
                   "https://ai.facebook.com/datasets/segment-anything-downloads/")


def redacted_url(url):
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def probe_route(url):
    result = dict(requested_url=redacted_url(url), checked_utc=recovery.now(),
                  permission_bypass_attempted=False, automatic_agreement_submission=False)
    try:
        with urlopen(url, timeout=20) as response:
            content = response.read()
            text = content.decode("utf-8", errors="replace")
            result.update(status_code=response.status, final_url=redacted_url(response.geturl()),
                content_bytes=len(content), content_sha256=hashlib.sha256(content).hexdigest(),
                content_type=response.headers.get("Content-Type"), reachable=True,
                successful_http_response=200 <= response.status < 300)
        title = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
        result["title"] = re.sub(r"\s+", " ", title.group(1))[:250] if title else None
        result["requested_shard_names_in_response"] = sorted(set(re.findall(r"sa_0000(?:14|16|17)\.tar", text)))
        result["gate_terms_observed"] = [term for term in ("login", "sign in", "agree", "license", "captcha")
                                          if term in text.lower()]
    except HTTPError as error:
        result.update(reachable=True, successful_http_response=False, status_code=error.code,
                      error_type=type(error).__name__)
    except (URLError, TimeoutError) as error:
        result.update(reachable=False, successful_http_response=False,
                      error_type=type(getattr(error, "reason", error)).__name__)
    return result


def source_document(raw_url, api_url):
    try:
        with urlopen(raw_url, timeout=20) as response:
            return response.read()
    except (URLError, TimeoutError):
        with urlopen(api_url, timeout=20) as response:
            payload = json.load(response)
        if payload.get("encoding") != "base64":
            raise ValueError("Official GitHub source response lacks base64 file content")
        return base64.b64decode(payload["content"])


def historical_comparison(state):
    rows = {row["filename"]: row for row in state["shards"]}
    results = []
    for position in SHARDS:
        name = f"sa_{position:06d}.tar"
        row = rows[name]
        proof_path = recovery.EVIDENCE / "sa1b-jpeg-rescue" / f"sa_{position:06d}" / "result.json"
        proof = recovery.load(proof_path, row.get("required_jpeg_rescue", {}))
        if proof and proof["training_index_sha256"] != recovery.RECORDS_SHA:
            raise ValueError("Rescue proof does not match the frozen index: " + name)
        failure_path = Path(row.get("tar_stderr", "/nonexistent"))
        historical = dict(size_bytes=row.get("size_bytes", row["expected_size_bytes"]),
            md5=row.get("observed_md5"), sha256=row.get("observed_sha256"),
            checklist_md5=row["expected_checklist_checksum"], lfs_sha256=row["expected_lfs_sha256"],
            tar_integrity_passed=row.get("tar_integrity_passed"),
            archive_failure_information=row.get("quarantine_reason"),
            gnu_tar_stderr_path=str(failure_path) if failure_path.is_file() else None,
            original_archive_present=Path(row["archive"]).is_file(), source="Existing HF/ODL historical records only")
        results.append(dict(filename=name, historical_hf_odl=historical,
            meta_official=dict(downloaded=False, size_bytes=None, md5=None, sha256=None,
                gnu_tar_passed=None, python_tarfile_passed=None, jpeg_member_count=None,
                required_jpeg_count=None, jpeg_decode_passed=None),
            bitwise_same=None, classification="META_OFFICIAL_COMPARISON_BLOCKED",
            required_images=proof.get("required_images"),
            recovered_required_images=proof.get("recovered_required_images"),
            missing_required_images=len(proof["missing_required_image_ids"]) if proof else None,
            missing_required_image_ids=proof.get("missing_required_image_ids"),
            existing_rescue_remaining_decode_failures=proof.get("remaining_decode_failures"),
            existing_rescue_evidence=str(proof_path)))
    return results


def report(audit):
    lines = ["# Meta official SA-1B audit", "", f"Checked UTC: {audit['checked_utc']}.", "",
        "## Documented official entry", "",
        f"SmartCLIP source: `{SMARTCLIP_URL}`.",
        f"Pinned source commit: `{SMARTCLIP_COMMIT}`.",
        f"SmartCLIP source SHA256: `{audit['smartclip_source_sha256']}`.",
        f"Documented Meta download page: `{DOWNLOAD_URL}`.",
        "SmartCLIP explicitly limits SAM training images to shards000000 through000050 and warns against resizing.",
        "The official facebookresearch/segment-anything README independently points to the ai.facebook.com download-page alias.",
        "", "## Access results", "",
        "| Official route | HTTP status | Result |", "|---|---|---|"]
    for result in audit["official_route_probes"]:
        status = result.get("status_code", "NOT RECEIVED")
        outcome = result.get("error_type", "HTTP response received; browser/terms/download links require inspection")
        lines.append(f"| {result['requested_url']} | {status} | {outcome} |")
    lines.extend(["", "No Meta official shard URL was obtained; no Meta archive has been downloaded.",
        "Connection failures do not establish whether login, agreement acceptance, or a signed URL is currently required.",
        "No permission bypass, login automation, agreement submission, guessed CDN object fetch, or HF/ODL object download was attempted.",
        "", "## Three-source comparison", "",
        "| Shard | Meta SHA256 | HF/ODL recorded SHA256 | Bitwise same? | Meta tar valid? | Existing required JPEGs recovered |",
        "|---|---|---|---|---|---|",])
    for row in audit["shards"]:
        lines.append(f"| {row['filename']} | NOT OBTAINED | `{row['historical_hf_odl']['sha256']}` | UNKNOWN | NOT TESTED | "
                     f"{row['recovered_required_images']}/{row['required_images']} |")
    lines.extend(["", "## Historical size / checksum / archive failures", ""])
    for row in audit["shards"]:
        old = row["historical_hf_odl"]
        lines.extend([f"### {row['filename']}", "", f"Size bytes: {old['size_bytes']}.",
            f"Recorded MD5: `{old['md5']}`; checklist MD5: `{old['checklist_md5']}`.",
            f"Recorded SHA256: `{old['sha256']}`; LFS SHA256: `{old['lfs_sha256']}`.",
            f"Historical archive integrity passed: {old['tar_integrity_passed']}.",
            f"Historical GNU stderr: `{old['gnu_tar_stderr_path']}`.",
            f"Original historical archive currently present: {old['original_archive_present']}.",
            f"Still missing required JPEGs: {row['missing_required_images']}.",
            f"Remaining decode failures among recovered JPEGs: {row['existing_rescue_remaining_decode_failures']}.", ""])
    lines.extend(["## Decision", "",
        "META_OFFICIAL_COMPARISON_BLOCKED. No caseA/B conclusion is possible without actual Meta bytes.",
        "Neither MIRROR_OBJECT_DIVERGENCE_CONFIRMED nor UPSTREAM_ARCHIVE_ANOMALY_CONFIRMED is asserted.",
        "All three requested shards still lack required JPEGs; the abnormal-shard blocker is not removed.",
        "Exact remaining image IDs are in MISSING_REQUIRED_SA_IMAGES.json and this directory's audit.json.",
        "Shard000014's full historical archive had already disappeared before the preceding rescue; only one surviving original JPEG was recovered.",
        "", "## Required user action", "",
        f"Open `{DOWNLOAD_URL}` in your browser. If prompted, log in and personally read/accept the dataset license.",
        "Select only sa_000014.tar, sa_000016.tar, and sa_000017.tar; provide the generated official download URLs or place the downloaded files in this audit directory.",
        "Signed URLs are credentials: do not commit them or include them in public reports. Expired URLs need to be regenerated.",
        "If your browser also cannot reach the official page, report that result; a login/access requirement must not be inferred from a network timeout.",
        "", "## Recovery isolation", "",
        "Existing normal HF/ODL supervisors remain running; neither their source plan nor archives were modified.",
        "No images were newly installed by this audit, so live completeness and shard state remain owned by the existing recovery supervisor.",
        "No smoke or training was started. recovery-operation-policy.json explicitly disables the smoke launcher, including a future automatic call by the existing supervisor.",
        f"Current full training_index_missing snapshot: {audit['training_index_missing']} (normal background recovery may change this).", ""])
    (recovery.RECOVERY / "META_OFFICIAL_SA1B_AUDIT.md").write_text("\n".join(lines))


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    source = source_document(SMARTCLIP_URL,
        f"https://api.github.com/repos/Mid-Push/SmartCLIP/contents/train/train.md?ref={SMARTCLIP_COMMIT}")
    if DOWNLOAD_URL.encode() not in source or b"000000~000050.tar" not in source:
        raise ValueError("Pinned SmartCLIP training document does not establish the expected official source")
    (FOLDER / "smartclip-train.md").write_bytes(source)
    readme = source_document(SAM_README_URL,
        "https://api.github.com/repos/facebookresearch/segment-anything/contents/README.md?ref=main")
    if OFFICIAL_ROUTES[2].encode() not in readme:
        raise ValueError("Official Segment Anything README does not establish the expected official alias")
    (FOLDER / "segment-anything-readme.md").write_bytes(readme)
    with ThreadPoolExecutor(max_workers=3) as pool:
        probes = list(pool.map(probe_route, OFFICIAL_ROUTES))
    if any(probe.get("successful_http_response") for probe in probes):
        raise RuntimeError("An official route is accessible; inspect its terms and original shard links before claiming a network blocker")
    state = recovery.load(recovery.STATE)
    completeness = recovery.load(recovery.RECOVERY / "TRAIN_IMAGE_COMPLETENESS.json")
    audit = dict(checked_utc=recovery.now(), smartclip_commit=SMARTCLIP_COMMIT,
        smartclip_source_url=SMARTCLIP_URL, smartclip_source_sha256=hashlib.sha256(source).hexdigest(),
        official_sam_readme_url=SAM_README_URL, official_sam_readme_sha256=hashlib.sha256(readme).hexdigest(),
        official_download_page=DOWNLOAD_URL, official_route_probes=probes,
        classification="META_OFFICIAL_COMPARISON_BLOCKED", shards=historical_comparison(state),
        training_index_sha256=recovery.RECORDS_SHA, training_index_missing=completeness["training_index_missing"],
        new_images_installed=0, smoke_started=False, formal_training_started=False,
        historical_mirror_download_attempted=False, existing_supervisors_modified=False)
    recovery.atomic_json(FOLDER / "audit.json", audit)
    report(audit)
    print(json.dumps({"classification": audit["classification"],
        "missing_required_images": {row["filename"]: row["missing_required_images"] for row in audit["shards"]},
        "report": str(recovery.RECOVERY / "META_OFFICIAL_SA1B_AUDIT.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
