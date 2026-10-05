"""Repair only frozen missing JPEG IDs after the200-reference source gate."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from PIL import Image
import torch

import sa1b_recovery as recovery
import sa1b_jpeg_rescue as rescue
import sa1b_modelscope_equivalence as equivalent
import sa1b_modelscope_source as source


FOLDER = equivalent.FOLDER
CACHE = Path("/root/.cache/said-recovery/modelscope-required")
SHARDS = (14, 16, 17)


def require_equivalence(audit):
    rows = audit.get("samples", [])
    if not (audit.get("passed") and audit.get("compared", 0) >= 200 and len(rows) == audit["compared"]
            and len({row["image_id"] for row in rows}) == len(rows)):
        raise ValueError("ModelScope source has not passed200 distinct reference comparisons")
    for row in rows:
        if not (row.get("classification") in equivalent.ALLOWED and row.get("passed")
            and row.get("id_basename_equal") and row.get("dimensions_equal")
            and row.get("rgb_pixels_exact_equal") and row.get("preprocess_tensor_exact_equal")
            and row.get("source_repository") == source.REPO and row.get("source_revision") == source.REVISION):
            raise ValueError("A failed or unrelated source comparison closes the repair gate")


def required_plan():
    original = recovery.load(FOLDER / "initial-missing-required-ids.json")
    if original["training_index_sha256"] != recovery.RECORDS_SHA:
        raise ValueError("Missing-ID snapshot differs from the frozen index")
    plan = {position: list(original["shards"][f"sa_{position:06d}.tar"]) for position in SHARDS}
    flat = [image_id for values in plan.values() for image_id in values]
    if len(flat) != len(set(flat)) or len(flat) != 11174:
        raise ValueError("Initial repair scope must be exactly11174 distinct missing required IDs")
    return plan


def fetch_and_install(image_id, preprocess, gate_sha):
    name = f"sa_{image_id}.jpg"
    url = source.FULL_IMAGE_PREFIX + name
    path = CACHE / name
    receipt_path = CACHE / (name + ".json")
    cached = recovery.load(receipt_path, {})
    if cached.get("installed"):
        target = recovery.SAM_ROOT / name
        if target.is_file() and recovery.digest(target) == cached["sha256"]:
            return cached
    if cached.get("http_status") in (403, 404):
        return cached
    priority = recovery.load(FOLDER / "priority-seven" / (name + ".json"), {})
    if priority.get("http_status") in (403, 404):
        result = dict(priority, installed=False, checked_utc=recovery.now(), equivalence_audit_sha256=gate_sha)
        recovery.atomic_json(receipt_path, result)
        return result
    result = dict(image_id=image_id, basename=name, source_url=url, source_path=str(path),
        source_repository=source.REPO, source_revision=source.REVISION, source_field="opensource_url",
        equivalence_audit_sha256=gate_sha, installed=False, image_transform="NONE")
    try:
        if not path.is_file() and priority.get("downloaded"):
            import shutil

            shutil.copyfile(priority["source_path"], path)
            if recovery.digest(path) != priority["sha256"]:
                raise ValueError("Priority source bytes changed during the copy")
        if not path.is_file():
            for attempt in range(3):
                try:
                    with urlopen(url, timeout=30) as response:
                        content = response.read(128 * 1024 * 1024 + 1)
                        if len(content) > 128 * 1024 * 1024 or not content.startswith(b"\xff\xd8"):
                            raise ValueError("Source object is not a bounded original JPEG")
                        result.update(http_status=response.status, etag=response.headers.get("ETag"), attempts=attempt+1)
                    temporary = path.with_suffix(".jpg.download")
                    temporary.write_bytes(content)
                    temporary.replace(path)
                    if recovery.digest(path) != hashlib.sha256(content).hexdigest():
                        raise ValueError("Downloaded source bytes changed during persistence")
                    break
                except HTTPError as error:
                    result["http_status"] = error.code
                    if error.code in (401, 403, 404) or attempt == 2:
                        raise
                    time.sleep(attempt+1)
                except (URLError, TimeoutError):
                    if attempt == 2:
                        raise
                    time.sleep(attempt+1)
        checked = rescue.check_jpeg(path)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            tensor = preprocess(rgb)
            if tuple(tensor.shape) != (3, 224, 224) or not torch.isfinite(tensor).all():
                raise ValueError("Native training image preprocessing failed")
            checked.update(rgb_pixels_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                preprocess_tensor_sha256=equivalent.tensor_sha(tensor), preprocess_passed=True)
        checked.update(source_repository=source.REPO, source_revision=source.REVISION, source_url=url,
            equivalence_audit_sha256=gate_sha,
            assurance="Source accepted by200-reference gate; missing image has no intact original for an individual comparison")
        inventory = {}
        rescue.install(path, name, inventory, "ModelScope original full-image opensource_url", verified=checked)
        result.update(inventory[name], installed=True, http_status=result.get("http_status", 200))
    except (OSError, ValueError, URLError, TimeoutError, HTTPException) as error:
        result.update(error_type=type(error).__name__, os_errno=getattr(error, "errno", None))
    result["checked_utc"] = recovery.now()
    recovery.atomic_json(receipt_path, result)
    return result


def publish_proofs(results, plan, gate_sha):
    installed = {row["image_id"]: row for row in results if row.get("installed")}
    for position in SHARDS:
        proof_path = rescue.FOLDER / f"sa_{position:06d}" / "result.json"
        proof = recovery.load(proof_path)
        inventory = {row["basename"]: row for row in map(json.loads, Path(proof["image_inventory"]).read_text().splitlines())}
        remaining = set(proof["missing_required_image_ids"])
        changed = []
        for image_id in remaining.copy():
            if image_id in installed and image_id in plan[position]:
                row = dict(installed[image_id])
                inventory[row["basename"]] = row
                remaining.remove(image_id)
                changed.append(image_id)
        if not changed:
            continue
        content = "".join(json.dumps(inventory[name], ensure_ascii=False)+"\n" for name in sorted(inventory))
        content_sha = hashlib.sha256(content.encode()).hexdigest()
        folder = FOLDER / "installed-inventories" / f"sa_{position:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"images-{len(inventory)}-{content_sha[:16]}.jsonl"
        if not target.exists():
            temporary = target.with_suffix(".jsonl.tmp")
            temporary.write_text(content)
            temporary.replace(target)
        if recovery.digest(target) != content_sha:
            raise ValueError("Installed-image inventory byte digest mismatch")
        proof.setdefault("original_rescue_inventory", dict(path=proof["image_inventory"], sha256=proof["image_inventory_sha256"]))
        repaired = proof.get("models_scope_repair", {})
        repaired_ids = sorted(set(repaired.get("installed_ids", [])) | set(changed))
        proof.update(image_inventory=str(target), image_inventory_sha256=content_sha,
            recovered_required_images=len(inventory), missing_required_image_ids=sorted(remaining),
            status="TRAINING_IMAGES_FULLY_RECOVERED" if not remaining else "REQUIRED_JPEGS_STILL_MISSING",
            models_scope_repair=dict(installed_ids=repaired_ids, repo=source.REPO, revision=source.REVISION,
                equivalence_audit=str(FOLDER / "equivalence-audit.json"), equivalence_audit_sha256=gate_sha),
            last_repaired_utc=recovery.now())
        if len(inventory)+len(remaining) != proof["required_images"]:
            raise ValueError("Required-image inventory accounting changed")
        recovery.atomic_json(proof_path, proof)
    proofs = [recovery.load(path) for path in rescue.FOLDER.glob("sa_*/result.json")]
    rescue.write_reports(sorted(proofs, key=lambda row: row["filename"]))


def write_progress(results, total, started, finished=False):
    installed = sum(row.get("installed", False) for row in results)
    status = dict(checked_utc=recovery.now(), worker_pid=os.getpid(), running=not finished, requested=total,
        processed=len(results), installed=installed, failed=len(results)-installed,
        http_status_counts=dict(Counter(str(row.get("http_status")) for row in results)),
        elapsed_seconds=time.monotonic()-started, source_repository=source.REPO,
        source_revision=source.REVISION, source_equivalence="200/200 BITWISE_EQUIVALENT",
        smoke_started=False, formal_training_started=False)
    recovery.atomic_json(FOLDER / "repair-state.json", status)
    (recovery.RECOVERY / "MODELSCOPE_SA1B_REPAIR.md").write_text("\n".join([
        "# ModelScope required-JPEG repair", "", f"Updated: {status['checked_utc']}.",
        "Source equivalence:200/200 BITWISE_EQUIVALENT, also RGB/native-preprocess exact equal.",
        f"Only the frozen11174 missing required IDs are in scope. Processed: {len(results)}; installed: {installed}; failed: {len(results)-installed}.",
        f"HTTP statuses: {json.dumps(status['http_status_counts'])}.",
        "Original basename and downloaded encoded bytes are preserved; no resize/reencoding/format conversion.",
        "All installations independently pass JPEG decode, RGB hashing, native training preprocessing and copied-byte verification.",
        "Source accepted by population equivalence audit; no per-missing-image original-byte equality is falsely asserted.",
        "Per-ID receipts and original bytes: /root/.cache/said-recovery/modelscope-required/; durable provenance: evidence/modelscope-sa1b/repair-results.jsonl.",
        "Current exact unrecovered IDs: MISSING_REQUIRED_SA_IMAGES.json.",
        "Normal HF/ODL recovery is not stopped or reassigned. Full decode/smoke gates remain mandatory; no formal training.", ""]))
    return status


def run():
    FOLDER.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / "repair.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit_path = FOLDER / "equivalence-audit.json"
        gate = recovery.load(audit_path)
        require_equivalence(gate)
        gate_sha = recovery.digest(audit_path)
        torch.set_num_threads(1)
        preprocess, provenance = equivalent.native_preprocess()
        if provenance["source_sha256"] != gate["preprocess_provenance"]["source_sha256"]:
            raise ValueError("Training image transform changed after the equivalence audit")
        plan = required_plan()
        required = {int(Path(path).stem[3:]) for path in recovery.requirements()["sam"]}
        if not {image_id for values in plan.values() for image_id in values}.issubset(required):
            raise ValueError("Repair plan contains images outside the frozen training index")
        ordered = plan[16] + plan[17] + plan[14]
        results = []
        started = time.monotonic()
        with (FOLDER / "repair-results.jsonl").open("a") as stream:
            for batch in (ordered[:7], ordered[7:]):
                with ThreadPoolExecutor(max_workers=8) as pool:
                    futures = [pool.submit(fetch_and_install, image_id, preprocess, gate_sha) for image_id in batch]
                    for future in as_completed(futures):
                        row = future.result()
                        results.append(row)
                        stream.write(json.dumps(row, ensure_ascii=False)+"\n")
                        if len(results) % 128 == 0:
                            stream.flush()
                            if recovery.digest(audit_path) != gate_sha:
                                raise ValueError("Equivalence audit was modified during repair")
                            publish_proofs(results, plan, gate_sha)
                            status = write_progress(results, len(ordered), started)
                            print("processed", len(results), "installed", status["installed"], "failed", status["failed"], flush=True)
                stream.flush()
                publish_proofs(results, plan, gate_sha)
                write_progress(results, len(ordered), started)
        final = write_progress(results, len(ordered), started, finished=True)
        missing = recovery.load(recovery.RECOVERY / "MISSING_REQUIRED_SA_IMAGES.json")
        final.update(remaining_quarantined_required_images=missing["missing_count"],
                     resume_status="NOT_READY_TO_START_S02_FULL")
        recovery.atomic_json(FOLDER / "repair-state.json", final)
        if missing["missing_count"] == 0:
            wait_for_final_gates()
        else:
            print("NOT_READY_TO_START_S02_FULL", "remaining quarantined JPEGs", missing["missing_count"], flush=True)


def wait_for_final_gates():
    live_path = recovery.EVIDENCE / "sa1b-fast-recovery/state.json"
    while True:
        live = recovery.load(live_path, {})
        active = False
        if live.get("pid"):
            try:
                os.kill(live["pid"], 0)
                active = True
            except ProcessLookupError:
                pass
        if live.get("supervisor_running") is False or not active:
            break
        time.sleep(60)
    subprocess.run([str(recovery.ROOT / ".venv/bin/python"), str(recovery.RECOVERY / "sa1b_recovery.py"), "audit"], check=True)
    completeness = recovery.load(recovery.RECOVERY / "TRAIN_IMAGE_COMPLETENESS.json")
    if not completeness.get("passed"):
        return
    subprocess.run([str(recovery.ROOT / ".venv/bin/python"), str(recovery.RECOVERY / "audit.py"), "training"], check=True)
    policy_path = recovery.EVIDENCE / "recovery-operation-policy.json"
    policy = recovery.load(policy_path)
    if not policy.get("conditional_smoke_authorized") or policy.get("maximum_smoke_updates") != 5 or policy.get("scheduler_horizon") != 4868:
        raise ValueError("Five-update smoke is not authorized by the current operation policy")
    policy.update(smoke_authorized=True, reason="All required original images and full decode now pass; only five updates authorized")
    recovery.atomic_json(policy_path, policy)
    environment = dict(os.environ)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    subprocess.run([str(recovery.ROOT / ".venv/bin/python"), str(recovery.RECOVERY / "validate.py"), "smoke"], check=True, env=environment)
    if recovery.load(recovery.EVIDENCE / "smoke-audit.json", {}).get("passed") and recovery.load(recovery.EVIDENCE / "export-audit.json", {}).get("passed"):
        state = recovery.load(FOLDER / "repair-state.json")
        state.update(resume_status="READY_TO_START_S02_FULL", smoke_started=True, smoke_passed=True,
                     formal_training_started=False, checked_utc=recovery.now())
        recovery.atomic_json(FOLDER / "repair-state.json", state)
        print("READY_TO_START_S02_FULL", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.parse_args()
    run()
