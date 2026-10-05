import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import sa1b_000014_forensics as forensic


def test_prefix_samples_preserve_encoded_bytes_without_accepting_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(forensic, "DIRECTORY", tmp_path)
    payload = io.BytesIO()
    original = b"untouched original encoded image bytes"
    with tarfile.open(fileobj=payload, mode="w") as archive:
        member = tarfile.TarInfo("./sa_160259.jpg")
        member.size = len(original)
        archive.addfile(member, io.BytesIO(original))
    result = forensic.inspect_prefix(gzip.compress(payload.getvalue()))
    assert result["compression"] == "gzip" and not result["whole_archive_integrity_passed"]
    assert result["samples"][0]["sha256"] == hashlib.sha256(original).hexdigest()
    assert (tmp_path/"original-prefix-samples/sa_160259.jpg").read_bytes() == original


def test_compressed_size_modulo512_does_not_classify_compression():
    assert forensic.SIZE % 512 == 156
    assert forensic.compression_signature(b"\x1f\x8banything") == "gzip"
    assert forensic.compression_signature(b"plain tar bytes") == "uncompressed_or_unknown"


def test_missing_full_archive_never_runs_tools_or_claims_integrity(tmp_path, monkeypatch):
    monkeypatch.setattr(forensic.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("No full archive exists")))
    results = forensic.local_diagnostics(tmp_path/"sa_000014.tar")
    assert len(results) == 5
    assert all(row["status"] == "NOT_RUN_ORIGINAL_ARCHIVE_MISSING" for row in results)


def test_main_defines_missing_path_evidence_before_namespace_publication(tmp_path, monkeypatch):
    huggingface_hub=SimpleNamespace(get_hf_file_metadata=lambda *args,**kwargs:None,
                                   hf_hub_url=lambda *args,**kwargs:"https://example.invalid/asset")
    monkeypatch.setitem(sys.modules,"huggingface_hub",huggingface_hub)
    monkeypatch.setitem(sys.modules,"requests",SimpleNamespace(Session=lambda:None))

    evidence = tmp_path/"evidence"
    (evidence/"sa1b-shards").mkdir(parents=True)
    monkeypatch.setattr(forensic,"EVIDENCE",evidence)
    monkeypatch.setattr(forensic,"RECOVERY",tmp_path)
    monkeypatch.setattr(forensic,"RESULT",evidence/"forensics.json")
    monkeypatch.setattr(forensic,"DIRECTORY",evidence/"fragments")
    monkeypatch.setattr(forensic,"STATE",tmp_path/"state.json")
    monkeypatch.setattr(forensic,"SAM_ROOT",tmp_path/"images")
    rows=[{} for position in range(16)]
    for position,image_id in [(13,1),(15,3)]:
        inventory=tmp_path/f"inventory{position}.jsonl"
        inventory.write_text(json.dumps(dict(basename=f"sa_{image_id}.jpg"))+"\n")
        rows[position]=dict(image_inventory=str(inventory),image_inventory_sha256=forensic.digest(inventory))
    rows[14]=dict(archive=str(tmp_path/"missing.tar"),repo_path="sa_000014.tar")
    forensic.STATE.write_text(json.dumps(dict(revision="fixed",endpoint="https://example.invalid",shards=rows)))
    (evidence/"sa1b-shards/sa_000014.tar.tar-members.txt").write_text("./sa_2.jpg\n")
    monkeypatch.setattr(forensic,"requirements",lambda:dict(sam={"sam/images/sa_2.jpg"}))
    monkeypatch.setattr(forensic,"configure_client",lambda endpoint:None)
    called=[]

    def namespace(state,families,record):
        assert record["missing_required_images"]==1
        assert Path(record["missing_ids_path"]).read_text()=="2\n"
        called.append(True)
        return dict(passed=False,reason="isolated unit fixture")

    monkeypatch.setattr(forensic,"audit_sam_namespace",namespace)
    monkeypatch.setattr(huggingface_hub,"get_hf_file_metadata",lambda *args,**kwargs:(_ for _ in ()).throw(RuntimeError("offline fixture")))
    forensic.main()
    record=json.loads(forensic.RESULT.read_text())
    assert called==[True] and record["missing_required_images"]==1
    assert not record["archive_accepted"] and record["remote_fragments"]["status"]=="failed"
