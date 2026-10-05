"""Audit recovered assets without starting training or changing historical code."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RECOVERY = ROOT / "recovery"
ASSETS = ROOT / "local_assets"
RUNTIME = ROOT / "runtime" / "SAID-nest-clip-v1"
EVIDENCE = RECOVERY / "evidence"
PINS = {
    "canonical": ("codex/nest-balanced-four-epoch-v1", "14653c92c6da9d552a2b624ab169eaaa275cdde8"),
    "dose": ("codex/nest-balanced-armb-summary-dose-500-v1", "00c088fe83a9529c25c83316013d82aea24aa607"),
    "gradient_reports": ("codex/nest-balanced-gradient-audit-v1", "246566fbe618daf36e54e809ea79e25515c0e3b7"),
    "research": ("codex/nest-balanced-armb-summary02-4epoch-v1", "52bb7ae2d77aae8c2b1f69877aed5f7cc5d85b18"),
}
ANNOTATION_SHA = "5c5f0f4ee58d7b7467f9e49eb5b17f930890a8a0c18a4e2a5be6b15714ef8b3c"
RECORDS_SHA = "0fed1fe12b625ba1f8e762b3545eb115ba084fb314743a74f9e90103e57a27c8"
CLIP_SHA = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
STEP0_SHA = "54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6"
ANNOTATION = ASSETS / "training/ShareGPT4V/share-captioner_coco_lcs_sam_1246k_1107.json"
PROTOCOLS = {
    "flickr30k_test1k": ("flickr30k_test1k.jsonl", "flickr30k", 1000, 5000, "113dbc616ca66db9400107ed4b97b56d33adae3c18608225902d97ee943600dc"),
    "docci_test": ("docci_test.jsonl", "docci", 5000, 5000, "e852a96b4efb9fa6585fd70b2686cb4e456409b1144c4a0e3c7bd8a24a36cb11"),
    "long_dci_reconstructed": ("long_dci_reconstructed.jsonl", "dci", 7602, 7602, "8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b"),
}


def digest(path):
    state = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            state.update(chunk)
    return state.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def git(*arguments):
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments])


def asset_manifest():
    assets = [
        dict(id="annotation", path=str(ANNOTATION), sha256=ANNOTATION_SHA, original_records=1246901,
             training_records=1245901, skip=1000, source="Lin-Chen/ShareGPT4V", urls=[
                 "https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/share-captioner_coco_lcs_sam_1246k_1107.json",
                 "https://hf-mirror.com/datasets/Lin-Chen/ShareGPT4V/resolve/main/share-captioner_coco_lcs_sam_1246k_1107.json"]),
        dict(id="llava_images", path=str(ASSETS / "training/ShareGPT4V/llava/llava_pretrain/images"),
             source="ShareGPT4V docs/Data.md; liuhaotian/LLaVA-Pretrain images.zip", sha256=None,
             urls=["https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip",
                   "https://hf-mirror.com/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip"],
             validation="Every path referenced by the frozen annotation; preserve original archive pixels."),
        dict(id="coco_train2017", path=str(ASSETS / "training/ShareGPT4V/coco/train2017"), sha256=None,
             source="Official COCO train2017 archive", urls=["https://images.cocodataset.org/zips/train2017.zip"],
             validation="Every path referenced by the frozen annotation, not merely archive image count."),
        dict(id="sam_training", path=str(ASSETS / "training/ShareGPT4V/sam/images"), sha256=None,
             source="Meta SA-1B; ShareGPT4V uses sa_000000.tar through sa_000050.tar inclusive", urls=[],
             source_page="https://ai.meta.com/datasets/segment-anything-downloads/",
             access="Obtain publisher-authorized SA-1B archive URLs; do not bypass license/access controls.",
             validation="Frozen annotation determines exact required IDs; do not substitute SAM9K or another subset."),
        dict(id="coco_val2017", path=str(ASSETS / "evaluation/coco/val2017"), sha256=None, n_images=5000,
             n_captions=25000, urls=["https://images.cocodataset.org/zips/val2017.zip"],
             source="Official COCO; sorted CocoCaptions IDs, first five original-order captions per image"),
        dict(id="coco_annotations", path=str(ASSETS / "evaluation/coco/annotations/captions_val2017.json"),
             sha256=None, urls=["https://images.cocodataset.org/annotations/annotations_trainval2017.zip"],
             source="Official COCO captions; 25014 source captions are NOT the 25000-caption candidate pool"),
        dict(id="urban1k", path=str(ASSETS / "evaluation/Urban1k/Urban1k"),
             archive_sha256="08e42b3fada77abf7f890a087ed6e9f9fbba1dc143f9f09801ed83187ae617b8", n_images=1000,
             n_captions=1000, source="BeichenZhang/Urban1k; historical docs/said_cls_cvssl/urban1k_500step_report.md",
             revision="953fcf1d1a3bde031d9424a91a0154e594b5a586",
             urls=["https://hf-mirror.com/datasets/BeichenZhang/Urban1k/resolve/953fcf1d1a3bde031d9424a91a0154e594b5a586/Urban1k.zip"],
             validation="Original image/ and caption/ stem pairing, first caption line, full diagonal pool"),
        dict(id="docci_annotations", path=str(ASSETS / "retrieval_benchmarks/docci/docci_descriptions.jsonlines"),
             sha256=None, urls=["https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines"],
             source="Google DOCCI release; repository parse_docci, test only"),
        dict(id="docci_images", path=str(ASSETS / "retrieval_benchmarks/docci/images"),
             archive_sha256="6a4f4f0ecc74e454702ffe6a24d5a32cf826fbeddc0aaf13cced1052d5befe72",
             source="Historical qihoo360/DOCCI-CN mirror; filenames must match original DOCCI manifest",
             urls=["https://hf-mirror.com/datasets/qihoo360/DOCCI-CN/resolve/main/images.zip"]),
        dict(id="dci_annotations", path=str(ASSETS / "retrieval_benchmarks/dci/annotations"),
             archive_sha256="d865c244150168d3f25daaad0bf5b70b2123cf3e83ca7b0e207d5b26943c5fc7",
             source="Official DCI archive; docs/extended_retrieval/status.json corrects old script SHA typo",
             urls=["https://dl.fbaipublicfiles.com/densely_captioned_images/dci.tar.gz"],
             validation="Use tools.prepare_retrieval_benchmarks.reconstruct_long_dci, extra_caption only"),
        dict(id="dci_images", path=str(ASSETS / "retrieval_benchmarks/dci/images"),
             archive_sha256s=["9e3b6bbe78747c74a12192d9b54895e187f0a37b2e2d230df583260e791e7dee",
                             "1d686645bd7886ed9e7f5c9ea1b3441b9513dd30d6203a11c89edb18cd940186"],
             source="Historical qihoo360/DCI-CN image archives; original SA-1B images",
             urls=["https://hf-mirror.com/datasets/qihoo360/DCI-CN/resolve/main/images/images_1.zip",
                   "https://hf-mirror.com/datasets/qihoo360/DCI-CN/resolve/main/images/images_2.zip"]),
        dict(id="flickr30k_test1k", path=str(ASSETS / "retrieval_benchmarks/flickr30k/images"),
             archive_sha256="b0fa9970cc1680a9334818b9da151a62024772b72b961e1454eb6e9866dd7cee",
             source="Historical nlphuji/flickr_1k_test_image_text_retrieval, explicit test1K only",
             urls=["https://hf-mirror.com/datasets/nlphuji/flickr_1k_test_image_text_retrieval/resolve/main/images_flickr_1k_test.zip",
                   "https://hf-mirror.com/datasets/nlphuji/flickr_1k_test_image_text_retrieval/resolve/main/test_1k_flickr.csv"]),
        dict(id="openai_clip_b16", path=str(ASSETS / "weights/openai/ViT-B-16.pt"), sha256=CLIP_SHA,
             source="Official OpenAI CLIP ViT-B/16", urls=[
                 f"https://openaipublic.azureedge.net/clip/models/{CLIP_SHA}/ViT-B-16.pt"]),
        dict(id="common_step0", path=str(RUNTIME / "shared/step000000.pt"), sha256=STEP0_SHA, urls=[],
             source="Regenerate with original tools.nest_clip prepare; seed0; original random MaskNetwork",
             optimizer_state="empty; no updates"),
        dict(id="data_index", path=str(RUNTIME / "data_index/records.jsonl"), sha256=RECORDS_SHA, urls=[],
             source="Original train.nested_semantic_data.prepare_index, skip first 1000", training_records=1245901),
    ]
    annotation = load(EVIDENCE / "annotation-audit.json", {})
    groups = annotation.get("image_path_prefix_counts", {})
    hf_training = load(RECOVERY / "HF_TRAINING_ASSETS.json", {}).get("assets", {})
    for asset in assets:
        path = Path(asset["path"])
        asset["present"] = path.exists()
        if asset.get("sha256") and path.is_file():
            asset["observed_sha256"] = digest(path)
            asset["sha256_matches"] = asset["observed_sha256"] == asset["sha256"]
        if asset["id"] == "llava_images":
            asset["required_training_paths"] = sum(count for prefix, count in groups.items() if prefix.startswith("llava/"))
        elif asset["id"] == "coco_train2017":
            asset["required_training_paths"] = groups.get("coco/train2017")
        elif asset["id"] == "sam_training":
            asset.update(required_training_paths=groups.get("sam/images"),
                         required_id_min=annotation.get("sam_id_min"), required_id_max=annotation.get("sam_id_max"))
            mirror = load(RECOVERY / "SA1B_SHARDS.json", {})
            if mirror:
                asset.update(mirror_repo=mirror["repo"], mirror_revision=mirror["revision"],
                    checklist=mirror["checklist"], shard_assets=[dict(filename=row["filename"],
                    expected_md5_candidate=row["expected_checklist_checksum"], expected_sha256=row.get("expected_lfs_sha256"),
                    expected_size_bytes=row["expected_size_bytes"], archive_path=row["archive"],
                    source=f"https://huggingface.co/datasets/{mirror['repo']}/resolve/{mirror['revision']}/{row['repo_path']}",
                    download_status=row["download_status"], extraction_status=row["extraction_status"])
                    for row in mirror["shards"]])
        family = {"coco_train2017": "coco", "llava_images": "llava"}.get(asset["id"])
        if family in hf_training:
            mirror = hf_training[family]
            archive = ASSETS / "downloads/hf_training" / family / mirror["repo"].replace("/", "--") / mirror["revision"] / mirror["filename"]
            asset.update(mirror_repo=mirror["repo"], mirror_revision=mirror["revision"],
                         archive_sha256=mirror["expected_sha256"], archive_path=str(archive),
                         expected_archive_bytes=mirror["expected_size_bytes"],
                         original_urls=asset["urls"], source_history=mirror["source_history"],
                         download_status=mirror["status"],
                         urls=[f"https://huggingface.co/datasets/{mirror['repo']}/resolve/{mirror['revision']}/{mirror['filename']}"])
    for protocol, (name, folder, images, captions, expected) in PROTOCOLS.items():
        assets.append(dict(id=protocol + "_manifest", path=str(ASSETS / "retrieval_benchmarks/manifests" / name),
                           sha256=expected, n_images=images, n_captions=captions,
                           source="Original tools/prepare_retrieval_benchmarks.py; never regenerate with a replacement split"))
    return assets


def scan():
    remote_lines = git("for-each-ref", "--format=%(refname) %(objectname)", "refs/remotes/origin").decode().splitlines()
    remotes = dict(line.split() for line in remote_lines)
    assert git("rev-parse", "--is-shallow-repository").strip() == b"false"
    subprocess.run(["git", "-C", str(ROOT), "fsck", "--full"], check=True)
    pins = {}
    for role, (branch, commit) in PINS.items():
        assert git("cat-file", "-t", commit).strip() == b"commit"
        assert "refs/remotes/origin/" + branch in remotes
        pins[role] = dict(branch=branch, pinned_commit=commit, current_remote_tip=remotes["refs/remotes/origin/" + branch])
    blobs = {}
    for reference in remotes:
        for line in git("ls-tree", "-r", reference).decode().splitlines():
            descriptor, path = line.split("\t", 1)
            _, kind, blob = descriptor.split()
            if kind == "blob" and path.endswith((".json", ".md", ".py", ".txt")):
                blobs.setdefault(blob, []).append(dict(ref=reference, path=path))
    index = []
    paths = set()
    urls = set()
    identity_records = []

    def identities(value, field=""):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from identities(child, field + "/" + key)
        elif isinstance(value, list):
            for position, child in enumerate(value):
                if isinstance(child, dict):
                    yield from identities(child, field + "/" + str(position))
        elif isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
            yield dict(field=field, sha256=value)
        elif isinstance(value, int) and not isinstance(value, bool) and field.rsplit("/", 1)[-1] in (
            "n_images", "n_captions", "rows", "count", "training_records", "original_records", "skip", "horizon", "batches_per_epoch"):
            yield dict(field=field, count=value)
    batch = subprocess.Popen(["git", "-C", str(ROOT), "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for blob, sources in blobs.items():
            batch.stdin.write((blob + "\n").encode())
            batch.stdin.flush()
            header = batch.stdout.readline().decode().split()
            content = batch.stdout.read(int(header[2]))
            assert batch.stdout.read(1) == b"\n"
            text = content.decode(errors="replace")
            hashes = sorted(set(re.findall(r"\b[0-9a-f]{64}\b", text)))
            paths.update(re.findall(r"/(?:root|data|datasets|mnt|opt|home)/[^\s\"'<>]+", text))
            urls.update(re.findall(r"https?://[^\s\"'<>]+", text))
            index.append(dict(blob=blob, content_sha256=hashlib.sha256(content).hexdigest(),
                              sources=sources, recorded_sha256=hashes))
            if any(source["path"].endswith(".json") for source in sources):
                try:
                    payload = json.loads(text)
                    recorded = list(identities(payload))
                    if recorded:
                        identity_records.append(dict(blob=blob, sources=sources, records=recorded))
                except json.JSONDecodeError:
                    pass
    finally:
        batch.stdin.close()
        batch.wait(timeout=15)
    manifest = dict(schema_version=1, root=str(ROOT), formal_training_started=False,
                    pins=pins, remote_refs=remotes, full_clone_verified=True, source_index=index,
                    recorded_external_paths=sorted(paths), recorded_source_urls=sorted(urls), identity_records=identity_records,
                    assets=asset_manifest(), frozen_protocols={name: dict(manifest=file, n_images=images,
                    n_captions=captions, sha256=expected) for name, (file, folder, images, captions, expected) in PROTOCOLS.items()},
                    deferred="Historical trained checkpoints and external baselines are NOT download targets; retain their reports.",
                    scan=dict(remote_refs=len(remotes), unique_evidence_blobs=len(blobs), truncated=False))
    dump(RECOVERY / "RECOVERY_MANIFEST.json", manifest)
    dump(EVIDENCE / "git-audit.json", dict(passed=True, pins=pins, remote_refs=len(remotes),
         full_clone=True, shallow=False, fsck_passed=True))
    print(json.dumps(manifest["scan"]))


def configs():
    fields = ("arm experiment_name condition_mode fusion visual full_native_mix sampling_mode sampling_seed shuffle_seed "
              "world_size batch_size accumulation epochs seed workers checkpoint_encoders checkpoint_pair_blocks "
              "image_chunk text_chunk score_chunk checkpoint_interval save_initial_checkpoint monitor_resources "
              "feasibility_abort_seconds four_epoch_followup fusion_lr visual_mask_lr_scale view_weights "
              "sparsity_scale inclusion_max hparam_search trial_id base_model").split()
    sources = {
        "randomk": (PINS["canonical"][1], "experiments/nest_clip_v1/three_followup_v1/evidence/6d44ae8d5c34/formal-4868-config.json"),
        "summary02": (PINS["research"][1], "experiments/nest_clip_v1/armb_summary02_4epoch_v1/config.json"),
    }
    report = {}
    for name, (commit, path) in sources.items():
        original = json.loads(git("show", commit + ":" + path))
        configuration = {key: original[key] for key in fields if key in original}
        assert configuration["world_size"] == 4 and configuration["batch_size"] == 256
        assert configuration["accumulation"] == 1 and configuration["seed"] == 0
        assert configuration["workers"] == 8 and configuration["epochs"] == 4
        assert configuration["condition_mode"] == "dual_branch"
        assert configuration["fusion"] == "balanced_stack" and configuration["visual"] == "patch"
        assert configuration["fusion_lr"] == 2e-4 and configuration["visual_mask_lr_scale"] == 1
        assert configuration["sparsity_scale"] == 1 and configuration["inclusion_max"] == 1
        assert configuration["sampling_mode"] == ("random_k" if name == "randomk" else "summary_random_detail")
        assert configuration["view_weights"] == ([1, 1, 1] if name == "randomk" else [1.4, 0.2, 1.4])
        destination = RECOVERY / "configs" / (name + ".json")
        dump(destination, configuration)
        source_root = ROOT / "worktrees/randomk" if name == "randomk" else ROOT
        source_identity = {}
        for filename, expected in original.get("code_sha256", {}).items():
            observed = digest(source_root / filename)
            assert observed == expected, f"Historical source SHA mismatch: {filename}"
            source_identity[filename] = dict(historical_sha256=expected, observed_sha256=observed, matched=True)
        report[name] = dict(source_commit=commit, source_file=path, config=str(destination),
                            horizon=4868, updates_per_epoch=1217, copied_fields=fields,
                            historical_code_sha256_checks=source_identity,
                            excluded="Old absolute paths, lost resume checkpoint, runtime/optimizer/adapter evidence; not hyperparameters.")
    dump(EVIDENCE / "config-audit.json", dict(passed=True, configurations=report))
    print(json.dumps(report, indent=2))


def environment():
    import torch
    import torchvision

    frozen = {}
    for line in (ROOT / "docs/environment_freeze.txt").read_text().splitlines():
        if "==" in line:
            package, version = line.split("==", 1)
            frozen[package] = version
    comparisons = {}
    for package, expected in frozen.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        comparisons[package] = dict(historical=expected, recovered=actual, matches=actual == expected)
    nccl = list(torch.cuda.nccl.version()) if torch.cuda.is_available() else None
    clip_url = load(Path(importlib.metadata.distribution("clip").locate_file("clip-1.0.dist-info/direct_url.json")), {})
    result = dict(passed=all(item["matches"] for item in comparisons.values()) and torch.cuda.device_count() == 4,
                  python=sys.version, historical_python="3.10; patch release not established", torch=torch.__version__,
                  torchvision=torchvision.__version__, cuda=torch.version.cuda, nccl=nccl,
                  visible_gpus=torch.cuda.device_count(), comparisons=comparisons, clip_source=clip_url,
                  driver=subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).decode().splitlines(),
                  unrecorded_versions={package: importlib.metadata.version(package) for package in ["pip", "setuptools", "pytest"]})
    result["passed"] = result["passed"] and result["cuda"] == "12.4" and nccl == [2, 21, 5]
    result["passed"] = result["passed"] and clip_url.get("vcs_info", {}).get("commit_id") == "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
    dump(EVIDENCE / "environment-audit.json", result)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
    (EVIDENCE / "recovered-pip-freeze.txt").write_text(freeze)
    print(json.dumps({key: value for key, value in result.items() if key != "comparisons"}, indent=2))


def training():
    annotation = load(EVIDENCE / "annotation-audit.json", {})
    metadata = load(RUNTIME / "data_index/metadata.json", {})
    records = RUNTIME / "data_index/records.jsonl"
    result = dict(annotation=annotation.get("annotation_sha256") == ANNOTATION_SHA,
                  original_records=annotation.get("original_records"), training_records=annotation.get("training_records"),
                  skip=annotation.get("skip"), index_present=records.is_file(),
                  index_sha256=digest(records) if records.is_file() else None, metadata=metadata,
                  missing_images=0, missing_examples=[])
    root = ASSETS / "training/ShareGPT4V"
    if records.is_file():
        roots = {prefix: (root / prefix).is_dir() for prefix in annotation.get("image_path_prefix_counts", {})}
        for line in records.open():
            path = json.loads(line)["image"]
            prefix = str(Path(path).parent)
            if not roots.get(prefix, False) or not (root / path).is_file():
                result["missing_images"] += 1
                if len(result["missing_examples"]) < 8:
                    result["missing_examples"].append(path)
    else:
        result["missing_images"] = annotation.get("training_records")
    result["passed"] = (result["annotation"] and result["training_records"] == 1245901 and
                        result["skip"] == 1000 and result["index_sha256"] == RECORDS_SHA and result["missing_images"] == 0)
    dump(EVIDENCE / "training-audit.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "metadata"}, indent=2))


def evaluators():
    sys.path.insert(0, str(ROOT))
    from PIL import Image
    from tools.prepare_retrieval_benchmarks import integrity

    def verify_images(paths):
        failures = []
        missing = 0
        for path in paths:
            if not Path(path).is_file():
                missing += 1
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as error:
                failures.append(dict(path=str(path), error=str(error)))
        return dict(missing_images=missing, decode_failures=failures)

    results = {}
    for name, (filename, folder, images, captions, expected) in PROTOCOLS.items():
        manifest = ASSETS / "retrieval_benchmarks/manifests" / filename
        result = dict(expected_images=images, expected_captions=captions, expected_sha256=expected, present=manifest.is_file(), passed=False)
        if manifest.is_file():
            rows = [json.loads(line) for line in manifest.open() if line.strip()]
            image_ids = {row["image_id"] for row in rows}
            caption_ids = {row["caption_id"] for row in rows}
            mapping = all(row["positive_image_id"] in image_ids for row in rows)
            result.update(sha256=digest(manifest), n_images=len(image_ids), n_captions=len(rows),
                          unique_caption_ids=len(caption_ids), all_gt_in_candidate_pool=mapping,
                          examples=[{key: row[key] for key in ["image_id", "caption_id", "positive_image_id", "image_path"]} for row in rows[:5]],
                          image_integrity=integrity(manifest, ASSETS / "retrieval_benchmarks" / folder / "images"))
            result["passed"] = (result["sha256"] == expected and len(image_ids) == images and len(rows) == captions
                                and len(caption_ids) == captions and mapping and not result["image_integrity"]["missing_images"]
                                and not result["image_integrity"]["decode_failures"])
        results[name] = result
    coco = ASSETS / "evaluation/coco"
    annotation = coco / "annotations/captions_val2017.json"
    results["coco"] = dict(expected_images=5000, expected_captions=25000, passed=False, present=annotation.is_file())
    if annotation.is_file():
        from torchvision.datasets import CocoCaptions

        dataset = CocoCaptions(str(coco / "val2017"), str(annotation))
        candidate_rows = []
        image_paths = []
        for image_id in dataset.ids:
            captions = dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=image_id))[:5]
            assert len(captions) == 5
            filename = dataset.coco.loadImgs(image_id)[0]["file_name"]
            image_paths.append(coco / "val2017" / filename)
            candidate_rows.extend(dict(image_id=image_id, caption_id=entry["id"], caption=entry["caption"]) for entry in captions)
        dump(EVIDENCE / "coco-canonical-candidate-map.json", candidate_rows)
        image_integrity = verify_images(image_paths)
        caption_ids = {row["caption_id"] for row in candidate_rows}
        results["coco"].update(n_images=len(dataset.ids), n_captions=len(candidate_rows),
            image_integrity=image_integrity, missing_images=image_integrity["missing_images"], unique_caption_ids=len(caption_ids),
            all_gt_in_candidate_pool=all(row["image_id"] in dataset.ids for row in candidate_rows),
            annotation_sha256=digest(annotation), candidate_map_sha256=digest(EVIDENCE / "coco-canonical-candidate-map.json"),
            historical_sha_comparison="Do not claim a historical match unless a comparable digest is found in RECOVERY_MANIFEST.json.",
            passed=len(dataset.ids) == 5000 and len(candidate_rows) == 25000 and len(caption_ids) == 25000
                and not image_integrity["missing_images"] and not image_integrity["decode_failures"])
    urban = ASSETS / "evaluation/Urban1k/Urban1k"
    results["urban1k"] = dict(expected_images=1000, expected_captions=1000, passed=False, present=urban.is_dir())
    if (urban / "image").is_dir() and (urban / "caption").is_dir():
        from tools.urban1k_retrieval import image_caption_pairs, read_captions

        pairs = image_caption_pairs(str(urban))
        captions = read_captions(pairs)
        image_integrity = verify_images([image for image, _ in pairs])
        mapping = [dict(candidate_id=position, positive_image_id=position,
                        image=Path(image).name, caption=caption)
                   for position, ((image, _), caption) in enumerate(zip(pairs, captions))]
        dump(EVIDENCE / "urban1k-candidate-map.json", mapping)
        expected_archive_sha = "08e42b3fada77abf7f890a087ed6e9f9fbba1dc143f9f09801ed83187ae617b8"
        archive_sha = digest(ASSETS / "downloads/Urban1k.zip")
        results["urban1k"].update(n_images=len(pairs), n_captions=len(captions),
            image_integrity=image_integrity, archive_sha256=archive_sha,
            historical_archive_sha256=expected_archive_sha,
            historical_archive_sha_matches=archive_sha == expected_archive_sha,
            historical_revision="953fcf1d1a3bde031d9424a91a0154e594b5a586",
            candidate_map_sha256=digest(EVIDENCE / "urban1k-candidate-map.json"),
            all_gt_in_candidate_pool=True,
            passed=len(pairs) == 1000 and len(captions) == 1000 and all(captions)
                and archive_sha == expected_archive_sha and not image_integrity["missing_images"]
                and not image_integrity["decode_failures"])
    dump(EVIDENCE / "evaluator-audit.json", dict(passed=all(result["passed"] for result in results.values()), protocols=results))
    print(json.dumps({name: {key: value for key, value in result.items() if key not in ("examples", "image_integrity")} for name, result in results.items()}, indent=2))


def status():
    required = {name: load(EVIDENCE / filename, {}).get("passed", False) for name, filename in {
        "code": "git-audit.json", "environment": "environment-audit.json", "configs": "config-audit.json",
        "training_data": "training-audit.json", "five_evaluation_sets": "evaluator-audit.json",
        "training_image_decode_completeness": "training-image-completeness-audit.json",
        "original_sam_shards": "sa1b-mirror-audit.json",
        "step0": "step0-audit.json", "cpu_tests": "cpu-tests.json", "sampling": "sampling-audit.json",
        "model_config_construction": "construction-audit.json",
        "four_gpu_five_step_smoke": "smoke-audit.json", "native_export": "export-audit.json",
        "native_evaluator_subset": "native-evaluator-subset.json"}.items()}
    full_dir = RECOVERY.parent / "experiments/nest_clip_v1/armb_summary02_4epoch_v1"
    full_progress = load(full_dir / "FULL_PROGRESS.json", {})
    if full_progress:
        required["s02_full_prefix"] = load(full_dir / "FIRST_FIVE_GATE.json", {}).get("passed", False)
    ready = all(required.values())
    result = dict(status="READY_TO_RESUME_RESEARCH" if ready else "NOT_READY_TO_RESUME_RESEARCH",
                  requirements=required, blocked_requirements=[name for name, passed in required.items() if not passed],
                  formal_training_started=full_progress.get("formal_training_started", False),
                  formal_training_authorized=load(EVIDENCE / "recovery-operation-policy.json", {}).get("formal_training_authorized", False),
                  s02_full_status=full_progress.get("status"),
                  lost_history="Trained checkpoint/optimizer/RNG states cannot be recovered from metrics or reports. A new run from step0 is not historical continuation.")
    dump(EVIDENCE / "readiness.json", result)
    lines = ["# Ready to resume", "", "**" + result["status"] + "**", "", "| Requirement | Verified |", "| --- | --- |"]
    lines.extend(f"| {name} | {'PASS' if passed else 'MISSING / NOT VERIFIED'} |" for name, passed in required.items())
    full_note = (f"S02 full status: {result['s02_full_status']}; completed updates: {full_progress.get('completed_updates', 0)}. "
                 "See experiments/nest_clip_v1/armb_summary02_4epoch_v1/FULL_RESULTS.md; no automatic restart/resume."
                 if full_progress else "No 500/4868-update training has been started or authorized by this recovery.")
    lines.extend(["", full_note, "", result["lost_history"], "",
                  "`native_export` means the trained five-update smoke checkpoint's verified bare export.",
                  "The separate step0 bare export/native evaluator subset has already passed when `native_evaluator_subset` is PASS.", "",
                  "Use `.venv/bin/python recovery/audit.py status` to re-evaluate the fail-closed gate."])
    (RECOVERY / "READY_TO_RESUME.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["scan", "configs", "environment", "training", "evaluators", "status"])
    args = parser.parse_args()
    globals()[args.command]()


if __name__ == "__main__":
    main()
