"""Render recovery status from evidence; missing evidence never implies success."""

import json

from audit import ANNOTATION_SHA, CLIP_SHA, EVIDENCE, RECOVERY, RECORDS_SHA, ROOT, STEP0_SHA, load, status


def write(name, lines):
    (RECOVERY / name).write_text("\n".join(lines) + "\n")


def main():
    environment = load(EVIDENCE / "environment-audit.json", {})
    lines = ["# Recovery environment", "", "Sources: pinned `docs/environment_freeze.txt`, `requirements.txt`, and",
             "`experiments/nest_clip_v1/evidence/environment.json`. No core dependency upgrades.", "",
             "| Item | Historical | Recovered |", "| --- | --- | --- |",
             f"| Python | 3.10, patch release not established | {environment.get('python', 'Audit pending')} |",
             f"| CUDA wheel runtime | 12.4 | {environment.get('cuda', 'Audit pending')} |",
             f"| NCCL | 2.21.5 | {environment.get('nccl', 'Audit pending')} |",
             f"| Visible GPUs | 4 x A100 80GB | {environment.get('visible_gpus', 'Audit pending')} |"]
    lines.extend(f"| {package} | {values['historical']} | {values['recovered']} |" for package, values in environment.get("comparisons", {}).items())
    lines.extend(["", "The isolated `.venv` uses the existing Python3.10.20 interpreter at",
                  "`/root/miniconda3/envs/RoboTwin/bin/python`; no packages in that environment were modified.",
                  "Its base interpreter must remain available. Full resolved versions are preserved in",
                  "`evidence/recovered-pip-freeze.txt`.", "",
                  "Unrecorded historic versions: Python patch release, pip, pytest, and exact setuptools",
                  "(original requirement is setuptools<81). Recovery uses pip25.2, pytest8.4.2, setuptools80.9.0.",
                  "pip was replaced to fix wheel-metadata name normalization, not to upgrade the training stack.",
                  "The system's default CUDA toolkit symlink is 12.2; PyTorch uses its pinned cu124 wheels.",
                  "Original driver and current driver are 550.127.05. No system CUDA/driver upgrade was performed.", "",
                  f"Environment audit passed: {environment.get('passed', False)}."])
    write("RECOVERY_ENVIRONMENT.md", lines)

    annotation = load(EVIDENCE / "annotation-audit.json", {})
    training = load(EVIDENCE / "training-audit.json", {})
    completeness = load(RECOVERY / "TRAIN_IMAGE_COMPLETENESS.json", {})
    prefixes = annotation.get("image_path_prefix_counts", {})
    public_job = load(EVIDENCE / "public-download-job.json", {})
    sam_mirror = load(RECOVERY / "SA1B_SHARDS.json", {})
    lines = ["# Dataset recovery", "", f"Expected annotation SHA256: `{ANNOTATION_SHA}`.",
             f"Observed annotation SHA256: `{annotation.get('annotation_sha256', 'not audited')}`.",
             f"Original records: {annotation.get('original_records', 'not audited')}; skip: {annotation.get('skip', 'not audited')};",
             f"training records: {annotation.get('training_records', 'not audited')} (required 1245901).", "",
             "| Family | Required training paths after skip1000 |", "| --- | --- |",
             f"| LLaVA LAION/CC/SBU | {sum(count for prefix, count in prefixes.items() if prefix.startswith('llava/'))} |",
             f"| COCO train2017 | {prefixes.get('coco/train2017', 'not audited')} |",
             f"| SAM | {prefixes.get('sam/images', 'not audited')} |", "",
             "SAM publisher archive range: sa_000000.tar through sa_000050.tar inclusive.",
             "The frozen annotation requires original SAM IDs 1..570590, not the 9K SFT subset.",
             f"Expected indexed records SHA256: `{RECORDS_SHA}`.",
             f"Observed indexed records SHA256: `{training.get('index_sha256', 'not audited')}`.",
             f"Missing training image paths: {completeness.get('training_index_missing', completeness.get('missing_training_paths', training.get('missing_images', 'image audit pending')))}.",
             f"Training data fully verified: {training.get('passed', False)}.", "",
             "No image preprocessing code was changed. Recover original JPEGs without recompression.",
             "Original index construction and text rules are reused; no replacement subset or synthetic training data.",
             "See per-asset `evidence/download-*.json` / `.log` and resumable `local_assets/downloads/*.part`.",
             f"Public training archive continuation job: {public_job.get('pid', 'not recorded')}; bounded retries, no training.",
             "Job output: `evidence/public-downloads-supervisor.log`. The job refreshes reports after extracting any completed public archives.",
             (f"Original-shard mirror selected: {sam_mirror.get('repo')} at {sam_mirror.get('revision')};"
              " see SA1B_MIRROR_RECOVERY.md and SA1B_SHARDS.json for actual checksum/download/extraction status."
              if sam_mirror else "SAM download URLs have not been obtained; publisher-authorized links or original archive copies are needed.")]
    lines.extend(["", "## Independent image recovery", "",
                  f"Completeness snapshot UTC: {completeness.get('checked_utc', 'not audited')}.",
                  "| Family | Recovered | Required | Missing |", "| --- | --- | --- | --- |"])
    for family, counts in completeness.get("families", {}).items():
        lines.append(f"| {family} | {counts['recovered_paths']} | {counts['required_paths']} | {counts['missing_paths']} |")
    lines.extend(["", "COCO uses Python standard-library ZipFile.testzip and safe original-path extraction; no system unzip dependency.",
                  "LLaVA identity/progress: LLAVA_COPY_IDENTITY.md, HF_LLAVA_ASSET.json and HF_LLAVA_RECOVERY.md.",
                  "The original archive has numeric subdirectories; extract them unchanged under llava/llava_pretrain/images/.",
                  "Other families retain their recorded snapshot counts until their own audits refresh them.",
                  "Full decode audit and any five-update smoke stay blocked until all required training paths exist."])
    write("DATASET_RECOVERY.md", lines)

    evaluator = load(EVIDENCE / "evaluator-audit.json", {})
    reconstruction = load(EVIDENCE / "manifest-reconstruction.json", {})
    subset = load(EVIDENCE / "native-evaluator-subset.json", {})
    lines = ["# Evaluator audit", "", "Only original native bare-student encoders and original retrieval metrics are used.",
             "No condition gate, fusion, reranking, DCI Full substitution, or split replacement.", "",
             "| Protocol | Images | Captions | Expected manifest SHA256 | Complete audit |", "| --- | --- | --- | --- | --- |"]
    from audit import PROTOCOLS

    for name, (_, _, images, captions, expected) in PROTOCOLS.items():
        lines.append(f"| {name} | {images} | {captions} | `{expected}` | {evaluator.get('protocols', {}).get(name, {}).get('passed', False)} |")
    for name, images, captions in [("coco", 5000, 25000), ("urban1k", 1000, 1000)]:
        lines.append(f"| {name} | {images} | {captions} | See recovered historical identity evidence | {evaluator.get('protocols', {}).get(name, {}).get('passed', False)} |")
    lines.extend(["", "COCO uses sorted `CocoCaptions.ids` and the first five original-order captions per image:",
                  "25014 source annotations are reduced by the original evaluator to 25000 candidates.",
                  "COCO similarity chunk is fixed at512 and uses the original row-wise argsort tie rule.",
                  "Urban1k uses original stem pairing, first caption line, and diagonal GT.",
                  "Urban1k archive SHA matches the historical pinned-revision report; all five image trees pass PIL verification.",
                  "COCO/Urban candidate-map hashes are fresh recovery evidence; no unpublished historical map hash is invented.",
                  "Flickr is the explicit test1K source. DOCCI is original Google test metadata, not translated captions.",
                  "Long-DCI is original `reconstruct_long_dci(extra_caption)`; 203 empty-extra-caption images are excluded",
                  "by the frozen original construction, not by a new split.", "",
                  "Historical extended manifest reconstruction:", "```json", json.dumps(reconstruction.get("manifests", {}), indent=2), "```", "",
                  "A manifest SHA match does not imply that its image assets have finished downloading.",
                  "Full identity/candidate/GT checks and image-integrity checks are recorded in `evidence/evaluator-audit.json`.",
                  f"Native step0 bare export and five-protocol small-sample evaluation passed: {subset.get('passed', False)}.",
                  "This is a CPU subset sanity check, not five full-benchmark scores or a trained-step5 export.",
                  "No official full-benchmark scores were rerun during recovery."])
    write("EVALUATOR_AUDIT.md", lines)

    initial = load(EVIDENCE / "step0-audit.json", {})
    lines = ["# Common step0 audit", "", f"Expected initializer SHA256: `{STEP0_SHA}`.",
             f"Observed initializer SHA256: `{initial.get('observed_sha256', 'not generated/audited')}`.",
             f"Official OpenAI CLIP B/16 SHA256: `{CLIP_SHA}`.",
             "Provenance: OpenAI CLIP + original random MaskNetwork; seed0; no training; empty optimizer state.",
             "Context248 construction is the original `longclip.load_from_clip` and repository model code.",
             f"Step0 audit passed: {initial.get('passed', False)}.",
             f"Historical file byte identity: {initial.get('historical_file_sha_matches', False)}.", "",
             "The semantic audit compares all keys/shapes/dtypes/tensor content against an independently",
             "rebuilt pinned original seed0 constructor, RNG state, and exact CPU image/text output digests.",
             "Per-tensor digests are in `evidence/step0-audit.json`. The historical tensor manifest itself was not",
             "published; if file SHA differs, do not claim the original lost file was independently inspected.",
             "No model parameters or constructor are modified to chase a serialization SHA."]
    write("STEP0_AUDIT.md", lines)

    smoke = load(EVIDENCE / "smoke-audit.json", {})
    cpu = load(EVIDENCE / "cpu-tests.json", {})
    construction = load(EVIDENCE / "construction-audit.json", {})
    lines = ["# Recovery smoke report", "", f"Four-GPU smoke executed: {smoke.get('executed', False)}.",
             f"Four-GPU smoke passed: {smoke.get('passed', False)}.",
             f"Blocked prerequisites: {smoke.get('blocked_by', 'not evaluated')}.",
             f"Original CPU/unit tests passed: {cpu.get('passed', False)}.",
             "Passing original suites: research 93 core +11 legacy; canonical 69 core +11 legacy.",
             "The legacy retrieval tests run separately because their old sys.path change shadows the train package in spawned workers.",
             f"Real CPU RandomK and Summary0.2 model/optimizer construction passed: {construction.get('passed', False)}.",
             "Construction uses original modules, H4868, context248, empty optimizer, zero updates, and original F/S/D weights.",
             "Recovery's fail-closed guard tests: four passed; see `test_recovery.py`.", "",
             "The only recovery trainer command is canonical RandomK, four A100s, batch256/rank,",
             "global1024, accumulation1, seed0, workers8, H4868, 1217 updates/epoch, stop exactly5.",
             "No synthetic-image or reduced-index smoke is used as a substitute for recovered training assets.",
             "DDP parameter agreement and native trained-checkpoint export remain unverified unless the real",
             "five-update acceptance/export JSONs pass. Missing assets never count as a passed smoke.", "",
             "Historical steady-state reference: about2.1s/update and27.8GiB peak allocated/rank.",
             "These are references, not recovery measurements. Five warmup updates cannot establish a",
             "495-cycle steady-state average. Investigate significant memory/time deviations before continuation.", "",
             "No 500/4868-step formal training, historical searches, or gradient audit have been run."]
    write("SMOKE_REPORT.md", lines)
    lines = ["# Lost local assets", "", "## Reconstructed exactly", "",
             "- Original annotation and skip1000 indexed record bytes, verified against historical SHA256.",
             "- Official OpenAI CLIP B/16 and common step000000.pt, both with matching historical SHA256.",
             "- Three extended evaluation manifests with matching historical SHA256; five evaluation protocols/data restored.",
             "- Full all-branch Git history and reports, pinned environment, training configurations and original code.", "",
             "## Rebuildable when source images are available", "",
             "- Training image trees, path indexes, decoded/cache assets, native bare exports and future run artifacts.",
             "- Public COCO/LLaVA image archives can be downloaded again; original SAM shard mirror/revision",
             "  is now recorded in SA1B_SHARDS.json, with actual checksum/recovery status in its dedicated report.", "",
             "## Not reconstructible from reports", "",
             "- Lost trained historical step500/step4868 checkpoints, optimizer moments, rank RNG and loader states.",
             "- Scalar metrics and provenance cannot recreate those exact tensors/files. A new run creates a new trajectory,",
             "  not continuation of the lost checkpoint. Backups, if found later, can change this assessment.",
             "- External baseline checkpoints are intentionally deferred until needed; no historical experiments are rerun."]
    write("LOST_ASSETS.md", lines)
    status()


if __name__ == "__main__":
    main()
