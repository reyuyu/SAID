# SAID extended retrieval

This directory preserves the v0.1 data preparation protocol and evaluator. Preparation commands remain model-free and run with `CUDA_VISIBLE_DEVICES=""`. Subsequent authorised native-student evaluations are complete for Clean v0.1 step3651, Full v0.1 step3651, and Full v0.1 step2000.

See the [results table](../../experiments/RESULTS.md) for all six protocols, actual R@1/5/10 and per-run evidence. The [status.json](status.json) is the historical preparation snapshot; its `MODEL_INFERENCE_NOT_RUN` fields describe that preparation task, not the later experiments.

## Protocols and sources

* DOCCI test: Google release, 5,000 examples, CC BY 4.0. Source: <https://google.github.io/docci/>.
* DCI: Facebook Research release, 7,805 images; annotations are downloadable from <https://github.com/facebookresearch/DCI> and the image archive requires the SA-1B licence. DCI is CC-BY-NC.
* Long-DCI: source manifest/repository at <https://huggingface.co/datasets/mderakhshani/Long-DCI>; preserve the published TSV IDs and captions.
* Flickr30k: source manifest at <https://huggingface.co/datasets/nlphuji/flickr30k>; keep all five captions per image and derive test1K only from its explicit split/ID list.

Data lives outside Git (default `/root/datasets/retrieval_benchmarks`). The preparation script records URL, local SHA256, extraction/check status and manifest SHA. A local SHA is an identity record when no upstream checksum is published; it is not an official verification.

## Commands

```bash
export CUDA_VISIBLE_DEVICES=""
python tools/prepare_retrieval_benchmarks.py --root /root/datasets/retrieval_benchmarks
python tools/prepare_retrieval_benchmarks.py --dataset docci --download --root /root/datasets/retrieval_benchmarks
python tools/prepare_retrieval_benchmarks.py --dataset docci --input /root/datasets/retrieval_benchmarks/docci/docci_descriptions.jsonlines --manifest /root/datasets/retrieval_benchmarks/manifests/docci_test.jsonl
pytest -q tests/test_extended_retrieval.py
```

An evaluation command requires an explicit model factory; replace the placeholder with a compatible native student loader:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval_extended_retrieval.py \
  --manifest /data/retrieval/manifests/flickr30k_test1k.jsonl \
  --image-root /data/retrieval/Flickr30k \
  --checkpoint /CHECKPOINT/BARE_STUDENT.pt \
  --model-factory your_adapter:load_bare_said_student --execute-model \
  --device cpu --batch-size 32 --score-chunk 1024 --output /tmp/flickr.json
```

The adapter must return the bare native student and implement native `encode_image`/`encode_text` with FP32 normalisation. No masks, gates, U, decoder, fusion, reranking or training are part of this evaluator. Cache identities include checkpoint, manifest/text SHA, tokenizer/context length, preprocessing and dtype.

## Preparation snapshot and completed evaluations

Base SHA: `11af80b344c623b27b93069f9be526970c9c950c` (SAID training branch snapshot; training worktree was not modified).

DOCCI test (5,000 pairs), Flickr Karpathy test1K (1,000 images / 5,000 captions), and DCI full (7,805 pairs) are ready and have been evaluated. Long-DCI uses 7,602 rows reconstructed from DCI `extra_caption`; it is explicitly labelled reconstructed and is not the unavailable official CSV. Dataset identities are in [ASSETS.md](../../experiments/ASSETS.md). Flickr full remains unavailable; this does not make the standard test1K evaluation incomplete.

## License and exclusions

Read each upstream license before downloading. Do not commit images, archives, full captions, checkpoints or feature caches. This directory contains preparation code documentation and historical status; small evaluation results are stored under `experiments/`.
