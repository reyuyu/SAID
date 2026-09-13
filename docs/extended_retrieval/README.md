# SAID extended retrieval setup v0.1

This branch prepares benchmark metadata and a model-free evaluator.  It does not load a checkpoint, import a model, initialise CUDA, extract features, or report real R@ metrics.  All preparation commands run with `CUDA_VISIBLE_DEVICES=""`.

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

Future model evaluation is explicit and must be run by a human after reviewing data and licensing:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval_extended_retrieval.py \
  --manifest /data/retrieval/manifests/flickr30k_test1k.jsonl \
  --image-root /data/retrieval/Flickr30k \
  --checkpoint /CHECKPOINT/BARE_STUDENT.pt \
  --model-factory your_adapter:load_bare_said_student --execute-model \
  --device cpu --batch-size 32 --score-chunk 1024 --output /tmp/flickr.json
```

The adapter must return the bare native student and implement native `encode_image`/`encode_text` with FP32 normalisation. No masks, gates, U, decoder, fusion, reranking or training are part of this evaluator. Cache identities include checkpoint, manifest/text SHA, tokenizer/context length, preprocessing and dtype.

## Current preparation record

Base SHA: `11af80b344c623b27b93069f9be526970c9c950c` (SAID training branch snapshot; training worktree was not modified).

At setup time the four benchmark roots were absent. Therefore no real manifest counts or image checks are claimed until the explicit download/parse commands complete. Status vocabulary is limited to `PREPARATION_NOT_STARTED`, `DOWNLOADED`, `PREPARED`, `BLOCKED_AUTH`, `BLOCKED_NETWORK`, `CPU_TESTED`, `MODEL_INFERENCE_NOT_RUN`, and `REAL_EVALUATION_NOT_RUN`.

## License and exclusions

Read each upstream license before downloading. Do not commit images, archives, full captions, checkpoints or feature caches. This branch contains only preparation code, tests and this README.
