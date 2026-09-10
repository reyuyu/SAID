"""Standard, deterministic ShareGPT4V validation retrieval.

The validation split is fixed by the audited manifest (first 1,000 JSON records,
see ``docs/phase27a2_full_data_gate.md``), and every validation image has exactly
one caption, so this is a 1,000-way image<->text retrieval computed with the
standard CLIP inference paths (``encode_image`` / ``encode_text``). The Said router
is never used.
"""
import torch

from eval.retrieval.coco_retrieval import DEFAULT_SIMILARITY_CHUNK, retrieval_metrics


@torch.inference_mode()
def evaluate_sharegpt4v(model, dataset, batch_size=64, device=None,
                        similarity_chunk=DEFAULT_SIMILARITY_CHUNK):
    """Retrieval metrics on the fixed ShareGPT4V validation split.

    Args:
        model: DDP-wrapped or plain model exposing ``encode_image`` / ``encode_text``.
        dataset: ``share4v_val_dataset``-like dataset returning ``(image, caption)``.
        batch_size: images per forward pass.
        device: torch device; defaults to the model's own device.

    Returns:
        dict with ``image2text_R{1,5,10}`` / ``text2image_R{1,5,10}`` in [0, 1].
    """
    if len(dataset) == 0:
        raise ValueError('empty validation dataset')
    from model import longclip

    core = model.module if hasattr(model, 'module') else model
    if device is None:
        device = next(core.parameters()).device
    device = torch.device(device)

    image_features = []
    text_features = []
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        batch = [dataset[i] for i in range(start, stop)]
        images = torch.stack([item[0] for item in batch]).to(device)
        captions = [item[1] for item in batch]
        image_features.append(core.encode_image(images).detach().cpu().float())
        tokens = longclip.tokenize(captions, truncate=True).to(device)
        text_features.append(core.encode_text(tokens).detach().cpu().float())

    return retrieval_metrics(
        torch.cat(image_features), torch.cat(text_features), captions_per_image=1,
        similarity_chunk=similarity_chunk,
    )
