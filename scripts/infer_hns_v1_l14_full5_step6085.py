#!/usr/bin/env python3
"""Native SAID HNS-v1 ViT-L/14 image/text inference."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from PIL import Image
from model import longclip
from model.backbone import load_native_state

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--texts", nargs="+", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--output", help="Optional JSON inference audit")
    p.add_argument("--cpu-threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.cpu_threads)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    model, _, spec = load_native_state(state, base_model="ViT-L/14", device=device)
    image = Image.open(a.image).convert("RGB")
    image_tensor = longclip._transform(224)(image).unsqueeze(0).to(device)
    tokens = longclip.tokenize(a.texts, context_length=248, truncate=True).to(device)
    with torch.no_grad():
        image_features = F.normalize(model.encode_image(image_tensor).float(), dim=-1)
        text_features = F.normalize(model.encode_text(tokens).float(), dim=-1)
        scores = (image_features @ text_features.T).squeeze(0).cpu().tolist()
    assert tokens.shape == (len(a.texts), 248)
    assert image_features.shape == (1, 768)
    assert text_features.shape == (len(a.texts), 768)
    assert torch.isfinite(image_features).all() and torch.isfinite(text_features).all()
    if a.output:
        Path(a.output).write_text(json.dumps({
            "passed": True, "device": device, "image": a.image,
            "image_embedding_shape": list(image_features.shape),
            "text_embedding_shape": list(text_features.shape),
            "token_shape": list(tokens.shape),
            "content_token_counts": [(row != 0).sum().item() - 2 for row in tokens],
            "texts": a.texts, "cosine_similarities": scores,
        }, indent=2) + "\n", encoding="utf-8")
    for rank, i in enumerate(sorted(range(len(scores)), key=lambda j: scores[j], reverse=True), 1):
        print(f"{rank}\t{scores[i]:.8f}\t{a.texts[i]}")
    print(f"embedding_dim={spec['embedding_dim']} device={device} context_length=248")

if __name__ == "__main__":
    main()
