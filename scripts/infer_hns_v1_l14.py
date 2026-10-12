#!/usr/bin/env python3
"""Native SAID HNS-v1 ViT-L/14 image/text inference."""
import argparse
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
    a = p.parse_args()
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
    for rank, i in enumerate(sorted(range(len(scores)), key=lambda j: scores[j], reverse=True), 1):
        print(f"{rank}\t{scores[i]:.8f}\t{a.texts[i]}")
    print(f"embedding_dim={spec['embedding_dim']} device={device} context_length=248")

if __name__ == "__main__":
    main()
