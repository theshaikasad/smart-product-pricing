"""Embedding generation for inference.

Both modalities are encoded with OpenCLIP ViT-B/32 (``laion2b_s34b_b79k``)
and L2-normalized to 512 dims — verified against the stored embeddings in
train_combined.csv (text cosine similarity 1.0000, image 0.998). The solution
writeup mentions Flan-T5 for text, but the shipped artifacts were produced
with the CLIP text tower; this module reproduces the artifacts.
"""

import io
from functools import lru_cache

import numpy as np
import requests
import torch
from PIL import Image

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
IMAGE_TIMEOUT_S = 10
_USER_AGENT = "smart-product-pricing/1.0"


@lru_cache(maxsize=1)
def _clip():
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_text(text: str) -> np.ndarray:
    model, _, tokenizer = _clip()
    tokens = tokenizer([text])
    feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].numpy().astype(np.float32)


@torch.no_grad()
def encode_image(image: Image.Image) -> np.ndarray:
    model, preprocess, _ = _clip()
    tensor = preprocess(image.convert("RGB")).unsqueeze(0)
    feats = model.encode_image(tensor)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].numpy().astype(np.float32)


def encode_image_bytes(data: bytes) -> np.ndarray | None:
    """Encode raw image bytes; None if they can't be decoded (the caller
    zero-fills, matching how training handled failed downloads)."""
    try:
        return encode_image(Image.open(io.BytesIO(data)))
    except Exception:
        return None


def encode_image_from_url(url: str) -> np.ndarray | None:
    """Download and encode a product image; None on any failure."""
    try:
        resp = requests.get(url, timeout=IMAGE_TIMEOUT_S, headers={"User-Agent": _USER_AGENT})
        resp.raise_for_status()
        return encode_image_bytes(resp.content)
    except Exception:
        return None
