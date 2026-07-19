"""End-to-end inference: raw product text (+ optional image URL) → price."""

import time
from dataclasses import dataclass

import numpy as np
import torch

from pricing import encoders
from pricing.artifacts import resolve_checkpoint
from pricing.models import AdvancedPriceModel


@dataclass
class Prediction:
    price: float
    log_price: float
    image_used: bool
    latency_ms: float


class PricePredictor:
    """Loads the trained checkpoint once and serves predictions.

    Mirrors the training-time preprocessing exactly: embeddings are produced by
    the same encoders used to build train_combined.csv, missing images are
    zero-filled, and both modalities pass through the persisted RobustScalers
    before the network. The model predicts log1p(price); expm1 inverts it.
    """

    def __init__(self, checkpoint_path=None, device: str = "cpu"):
        self.device = torch.device(device)
        path = checkpoint_path or resolve_checkpoint()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        self.scaler_img = ckpt["scaler_img"]
        self.scaler_txt = ckpt["scaler_txt"]
        self.config = ckpt["config"]

        self.model = AdvancedPriceModel(**self.config)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

    def warmup(self):
        """Load encoder weights and run one dummy forward pass."""
        encoders.encode_text("warmup")
        self.predict("warmup", None)

    @torch.no_grad()
    def predict(self, text: str, image_url: str | None = None) -> Prediction:
        start = time.perf_counter()

        txt_vec = encoders.encode_text(text)

        image_used = False
        img_vec = None
        if image_url:
            img_vec = encoders.encode_image_from_url(image_url)
            image_used = img_vec is not None
        if img_vec is None:
            # Training zero-filled missing image embeddings; do the same here.
            img_vec = np.zeros(self.config["dim_img"], dtype=np.float32)

        img_scaled = self.scaler_img.transform(img_vec.reshape(1, -1)).astype(np.float32)
        txt_scaled = self.scaler_txt.transform(txt_vec.reshape(1, -1)).astype(np.float32)

        log_price = float(self.model(
            torch.from_numpy(img_scaled).to(self.device),
            torch.from_numpy(txt_scaled).to(self.device),
        ))
        price = max(float(np.expm1(log_price)), 0.0)

        return Prediction(
            price=price,
            log_price=log_price,
            image_used=image_used,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
