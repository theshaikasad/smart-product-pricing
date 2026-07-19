"""Slim the full training checkpoint for deployment.

The training checkpoint carries the SWA weights and the full training history;
inference only needs the standard weights, the two RobustScalers, and the
model config. This roughly halves the file size.

Usage: python scripts/export_deploy_checkpoint.py
"""

from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "artifacts" / "advanced_price_model.pt"
DST = REPO_ROOT / "artifacts" / "price_model_deploy.pt"


def main():
    ckpt = torch.load(SRC, map_location="cpu", weights_only=False)
    slim = {
        "model_state_dict": ckpt["model_state_dict"],
        "scaler_img": ckpt["scaler_img"],
        "scaler_txt": ckpt["scaler_txt"],
        "config": ckpt["config"],
    }
    torch.save(slim, DST)
    print(f"Wrote {DST} ({DST.stat().st_size / 1e6:.1f} MB, from {SRC.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
