"""Locate or download the model checkpoint.

Resolution order:
1. ``PRICING_CHECKPOINT`` env var (explicit path)
2. ``artifacts/price_model_deploy.pt`` (slim deploy checkpoint)
3. ``artifacts/advanced_price_model.pt`` (full training checkpoint)
4. Download the slim checkpoint from the Hugging Face Hub
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts"

HF_REPO_ID = os.environ.get("PRICING_HF_REPO", "theshaikasad/smart-product-pricing-artifacts")
DEPLOY_CKPT_NAME = "price_model_deploy.pt"
FULL_CKPT_NAME = "advanced_price_model.pt"


def resolve_checkpoint() -> Path:
    env_path = os.environ.get("PRICING_CHECKPOINT")
    if env_path:
        return Path(env_path)

    for name in (DEPLOY_CKPT_NAME, FULL_CKPT_NAME):
        candidate = ARTIFACT_DIR / name
        if candidate.exists():
            return candidate

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=HF_REPO_ID, filename=DEPLOY_CKPT_NAME))
