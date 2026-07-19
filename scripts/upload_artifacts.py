"""Upload the deploy checkpoint to the Hugging Face Hub.

Requires `huggingface-cli login` first.

Usage: python scripts/upload_artifacts.py [--repo-id USER/smart-product-pricing-artifacts]
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, whoami

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT = REPO_ROOT / "artifacts" / "price_model_deploy.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=None)
    args = parser.parse_args()

    repo_id = args.repo_id or f"{whoami()['name']}/smart-product-pricing-artifacts"

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=CKPT,
        path_in_repo=CKPT.name,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded {CKPT.name} to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
