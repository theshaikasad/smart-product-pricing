"""Modal deployment — serves the FastAPI app serverlessly.

    modal deploy modal_app.py

Model weights (OpenCLIP encoder + price checkpoint) are baked into the image
at build time so cold starts don't download anything.
"""

import modal

app = modal.App("smart-product-pricing")


def _download_weights():
    import open_clip
    from huggingface_hub import hf_hub_download

    open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    hf_hub_download(
        repo_id="theshaikasad/smart-product-pricing-artifacts",
        filename="price_model_deploy.pt",
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2,<3",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "open-clip-torch>=2.24",
        "scikit-learn>=1.4,<1.7",
        "numpy>=1.26,<2",
        "pillow>=10",
        "requests>=2.31",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.29",
        "huggingface_hub>=0.23",
    )
    .run_function(_download_weights)
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("app", remote_path="/root/app")
)


@app.function(image=image, memory=3072, cpu=2, max_containers=1, scaledown_window=300)
@modal.asgi_app(label="smart-product-pricing")
def web():
    import sys

    sys.path.insert(0, "/root/src")
    from app.main import app as fastapi_app

    return fastapi_app
