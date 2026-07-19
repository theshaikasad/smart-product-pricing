# Hugging Face Space (Docker SDK) — serves the FastAPI app on port 7860.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    PIP_NO_CACHE_DIR=1

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU-only torch keeps the image ~5GB smaller than the CUDA default
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY --chown=user . .
USER user

# Bake model weights into the image so cold starts don't re-download:
# the OpenCLIP encoder and the price-model checkpoint from the Hub.
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')" && \
    python -c "import sys; sys.path.insert(0, 'src'); from pricing.artifacts import resolve_checkpoint; print(resolve_checkpoint())"

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
