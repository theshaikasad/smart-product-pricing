"""FastAPI server: serves the frontend and the prediction API."""

import base64
import binascii
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pricing.pipeline import PricePredictor

STATIC_DIR = Path(__file__).resolve().parent / "static"

predictor: PricePredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = PricePredictor()
    predictor.warmup()
    yield


app = FastAPI(title="Smart Product Pricing", lifespan=lifespan)


MAX_IMAGE_BYTES = 8 * 1024 * 1024


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Product catalog text")
    image_url: str | None = Field(None, max_length=2000, description="Optional product image URL")
    image_b64: str | None = Field(
        None,
        max_length=11_500_000,  # ~8MB after base64 expansion
        description="Optional uploaded image as base64 (raw or data: URL); takes precedence over image_url",
    )


class PredictResponse(BaseModel):
    price: float
    log_price: float
    image_used: bool
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Product text must not be empty")

    image_bytes = None
    if req.image_b64:
        payload = req.image_b64.split(",", 1)[-1]  # tolerate data: URL prefixes
        try:
            image_bytes = base64.b64decode(payload, validate=True)
        except binascii.Error:
            raise HTTPException(status_code=422, detail="Uploaded image is not valid base64")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=422, detail="Uploaded image exceeds 8 MB")

    result = predictor.predict(req.text, req.image_url or None, image_bytes)
    return PredictResponse(
        price=round(result.price, 2),
        log_price=result.log_price,
        image_used=result.image_used,
        latency_ms=round(result.latency_ms, 1),
    )


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
