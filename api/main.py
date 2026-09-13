"""AutiLens AI - FastAPI inference backend (PDF section 12).

Endpoints:
  GET  /health                 -> service status + loaded checkpoint
  POST /predict  (video file)  -> behavior probabilities + evidence
  POST /report   (video file)  -> prediction + template screening report

Run:
  AUTILENS_CKPT=models/av_transformer.pt \
    python -m uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.inference import AutiLensPredictor
from src.report import build_report

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MAX_MB = float(os.environ.get("AUTILENS_MAX_MB", "50"))
ALLOWED = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

app = FastAPI(title="AutiLens AI", version="0.1.0",
              description="Multimodal autism-related behavior recognition (research prototype, not a diagnostic tool).")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_predictor: AutiLensPredictor | None = None


def _resolve_ckpt() -> str:
    env = os.environ.get("AUTILENS_CKPT")
    if env:
        return env
    for name in ("av_transformer.pt", "vision_transformer.pt"):
        if (MODELS_DIR / name).exists():
            return str(MODELS_DIR / name)
    cands = sorted(MODELS_DIR.glob("*.pt"))
    if not cands:
        raise RuntimeError("No checkpoint found in models/. Train one first (python -m src.train).")
    return str(cands[0])


@app.on_event("startup")
def _load() -> None:
    global _predictor
    _predictor = AutiLensPredictor(_resolve_ckpt())


def _save_upload(file: UploadFile) -> Path:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(415, f"Unsupported type {ext!r}. Allowed: {sorted(ALLOWED)}")
    data = file.file.read()
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_MB:.0f} MB limit")
    tmp = Path(tempfile.mkdtemp(prefix="autilens_")) / (file.filename or f"upload{ext}")
    tmp.write_bytes(data)
    return tmp


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _predictor else "loading",
        "checkpoint": Path(_resolve_ckpt()).name,
        "modality": _predictor.modality if _predictor else None,
        "labels": _predictor.labels if _predictor else None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if _predictor is None:
        raise HTTPException(503, "model not loaded")
    path = _save_upload(file)
    try:
        return _predictor.predict(path).to_dict()
    finally:
        _cleanup(path)


@app.post("/report")
async def report(file: UploadFile = File(...)) -> dict:
    if _predictor is None:
        raise HTTPException(503, "model not loaded")
    path = _save_upload(file)
    try:
        pred = _predictor.predict(path)
        return {"prediction": pred.to_dict(), "report": build_report(pred)}
    finally:
        _cleanup(path)


def _cleanup(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
        path.parent.rmdir()
    except OSError:
        pass
