"""AutiLens AI - FastAPI inference backend (bootcamp guide §11).

Endpoints:
  GET  /health                 -> service status + loaded checkpoint
  POST /predict  (video file)  -> CV model output only: behavior probabilities + evidence
  POST /report   (video file)  -> CV output PASSED TO the LLM (bootcamp guide §10
                                   integration requirement) -> generated report.
                                   Falls back to a deterministic template if
                                   ANTHROPIC_API_KEY is not set / the call fails.

Run:
  AUTILENS_CKPT=models/vision_r2p1d_cv_fold3.pt ANTHROPIC_API_KEY=sk-... \
    python -m uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.inference import AutiLensPredictor
from src.inference_fast import FastEnsemblePredictor, TwoStagePredictor
from src.model_registry import select_best
from src.llm_report import generate_llm_report

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MAX_MB = float(os.environ.get("AUTILENS_MAX_MB", "50"))
ALLOWED = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

app = FastAPI(title="AutiLens AI", version="0.1.0",
              description="Multimodal autism-related behavior recognition (research prototype, not a diagnostic tool).")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_predictor: AutiLensPredictor | FastEnsemblePredictor | None = None
_n_folds: int = 0
_two_stage: TwoStagePredictor | None = None


def _resolve_card():
    """The checkpoint to serve: highest measured test accuracy, not best filename.

    Shared with the Streamlit UI through ``src.model_registry`` so both entry
    points serve the same model. ``AUTILENS_CKPT`` still overrides.
    """
    # Single-model serving is the 9-behavior scope. Ranking is on OUT-OF-FOLD
    # score; without the scope filter a 4-family Stage-1 checkpoint would win
    # on a macro-F1 from a different (easier) exam.
    card = select_best(MODELS_DIR, n_classes=9)
    if card is None:
        raise RuntimeError("No checkpoint found in models/. Train one first "
                           "(python -m src.cv_fast).")
    return card


def _resolve_ckpt() -> str:
    return str(_resolve_card().path)


def _build_predictor(path: str):
    """Fast ensemble if the checkpoint holds fold heads, else the legacy model."""
    import torch

    try:
        st = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(st, dict) and isinstance(st.get("folds"), list):
            return FastEnsemblePredictor(path), len(st["folds"])
    except Exception:
        pass
    return AutiLensPredictor(path), 1


#: Stage 2 is the existing 9-behavior model, used unchanged.
STAGE2_CKPT = "autilens_v2.pt"


def _build_two_stage():
    """Stage 1 (4 behavior families) + Stage 2 (9 behaviors), one backbone pass.

    Returns ``None`` when either half is missing, so the service degrades to
    single-model serving rather than failing to start.
    """
    stage1 = select_best(MODELS_DIR, n_classes=4)
    stage2 = MODELS_DIR / STAGE2_CKPT
    if stage1 is None or not stage2.exists():
        return None
    try:
        return TwoStagePredictor(stage1.path, stage2)
    except Exception as exc:                      # incompatible feature setups etc.
        print(f"[autilens] two-stage unavailable, falling back to single model: {exc}")
        return None


@app.on_event("startup")
def _load() -> None:
    global _predictor, _n_folds, _two_stage
    _two_stage = None if os.environ.get("AUTILENS_CKPT") else _build_two_stage()
    _predictor, _n_folds = _build_predictor(_resolve_ckpt())


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
    card = _resolve_card()
    return {
        "status": "ok" if _predictor else "loading",
        "checkpoint": card.path.name,
        "selected_by": "highest test score in reports/ (override: AUTILENS_CKPT)",
        "test_score": card.macro_f1,
        "test_metric": card.metric_name,
        "calibrated": card.calibrated,
        "task": getattr(_predictor, "task", None) if _predictor else None,
        "modality": _predictor.modality if _predictor else None,
        "ensemble_folds": _n_folds,
        "two_stage": _two_stage is not None,
        "families": (_two_stage.stage1.labels if _two_stage else None),
        "labels": _predictor.labels if _predictor else None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  two_stage: bool = Query(False,
                      description="Return Stage 1 behavior families alongside the "
                                  "9 behaviors, with Stage 2 masked to the detected "
                                  "families. Adds a `families` key; every existing "
                                  "key keeps its meaning.")) -> dict:
    if _predictor is None:
        raise HTTPException(503, "model not loaded")
    if two_stage and _two_stage is None:
        raise HTTPException(409, "two-stage unavailable: needs a 4-family Stage 1 "
                                 f"checkpoint and models/{STAGE2_CKPT}")
    path = _save_upload(file)
    try:
        engine = _two_stage if two_stage else _predictor
        return engine.predict(path).to_dict()
    finally:
        _cleanup(path)


@app.post("/report")
async def report(file: UploadFile = File(...), prompt_version: str = Query("v2", pattern="^(v1|v2)$")) -> dict:
    """CV model -> structured JSON -> LLM prompt -> report (the required integration, §10).

    ``prompt_version`` selects between the two documented prompt iterations in
    src/llm_report.py (v2 is the hardened default). Automatically falls back to
    the deterministic template in src/report.py if no ANTHROPIC_API_KEY is set.
    """
    if _predictor is None:
        raise HTTPException(503, "model not loaded")
    path = _save_upload(file)
    try:
        pred = _predictor.predict(path)
        return {"prediction": pred.to_dict(), "report": generate_llm_report(pred, prompt_version)}
    finally:
        _cleanup(path)


def _cleanup(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
        path.parent.rmdir()
    except OSError:
        pass
