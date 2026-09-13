"""AutiLens AI - Streamlit demo UI (PDF section 11).

Screens: Home / consent -> Upload -> Processing -> Results -> Evidence -> Report.
Runs inference in-process (no separate API needed); set AUTILENS_API to call the
FastAPI backend instead.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

st.set_page_config(page_title="AutiLens AI", page_icon="🔎", layout="centered")
st.title("AutiLens AI")
st.caption("A multimodal AI system for autism-related **behavioral screening** — research prototype.")

st.warning(
    "**Not a diagnostic tool.** AutiLens AI recognises dataset-defined behaviors in "
    "short video clips and produces an explainable research report. It does not "
    "diagnose autism. Autism diagnosis requires qualified professionals and broader "
    "clinical evidence. Do not upload identifiable video of children without consent."
)


@st.cache_resource
def get_predictor(ckpt: str):
    from src.inference import AutiLensPredictor

    return AutiLensPredictor(ckpt)


ckpts = sorted(p.name for p in MODELS.glob("*.pt"))
if not ckpts:
    st.error("No trained checkpoint in `models/`. Run `python -m src.train` first.")
    st.stop()

with st.sidebar:
    st.header("Model")
    default = next((c for c in ckpts if c.startswith("av_")), ckpts[0])
    ckpt = st.selectbox("Checkpoint", ckpts, index=ckpts.index(default))
    consent = st.checkbox("I confirm I have the right to analyse this video and understand this is not a diagnosis.")

uploaded = st.file_uploader("Upload a short video clip", type=["mp4", "mov", "mkv", "webm", "avi", "m4v"])

if uploaded and not consent:
    st.info("Please confirm the consent checkbox in the sidebar to continue.")

if uploaded and consent:
    tmp = Path(tempfile.mkdtemp(prefix="autilens_")) / uploaded.name
    tmp.write_bytes(uploaded.getvalue())
    st.video(str(tmp))

    with st.spinner("Preprocessing and running inference…"):
        predictor = get_predictor(str(MODELS / ckpt))
        pred = predictor.predict(tmp)
        from src.report import build_report

        rep = build_report(pred)

    st.subheader("Results — behavior probabilities")
    df = pd.DataFrame(pred.behaviors)
    df["probability"] = (df["probability"] * 100).round(1)
    st.dataframe(
        df.rename(columns={"probability": "probability %"}),
        hide_index=True, use_container_width=True,
    )
    st.bar_chart(df.set_index("label")["probability %"])

    st.subheader("Evidence")
    if pred.evidence:
        for e in pred.evidence:
            st.markdown(
                f"- **{e['label']}** — approx. window "
                f"`{e['window_s'][0]:.1f}s – {e['window_s'][1]:.1f}s` "
                f"(peak activation {e['peak_frame_score']:.2f})"
            )
        st.caption("Evidence windows are approximate model saliency, not verified events.")
    else:
        st.write("No behavior crossed its detection threshold, so no evidence windows are shown.")

    st.subheader("Screening report")
    st.code(rep["text"], language="text")
    st.download_button("Download report (.txt)", rep["text"], file_name=f"{tmp.stem}_autilens_report.txt")

    try:
        tmp.unlink()
        tmp.parent.rmdir()
    except OSError:
        pass
