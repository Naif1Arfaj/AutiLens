"""AutiLens AI - demo UI (bootcamp guide section 11, MVP prototype).

Flow: consent -> upload -> analyse -> behavior results + evidence -> LLM report.
Inference runs in-process; no separate API is needed.

Target user (docs/problem_and_use_case.md): a behavioral therapist or researcher
doing a first-pass review of home/session video, not a diagnosing clinician.

Two deliberate design decisions:

* **No model picker.** The old sidebar listed every ``.pt`` in ``models/`` and
  defaulted by filename, which served a checkpoint scoring 0.458 while a 0.494
  one sat next to it. ``src/model_registry`` now ranks by measured out-of-fold accuracy
  and the app serves the winner, showing which and why instead of asking.
* **Head-type aware results.** Multi-label (independent sigmoids, per-class
  thresholds) and multi-class (softmax, one winning class) checkpoints render
  through different views, chosen from ``Prediction.task``.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import components as ui                      # noqa: E402
from app.theme import inject                          # noqa: E402
from src.inference import MULTICLASS                  # noqa: E402
from src.model_registry import (discover_models, models_signature,  # noqa: E402
                                select_best)

MODELS = ROOT / "models"
STEPS = ["Consent", "Upload", "Analyse", "Results", "Report"]

# Mirrors api/main.py so the two entry points advertise identical limits.
MAX_MB = float(os.environ.get("AUTILENS_MAX_MB", "50"))
VIDEO_TYPES = ["mp4", "mov", "mkv", "webm", "avi", "m4v"]

st.set_page_config(page_title="AutiLens AI", page_icon="🔎",
                   layout="centered", initial_sidebar_state="expanded")
inject()


# --------------------------------------------------------------------------- #
# model resolution -- ranked once per change to models/
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def ranked_models(_signature: str):
    """``_signature`` changes when a checkpoint is added, removed or retrained,
    which is what invalidates this cache."""
    return discover_models(MODELS), select_best(MODELS, n_classes=9)


@st.cache_resource(show_spinner=False)
def get_predictor(ckpt: str):
    """Load a k-fold ensemble when the checkpoint holds fold heads, else the
    legacy single model. Serving one fold scored below the reported numbers."""
    import torch

    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and isinstance(state.get("folds"), list):
            from src.inference_fast import FastEnsemblePredictor

            return FastEnsemblePredictor(ckpt), len(state["folds"])
    except Exception:
        pass
    from src.inference import AutiLensPredictor

    return AutiLensPredictor(ckpt), 1


@st.cache_data(show_spinner=False)
def clip_duration(path: str) -> float:
    from src.inference import _duration_s

    return _duration_s(Path(path))


# --------------------------------------------------------------------------- #
# chrome
# --------------------------------------------------------------------------- #
ui.masthead()
ui.safety_banner()

cards, best = ranked_models(models_signature(MODELS))
if best is None:
    ui.error_card("No trained checkpoint found in models/", [
        "Train one with  python -m src.cv_fast",
        "Or point AUTILENS_CKPT at an existing .pt file",
    ])
    st.stop()

with st.sidebar:
    st.markdown("### Report settings")
    prompt_version = st.selectbox(
        "Prompt version", ["v2", "v1"],
        help="v2 is the hardened prompt (default). See docs/PROMPT_VERSIONS_NOTES.md "
             "for the v1 to v2 changes.")
    if os.environ.get("ANTHROPIC_API_KEY"):
        st.success("Language model enabled.", icon=":material/check_circle:")
    else:
        st.warning("ANTHROPIC_API_KEY not set — reports fall back to the "
                   "deterministic template.", icon=":material/info:")

    st.markdown("### Privacy")
    st.caption("Uploads are written to a temporary directory, used for one "
               "inference pass and deleted immediately. Nothing is retained, and "
               "the language model receives only the numeric results — never the video.")

# Step 1 -- consent, in the flow rather than hidden in the sidebar.
consent = st.checkbox(
    "I confirm I have the right to analyse this video, and I understand this is "
    "a research screening aid and not a diagnosis.",
    key="consent")

uploaded = None
if consent:
    uploaded = st.file_uploader(
        f"Upload a short video clip  ·  {', '.join(VIDEO_TYPES)}  ·  up to {MAX_MB:.0f} MB",
        type=VIDEO_TYPES)

pred = st.session_state.get("pred")
current = 0 if not consent else (1 if uploaded is None else (3 if pred is not None else 2))
ui.stepper(STEPS, min(current, len(STEPS) - 1))

ui.model_card(best)
with st.expander(f"Why {best.tag}? See all {len(cards)} candidates"):
    ui.model_table(cards, best)

if not consent:
    st.html('<div class="al-empty">Confirm the statement above to enable the '
                'upload step.</div>')
    st.stop()
if uploaded is None:
    st.stop()

# --------------------------------------------------------------------------- #
# analyse
# --------------------------------------------------------------------------- #
tmp = Path(tempfile.mkdtemp(prefix="autilens_")) / uploaded.name
tmp.write_bytes(uploaded.getvalue())
st.video(str(tmp))

# Re-running inference on every widget interaction (e.g. switching the prompt
# version) would cost ~20s each time, so results are keyed to the clip.
job = f"{best.tag}:{uploaded.name}:{uploaded.size}"
if st.session_state.get("job") != job:
    st.session_state.pop("pred", None)
    st.session_state.pop("llm", None)
    st.session_state["job"] = job

if "pred" not in st.session_state:
    try:
        with st.status("Analysing clip…", expanded=True) as status:
            st.write(f"Loading **{best.tag}** — frozen {best.backbone} backbone"
                     f"{f' + {best.folds} fold heads' if best.folds > 1 else ''}")
            predictor, n_folds = get_predictor(str(best.path))
            st.write("Sampling windows, running the backbone and scoring each fold…")
            st.session_state["pred"] = predictor.predict(tmp)
            status.update(label="Analysis complete", state="complete", expanded=False)
    except Exception:
        ui.error_card("The computer-vision step could not process this file.", [
            "The clip may be corrupted, shorter than one sampling window, or in an "
            "unsupported codec.",
            "Re-encoding to H.264 MP4 usually fixes codec problems.",
            "Check that ffmpeg is installed and on PATH — it is used to pull the audio track.",
        ])
        with st.expander("Technical details"):
            st.code(traceback.format_exc(limit=3), language="text")
        st.stop()

pred = st.session_state["pred"]

# --------------------------------------------------------------------------- #
# results -- the view depends on the head type, not on an assumption
# --------------------------------------------------------------------------- #
if pred.task == MULTICLASS:
    ui.section("Model result", "One class per clip: these classes are mutually exclusive.")
    ui.verdict(pred.top_class, pred.behaviors)
else:
    n_hit = sum(1 for b in pred.behaviors if b["detected"])
    ui.section("Model result",
               f"{n_hit} of {len(pred.behaviors)} behaviors crossed their threshold. "
               "Behaviors are independent — any number can fire at once.")
    ui.behavior_rows(pred.behaviors)

ui.section("Evidence")
ui.evidence_timeline(pred.evidence, clip_duration(str(tmp)))

# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
ui.section("Generated report",
           "The results above are sent to the language model as structured JSON. "
           "It never sees the raw video.")

if "llm" not in st.session_state:
    try:
        with st.spinner("Generating report…"):
            from src.llm_report import generate_llm_report

            st.session_state["llm"] = generate_llm_report(pred, version=prompt_version)
    except Exception:
        ui.error_card("The language-model step failed; the results above still stand.", [
            "Check ANTHROPIC_API_KEY and network access.",
            "Without a key the deterministic template report is used instead.",
        ])
        with st.expander("Technical details"):
            st.code(traceback.format_exc(limit=3), language="text")
        st.session_state["llm"] = None

llm_out = st.session_state.get("llm")
download_text = None

if llm_out is None:
    pass
elif llm_out["source"] == "template":
    st.info(f"Language model unavailable ({llm_out['reason']}) — showing the "
            "deterministic fallback report.")
    st.code(llm_out["text"], language="text")
    download_text = llm_out["text"]
else:
    st.caption(f"Generated by {llm_out['model']} · prompt {llm_out['version']} · "
               f"{llm_out['latency_s']}s")
    s = llm_out.get("structured")
    if s is None:
        st.code(llm_out["text"], language="text")
        download_text = llm_out["text"]
    elif s.get("parse_error"):
        st.warning("The model did not return valid JSON; showing its raw output.")
        st.code(s["raw"], language="text")
        download_text = s["raw"]
    else:
        with st.container(border=True):
            st.markdown("**Detected behaviors**")
            for b in s.get("detected_behaviors", []) or ["None above threshold."]:
                st.markdown(f"- {b}")
            st.markdown("**Interpretation**")
            st.write(s.get("interpretation", ""))
            st.markdown("**Limitations**")
            for l in s.get("limitations", []):
                st.markdown(f"- {l}")
        if s.get("disclaimer"):
            st.warning(s["disclaimer"], icon=":material/warning:")
        download_text = json.dumps(s, indent=2)

if download_text is not None:
    c1, c2, _ = st.columns([1, 1, 2])
    c1.download_button("Download report", download_text, type="primary",
                       file_name=f"{tmp.stem}_autilens_report.txt", use_container_width=True)
    c2.download_button(
        "Download raw results",
        json.dumps({"checkpoint": best.tag, "test_score": best.macro_f1,
                    "prediction": pred.to_dict()}, indent=2),
        file_name=f"{tmp.stem}_autilens_results.json", use_container_width=True)

try:
    tmp.unlink()
    tmp.parent.rmdir()
except OSError:
    pass
