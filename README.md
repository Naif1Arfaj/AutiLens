# AutiLens AI

**A multimodal AI system for autism-related behavioral screening — research prototype.**

AutiLens AI recognises observable, dataset-defined behaviors in short video clips
(audio + visual) and produces an explainable screening-style report.

> ⚠️ **Not a diagnostic tool.** It reports *behavior recognition* only. It does not
> diagnose autism. Autism diagnosis requires qualified professionals and broader
> clinical evidence. Do not expose identifiable participant video in demos.

Built from the *AutiLens AI Complete Project Guide* on the **AV-ASD** dataset
([repo](https://github.com/ShijianDeng/AV-ASD), [paper](https://arxiv.org/abs/2406.02554)).

---

## What is implemented

| PDF section | Component | Location |
|---|---|---|
| §6 Data preparation | Metadata table, subject-disjoint split verification, frame + log-mel caching | `src/preprocessing/` |
| §4/§7 Visual baseline | Pretrained ResNet/EfficientNet frame encoder + temporal head (mean / LSTM / Transformer) | `src/vision/model.py` |
| §8 Audio branch | Log-mel CNN encoder | `src/audio/model.py` |
| §4/§8 Multimodal fusion | Gated late fusion of L2-normalised modality embeddings | `src/fusion/model.py` |
| §9 Training plan | Multi-label BCE w/ capped pos-weights, cosine schedule, early stopping, val-tuned per-class thresholds | `src/train.py` |
| §10 Evaluation | Macro/micro-F1, precision/recall, AUROC, AUPRC, ECE, per-class confusion, ablation study | `src/evaluate.py`, `src/evaluation/` |
| §11 Explainable report | **Deterministic template** report from model outputs (no LLM — cannot invent observations) | `src/report.py` |
| §12 Backend | FastAPI: `/health`, `/predict`, `/report` | `api/main.py` |
| §11/§13 Frontend | Streamlit demo: consent → upload → results → evidence → report. No model picker — the most accurate checkpoint is served automatically | `app/streamlit_app.py`, `app/theme.py`, `app/components.py` |

The LLM report layer lives in `src/llm_report.py` and receives only the structured
model output (probabilities + evidence windows), never the raw video. Without an
`ANTHROPIC_API_KEY` it falls back to the deterministic template in `src/report.py`,
so the report can never invent observations the model did not produce.

### v2 pipeline — frozen backbone + cached features (current)

| Change | Where | Why |
|---|---|---|
| **Windowed fixed-fps sampling** (3 x 16 consecutive frames @ 10 fps) | `src/preprocessing/windows.py` | v1 spread 16 frames over the WHOLE clip; the 887 s clip was sampled at 1 frame / 55 s, aliasing away the 2-4 Hz stereotypy motion entirely |
| **Frozen-backbone feature cache** (Swin3D-T, K400 77.7%) | `src/preprocessing/extract_features.py` | The backbone never trains, so it is run ONCE instead of every epoch — 5-fold runs went ~45 min -> ~20 s |
| **Precision floor on thresholds** | `src/evaluation/metrics.py` | v1 maximised F1 unconstrained and collapsed to "predict everything" (recall 1.00 / precision 0.23) |
| **Probability calibration** (isotonic/Platt on OOF) | `src/evaluation/calibration.py` | ECE 0.204 -> 0.033; probabilities shown to users now mean something. Clamped to (0.005, 0.995) so a screening tool never displays "100%" |
| **Feature-space mixup** | `src/fusion/head.py` | Free with a cache; restores continuous augmentation diversity |
| **SpecAugment on audio** | `src/preprocessing/extract_features.py` | The audio branch previously had *zero* augmentation |
| **focal alpha replaces pos_weight** | `src/fusion/head.py` | Using both double-counted positives — a direct cause of the low precision |
| **API/UI serve the k-fold ensemble** | `src/inference_fast.py`, `api/main.py` | The demo was resolving to a single fold, scoring below the reported numbers |
| **Checkpoint chosen by accuracy, not filename** | `src/model_registry.py` | Selection by name served `autilens_v2` (test macro-F1 0.458) while `vision_swin_fast` (0.494) sat unused in the same directory. Both the API and the UI now rank on the **out-of-fold** scores in `reports/`, nested-CV runs first — the held-out test set is reporting-only and must never drive selection |
| **Head-type-aware results** | `src/inference_fast.py`, `app/components.py` | Multi-label (sigmoid + per-class thresholds) and multi-class (softmax, one winning class) checkpoints render through different views, selected from `Prediction.task` — see `docs/MULTICLASS_CHECKPOINTS.md` |

```bash
python3 -m src.preprocessing.windows            # one-off, ~6 min
python3 -m src.preprocessing.extract_features   # one-off, ~5 min
bash scripts/run_fast.sh                        # all configs + ablations, seconds each
```

### v1 improvements (superseded, kept for the ablation history)

| Improvement | Where | Effect |
|---|---|---|
| **Kinetics-400 pretrained 3D video backbone** (`r2plus1d_18` / `mc3_18` / `r3d_18`) | `src/vision/model.py` (`VideoEncoder`) | Motion-aware features instead of per-frame ImageNet; set `model.vision_backbone` |
| **Focal loss** (γ=2) | `src/train.py` (`FocalLoss`) | Better learning on rare classes (`Object Lining-Up` n=10); `train.loss: focal` |
| **Subject-disjoint k-fold CV** + OOF threshold tuning | `src/cv.py` | Replaces the noisy 27-clip val split; metrics as mean ± std |
| **k-model ensemble** | `src/cv.py` | Averages fold models on the test split |
| **Test-time augmentation** (h-flip) | `src/train.py::_infer`, `src/inference.py` | Small, free robustness gain; `eval.tta: true` |
| Stronger train aug (brightness, temporal dropout) | `src/datasets/avasd.py` | Regularisation for the tiny training set |

```bash
bash scripts/run_improved.sh          # -> reports/cv_summary.md  (CV + ensemble, all 3 modalities)
python -m src.cv --modality av --backbone r2plus1d_18 --loss focal --tag av_r2p1d_cv
```

Still open (need more data / more compute): recovering more AV-ASD source videos,
adding SSBD / 3D-Movement / Autism-Action datasets, a pose-keypoint stream, a
Whisper-transcript text branch, and cross-attention fusion. See the "How to
improve" notes.

---

## Dataset layout expected

`configs/default.yaml` points at the **full AV-ASD download** (`Suluk_AVASD/`):

```
Suluk_AVASD/
├── clips_video/   *.mp4   (427 usable clips)
├── clips_audio/   *.wav
└── csvs/          train.csv val.csv test.csv   (multi-label, 9 classes, Original_ID = subject)
```

427 of 428 clips are usable (1 truncated/corrupt file is auto-excluded by
`build_metadata`, which probes every file and drops anything with 0 decodable
frames). Splits are **subject-disjoint** (`Original_ID` column; no source video
shared across train/val/test) — verified by `build_metadata`, train 246 / val 89 /
test 92.

**9 behavior labels** (this full download drops the `Background` column that the
earlier partial download had): Absence/Avoidance of Eye Contact · Aggressive
Behavior · Hyper-/Hyporeactivity to Sensory Input · Non-Responsiveness to Verbal
Interaction · Non-Typical Language · Object Lining-Up · Self-Hitting / Self-Injurious
Behavior · Self-Spinning / Spinning Objects · Upper Limb Stereotypies.

> An earlier, partial download (195/928 clips, 10 labels incl. `Background`) was used
> for initial development. Those models/results are archived under
> `models/archive_195clip/` and `reports/archive_195clip/` for comparison — see
> "Results" below for the before/after.

---

## Setup

```bash
cd AutiLens
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Runs on CPU, Apple Silicon (MPS), or CUDA — auto-detected (`train.device: auto`).

---

## Run the pipeline

```bash
# 1. Data prep: metadata table, split checks, frame + mel cache  (~5 min for 427 clips)
bash scripts/prepare_data.sh

# 2. RECOMMENDED (v2): build the caches once, then every experiment takes seconds
python -m src.preprocessing.windows              # windowed frames  (~6 min)
python -m src.preprocessing.extract_features     # frozen features  (~5 min)
bash scripts/run_fast.sh                         # -> reports/model_comparison.md

# Train one config (~20 s) / reproduce the production model:
python -m src.cv_fast --modality av --backbone swin3d_t --min-precision 0.4 --tag autilens_v2

# v1 pipeline (slow, ~1-2 h) — kept for comparison:
EPOCHS=30 FOLDS=5 bash scripts/run_improved.sh   # -> reports/cv_summary.md
```

### Serve

```bash
# Which checkpoint gets served, and why (ranked by OOF score from reports/)
python -m src.model_registry

# FastAPI backend — serves the highest-scoring checkpoint; AUTILENS_CKPT overrides.
python -m uvicorn api.main:app --port 8000
curl -s http://localhost:8000/health          # reports the checkpoint and its score
curl -F "file=@/Users/naifalarfaj/Downloads/Suluk_AVASD/clips_video/SOME_CLIP.mp4" http://localhost:8000/report

# Streamlit demo (in-process inference, no API needed)
streamlit run app/streamlit_app.py

# Result rendering for both head shapes, without loading a backbone (<1 s)
python3 scripts/check_result_shapes.py

# UI: asserts the stylesheet is delivered via st.html and never through the
# markdown parser, which silently printed the whole sheet onto the page as text
python3 scripts/check_ui_render.py
```

Selection is by **measured accuracy**: every non-fold `models/*.pt` is matched to its
`reports/<tag>_cv.json` and ranked on **out-of-fold** macro-F1 (accuracy for multi-class
heads), with nested-CV runs ahead of pre-nested ones and ties broken by lower calibration
error. Selecting on the held-out test set would leak it, so that number is reported but
never used to choose. A newly trained checkpoint is served as
soon as its report lands — no code change, and no dropdown asking a reviewer to pick a
`.pt` file. The UI shows the winner and the full ranked candidate list instead.

Multi-class checkpoints are supported alongside the multi-label ones; the required
checkpoint keys are documented in [`docs/MULTICLASS_CHECKPOINTS.md`](docs/MULTICLASS_CHECKPOINTS.md).

---

## Configuration

Everything is in `configs/default.yaml` — frame count/size, audio mel params,
backbone, temporal model, fusion modality (`vision` | `audio` | `av`), training
hyper-parameters, threshold tuning. CLI flags on `src.train`
(`--modality`, `--temporal`, `--backbone`, `--epochs`) override it per run.

---

## Results

### v2 pipeline (current)

Production model: **`models/autilens_v2.pt`** — frozen Swin3D-T + gated audio fusion,
5-fold ensemble, calibrated, precision floor 0.4. Full table in
`reports/model_comparison.md`.

Because MPS introduces run-to-run nondeterminism, single runs vary by roughly
±0.03 macro-F1. These are means ± std over **5 seeds**, which is the honest
comparison:

> ⚠️ **The v2 figures below were measured under a validation leak and are optimistic.**
> The epoch was selected on the outer validation fold, and thresholds/calibrators were
> then fitted on the pooled out-of-fold predictions they scored. Re-running the same
> configuration under nested CV gives **9-class OOF macro-F1 0.413, not 0.489** — the
> bias was **0.076**. Treat every v2 number here as an upper bound, and see
> `docs/v3_status.md` for the corrected results.
>
> | corrected (nested CV) | scope | OOF macro-F1 | precision | AUROC |
> |---|---|---|---|---|
> | 9-behavior (same config as v2) | 9-class | **0.413** | 0.395 | 0.723 |
> | Stage 1 families (direct, 3 seeds) | 4-family | **0.571 ± 0.020** | 0.533 ± 0.011 | 0.731 ± 0.008 |
> | derived 9→4 baseline (3 seeds) | 4-family | 0.540 ± 0.022 | 0.492 ± 0.018 | 0.719 ± 0.006 |
>
> Scopes are different exams: the 4-family and 9-behavior numbers are **not** comparable.

| config | macro-F1 | precision | recall | AUPRC | AUROC | ECE |
|---|---|---|---|---|---|---|
| **v2 vision+audio** | 0.419 ± 0.026 | **0.572 ± 0.059** | 0.448 | **0.507 ± 0.008** | **0.751 ± 0.007** | **0.033** |
| v2 vision-only | 0.439 ± 0.024 | 0.488 ± 0.055 | 0.550 | 0.505 ± 0.010 | 0.737 ± 0.007 | 0.040 |
| v1 baseline (R(2+1)D-18 fusion) | 0.438 | 0.420 | 0.524 | 0.457 | 0.723 | 0.204 |

**What actually improved, stated plainly:**

- **macro-F1 did not improve.** 0.419 ± 0.026 vs the 0.438 baseline is within noise.
  Anyone quoting a single lucky run at 0.51 would be cherry-picking.
- **Precision improved a lot: 0.420 → 0.572 (+0.15).** This was the goal — the v1
  model degenerated toward "predict everything" (one class sat at recall 1.00 /
  precision 0.23).
- **Ranking improved: AUPRC 0.457 → 0.507, AUROC 0.723 → 0.751.** Small but
  consistent across all 5 seeds.
- **Calibration improved ~6x: ECE 0.204 → 0.033.** Reported probabilities now
  roughly match observed frequencies, so they are meaningful to show a user.
- **Iteration got ~135x faster: ~45 min → ~20 s per 5-fold run**, because the frozen
  backbone is now run once and cached instead of re-run every epoch.

### What did *not* work

Replacing the from-scratch mel-CNN with a **frozen ImageNet ResNet18 over the mel
spectrogram made audio-only much worse** — macro-F1 0.337 → 0.087, AUROC 0.475
(below chance). Generic ImageNet texture features do not transfer to spectrograms
without fine-tuning. Audio still helps *in fusion* (it is why v2 fusion beats v2
vision-only on precision, AUROC and ECE), but a real speech encoder
(`torchaudio.pipelines.WAV2VEC2_BASE`) is the obvious next step — especially for
`Non-Typical Language` and `Non-Responsiveness to Verbal Interaction`, which video
cannot observe at all.

### Ablations (single runs, so read alongside the ±0.03 noise)

| variant | macro-F1 | precision | ECE |
|---|---|---|---|
| full v2 | 0.458 | 0.462 | 0.046 |
| no precision floor | 0.458 | 0.461 | 0.043 |
| no mixup | 0.435 | 0.485 | 0.028 |
| no calibration | 0.433 | 0.467 | 0.045 |
| r2plus1d_18 instead of Swin3D-T | 0.472 | 0.476 | 0.015 |

The backbone swap is **not** clearly responsible for the gains — `av_r2p1d_fast`
scores comparably. Most of the improvement came from the windowed sampling,
calibration and threshold changes, which apply to either backbone.

### Effect of the full dataset (427 vs. 195 usable clips)

| model | dataset | macro-F1 | macro-AUROC |
|---|---|---|---|
| vision | 195 clips (`archive_195clip/`) | 0.347 | 0.648 |
| vision | **427 clips** | 0.417 | 0.711 |
| vision + audio | 195 clips | 0.327 | 0.652 |
| vision + audio | **427 clips** | 0.438 | 0.723 |

More data remains the single largest lever measured in this project — larger than
any architecture or loss change tried here.

This is the clean §10 result: audio + visual > visual-only, once there's enough
data for both branches to actually learn something. Old MVP baseline (single
split, ImageNet ResNet18, 195 clips): per-label acc 53.8%, macro-F1 0.264 —
see `reports/archive_195clip/ablation_summary.md`.

Fold spread on the new run is tighter (macro-F1 std ±0.02–0.04 vs. ±0.03–0.07
before) — another sign the bigger dataset is reducing variance, not just
shifting the mean. Still a research prototype: macro-F1 in the 0.3–0.4 range and
per-label accuracy are both modest in absolute terms, and accuracy alone is a
weak yardstick here (sparse labels mean an all-negative model already scores
~80%) — macro-F1 / AUPRC / AUROC are the metrics to trust.

---

## Language model component (bootcamp guide §9-10)

`src/llm_report.py` calls the Claude API to turn the CV model's **structured JSON
output** (probabilities + evidence windows, never the raw video) into a plain-language,
non-diagnostic report. Two prompt versions are implemented and compared —
see [`docs/PROMPT_VERSIONS_NOTES.md`](docs/PROMPT_VERSIONS_NOTES.md). Requires
`ANTHROPIC_API_KEY`; without it, both `api/main.py` and `app/streamlit_app.py`
automatically fall back to the deterministic template in `src/report.py` so the
prototype still runs end-to-end.

```bash
export ANTHROPIC_API_KEY=sk-...
python -m src.llm_report_eval          # manual-evaluation samples, both prompt versions
```

## Bootcamp deliverables

| Deliverable | File |
|---|---|
| Problem & use case (§5) | [`docs/problem_and_use_case.md`](docs/problem_and_use_case.md) |
| Short report (§16) | [`docs/short_report.md`](docs/short_report.md) |
| Impact card | [`docs/impact_card.md`](docs/impact_card.md) |
| Prompt design notes (§9) | [`docs/PROMPT_VERSIONS_NOTES.md`](docs/PROMPT_VERSIONS_NOTES.md) |
| Correct/failed CV examples (§7-8) | `reports/vision_r2p1d_cv_examples.md` (run `python -m src.evaluation.examples <tag>`) |

---

## Ethics & privacy (bootcamp guide §13)

- Presented as research/screening **support**, never a final decision or clinical diagnosis.
- No facial-appearance-based inference; labels are observable behaviors, not identity.
- Subject-disjoint splits to reduce identity leakage between train/val/test.
- Demo keeps uploads in a temp dir and deletes them after inference; no retention.
- Consent checkbox required in the Streamlit demo before any upload is processed.
- LLM system prompt explicitly forbids inventing information, guessing identity/
  appearance, or using the words "diagnose"/"diagnosis" for what the system does.
- Follow the AV-ASD licence and source-video terms; do not redistribute media.
- Report limitations and data-source bias (see `docs/short_report.md` §9);
  avoid high-stakes decisions from output; any real concern is routed to a human
  specialist, not resolved by the app.
