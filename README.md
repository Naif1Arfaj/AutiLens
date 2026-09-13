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
| §11/§13 Frontend | Streamlit demo: consent → upload → results → evidence → report | `app/streamlit_app.py` |

The LLM explanation layer (§8 optional / §11) is intentionally **not** included —
the report is generated deterministically from structured model outputs so it
stays faithful to what the model actually predicted.

### Improvements implemented on top of the MVP

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

`configs/default.yaml` points at the downloaded AV-ASD folder (default `../avasd_data`):

```
avasd_data/
├── clips_video/   *.mp4   (195 usable clips)
├── clips_audio/   *.wav
└── csvs/          train.csv val.csv test.csv dataset.csv   (multi-label, 10 classes)
```

Only 195 of 928 annotated clips have a downloadable source video
(train 121 / val 27 / test 47). Splits are **subject-disjoint** (no source video
shared across splits) — verified by `build_metadata`.

**10 behavior labels:** Absence/Avoidance of Eye Contact · Aggressive Behavior ·
Hyper-/Hyporeactivity to Sensory Input · Non-Responsiveness to Verbal Interaction ·
Non-Typical Language · Object Lining-Up · Self-Hitting / Self-Injurious Behavior ·
Self-Spinning / Spinning Objects · Upper Limb Stereotypies · Background.

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
# 1. Data prep: metadata table, split checks, frame + mel cache  (~2 min)
bash scripts/prepare_data.sh

# 2. Train one model
python -m src.train --modality av --temporal transformer --tag av_transformer

# 3. Evaluate on the held-out test split (writes reports/*_test_metrics.json + confusion PNG)
python -m src.evaluate models/av_transformer.pt --split test

# 4. Full ablation study: vision-only / audio-only / vision+audio + temporal comparison
bash scripts/run_ablation.sh          # -> reports/ablation_summary.md
```

### Serve

```bash
# FastAPI backend
AUTILENS_CKPT=models/av_transformer.pt python -m uvicorn api.main:app --port 8000
curl -F "file=@../avasd_data/clips_video/SOME_CLIP.mp4" http://localhost:8000/report

# Streamlit demo (in-process inference, no API needed)
streamlit run app/streamlit_app.py
```

---

## Configuration

Everything is in `configs/default.yaml` — frame count/size, audio mel params,
backbone, temporal model, fusion modality (`vision` | `audio` | `av`), training
hyper-parameters, threshold tuning. CLI flags on `src.train`
(`--modality`, `--temporal`, `--backbone`, `--epochs`) override it per run.

---

## Results

**Cross-validated** (5-fold subject-disjoint on train+val, official test split held
out, k-model ensemble + TTA) — `reports/cv_summary.md`:

| model | per-label acc | macro-F1 | macro-AUPRC | macro-AUROC | ECE |
|---|---|---|---|---|---|
| vision — R(2+1)D-18 + focal | **64.9%** | **0.347** | 0.374 | 0.648 | 0.188 |
| audio — mel-CNN + focal | 44.7% | 0.277 | 0.303 | 0.547 | 0.267 |
| vision + audio — late fusion | 60.6% | 0.327 | **0.490** | **0.652** | 0.206 |

MVP baseline (single split, ImageNet ResNet18 + Transformer): per-label acc 53.8%,
macro-F1 0.264, AUROC 0.579 — see `reports/ablation_summary.md`.

Takeaways: the Kinetics video backbone is the biggest win (+11 pts accuracy).
Vision-only is best on thresholded F1; fusion wins clearly on ranking quality
(AUPRC 0.37 → 0.49) — the §10 result that audio carries complementary signal.
With 148 training clips, fold spread is ±0.03–0.07 F1, so treat these as
directional, not a benchmark. Accuracy is a weak yardstick here (sparse labels →
an all-negative model scores ~80%); macro-F1 / AUPRC / AUROC are the ones to read.

---

## Ethics & privacy (PDF §14)

- Presented as research/screening support, **never** clinical diagnosis.
- No facial-appearance-based inference; labels are observable behaviors.
- Subject-disjoint splits to reduce identity leakage.
- Demo keeps uploads in a temp dir and deletes them after inference; no retention.
- Follow the AV-ASD licence and source-video terms; do not redistribute media.
- Report limitations and data-source bias; avoid high-stakes decisions from output.
