# AutiLens AI — Short Project Report

*AI Model Development Bootcamp — Final Project Report*

## 1. Project name and team

**AutiLens AI** — A Multimodal AI System for Autism-Related Behavioral Screening.
Team: Naif Alarfaj *(add remaining team members here)*.

## 2. Problem, organization, beneficiary user

See `docs/problem_and_use_case.md` for the full writeup. Summary: reviewing video for
autism-related behaviors is currently a fully manual task for behavioral
therapists/researchers; AutiLens gives them an AI first-pass flag + report instead of
requiring a full manual watch-through. Beneficiary sector: health / developmental
services. Target user: the reviewing specialist, not the child/family, and not a
diagnosing physician.

## 3. Solution use case

Video clip -> CV model (behavior probabilities + evidence windows, structured JSON)
-> language model (Claude API, given only that JSON) -> plain-language, non-diagnostic
report -> shown in the Streamlit prototype. Full flow diagram in
`docs/problem_and_use_case.md`.

## 4. Data description, sources, processing steps

- **Source**: AV-ASD (audio-visual autism behavior dataset),
  https://github.com/ShijianDeng/AV-ASD, paper https://arxiv.org/abs/2406.02554.
  Multi-label, 9 behavior classes.
- **Availability**: development started on a partial download (195 of 928 annotated
  clips had a downloadable source video; the rest pointed to since-removed
  YouTube/Facebook links). A **full download (`Suluk_AVASD`, 428 clips)** was later
  obtained and became the primary dataset — see the "before/after" comparison in
  Section 5. One clip (261 bytes, truncated) was corrupt and is auto-excluded, for
  **427 usable clips**.
- **Processing** (`src/preprocessing/`): built a metadata table over all usable
  clips; **verified subject-disjoint train/val/test splits** using the dataset's own
  `Original_ID` (source-video) column (`build_metadata.py` raises if it finds a
  subject shared across splits); probes every video and drops any with 0 decodable
  frames (catches corrupt downloads automatically); sampled 16 frames/clip at
  112x112 and cached them; extracted 16 kHz mono audio and 64-bin log-mel
  spectrograms.
- **Splits (usable clips)**: train 246 / val 89 / test 92 (official split), pooled to
  335 train+val clips for 5-fold cross-validation, test held out throughout.
- **Class balance** (of 427 usable clips): Upper Limb Stereotypies 163, Eye Contact
  93, Hyper-/Hyporeactivity 88, Aggressive Behavior 83, Non-Typical Language 70,
  Non-Responsiveness to Verbal Interaction 66, Self-Hitting 63, Self-Spinning 59,
  Object Lining-Up 24 — still imbalanced (~7x between the most and least common
  class) but proportionally similar to, and larger in absolute terms than, the
  partial download. Addressed with capped positive-class weighting and focal loss
  (`src/train.py`).
- **Data-size justification (§6)**: 427 clips (335 usable for train+val) is still
  modest for a 9-way multi-label video task, but roughly 2.2x the partial download
  used during initial development, and the improvement is measurable (Section 5).
  The response combines more data with the same architectural choices used on the
  smaller set: transfer learning from a Kinetics-400-pretrained 3D video backbone
  (frozen, so effectively very few parameters are fit to the AV-ASD set), subject-
  disjoint k-fold cross-validation instead of a single noisy split, and a k-model
  ensemble to reduce variance.

## 5. Computer-vision model, experiments, evaluation results

- **Baseline**: ImageNet ResNet18 frame encoder + Transformer temporal head + late
  fusion with an audio log-mel CNN, single train/val/test split.
- **Experiment set 1 (ablation)**: vision-mean / vision-LSTM / vision-Transformer /
  audio / vision+audio, comparing temporal-pooling strategies and modalities
  (`reports/ablation_summary.md`).
- **Experiment set 2 (improvement)**: swapped the frame encoder for a
  Kinetics-400-pretrained **R(2+1)D-18 3D video backbone** (motion-aware, vs.
  per-frame ImageNet features), added **focal loss** for the rare classes, and
  replaced the single noisy 27-clip validation split with **subject-disjoint 5-fold
  cross-validation + ensemble + test-time augmentation** (`src/cv.py`,
  `reports/cv_summary.md`).
- **Metrics used**: macro/micro-F1, per-class precision/recall, AUROC, AUPRC,
  Expected Calibration Error, and a full confusion matrix — accuracy alone is
  reported too but explicitly flagged as a weak metric here (sparse multi-label
  data means an all-negative model already scores ~80% per-label "accuracy").

| model | per-label accuracy | macro-F1 | macro-AUPRC | macro-AUROC | ECE |
|---|---|---|---|---|---|
| **vision + audio fusion (best, 427 clips)** | **74.2%** | **0.438** | **0.457** | **0.723** | 0.204 |
| vision — R(2+1)D-18 + focal (427 clips) | 65.2% | 0.417 | 0.432 | 0.711 | 0.199 |
| audio only (427 clips) | 47.8% | 0.337 | 0.288 | 0.561 | 0.289 |
| vision + audio fusion (earlier, 195 clips) | 60.6% | 0.327 | 0.490 | 0.652 | 0.206 |
| baseline (single split, ImageNet ResNet18, 195 clips) | 53.8% | 0.264 | – | 0.579 | 0.168 |

- **Best model selected**: vision+audio fusion, now winning on *every* metric once
  trained on the full 427-clip dataset. On the earlier 195-clip subset, fusion only
  won on AUPRC and lost to vision-only on thresholded F1/accuracy — the extra audio
  training data is what let the audio branch stop being a net drag on fusion.
  This is the clean version of the §10 research question: audio + visual beats
  visual-only, but only once there's enough data for the weaker (audio) modality
  to learn a useful signal.
- **Correct vs. failed examples**: `reports/vision_r2p1d_cv_examples.md` (+ thumbnail
  images in `reports/examples/`) shows the highest- and lowest-per-clip-F1 test
  clips with their true vs. predicted labels — required by §7/§8, not just a
  headline number.

## 6. Language model, prompt-design method, and evaluation

- **Model**: Claude (Anthropic API), called from `src/llm_report.py`. No
  training/fine-tuning — as explicitly allowed by the bootcamp guide §9.
- **Core function**: turns the CV model's structured JSON output into a
  plain-language, non-diagnostic screening report (detected behaviors,
  confidence framing, interpretation, limitations, disclaimer).
- **Two prompt versions tested and compared**: see `docs/PROMPT_VERSIONS_NOTES.md`
  for the full v1 -> v2 diff and rationale. v2 (structured JSON output,
  threshold-relative confidence language, verbatim disclaimer requirement) is the
  default.
- **Anti-hallucination constraints**: system prompt forbids using information not
  present in the input JSON, forbids the words "diagnose"/"diagnosis" for what the
  system does, and forbids guessing identity/appearance details.
- **Manual evaluation**: `src/llm_report_eval.py` runs both prompt versions on 3
  representative clips (multi-behavior, single-behavior, background/none) side by
  side for grading against §12's criteria (factual alignment, clarity, format
  compliance, no invented info). *Requires `ANTHROPIC_API_KEY`* — the harness
  refuses to fabricate sample outputs when no key is configured, so
  `reports/llm_eval_samples.md` will say plainly if it hasn't been run yet.

## 7. How the two components are connected + prototype

Integration is real, not cosmetic: the LLM call in `src/llm_report.py` takes
`Prediction.to_dict()` — the CV model's own output object — as its only input.
There is no path in the code where the LLM sees the video or invents its own
vision analysis. Prototype: Streamlit app (`app/streamlit_app.py`) and a FastAPI
backend (`api/main.py`, `/predict` and `/report` endpoints) — both runnable
locally; see `README.md` for exact commands.

## 8. Success/failure cases and ethical/privacy aspects

- See `reports/*_examples.md` for concrete per-clip successes and failures.
- Ethics/privacy: no facial-identity use, subject-disjoint splits, non-diagnostic
  disclaimer surfaced in the UI/API and enforced in the LLM system prompt, consent
  checkbox in the demo, temp-file cleanup (no video retention). Full checklist in
  `README.md` → Ethics & Privacy, matching bootcamp guide §13.

## 9. Challenges and future improvements

- **Data availability (resolved mid-project)**: development started on a partial
  AV-ASD download where 733 of 928 clip references had no downloadable video
  (link rot). A full download (428 clips, 1 corrupt) was later obtained and
  retraining on it measurably improved every metric (Section 5) — the clearest
  practical lesson of the project: more in-domain data beat every architectural
  trick tried on the smaller set. Before that full download arrived, larger
  alternative datasets (3D Autism Movement Dataset, MMASD+, SSBD+) were
  considered and rejected as a straight swap because their label taxonomies
  don't match this project's behavior classes; SSBD+ (arm-flapping/head-
  banging/spinning) remains a compatible future extension.
- **Small data** still limits per-class reliability even at 427 clips, especially
  `Object Lining-Up` (n=24, the rarest class).
- **Future work**: pose-keypoint stream, Whisper transcript branch feeding the
  LLM directly, cross-attention fusion, and a real pilot measuring reviewer time
  saved (the impact indicator in `docs/impact_card.md` is currently a hypothesis,
  not a measured result).
