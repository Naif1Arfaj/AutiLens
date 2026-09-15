# Problem Definition and Use Case (bootcamp guide §5)

## What problem does the project address?

Behavioral screening for autism-spectrum-related behaviors (repetitive motor
movements, reduced eye contact, atypical response to sensory input, etc.) currently
depends on a human specialist watching video/observation sessions in full and manually
noting when and which behaviors occur. This is slow, inconsistent between raters, and
does not scale to the volume of home-video or session-recording review that early
screening and intervention-tracking programs increasingly rely on.

## Which organization or sector faces the problem?

Health & developmental-services sector — specifically early-childhood behavioral
screening and therapy-progress review (e.g., autism assessment clinics, early
intervention centers, school-based support services).

## Who is the target user?

A **behavioral therapist, special-education support worker, or clinical researcher**
doing a first-pass review of a video clip — not a diagnosing physician, and not the
child or family directly. The system is a triage/review aid for a professional who
will make the actual judgment.

## How is the task currently performed?

A specialist watches the full video (or session recording) in real time or faster
playback, manually timestamps behaviors of interest against a checklist, and writes
up notes/a report by hand. For long recordings or high caseloads this is the
bottleneck step.

## What is the problem's impact (time, cost, quality, safety)?

- **Time**: manual review takes as long as (often longer than) the recording itself.
- **Cost**: specialist time is the scarcest and most expensive resource in this workflow.
- **Quality/consistency**: manual behavior-tagging is subject to inter-rater variation;
  a first-pass automated flag with confidence scores gives a consistent starting point.
- **Safety**: none directly — this is a screening-support tool, not a safety-critical
  control system.

## What value will the solution provide?

A first-pass, timestamped, confidence-scored flag of which of 10 defined behaviors
appear in a clip, plus a plain-language written summary — so the specialist spends
their time verifying and interpreting flagged moments instead of watching the whole
clip cold. It does **not** replace their judgment and is explicitly labeled as
non-diagnostic (see `docs/impact_card.md` and the in-app disclaimer).

## Use-case scenario (bootcamp guide §5 flow)

1. The user (therapist/researcher) uploads a short video clip via the Streamlit app.
2. The **computer-vision model** (R(2+1)D-18 video backbone + temporal/fusion head,
   `src/fusion/model.py`) analyzes the clip and extracts behavior probabilities +
   approximate evidence windows.
3. This becomes **structured data**: a JSON object with `{label, probability,
   detected, threshold}` per behavior class and `{label, window_s, peak_frame_score}`
   evidence entries (`src/inference.py::Prediction.to_dict()`).
4. The **language model** (Claude API, `src/llm_report.py`) receives exactly that
   JSON — never the raw video — and produces a structured explanation: detected
   behaviors in plain language, confidence framing, an interpretation paragraph, and
   limitations, always ending in the non-diagnostic disclaimer.
5. The final result (CV probabilities + evidence + LLM report) is shown to the user
   inside the Streamlit app, with a clear non-diagnostic banner and error handling if
   either step fails.

## Idea-requirements checklist (bootcamp guide §3)

| Requirement | How AutiLens meets it |
|---|---|
| Real problem, real organization/sector | Behavioral screening review workload, health/developmental-services sector |
| Clear target user + task | Therapist/researcher doing first-pass clip review |
| Image/video/document as primary input | Short video clips (+ audio) |
| LLM produces a useful report/explanation | `src/llm_report.py` -> structured screening report |
| CV -> LLM data path explained | Structured JSON only, see step 3-4 above and `src/inference.py` |
| Evaluable, comparable outputs | Test-set metrics (`reports/cv_summary.md`), per-clip qualitative examples (`reports/*_examples.md`) |
