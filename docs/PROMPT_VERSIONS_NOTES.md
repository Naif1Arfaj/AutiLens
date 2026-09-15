# Prompt design notes (bootcamp guide §9: "design a clear prompt and test at least two versions")

Both versions live in [`src/llm_report.py`](../src/llm_report.py) as `PROMPT_V1` / `PROMPT_V2`,
selectable via the `prompt_version` query param on `/report` or the Streamlit sidebar.
Both receive **only** the CV model's structured JSON (`Prediction.to_dict()`) — never the
raw video — which is the integration required by §10.

## v1 (first draft)

```
Here is the model output for one video clip:
{payload}

Write a screening report with these sections: Detected Behaviors, Evidence,
Interpretation, Limitations. Keep it brief.
```

Problems observed reviewing v1-style outputs against the system prompt's rules:
1. **Free text is hard to grade/render.** The frontend had to parse prose to pull out
   sections, and it was easy for the model to drift from the requested structure.
2. **Confidence language wasn't grounded.** Nothing forced the model to relate its wording
   ("clearly shows...", "strongly suggests...") to the actual probability vs. threshold gap
   in the JSON — a borderline 0.51-probability detection could get described as confidently
   as a 0.95 one.
3. **No explicit "verbatim disclaimer" requirement** — relying on the system prompt alone
   to reproduce the disclaimer field exactly was inconsistent risk.

## v2 (hardened, default)

```
Structured computer-vision model output for one video clip (JSON):
{payload}

Using ONLY the facts above (no outside knowledge about autism in general), return a
JSON object with exactly these keys:
  "detected_behaviors": ...
  "confidence_language": for each detected behavior, describe confidence as
      "low" / "moderate" / "higher" relative to its own threshold ...
  "interpretation": ...
  "limitations": ...
  "disclaimer": copy the "disclaimer" field from the input JSON verbatim.
Return JSON only, no markdown fences, no extra keys.
```

Changes and why:
1. **Structured JSON output** instead of free text -> directly renderable by the UI,
   diffable for manual evaluation, and much easier to catch a bad/hallucinated field.
2. **Threshold-relative confidence buckets** (`low`/`moderate`/`higher`) computed from
   numbers already in the input -> stops the model from inventing precision it wasn't given.
3. **Explicit "copy verbatim" instruction** for the disclaimer, closing the one part of the
   output where paraphrasing would be a compliance risk.
4. **"No outside knowledge about autism in general"** line added after v1 runs occasionally
   pulled in generic textbook facts not grounded in this clip's JSON.

**Decision: v2 is the default.** v1 is kept selectable in the API/UI so the before/after
difference stays visible and testable for the required "test >= 2 versions" comparison.

## Manual evaluation

Run once `ANTHROPIC_API_KEY` is set:

```bash
python -m src.llm_report_eval    # writes reports/llm_eval_samples.md
```

That script (to be run with a live key) samples a handful of test clips, calls both prompt
versions, and writes the raw outputs side by side for manual review against the checklist
in bootcamp guide §12: factual alignment with the CV JSON, clarity, format compliance, no
invented information, and appropriate hedging language.
