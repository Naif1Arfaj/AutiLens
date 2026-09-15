"""Language-model report generation (bootcamp guide §9-10).

Core function of the LLM in this project: turn the computer-vision model's
STRUCTURED output (behavior probabilities + evidence windows, see
``src/inference.py::Prediction``) into a plain-language screening report.

Design choices required by the guide:
  * The LLM never sees the raw video — only structured JSON from the CV model
    (§10 integration requirement: "the computer-vision result must become an
    actual input used by the language model").
  * Two prompt versions are implemented (PROMPT_V1, PROMPT_V2) so the project
    can show real prompt iteration (§9 "test at least two versions"). V2 is the
    default — see PROMPT_VERSIONS_NOTES.md for why.
  * Explicit anti-hallucination constraints: the model is told to use ONLY the
    given JSON, to say so when evidence is weak, and to never claim a
    diagnosis.
  * No training/fine-tuning: this calls the Claude API (`ANTHROPIC_API_KEY`
    env var). If the key is missing or the call fails, it falls back to the
    deterministic template in ``src/report.py`` so the prototype still works
    end-to-end without an API key.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    # Loads AutiLens/.env into the process environment if present. .env is
    # git-ignored -- this is the file-based alternative to `export
    # ANTHROPIC_API_KEY=...` in your shell profile. Never commit it.
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass  # python-dotenv not installed -- shell-exported env vars still work
import time
from dataclasses import asdict

from src.inference import Prediction
from src.report import build_report as build_template_report

MODEL = os.environ.get("AUTILENS_LLM_MODEL", "claude-sonnet-5")

SYSTEM_PROMPT = """You write short, cautious research-screening reports for AutiLens AI, \
a behavior-recognition prototype (NOT a diagnostic tool).

Hard rules:
- Use ONLY the facts in the JSON you are given. Never invent a behavior, cause, \
severity, or clinical interpretation that is not directly supported by the JSON.
- If a behavior's probability is close to its threshold or evidence is weak, say so \
explicitly instead of asserting confidence.
- Never use the word "diagnose"/"diagnosis" to describe what this system does. Always \
make clear the result is not a diagnosis and requires professional follow-up for any \
real concern.
- Do not mention the child's identity, gender, or appearance — you were not given that \
information and must not guess it.
- Output must follow the exact section headers requested. No extra commentary outside \
those sections."""

# --- Prompt v1: first draft -------------------------------------------------
PROMPT_V1 = """Here is the model output for one video clip:
{payload}

Write a screening report with these sections: Detected Behaviors, Evidence, \
Interpretation, Limitations. Keep it brief."""

# --- Prompt v2: hardened after reviewing v1 outputs -------------------------
# Changes vs v1 (documented for the "test >=2 versions" requirement):
#  1. Forces JSON-only grounding with an explicit "do not use outside
#     knowledge" line (v1 outputs sometimes added generic ASD facts).
#  2. Adds a numeric confidence-language rule so wording matches the actual
#     probability instead of always sounding confident.
#  3. Requests structured JSON OUTPUT (not just free text) so the frontend can
#     render sections independently and so outputs are diffable for eval.
#  4. Adds an explicit disclaimer field that must be reproduced verbatim.
PROMPT_V2 = """Structured computer-vision model output for one video clip (JSON):
{payload}

Using ONLY the facts above (no outside knowledge about autism in general), return a \
JSON object with exactly these keys:
  "detected_behaviors": short bullet list of behaviors with probability >= threshold,
      each with a one-clause plain-language note grounded in the "evidence" field
      when available.
  "confidence_language": for each detected behavior, describe confidence as
      "low" (prob < threshold+0.1), "moderate" (< threshold+0.25), or
      "higher" (>= threshold+0.25) relative to its own threshold -- never invent
      a percentage that isn't in the input.
  "interpretation": 2-3 sentences summarizing the pattern across behaviors, citing
      only what's in the JSON.
  "limitations": 2-3 bullet points (small research dataset, approximate evidence
      windows, clip-only context).
  "disclaimer": copy the "disclaimer" field from the input JSON verbatim.
Return JSON only, no markdown fences, no extra keys."""

PROMPT_VERSIONS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


class LLMUnavailable(RuntimeError):
    pass


def _call_claude(prompt: str) -> str:
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise LLMUnavailable("anthropic package not installed") from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMUnavailable("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1200,
        # temperature/top_p/top_k are rejected (400) on Claude Sonnet 5 -- the
        # model no longer takes a sampling knob, so we don't pass one.
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def generate_llm_report(pred: Prediction, version: str = "v2") -> dict:
    """Returns {"source": "llm"|"template", "version", "text"/"structured", "latency_s", ...}."""
    payload = json.dumps(pred.to_dict(), indent=2)
    prompt = PROMPT_VERSIONS[version].format(payload=payload)

    t0 = time.time()
    try:
        raw = _call_claude(prompt)
        latency = round(time.time() - t0, 2)
        if version == "v2":
            try:
                structured = json.loads(raw)
            except json.JSONDecodeError:
                structured = {"raw": raw, "parse_error": True}
            return {"source": "llm", "version": version, "model": MODEL,
                    "latency_s": latency, "structured": structured}
        return {"source": "llm", "version": version, "model": MODEL,
                "latency_s": latency, "text": raw}
    except LLMUnavailable as e:
        return _fallback(pred, str(e))
    except Exception as e:  # noqa: BLE001 - any API-side failure (bad/revoked key,
        # rate limit, network, request error, etc.) should degrade to the
        # template, not crash the app. Keep the API's own message -- it names
        # exactly what was wrong (e.g. a rejected parameter, a bad model id).
        reason = f"{type(e).__name__}: {e}"
        try:
            import anthropic

            if isinstance(e, anthropic.AuthenticationError):
                reason = "invalid or revoked ANTHROPIC_API_KEY (401)"
            elif isinstance(e, anthropic.RateLimitError):
                reason = "rate limited / no credit balance"
        except ImportError:
            pass
        return _fallback(pred, reason)


def _fallback(pred: Prediction, reason: str) -> dict:
    fallback = build_template_report(pred)
    return {"source": "template", "reason": reason, "text": fallback["text"],
            "detected_behaviors": fallback["detected_behaviors"],
            "disclaimer": fallback["disclaimer"]}
