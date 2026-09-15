"""Presentational pieces for the AutiLens UI.

Each function returns or renders self-contained HTML that leans on the tokens in
``app/theme.py``. Keeping them here means ``streamlit_app.py`` stays a readable
description of the flow rather than a wall of markup.

Icons are inline SVG, never emoji: emoji render inconsistently across platforms,
carry no accessible name, and read as decoration in a clinical context.
"""
from __future__ import annotations

import html
from typing import Iterable

import streamlit as st

from app.theme import series

#: Streamlit's ``st.html`` sanitizer strips <svg> outright -- every inline icon
#: rendered as an empty box. Streamlit already ships the Material Symbols
#: Rounded font for its own icons, so we draw ours the same way: a span whose
#: text is a ligature name. That is plain text, so nothing can strip it.
_ICON = {
    "lens": "visibility",
    "check": "check",
    "alert": "warning",
    "schedule": "schedule",
    "person": "person",
    "compare": "compare_arrows",
}


def icon(name: str, size: int = 18, color: str = "currentColor", label: str = "") -> str:
    """A Material Symbols ligature span. ``label`` names it for screen readers."""
    a11y = (f'role="img" aria-label="{html.escape(label)}"' if label
            else 'aria-hidden="true"')
    return (f'<span class="al-ico" translate="no" {a11y} '
            f'style="font-size:{size}px;color:{color}">{_ICON.get(name, name)}</span>')


def _e(v) -> str:
    return html.escape(str(v))


def _pct(v: float) -> float:
    return max(0.0, min(100.0, float(v) * 100.0))


# --------------------------------------------------------------------------- #
# chrome
# --------------------------------------------------------------------------- #
def masthead() -> None:
    st.html(f"""
<div class="al-mast">
  <div class="al-mark">{icon('lens', 23, '#fff')}</div>
  <div>
    <h1>AutiLens AI</h1>
    <p class="al-sub">Multimodal behavioral screening support &mdash; research prototype</p>
  </div>
</div>""")
    # The landing gate is otherwise one-way: once "started" is set there is no
    # route back to the overview short of restarting the app. A link cannot
    # clear session state (st.html runs no JavaScript), so this is a real button.
    if st.session_state.get("started"):
        if st.button("← Overview", key="al_back",
                     help="Back to what this project is"):
            st.session_state["started"] = False
            st.rerun()


def safety_banner() -> None:
    """The one thing on this page that must never be missed."""
    st.html(f"""
<div class="al-safety" role="note">
  <div style="flex:0 0 auto;padding-top:1px">{icon('alert', 20, 'var(--al-warn)')}</div>
  <div>
    <strong>Not a diagnostic tool.</strong>
    <p>AutiLens recognises dataset-defined behaviors in short video clips and produces an
    explainable research report. It does not diagnose autism &mdash; that requires qualified
    professionals and far broader clinical evidence. Do not upload identifiable video of
    children without consent.</p>
  </div>
</div>""")


def stepper(steps: Iterable[str], current: int) -> None:
    out = []
    for i, name in enumerate(steps):
        cls = "done" if i < current else ("now" if i == current else "")
        badge = icon("check", 12, "currentColor") if i < current else str(i + 1)
        aria = ' aria-current="step"' if i == current else ""
        out.append(f'<div class="al-step {cls}"{aria}><span class="n">{badge}</span>{_e(name)}</div>')
    st.html(f'<nav class="al-steps" aria-label="Progress">{"".join(out)}</nav>')


# --------------------------------------------------------------------------- #
# model transparency -- show which model and why, instead of asking
# --------------------------------------------------------------------------- #
def model_card(card) -> None:
    score = f"{card.macro_f1:.3f}" if card.scored else "—"
    meta = [card.modality_label, card.backbone]
    if card.folds > 1:
        meta.append(f"{card.folds}-fold ensemble")
    meta.append("calibrated" if card.calibrated else "uncalibrated")
    if card.task == "multiclass":
        meta.append("multi-class head")
    # src/model_registry flags runs evaluated before nested CV as optimistic.
    # That caveat belongs next to the number, not buried in the registry.
    if card.scored and not getattr(card, "nested_cv", True):
        meta.append("⚠ pre-nested-CV (optimistic)")
    st.html(f"""
<div class="al-model">
  <div>
    <div class="al-eyebrow">Serving the most accurate checkpoint</div>
    <div class="al-name">{_e(card.tag)}</div>
    <div class="al-meta">{_e(' · '.join(meta))}</div>
  </div>
  <div class="al-score"><div class="v">{score}</div>
    <div class="k">OOF {_e(card.metric_name)}</div></div>
</div>""")


def model_table(cards, best) -> None:
    rows = []
    for c in cards:
        sel = c.path == best.path
        score = f"{c.macro_f1:.3f}" if c.scored else "not evaluated"
        ece = f"{c.ece:.3f}" if c.ece is not None else "—"
        rows.append(
            f'<tr style="{"font-weight:600;color:var(--al-primary)" if sel else ""}">'
            f'<td style="padding:.35rem .7rem .35rem 0">{"→ " if sel else ""}{_e(c.tag)}</td>'
            f'<td style="padding:.35rem .7rem">{_e(c.modality_label)}</td>'
            f'<td style="padding:.35rem .7rem">{_e(c.backbone)}</td>'
            f'<td style="padding:.35rem .7rem;text-align:right;'
            f'font-family:JetBrains Mono,monospace">{score}</td>'
            f'<td style="padding:.35rem 0;text-align:right;'
            f'font-family:JetBrains Mono,monospace">{ece}</td></tr>')
    st.html(f"""
<p style="font-size:.9rem;color:var(--al-muted);margin:.2rem 0 .8rem">
Every checkpoint in <code>models/</code> is ranked by its <strong>out-of-fold</strong>
score from <code>reports/</code> &mdash; nested-CV runs first, ties broken by lower
calibration error. The held-out test set is for reporting only and never drives
selection, which would leak it. The top row is
served automatically &mdash; choosing a checkpoint by hand is not a decision a reviewer
should have to make, and picking by filename served a weaker model than this one.</p>
<table style="width:100%;border-collapse:collapse;font-size:.88rem">
<thead><tr style="border-bottom:1px solid var(--al-border);color:var(--al-muted);
  text-align:left;font-size:.76rem;text-transform:uppercase;letter-spacing:.07em">
<th style="padding:0 .7rem .4rem 0">checkpoint</th><th style="padding:0 .7rem .4rem">modality</th>
<th style="padding:0 .7rem .4rem">backbone</th>
<th style="padding:0 .7rem .4rem;text-align:right">OOF score</th>
<th style="padding:0 0 .4rem;text-align:right">ECE</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>""")


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
def behavior_rows(behaviors: list[dict]) -> None:
    """Multi-label view: one row per behavior, threshold drawn on the bar.

    The old table hid ``threshold`` entirely, so a 0.31 probability against a
    0.28 threshold looked like a miss. Here the decision point is visible.
    """
    rows = []
    for b in behaviors:
        hit = bool(b.get("detected"))
        p = _pct(b["probability"])
        thr = b.get("threshold")
        tick = (f'<span class="thr" style="left:{_pct(thr):.1f}%" '
                f'title="decision threshold {thr:.2f}"></span>') if thr is not None else ""
        chip = (f'<span class="al-chip on">{icon("check", 11)} detected</span>' if hit
                else '<span class="al-chip off">below threshold</span>')
        rows.append(f"""
<div class="al-row {'hit' if hit else ''}">
  <div class="al-row-top">
    <span class="al-row-label">{_e(b['label'])}</span>{chip}
    <span class="al-row-val">{p:.1f}%</span>
  </div>
  <div class="al-bar"><span class="fill" style="width:{p:.1f}%"></span>{tick}</div>
</div>""")
    legend = ('<p style="font-size:.82rem;color:var(--al-muted);margin:.75rem 0 0">'
              'The vertical tick on each bar is that behavior&rsquo;s decision threshold, '
              'tuned on out-of-fold predictions. A bar crossing its tick counts as detected.</p>')
    st.html(f'<div class="al-card">{"".join(rows)}{legend}</div>')


def verdict(top: dict, behaviors: list[dict]) -> None:
    """Multi-class view: one winning class plus the full distribution."""
    palette = series()
    segs, legend = [], []
    for i, b in enumerate(behaviors):
        w = _pct(b["probability"])
        if w <= 0:
            continue
        c = palette[i % len(palette)]
        segs.append(f'<span style="width:{w:.2f}%;background:{c}" '
                    f'title="{_e(b["label"])} {w:.1f}%"></span>')
        legend.append(f'<span><i style="background:{c}"></i>{_e(b["label"])} '
                      f'&middot; {w:.1f}%</span>')
    st.html(f"""
<div class="al-verdict">
  <div class="k">Predicted class</div>
  <div class="v">{_e(top['label'])}</div>
  <div class="k">Confidence {_pct(top['probability']):.1f}% &middot; across
    {len(behaviors)} mutually exclusive classes</div>
  <div class="al-dist" role="img"
       aria-label="Probability distribution across classes">{''.join(segs)}</div>
  <div class="al-legend">{''.join(legend)}</div>
</div>""")


def evidence_timeline(evidence: list[dict], duration: float) -> None:
    if not evidence:
        st.html('<div class="al-empty">No behavior crossed its detection threshold, '
                    'so there are no evidence windows to show for this clip.</div>')
        return

    span = duration if duration and duration > 0 else max(
        (e["window_s"][1] for e in evidence), default=1.0) or 1.0

    # Behaviors often peak in the SAME window, and drawing those on one row hides
    # all but the last. Pack them into lanes instead, so every window stays
    # visible and the block count matches the list underneath.
    lanes: list[float] = []
    placed = []
    for e in evidence:
        s0, s1 = float(e["window_s"][0]), float(e["window_s"][1])
        lane = next((i for i, end_s in enumerate(lanes) if s0 >= end_s), len(lanes))
        if lane == len(lanes):
            lanes.append(s1)
        else:
            lanes[lane] = s1
        placed.append((e, s0, s1, lane))

    n_lanes = max(len(lanes), 1)
    lane_h, gap, pad = 16, 5, 8
    height = pad * 2 + n_lanes * lane_h + (n_lanes - 1) * gap

    palette = series()
    blocks, items = [], []
    for i, (e, s0, s1, lane) in enumerate(placed):
        left = max(0.0, min(100.0, s0 / span * 100))
        width = max(1.2, min(100.0 - left, (s1 - s0) / span * 100))
        c = palette[i % len(palette)]
        top = pad + lane * (lane_h + gap)
        blocks.append(
            f'<span class="blk" style="left:{left:.2f}%;width:{width:.2f}%;'
            f'top:{top}px;height:{lane_h}px;bottom:auto;background:{c}" '
            f'title="{_e(e["label"])} {s0:.1f}-{s1:.1f}s"></span>')
        items.append(
            f'<div class="al-ev"><i style="width:9px;height:9px;border-radius:3px;'
            f'background:{c};flex:0 0 auto"></i>'
            f'<span class="t">{s0:.1f}s &ndash; {s1:.1f}s</span>'
            f'<span style="flex:1">{_e(e["label"])}</span>'
            f'<span style="color:var(--al-muted);font-size:.84rem">peak '
            f'{float(e.get("peak_frame_score", 0)):.2f}</span></div>')

    st.html(f"""
<div class="al-card">
  <div class="al-eyebrow">Where in the clip</div>
  <div class="al-tl" style="height:{height}px"
       role="img" aria-label="Timeline of {len(placed)} evidence windows across
       {span:.1f} seconds">{''.join(blocks)}</div>
  <div class="al-tl-axis"><span>0.0s</span><span>{span:.1f}s</span></div>
  <div style="margin-top:.9rem">{''.join(items)}</div>
  <p style="font-size:.82rem;color:var(--al-muted);margin:.75rem 0 0">
  Each block is a real time span the model scored highest for that behavior. It tells you
  where to look &mdash; it is not a verified event.</p>
</div>""")


def error_card(title: str, hints: list[str]) -> None:
    lis = "".join(f"<li>{_e(h)}</li>" for h in hints)
    st.html(f'<div class="al-error"><h4>{_e(title)}</h4>'
                f'<ul>{lis}</ul></div>')


def section(title: str, caption: str = "") -> None:
    cap = (f'<p style="color:var(--al-muted);font-size:.92rem;margin:.1rem 0 0">{_e(caption)}</p>'
           if caption else "")
    st.html(f'<div style="margin:1.9rem 0 .8rem"><h3 style="margin:0">{_e(title)}</h3>{cap}</div>')
