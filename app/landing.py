"""Landing screen — what AutiLens is, shown before the tool.

Every claim here traces to a file in ``docs/``; nothing is written to sound
impressive. Sources are named per section below.

**No accuracy numbers on this page, deliberately.** ``docs/v3_status.md`` records
that nested cross-validation exposed a selection-on-test leak and voids every
pre-nested figure the older docs quote. Rather than police which numbers are
still live on a page that reviewers will read first, this page states no metric
at all, and ``scripts/check_ui_render.py`` asserts that none creeps back in. The
served model's real, current score is on the model card inside the tool, where it
comes from the registry at runtime and cannot go stale.
"""
from __future__ import annotations

import streamlit as st

from app.components import _e, icon
from src.config import load_config

# --- copy -------------------------------------------------------------------
# docs/impact_card.md, README.md
LEDE = ("A multimodal AI that gives a specialist a first-pass read of a short "
        "video clip — which autism-related behaviors appear, roughly when, and a "
        "plain-language write-up — so their time goes to the moments that matter "
        "instead of watching every clip cold.")

# docs/problem_and_use_case.md, "What is the problem's impact"
PROBLEM = [
    ("schedule", "Review costs as much time as the recording",
     "A specialist watches the session in full and timestamps behaviors by hand. "
     "For long recordings and high caseloads this is the bottleneck step."),
    ("person", "Specialist time is the scarcest resource",
     "It is the most expensive input in the workflow, and it is spent watching "
     "footage in which most minutes turn out to be unremarkable."),
    ("compare", "Two reviewers tag differently",
     "Manual behavior-tagging varies between raters. A consistent, "
     "confidence-scored first pass gives everyone the same starting point."),
]

# docs/problem_and_use_case.md, "Who is the target user?"
AUDIENCE = ("Built for a <strong>behavioral therapist, early-intervention specialist, "
            "or clinical researcher</strong> doing a first-pass review — deciding what deserves "
            "closer attention. Not a diagnosing physician, and not the child or family. "
            "The professional makes every judgment; this only decides where they look "
            "first.")

# docs/v3_status.md (architecture) + docs/problem_and_use_case.md (steps 3-4)
FLOW = [
    ("Clip in", "A short video clip with its audio. It is held in a temporary "
                "directory for one inference pass and deleted immediately."),
    ("Which family", "A first model decides which broad family of behavior is "
                     "present, from the four below."),
    ("Which behavior", "A second model narrows that to the specific behaviors "
                       "within the family it detected, with a time window for each."),
    ("Written up", "Those numbers — and only those numbers — go to a language "
                   "model, which writes the report. It never sees the video."),
]

# docs/impact_card.md, "What it is NOT" + "Responsible-use guardrails"
NOT = [
    "<strong>It does not diagnose autism.</strong> Diagnosis requires qualified "
    "professionals and far broader clinical evidence than a short clip.",
    "Its output is not a decision. It is a pointer to moments worth a human look.",
    "The language model can only describe what the vision model measured — it is "
    "given the numbers, never the footage, so it cannot invent a clinical claim.",
    "No facial identity is used, and the demo will not process a clip until you "
    "confirm you have the right to analyse it.",
]

# docs/impact_card.md "Key limitation" + docs/v3_status.md per-family finding
LIMITS = [
    "A research prototype trained on a few hundred clips — a demonstration that "
    "this is feasible, not a production screening system.",
    "Reliability varies by behavior. Some are recognised far more dependably than "
    "others, and the tool shows its confidence and decision threshold for each so "
    "you can see which is which.",
    "The social-communicative and sensory-reactivity families depend heavily on "
    "speech and context that video alone cannot supply, and are the weakest.",
    "Every window it reports is a place to look, not a verified event.",
]


#: Landing sections, in page order: (anchor id, nav label).
SECTIONS = [
    ("problem", "The problem"),
    ("how", "How it works"),
    ("behaviors", "Behaviors"),
    ("limits", "Limits"),
]


def _nav() -> str:
    """Sticky section nav.

    Anchor links only: ``st.html`` does not execute JavaScript, so there is no
    smooth-scroll and no scroll-spy to highlight the current section. The CTA is
    an anchor down to the real Streamlit button rather than a second button,
    because a link cannot change session state.
    """
    links = "".join(
        f'<a href="#{i}">{_e(label)}</a>' for i, label in SECTIONS)
    return (f'<nav class="al-nav" aria-label="Sections">'
            f'<span class="al-nav-brand">{icon("lens", 15)} AutiLens</span>'
            f'<span class="al-nav-links">{links}</span>'
            f'<a class="al-nav-cta" href="#start">Start screening</a></nav>')


def _scroll_shim() -> None:
    """Make the nav links actually navigate.

    ``st.html`` runs no JavaScript, and Streamlit is a single-page app: by the
    time it has rendered the sections, the browser has long since given up on
    any ``#fragment`` in the URL. Plain anchors therefore do nothing at all --
    they looked like navigation and were decoration.

    ``components.html`` runs in a same-origin iframe, so a script inside it can
    reach ``window.parent.document`` and wire the links up for real. The
    MutationObserver re-wires them because Streamlit rebuilds the DOM on every
    rerun, which would otherwise drop the handlers.
    """
    import streamlit.components.v1 as components

    components.html("""
<script>
(function () {
  try {
    var doc = window.parent.document;
    var reduce = window.parent.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var wire = function () {
      doc.querySelectorAll('.al-nav a[href^="#"]').forEach(function (a) {
        if (a.dataset.alWired) { return; }
        a.dataset.alWired = '1';
        a.addEventListener('click', function (e) {
          e.preventDefault();
          var el = doc.getElementById(a.getAttribute('href').slice(1));
          if (el) {
            el.scrollIntoView({behavior: reduce ? 'auto' : 'smooth', block: 'start'});
          }
        });
      });
    };
    wire();
    new MutationObserver(wire).observe(doc.body, {childList: true, subtree: true});
  } catch (err) {
    /* Cross-origin or no parent: links stay inert rather than breaking the page. */
  }
})();
</script>""", height=0)


def _tile(ic: str, title: str, body: str) -> str:
    return (f'<div class="al-tile">{icon(ic, 20, "var(--al-primary)")}'
            f'<h4>{_e(title)}</h4><p>{_e(body)}</p></div>')


def render() -> None:
    """Draw the landing screen. The button hands over to the tool."""
    cfg = load_config()

    st.html(_nav())
    _scroll_shim()

    st.html(f"""
<div class="al-hero">
  <span class="al-tagline">{icon('lens', 13)} Research prototype</span>
  <h2>Spend the review on the moments that matter.</h2>
  <p class="al-lede">{_e(LEDE)}</p>
</div>""")

    st.html('<div class="al-eyebrow al-anchor" id="problem" style="margin-top:.6rem">The problem</div>'
            '<div class="al-grid">'
            + "".join(_tile(*p) for p in PROBLEM) + "</div>")

    st.html(f'<div class="al-card" style="margin-top:1.1rem">'
            f'<div class="al-eyebrow">Who it is for</div>'
            f'<p style="margin:0;line-height:1.62">{AUDIENCE}</p></div>')

    steps = "".join(
        f'<div class="s"><div class="n">{i + 1}</div><strong>{_e(t)}</strong>'
        f'<span>{_e(b)}</span></div>'
        for i, (t, b) in enumerate(FLOW))
    st.html('<div class="al-eyebrow al-anchor" id="how" style="margin-top:1.6rem">How it works</div>'
            f'<div class="al-flow">{steps}</div>')

    # Read live from configs/default.yaml so this cannot drift from the taxonomy
    # the models are actually trained against.
    fams = cfg.get("families") or {}
    if fams:
        blocks = "".join(
            f'<div class="al-fam"><strong>{_e(name)}</strong><ul>'
            + "".join(f"<li>{_e(b)}</li>" for b in behaviors) + "</ul></div>"
            for name, behaviors in fams.items())
        n_beh = sum(len(v) for v in fams.values())
        st.html(f'<div class="al-eyebrow al-anchor" id="behaviors" style="margin-top:1.6rem">What it looks for</div>'
                f'<p style="color:var(--al-muted);font-size:.93rem;margin:.1rem 0 .7rem">'
                f'{n_beh} behaviors, grouped into {len(fams)} families.</p>{blocks}')

    st.html('<div class="al-anchor" id="limits" style="margin-top:1.6rem"></div><div class="al-not">'
            f'<h4>{icon("alert", 17, "var(--al-warn)")} What it is not</h4><ul>'
            + "".join(f"<li>{n}</li>" for n in NOT) + "</ul></div>")

    st.html('<div class="al-card al-limits" style="margin-top:1.1rem">'
            '<div class="al-eyebrow">Honest limits</div><ul>'
            + "".join(f"<li>{_e(l)}</li>" for l in LIMITS) + "</ul></div>")

    st.html('<div class="al-anchor" id="start" style="margin-top:1.4rem"></div>')
    if st.button("Start screening", type="primary", key="al_start"):
        st.session_state["started"] = True
        st.rerun()
    st.caption("You will be asked to confirm you have the right to analyse the "
               "clip before anything is uploaded.")
