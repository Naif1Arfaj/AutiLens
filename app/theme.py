"""Design tokens and CSS for the AutiLens demo UI.

Direction (from the ui-ux-pro-max design data installed in ~/.claude/skills):

  style       Minimalism & Swiss   -- WCAG AAA, the style that data lists as
                                      "best for: dashboards, professional tools"
  palette     Healthcare App       -- calm cyan + health green on near-white
  typography  Corporate Trust      -- Lexend headings / Source Sans 3 body,
                                      chosen for readability and accessibility

Glassmorphism, claymorphism and aurora gradients were all rejected on purpose:
that same data lists "serious/medical, data-critical, critical accessibility"
under their Do-Not-Use-For column. For a tool whose first message is that it is
NOT a diagnosis, restraint is the correct design.
"""
from __future__ import annotations

import streamlit as st

FONTS = ("https://fonts.googleapis.com/css2?"
         "family=Lexend:wght@300;400;500;600;700"
         "&family=Source+Sans+3:wght@300;400;500;600;700"
         "&family=JetBrains+Mono:wght@400;500&display=swap")

LIGHT = {
    "bg": "#F2FBFD", "surface": "#FFFFFF", "surface_alt": "#ECFEFF",
    "border": "#CBEAF2", "border_strong": "#A5F3FC",
    "text": "#123C4A", "muted": "#4A6B75",
    "primary": "#0E7490", "primary_soft": "#E0F6FB", "secondary": "#0891B2",
    "positive": "#047857", "positive_soft": "#E7F7F0",
    "warn": "#92400E", "warn_soft": "#FEF6E7", "warn_border": "#F5CF8E",
    "danger": "#9F1239", "danger_soft": "#FDF2F5",
    "track": "#E3EEF2", "shadow": "0 1px 2px rgba(18,60,74,.06), 0 8px 24px -16px rgba(18,60,74,.24)",
}

DARK = {
    "bg": "#081B21", "surface": "#0F2A33", "surface_alt": "#123642",
    "border": "#1D4655", "border_strong": "#286d80",
    "text": "#E4F6FA", "muted": "#9CBDC7",
    "primary": "#3FD0E8", "primary_soft": "#10333E", "secondary": "#22D3EE",
    "positive": "#34D399", "positive_soft": "#0D3129",
    "warn": "#FCD34D", "warn_soft": "#2E2513", "warn_border": "#6B5320",
    "danger": "#FB7185", "danger_soft": "#2E1620",
    "track": "#17414F", "shadow": "0 1px 2px rgba(0,0,0,.30), 0 8px 24px -16px rgba(0,0,0,.65)",
}


def active_theme() -> str:
    """Follow whatever theme Streamlit is actually rendering.

    Using a CSS ``prefers-color-scheme`` block instead would desynchronise the
    custom components from Streamlit's own chrome whenever the user flips the
    in-app theme toggle, which is the one case that matters here.
    """
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def tokens() -> dict:
    return DARK if active_theme() == "dark" else LIGHT


def _flatten(css: str) -> str:
    """Drop blank lines so markdown treats the sheet as one unbroken HTML block."""
    return "\n".join(line for line in css.splitlines() if line.strip())


def inject() -> None:
    """Emit the stylesheet.

    Two delivery bugs bound this together, so read both before changing it.

    1. ``st.markdown(unsafe_allow_html=True)`` runs the sheet through a markdown
       parser, where a **blank line closes an HTML block**. The sheet was
       accepted only as far as its first blank line and every rule after that was
       printed onto the page as prose.
    2. ``st.html`` skips markdown, but its sanitizer **strips the whole
       ``<style>`` element**, so the page rendered with no styling at all.

    So: markdown delivery, with the blank lines stripped on the way out. They stay
    in the source below because they are what makes 300 lines of CSS readable;
    ``_flatten`` removes them at the boundary. ``scripts/check_ui_render.py``
    asserts the sheet arrives whole, which is what catches either bug returning.
    """
    t = tokens()
    css = f"""
<style>
@import url('{FONTS}');

:root {{
  --al-bg:{t['bg']}; --al-surface:{t['surface']}; --al-surface-alt:{t['surface_alt']};
  --al-border:{t['border']}; --al-border-strong:{t['border_strong']};
  --al-text:{t['text']}; --al-muted:{t['muted']};
  --al-primary:{t['primary']}; --al-primary-soft:{t['primary_soft']}; --al-secondary:{t['secondary']};
  --al-positive:{t['positive']}; --al-positive-soft:{t['positive_soft']};
  --al-warn:{t['warn']}; --al-warn-soft:{t['warn_soft']}; --al-warn-border:{t['warn_border']};
  --al-danger:{t['danger']}; --al-danger-soft:{t['danger_soft']};
  --al-track:{t['track']}; --al-shadow:{t['shadow']};
  --al-r:14px; --al-r-sm:9px;
  --z-base:10; --z-sticky:20; --z-overlay:30; --z-modal:50;
  /* above Streamlit's own header, which sits at 999990 */
  --z-nav:999991;
}}

/* ---------- base ------------------------------------------------------- */
/* Do NOT add [class*="st-"] here. Every Streamlit node carries an
   st-emotion-cache-* class, including the Material Symbols icon spans, so that
   selector overrode the icon font and made each icon render as its literal
   ligature name ("info", "keyboard_arrow_right") on top of the label. Body
   inheritance covers the text without touching the icons. */
html, body, .stApp {{
  font-family:'Source Sans 3',system-ui,-apple-system,sans-serif;
  color:var(--al-text);
}}
/* Belt and braces: whatever else is set, icons keep their own font. */
[data-testid="stIconMaterial"], [data-testid="stAlertDynamicIcon"],
span[class*="material-symbols"] {{
  font-family:'Material Symbols Rounded','Material Symbols Outlined' !important;
}}
.stApp {{ background:var(--al-bg); }}
.block-container {{ max-width:1020px; padding-top:2.2rem; padding-bottom:5rem; }}
h1,h2,h3,h4,h5 {{ font-family:'Lexend',system-ui,sans-serif !important; color:var(--al-text) !important;
  letter-spacing:-.018em; }}
/* 16px floor on mobile body text (ui-ux-pro-max: readable-font-size) */
p, li, .al-body {{ font-size:1rem; line-height:1.62; }}
#MainMenu, footer {{ visibility:hidden; }}
header[data-testid="stHeader"] {{ background:transparent; }}

/* Widget labels take their colour from config.toml's textColor, which is pinned
   to the light palette. On a dark-mode browser that put near-black text on a
   near-black background -- the consent checkbox was barely legible. Drive them
   from our own tokens instead, same reasoning as the sidebar above. */
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
.stCheckbox label, .stCheckbox label span, .stRadio label {{
  color:var(--al-text) !important; opacity:1 !important;
}}

/* Visible focus ring on every interactive element (priority-1 rule). */
a:focus-visible, button:focus-visible, input:focus-visible,
[role="button"]:focus-visible, [tabindex]:focus-visible,
.stCheckbox input:focus-visible + div {{
  outline:3px solid var(--al-secondary) !important;
  outline-offset:2px !important; border-radius:6px;
}}

/* Sidebar: styled from our own tokens on purpose. Left to Streamlit it takes
   its colours from .streamlit/config.toml, which is pinned light, so on a
   dark-mode browser the chrome went light while the page went dark and the
   sidebar text dropped to unreadable contrast. */
[data-testid="stSidebar"] {{ background:var(--al-surface) !important;
  border-right:1px solid var(--al-border); }}
[data-testid="stSidebar"] * {{ color:var(--al-text); }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{ color:var(--al-text) !important; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] small {{ color:var(--al-muted) !important; }}
[data-testid="stSidebar"] [data-baseweb="select"] > div {{
  background:var(--al-surface-alt); border-color:var(--al-border); }}

/* Our own icons, drawn with the Material Symbols font Streamlit already loads
   (st.html strips <svg>, so inline SVG is not an option here). */
.al-ico {{ font-family:'Material Symbols Rounded','Material Symbols Outlined';
  font-weight:normal; font-style:normal; line-height:1; letter-spacing:normal;
  text-transform:none; white-space:nowrap; direction:ltr; display:inline-block;
  vertical-align:middle; -webkit-font-feature-settings:'liga';
  -webkit-font-smoothing:antialiased; font-variation-settings:'wght' 500; }}

/* ---------- masthead --------------------------------------------------- */
.al-mast {{ display:flex; align-items:center; gap:.9rem; margin-bottom:.35rem; }}
.al-mark {{ width:42px; height:42px; flex:0 0 42px; border-radius:12px;
  background:linear-gradient(140deg,var(--al-secondary),var(--al-primary));
  display:grid; place-items:center; box-shadow:var(--al-shadow); }}
.al-mast h1 {{ font-size:1.72rem !important; margin:0 !important; line-height:1.15; }}
.al-mast .al-sub {{ color:var(--al-muted); font-size:.95rem; margin:.15rem 0 0; }}

/* ---------- safety banner (must never look like decoration) ------------ */
.al-safety {{ display:flex; gap:.8rem; align-items:flex-start;
  background:var(--al-warn-soft); border:1px solid var(--al-warn-border);
  border-left:5px solid var(--al-warn); border-radius:var(--al-r-sm);
  padding:.95rem 1.1rem; margin:1.1rem 0 1.6rem; }}
.al-safety strong {{ color:var(--al-warn); }}
.al-safety p {{ margin:.25rem 0 0; font-size:.94rem; color:var(--al-text); }}

/* ---------- stepper ---------------------------------------------------- */
.al-steps {{ display:flex; gap:.4rem; margin:0 0 1.6rem; flex-wrap:wrap; }}
.al-step {{ display:flex; align-items:center; gap:.5rem; padding:.45rem .8rem;
  border-radius:999px; font-size:.86rem; font-weight:500; white-space:nowrap;
  background:var(--al-surface); border:1px solid var(--al-border); color:var(--al-muted); }}
.al-step .n {{ width:20px; height:20px; border-radius:50%; display:grid; place-items:center;
  font-size:.72rem; font-weight:700; background:var(--al-track); color:var(--al-muted); }}
.al-step.done {{ color:var(--al-positive); border-color:var(--al-positive); background:var(--al-positive-soft); }}
.al-step.done .n {{ background:var(--al-positive); color:var(--al-surface); }}
.al-step.now {{ color:var(--al-surface); background:var(--al-primary); border-color:var(--al-primary); }}
.al-step.now .n {{ background:rgba(255,255,255,.25); color:var(--al-surface); }}

/* ---------- cards ------------------------------------------------------ */
.al-card {{ background:var(--al-surface); border:1px solid var(--al-border);
  border-radius:var(--al-r); padding:1.15rem 1.3rem; box-shadow:var(--al-shadow); margin-bottom:1rem; }}
.al-card h3 {{ margin:0 0 .2rem !important; font-size:1.06rem !important; }}
.al-eyebrow {{ text-transform:uppercase; letter-spacing:.1em; font-size:.7rem;
  font-weight:700; color:var(--al-muted); margin-bottom:.55rem; }}

/* model card */
.al-model {{ display:flex; gap:1rem; align-items:center; background:var(--al-surface);
  border:1px solid var(--al-border); border-left:4px solid var(--al-primary);
  border-radius:var(--al-r); padding:.95rem 1.15rem; box-shadow:var(--al-shadow); }}
.al-model .al-name {{ font-family:'Lexend',sans-serif; font-weight:600; font-size:1.02rem; }}
.al-model .al-meta {{ color:var(--al-muted); font-size:.87rem; margin-top:.15rem; }}
.al-score {{ margin-left:auto; text-align:right; flex:0 0 auto; }}
.al-score .v {{ font-family:'Lexend',sans-serif; font-size:1.5rem; font-weight:700;
  color:var(--al-primary); line-height:1; }}
.al-score .k {{ font-size:.7rem; text-transform:uppercase; letter-spacing:.08em;
  color:var(--al-muted); margin-top:.2rem; }}

/* ---------- behaviour rows -------------------------------------------- */
.al-row {{ padding:.7rem 0; border-bottom:1px solid var(--al-border); }}
.al-row:last-child {{ border-bottom:none; }}
.al-row-top {{ display:flex; align-items:baseline; gap:.6rem; margin-bottom:.4rem; }}
.al-row-label {{ font-weight:600; font-size:.96rem; }}
.al-row-val {{ margin-left:auto; font-family:'JetBrains Mono',ui-monospace,monospace;
  font-size:.9rem; font-variant-numeric:tabular-nums; color:var(--al-muted); }}
.al-row.hit .al-row-val {{ color:var(--al-primary); font-weight:600; }}

.al-bar {{ position:relative; height:9px; border-radius:99px; background:var(--al-track); overflow:visible; }}
.al-bar .fill {{ position:absolute; inset:0 auto 0 0; border-radius:99px;
  background:var(--al-border-strong); transition:width .45s cubic-bezier(.2,.7,.3,1); }}
.al-row.hit .al-bar .fill {{ background:linear-gradient(90deg,var(--al-secondary),var(--al-primary)); }}
/* the per-class decision threshold, drawn where it actually sits */
.al-bar .thr {{ position:absolute; top:-4px; bottom:-4px; width:2px;
  background:var(--al-text); opacity:.55; border-radius:2px; }}
.al-bar .thr::after {{ content:''; position:absolute; top:-3px; left:-2px;
  width:6px; height:6px; border-radius:50%; background:var(--al-text); opacity:.75; }}

.al-chip {{ display:inline-flex; align-items:center; gap:.35rem; padding:.16rem .55rem;
  border-radius:999px; font-size:.73rem; font-weight:600; letter-spacing:.01em; }}
.al-chip.on {{ background:var(--al-positive-soft); color:var(--al-positive);
  border:1px solid var(--al-positive); }}
.al-chip.off {{ background:var(--al-surface-alt); color:var(--al-muted);
  border:1px solid var(--al-border); }}

/* ---------- multi-class verdict --------------------------------------- */
.al-verdict {{ background:linear-gradient(150deg,var(--al-primary-soft),var(--al-surface));
  border:1px solid var(--al-border-strong); border-radius:var(--al-r);
  padding:1.5rem 1.6rem; box-shadow:var(--al-shadow); }}
.al-verdict .k {{ text-transform:uppercase; letter-spacing:.1em; font-size:.7rem;
  font-weight:700; color:var(--al-muted); }}
.al-verdict .v {{ font-family:'Lexend',sans-serif; font-size:1.85rem; font-weight:600;
  line-height:1.2; margin:.35rem 0 .7rem; color:var(--al-text); }}
.al-dist {{ display:flex; height:11px; border-radius:99px; overflow:hidden;
  background:var(--al-track); margin-top:.3rem; }}
.al-dist span {{ display:block; height:100%; }}
.al-legend {{ display:flex; flex-wrap:wrap; gap:.5rem 1rem; margin-top:.65rem;
  font-size:.82rem; color:var(--al-muted); }}
.al-legend i {{ width:9px; height:9px; border-radius:3px; display:inline-block; margin-right:.35rem; }}

/* ---------- evidence timeline ----------------------------------------- */
.al-tl {{ position:relative; height:46px; border-radius:var(--al-r-sm);
  background:var(--al-surface-alt); border:1px solid var(--al-border); margin:.5rem 0 .3rem; }}
.al-tl .blk {{ position:absolute; top:7px; bottom:7px; border-radius:5px;
  background:linear-gradient(180deg,var(--al-secondary),var(--al-primary));
  min-width:4px; box-shadow:0 1px 3px rgba(0,0,0,.18); }}
.al-tl-axis {{ display:flex; justify-content:space-between; font-size:.74rem;
  color:var(--al-muted); font-family:'JetBrains Mono',monospace; }}
.al-ev {{ display:flex; gap:.6rem; align-items:center; padding:.5rem 0;
  border-bottom:1px solid var(--al-border); font-size:.92rem; }}
.al-ev:last-child {{ border-bottom:none; }}
.al-ev .t {{ font-family:'JetBrains Mono',monospace; font-size:.82rem;
  color:var(--al-primary); font-weight:600; white-space:nowrap; }}

/* ---------- empty + error states -------------------------------------- */
.al-empty {{ text-align:center; padding:1.8rem 1rem; color:var(--al-muted);
  background:var(--al-surface-alt); border:1px dashed var(--al-border-strong);
  border-radius:var(--al-r); font-size:.94rem; }}
.al-error {{ background:var(--al-danger-soft); border:1px solid var(--al-danger);
  border-left:5px solid var(--al-danger); border-radius:var(--al-r-sm); padding:1rem 1.15rem; }}
.al-error h4 {{ margin:0 0 .35rem !important; color:var(--al-danger) !important; font-size:1rem !important; }}
.al-error ul {{ margin:.4rem 0 0 1.05rem; font-size:.91rem; }}

/* ---------- streamlit widget polish ------------------------------------ */
.stButton>button, .stDownloadButton>button {{
  font-family:'Lexend',sans-serif; font-weight:500; border-radius:10px;
  border:1px solid var(--al-border-strong); padding:.5rem 1.15rem;
  min-height:44px; /* touch-target-size */
  transition:transform .15s ease, box-shadow .15s ease, background .15s ease; }}
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"] {{
  background:var(--al-primary); border-color:var(--al-primary); color:#fff; }}
.stButton>button:hover, .stDownloadButton>button:hover {{
  transform:translateY(-1px); box-shadow:var(--al-shadow); cursor:pointer; }}
[data-testid="stFileUploaderDropzone"] {{
  background:var(--al-surface); border:1.5px dashed var(--al-border-strong);
  border-radius:var(--al-r); transition:border-color .2s ease, background .2s ease; }}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color:var(--al-primary); background:var(--al-surface-alt); }}
[data-testid="stExpander"] {{ border:1px solid var(--al-border); border-radius:var(--al-r);
  background:var(--al-surface); }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:var(--al-r); }}
.stTabs [data-baseweb="tab"] {{ font-family:'Lexend',sans-serif; font-weight:500; }}

/* The scroll shim is a zero-height component iframe; keep it from reserving
   vertical space in the layout. */
.stApp:has(.al-nav) [data-testid="stIFrame"] {{ height:0 !important; display:block; }}
.stApp:has(.al-nav) [data-testid="stElementContainer"]:has([data-testid="stIFrame"]) {{
  height:0 !important; min-height:0 !important; margin:0 !important; overflow:hidden; }}

/* ---------- landing nav ------------------------------------------------ */
/* position:sticky is unreliable here: Streamlit wraps every element in its own
   short container, so a sticky child has nothing tall to stick within. Fixed
   positioning is anchored to the viewport instead, and .block-container gets
   extra top padding on the landing so nothing hides underneath. */
/* Streamlit's header is position:absolute at z-index 999990 over the top 60px --
   exactly where this bar sits. Transparent, so the nav showed through it, but it
   still won every hit-test and swallowed every click: the links were visible and
   completely inert. The nav has to outrank it, and the header itself must stop
   capturing pointer events -- with the toolbar exempted so its menu still works. */
header[data-testid="stHeader"] {{ pointer-events:none; }}
header[data-testid="stHeader"] [data-testid="stToolbar"] {{ pointer-events:auto; }}
[data-testid="stAppDeployButton"] {{ display:none !important; }}
.al-nav {{ position:fixed; top:0; left:0; right:0; z-index:var(--z-nav);
  display:flex; align-items:center; gap:1.1rem; flex-wrap:wrap;
  padding:.62rem clamp(1rem,6vw,2.2rem);
  padding-right:calc(clamp(1rem,6vw,2.2rem) + 44px);
  background:color-mix(in srgb, var(--al-bg) 92%, transparent);
  backdrop-filter:saturate(1.4) blur(9px);
  -webkit-backdrop-filter:saturate(1.4) blur(9px);
  border-bottom:1px solid var(--al-border); }}
.al-nav-brand {{ display:inline-flex; align-items:center; gap:.4rem;
  font-family:'Lexend',sans-serif; font-weight:600; font-size:.95rem;
  color:var(--al-text); }}
.al-nav-links {{ display:flex; gap:.25rem; flex-wrap:wrap; margin-left:auto; }}
.al-nav-links a {{ color:var(--al-muted); text-decoration:none; font-size:.88rem;
  font-weight:500; padding:.42rem .6rem; border-radius:8px;
  transition:color .15s ease, background .15s ease; }}
.al-nav-links a:hover {{ color:var(--al-text); background:var(--al-surface-alt); }}
.al-nav-cta {{ background:var(--al-primary); color:var(--al-surface) !important;
  text-decoration:none; font-family:'Lexend',sans-serif; font-weight:500;
  font-size:.88rem; padding:.5rem .95rem; border-radius:9px;
  display:inline-flex; align-items:center; min-height:38px;
  transition:transform .15s ease, box-shadow .15s ease; }}
.al-nav-cta:hover {{ transform:translateY(-1px); box-shadow:var(--al-shadow); }}
/* Land anchored sections below the fixed bar rather than under it. */
.al-anchor {{ scroll-margin-top:78px; }}
/* The nav is the landing's first element; give the page room for it. */
.stApp:has(.al-nav) .block-container {{ padding-top:4.6rem; }}
@media (max-width: 680px) {{
  .al-nav {{ gap:.5rem; padding:.5rem .8rem; }}
  .al-nav-links {{ order:3; width:100%; margin-left:0; }}
  .al-nav-cta {{ margin-left:auto; }}
  .stApp:has(.al-nav) .block-container {{ padding-top:7rem; }}
}}

/* ---------- landing screen -------------------------------------------- */
.al-hero {{ padding:.5rem 0 1.4rem; }}
.al-hero h2 {{ font-size:2.05rem !important; line-height:1.18; margin:.2rem 0 .6rem !important;
  letter-spacing:-.025em; max-width:20ch; }}
.al-hero .al-lede {{ font-size:1.08rem; line-height:1.6; color:var(--al-muted);
  max-width:62ch; margin:0; }}
.al-tagline {{ display:inline-flex; align-items:center; gap:.45rem; font-size:.74rem;
  font-weight:700; letter-spacing:.11em; text-transform:uppercase;
  color:var(--al-primary); background:var(--al-primary-soft);
  border:1px solid var(--al-border-strong); border-radius:999px; padding:.3rem .75rem; }}

/* Two-column prose blocks; collapse to one column on narrow screens. */
.al-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(248px,1fr));
  gap:.9rem; margin:.3rem 0 .4rem; }}
.al-tile {{ background:var(--al-surface); border:1px solid var(--al-border);
  border-radius:var(--al-r); padding:1.05rem 1.15rem; }}
.al-tile h4 {{ margin:.55rem 0 .3rem !important; font-size:1rem !important; }}
.al-tile p {{ margin:0; font-size:.93rem; line-height:1.58; color:var(--al-muted); }}

/* Numbered pipeline: clip -> stage 1 -> stage 2 -> report. */
.al-flow {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:.8rem; }}
.al-flow .s {{ position:relative; background:var(--al-surface-alt);
  border:1px solid var(--al-border); border-radius:var(--al-r); padding:1rem 1.05rem; }}
.al-flow .s .n {{ width:24px; height:24px; border-radius:50%; display:grid; place-items:center;
  font-family:'Lexend',sans-serif; font-size:.76rem; font-weight:700;
  background:var(--al-primary); color:var(--al-surface); margin-bottom:.55rem; }}
.al-flow .s strong {{ display:block; font-size:.95rem; margin-bottom:.22rem; }}
.al-flow .s span {{ font-size:.89rem; line-height:1.55; color:var(--al-muted); }}

/* Behavior families, colour-keyed to match the results views. */
.al-fam {{ border:1px solid var(--al-border); border-left:4px solid var(--al-primary);
  border-radius:var(--al-r-sm); background:var(--al-surface); padding:.85rem 1rem;
  margin-bottom:.6rem; }}
.al-fam strong {{ font-size:.96rem; }}
.al-fam ul {{ margin:.4rem 0 0; padding-left:1.05rem; }}
.al-fam li {{ font-size:.89rem; color:var(--al-muted); line-height:1.5; }}

/* "What it is not" — the guardrails, styled as a deliberate stop, not a footnote. */
.al-not {{ background:var(--al-warn-soft); border:1px solid var(--al-warn-border);
  border-radius:var(--al-r); padding:1.1rem 1.25rem; }}
.al-not h4 {{ margin:0 0 .55rem !important; color:var(--al-warn) !important;
  font-size:1rem !important; }}
.al-not li {{ font-size:.93rem; line-height:1.6; margin-bottom:.28rem; }}
.al-not ul {{ margin:0; padding-left:1.1rem; }}

.al-limits li {{ font-size:.93rem; line-height:1.6; color:var(--al-muted); margin-bottom:.3rem; }}
.al-limits ul {{ margin:0; padding-left:1.1rem; }}
@media (max-width: 640px) {{
  .al-hero h2 {{ font-size:1.6rem !important; }}
}}

/* Respect the user's motion setting (ui-ux-pro-max: reduced-motion). */
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation:none !important; transition:none !important; }}
}}
@media (max-width: 640px) {{
  .block-container {{ padding-left:1rem; padding-right:1rem; }}
  .al-model {{ flex-wrap:wrap; }} .al-score {{ margin-left:0; text-align:left; }}
  .al-steps {{ gap:.3rem; }} .al-step {{ font-size:.78rem; padding:.35rem .6rem; }}
}}
</style>
"""
    st.markdown(_flatten(css), unsafe_allow_html=True)


#: Categorical colours for evidence blocks and the multi-class distribution.
#: Ordered so adjacent entries stay distinguishable, including for the common
#: red-green deficiencies. Two sets: the light ramp is too dark to separate from
#: the dark surface (#0F2A33), so dark mode gets lifted, desaturated variants.
SERIES_LIGHT = ["#0E7490", "#F97316", "#7C3AED", "#047857", "#DB2777",
                "#0369A1", "#B45309", "#4338CA", "#15803D", "#BE123C"]
SERIES_DARK = ["#38BDF8", "#FB923C", "#C084FC", "#34D399", "#F472B6",
               "#7DD3FC", "#FBBF24", "#A5B4FC", "#86EFAC", "#FDA4AF"]


def series() -> list[str]:
    """Categorical palette for the theme currently being rendered."""
    return SERIES_DARK if active_theme() == "dark" else SERIES_LIGHT


#: Backwards-compatible alias; prefer ``series()``, which is theme-aware.
SERIES = SERIES_LIGHT
