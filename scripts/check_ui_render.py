#!/usr/bin/env python3
"""Assert the UI's markup is delivered as HTML, not through the markdown parser.

This exists because of a bug that two other checks failed to catch. The
stylesheet was shipped through ``st.markdown(unsafe_allow_html=True)``. A blank
line closes an HTML block in markdown, so every rule after the sheet's first
blank line was parsed as prose and printed onto the page as a wall of CSS text.

The static-HTML screenshot missed it (that bypasses Streamlit entirely) and the
AppTest run missed it (it only read the source string back). What neither
asserted is the thing that actually broke: *which element type* carries the
markup. ``st.html`` never reaches the markdown parser; ``st.markdown`` always
does. So that is what this checks.

    python3 scripts/check_ui_render.py

Runs in about 10 seconds. No browser, no video, no inference.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest                                # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
#: Markup that must never appear in a markdown element, because markdown would
#: mangle it exactly the way the original bug did.
RAW_MARKUP = re.compile(r"--al-[a-z-]+\s*:|<div\s|<style|<nav\s|<table\s|<svg\s")
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        if detail:
            print(f"        {detail}")
        failures.append(name)


#: Any metric on the landing page is a bug. docs/v3_status.md voids every
#: pre-nested-CV figure the older docs quote, so the page states none at all and
#: this is what keeps it that way.
METRIC = re.compile(r"macro-F1|AUROC|AUPRC|\bF1\b|\bECE\b|\b0\.\d{2,}\b")


def main() -> int:
    # --- landing screen ---------------------------------------------------- #
    at = AppTest.from_file(APP, default_timeout=180).run()
    check("app runs without exceptions", not at.exception,
          "; ".join(str(e.value) for e in at.exception))

    # The stylesheet is an html element too, and it defines .al-model{...} and
    # carries numbers of its own -- checking it would match the rules rather than
    # the page. Only the content elements are the landing page.
    landing = "\n".join(h.proto.body for h in at.get("html") if "<style>" not in h.proto.body)
    check("landing screen renders first", "al-hero" in landing)
    check("tool chrome is gated behind it", 'class="al-model' not in landing,
          "the model card renders before the reader has started")

    metrics = sorted(set(METRIC.findall(landing)))
    check("landing states no accuracy numbers", not metrics,
          f"found {metrics} — docs/v3_status.md voids the pre-nested figures")

    # Nav: every link must have a target on the page. A nav link pointing at an
    # id that does not exist is a dead link the browser silently ignores.
    import re as _re
    hrefs = set(_re.findall(r'href="#([\w-]+)"', landing))
    ids = set(_re.findall(r'id="([\w-]+)"', landing))
    check("section nav renders", 'class="al-nav"' in landing)
    check("every nav link has a target on the page", hrefs and hrefs <= ids,
          f"links {sorted(hrefs)} vs targets {sorted(ids)}")

    # --- click through to the tool ----------------------------------------- #
    at.button(key="al_start").click().run()
    check("Start screening reaches the tool", not at.exception,
          "; ".join(str(e.value) for e in at.exception))

    at.checkbox[0].check().run()   # consent gate -> renders the full first screen
    check("app runs without exceptions after consent", not at.exception,
          "; ".join(str(e.value) for e in at.exception))

    html = [h.proto.body for h in at.get("html")]
    md = [m.value for m in at.get("markdown")]
    print(f"\n  {len(html)} html element(s), {len(md)} markdown element(s)\n")

    # 1. The stylesheet must exist. It rides on a markdown element by necessity:
    #    st.html's sanitizer strips <style> outright.
    sheet = [b for b in html + md if "--al-primary" in b and "<style>" in b]
    check("stylesheet is delivered", len(sheet) == 1,
          f"found {len(sheet)} stylesheet elements")

    # 2. It must arrive WHOLE. Markdown ends an HTML block at the first blank
    #    line, which once truncated this sheet and printed the rest as prose.
    check("stylesheet is not truncated",
          bool(sheet) and "prefers-reduced-motion" in sheet[0],
          "the last rule is missing — the sheet was cut short in transit")

    # 3. ...which is only true while it carries no blank lines. This is the
    #    invariant that keeps the truncation bug from coming back.
    check("stylesheet carries no blank lines",
          bool(sheet) and not any(not ln.strip() for ln in sheet[0].splitlines()),
          "a blank line will end the HTML block and dump the rest onto the page")

    # 4. Nothing else raw may ride on a markdown element.
    offenders = [m[:90].replace("\n", " ") for m in md
                 if "<style>" not in m and RAW_MARKUP.search(m)]
    check("no other raw HTML or CSS on a markdown element", not offenders,
          f"{len(offenders)} offender(s), first: {offenders[0] if offenders else ''}")

    # 4. The components rendered, and rendered as html.
    joined = "\n".join(html)
    for cls in ("al-mast", "al-safety", "al-steps", "al-model"):
        check(f"component .{cls} rendered as html", f'class="{cls}' in joined)

    # The gate must not be one-way: there has to be a route back.
    back = [b for b in at.button if b.key == "al_back"]
    check("tool offers a way back to the overview", len(back) == 1)
    if back:
        back[0].click().run()
        returned = "\n".join(h.proto.body for h in at.get("html")
                             if "<style>" not in h.proto.body)
        check("back reaches the landing again", "al-hero" in returned,
              "clicking back did not restore the overview")
        at.button(key="al_start").click().run()   # forward again for the rest
        joined = "\n".join(h.proto.body for h in at.get("html"))

    # 5. The auto-selected checkpoint reached the page. Resolved from the
    #    registry at runtime, never hardcoded: the selection rule is owned by
    #    src/model_registry.py and both the rule and the winner change as new
    #    checkpoints are trained.
    from src.model_registry import select_best

    best = select_best()
    check("served checkpoint shown on the page",
          best is not None and best.tag in joined,
          f"registry selects {best.tag if best else None}, which is absent from the page")

    print(f"\n{'ALL CHECKS PASSED' if not failures else str(len(failures)) + ' FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
