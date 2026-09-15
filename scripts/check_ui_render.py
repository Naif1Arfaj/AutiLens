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


def main() -> int:
    at = AppTest.from_file(APP, default_timeout=180).run()
    check("app runs without exceptions", not at.exception,
          "; ".join(str(e.value) for e in at.exception))

    at.checkbox[0].check().run()   # consent gate -> renders the full first screen
    check("app runs without exceptions after consent", not at.exception,
          "; ".join(str(e.value) for e in at.exception))

    html = [h.proto.body for h in at.get("html")]
    md = [m.value for m in at.get("markdown")]
    print(f"\n  {len(html)} html element(s), {len(md)} markdown element(s)\n")

    # 1. THE bug: the stylesheet must be an html element, never markdown.
    sheet = [h for h in html if "--al-primary" in h and "<style>" in h]
    check("stylesheet delivered via st.html", len(sheet) == 1,
          f"found {len(sheet)} stylesheet html elements")

    # 2. It must arrive whole. The markdown route truncated it at the first
    #    blank line, so check a rule from the very end of the sheet.
    check("stylesheet is not truncated",
          bool(sheet) and "prefers-reduced-motion" in sheet[0],
          "the last section of the sheet is missing — it was cut short in transit")

    # 3. Nothing raw may ride on a markdown element anywhere in the app.
    offenders = [m[:90].replace("\n", " ") for m in md if RAW_MARKUP.search(m)]
    check("no raw HTML or CSS on a markdown element", not offenders,
          f"{len(offenders)} offender(s), first: {offenders[0] if offenders else ''}")

    # 4. The components rendered, and rendered as html.
    joined = "\n".join(html)
    for cls in ("al-mast", "al-safety", "al-steps", "al-model"):
        check(f"component .{cls} rendered as html", f'class="{cls}' in joined)

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
