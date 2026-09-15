#!/usr/bin/env python3
"""Click a nav link in a real browser and prove the page moves.

The nav links were shipped once as plain anchors and did nothing: st.html runs
no JavaScript, and Streamlit's SPA has already discarded any #fragment by the
time the sections exist. "The markup is present" is therefore NOT evidence that
navigation works -- only a real click is. This drives Chrome over the DevTools
Protocol and measures scroll position before and after.

    python3 scripts/check_nav_click.py [--port 8501]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import socket
import subprocess
import sys
import time
import urllib.request

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
# Streamlit scrolls an inner element, not the window, so ask the page which one
# actually moved rather than assuming window.scrollY.
SCROLL_Y = """(() => {
  const m = document.querySelector('section.stMain') ||
            document.querySelector('[data-testid="stMain"]');
  return Math.round((m && m.scrollTop) || window.scrollY || 0);
})()"""


async def cdp(ws_url: str, app_url: str) -> dict:
    import websockets

    async with websockets.connect(ws_url, max_size=None) as ws:
        n = 0

        async def send(method, **params):
            nonlocal n
            n += 1
            await ws.send(json.dumps({"id": n, "method": method, "params": params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == n:
                    return msg.get("result", {})

        async def js(expr):
            r = await send("Runtime.evaluate", expression=expr, returnByValue=True,
                           awaitPromise=True)
            return r.get("result", {}).get("value")

        await send("Page.enable")
        await send("Runtime.enable")
        await send("Page.navigate", url=app_url)

        # CSS allows an unquoted identifier in an attribute selector, so
        # [href$=behaviors] avoids nesting quotes inside Python inside JS.
        link = "document.querySelector('.al-nav a[href$=behaviors]')"

        # Wait for Streamlit to hydrate and the landing to exist.
        for _ in range(60):
            await asyncio.sleep(1)
            if await js("!!" + link) and await js("!!document.getElementById('behaviors')"):
                break
        else:
            return {"error": "landing never rendered"}

        await asyncio.sleep(3)          # let the shim's iframe load and wire up
        wired = await js("!!" + link + ".dataset.alWired")
        before = await js(SCROLL_Y)
        await js(link + ".click()")
        await asyncio.sleep(2)          # smooth scroll needs time to settle
        after = await js(SCROLL_Y)
        top = await js("Math.round(document.getElementById('behaviors')"
                       ".getBoundingClientRect().top)")
        return {"wired": wired, "before": before, "after": after, "target_top": top}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8501)
    args = ap.parse_args()
    app_url = f"http://127.0.0.1:{args.port}/"

    try:
        urllib.request.urlopen(app_url, timeout=5)
    except Exception:
        sys.exit(f"No Streamlit app on {app_url} — start it first.")

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dbg = s.getsockname()[1]

    chrome = subprocess.Popen(
        [CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
         f"--remote-debugging-port={dbg}", "--window-size=1250,900",
         "--user-data-dir=/tmp/al_cdp_profile", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ws_url = None
        for _ in range(40):
            try:
                tabs = json.load(urllib.request.urlopen(
                    f"http://127.0.0.1:{dbg}/json", timeout=2))
                pages = [t for t in tabs if t.get("type") == "page"]
                if pages:
                    ws_url = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                pass
            time.sleep(0.5)
        if not ws_url:
            sys.exit("Could not attach to Chrome.")

        r = asyncio.run(cdp(ws_url, app_url))
        if r.get("error"):
            print("FAIL ", r["error"])
            return 1

        moved = r["after"] - r["before"]
        print(f"\n  handler attached to the link : {bool(r['wired'])}")
        print(f"  scroll before click          : {r['before']}px")
        print(f"  scroll after click           : {r['after']}px  (moved {moved}px)")
        print(f"  target distance from viewport: {r['target_top']}px\n")

        ok = bool(r["wired"]) and moved > 100 and abs(r["target_top"]) < 140
        print("PASS  clicking a nav link scrolls to its section" if ok else
              "FAIL  the nav link did not navigate")
        return 0 if ok else 1
    finally:
        chrome.terminate()
        try:
            chrome.wait(timeout=10)
        except subprocess.TimeoutExpired:
            chrome.kill()


if __name__ == "__main__":
    sys.exit(main())
