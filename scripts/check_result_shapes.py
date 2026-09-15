#!/usr/bin/env python3
"""Exercise both head shapes end-to-end below the backbone.

The multi-class checkpoints this UI must serve do not exist in the repo yet, so
this is the only thing standing between "the multi-class path is written" and
"the multi-class path works". It uses plain arrays -- no torch backbone, no
video -- so it runs in well under a second.

    python3 scripts/check_result_shapes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config                                    # noqa: E402
from src.inference import MULTICLASS, MULTILABEL                      # noqa: E402
from src.inference_fast import build_prediction, detect_task          # noqa: E402

CFG = load_config()
failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail and not cond else ''}")
    if not cond:
        failures.append(name)


# --------------------------------------------------------------------------- #
print("\ndetect_task")
cases = [
    ("explicit multiclass",      {"task": "multiclass", "classes": ["a", "b"]},           MULTICLASS, 2),
    ("explicit multi-class",     {"task": "multi-class", "class_names": ["a", "b", "c"]}, MULTICLASS, 3),
    ("nested in cfg.model",      {"cfg": {"model": {"task": "multiclass"}},
                                  "classes": ["x", "y"]},                                 MULTICLASS, 2),
    ("inferred: classes, no thr", {"classes": ["p", "q", "r"]},                            MULTICLASS, 3),
    ("thresholds win over guess", {"classes": ["p", "q"], "thresholds": [0.3, 0.4]},       MULTILABEL, 2),
    ("explicit multilabel",      {"task": "multilabel"},                                   MULTILABEL, 9),
    ("bare legacy checkpoint",   {"folds": [], "thresholds": [0.5] * 9},                   MULTILABEL, 9),
]
for name, state, want_task, want_n in cases:
    task, labels = detect_task(state, CFG)
    check(name, task == want_task and len(labels) == want_n,
          f"got {task} with {len(labels)} labels, wanted {want_task} with {want_n}")

# --------------------------------------------------------------------------- #
print("\nbuild_prediction -- multi-label (the 9-behavior head trained here)")
labels = list(CFG["labels"])
overall = np.array([.81, .62, .45, .30, .22, .18, .11, .07, .04])
per_window = np.stack([overall * .6, overall, overall * .8])
thr = np.array([.50, .70, .40, .35, .60, .50, .50, .50, .50])
spans = [(0.0, 1.6), (1.6, 3.2), (3.2, 4.8)]

p = build_prediction(labels, MULTILABEL, "av", overall, per_window, spans, thr)
hits = [b["label"] for b in p.behaviors if b["detected"]]
check("task recorded", p.task == MULTILABEL)
check("no top_class for multi-label", p.top_class is None)
check("all classes listed", len(p.behaviors) == 9)
check("sorted by probability",
      all(a["probability"] >= b["probability"] for a, b in zip(p.behaviors, p.behaviors[1:])))
check("thresholds exposed to the UI", all(b["threshold"] is not None for b in p.behaviors))
check("independent firing (.81>=.50 and .45>=.40; .62<.70 does not)",
      hits == [labels[0], labels[2]], str(hits))
check("one evidence window per detected behavior", len(p.evidence) == len(hits))
check("evidence picks the peak window", list(p.evidence[0]["window_s"]) == [1.6, 3.2],
      str(p.evidence[0]["window_s"]))

# --------------------------------------------------------------------------- #
print("\nbuild_prediction -- multi-class (the two checkpoints still to come)")
classes = ["No target behavior", "Upper limb stereotypies", "Self-spinning"]
soft = np.array([0.18, 0.67, 0.15])
per_window_mc = np.stack([np.array([.5, .3, .2]), soft, np.array([.25, .55, .20])])

m = build_prediction(classes, MULTICLASS, "vision", soft, per_window_mc, spans)
check("task recorded", m.task == MULTICLASS)
check("top_class is the argmax", m.top_class["label"] == "Upper limb stereotypies",
      str(m.top_class))
check("confidence normalised", abs(m.top_class["probability"] - 0.67) < 1e-3,
      str(m.top_class))
check("distribution sums to 1",
      abs(sum(b["probability"] for b in m.behaviors) - 1.0) < 1e-3)
check("exactly one class detected",
      sum(1 for b in m.behaviors if b["detected"]) == 1)
check("no thresholds on a softmax head",
      all(b["threshold"] is None for b in m.behaviors))
check("evidence only for the winner", len(m.evidence) == 1)
check("evidence wording says class, not behavior", "class" in m.evidence[0]["note"])

# --------------------------------------------------------------------------- #
print("\nto_dict stays backward compatible")
d = m.to_dict()
check("legacy keys intact",
      all(k in d for k in ("behaviors", "evidence", "modality", "disclaimer")))
check("task surfaced to API clients", d["task"] == MULTICLASS)
check("top_class included when present", d["top_class"]["label"] == "Upper limb stereotypies")
check("top_class omitted for multi-label", "top_class" not in p.to_dict())

print(f"\n{'ALL CHECKS PASSED' if not failures else str(len(failures)) + ' FAILED: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
