"""Compare runs — ranked by OUT-OF-FOLD performance only.

Two rules this module enforces, both of which the previous version violated:

1. **The 92-clip test set is never a selection signal.** The old summary sorted rows
   by `test_ensemble_metrics.macro_f1`, which is selection on the held-out set. Here
   the test column is hidden unless `--locked <tag>` names the candidate that was
   already chosen on OOF, and even then it is shown for that row alone.
2. **Never rank across class scopes.** A 4-family macro-F1 and a 9-behavior macro-F1
   are different exams. Rows are grouped by scope and ranked only within a group.

Runs predating nested CV are flagged: their OOF figures were inflated by a two-part
leak (epoch chosen on the outer fold, thresholds/calibrators fitted on the pooled OOF
they scored), measured at −0.076 macro-F1 on the 9-class configuration.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from src.config import load_config


def _fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return "-" if v != v else f"{v:.{nd}f}"
    return str(v)


def collect(reports):
    rows = []
    for f in sorted(reports.glob("*_cv.json")):
        d = json.loads(f.read_text())
        oof = d.get("oof_metrics") or {}
        nested = bool(d.get("nested_cv", False))
        rows.append({
            "tag": d.get("tag", f.stem),
            "scope": d.get("scope", f"{len(oof.get('per_class', {})) or '?'}-class (legacy)"),
            "nested": nested,
            "modality": d.get("modality", "?"),
            "backbone": d.get("backbone", "?"),
            "oof_f1": oof.get("macro_f1", d.get("oof_macro_f1")),
            "oof_p": oof.get("macro_precision"),
            "oof_r": oof.get("macro_recall"),
            "oof_auprc": oof.get("macro_auprc", d.get("oof_macro_auprc")),
            "oof_auroc": oof.get("macro_auroc"),
            "oof_ece": oof.get("ece"),
            "derived": (d.get("derived_baseline_oof_metrics") or {}).get("macro_f1"),
            "secs": d.get("seconds"),
            "test_evaluated": d.get("test_evaluated", "test_ensemble_metrics" in d),
            "test": (d.get("test_ensemble_metrics") or {}).get("macro_f1"),
        })
    return rows


def main(config_path: str | None = None, locked: str | None = None) -> None:
    cfg = load_config(config_path)
    reports = cfg.resolve_path("reports_dir")
    rows = collect(reports)

    by_scope = defaultdict(list)
    for r in rows:
        by_scope[r["scope"]].append(r)

    L = ["# AutiLens — run comparison (**ranked by out-of-fold, not test**)", "",
         "Selection uses outer-OOF performance only. The 92-clip held-out test set is "
         "reporting-only and is shown for the locked candidate alone, after selection.",
         "", "Rows are grouped by class scope; scopes are never ranked against each other.", ""]

    for scope in sorted(by_scope):
        group = sorted(by_scope[scope],
                       key=lambda r: -(r["oof_f1"] if isinstance(r["oof_f1"], float)
                                       and r["oof_f1"] == r["oof_f1"] else -1))
        L += [f"## scope: {scope}", "",
              "| run | nested CV | modality | backbone | **OOF F1** | OOF P | OOF R | OOF AUPRC | OOF AUROC | OOF ECE | 9→4 baseline | time |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in group:
            secs = "-" if r["secs"] is None else (f"{r['secs']:.0f}s" if r["secs"] < 120
                                                  else f"{r['secs'] / 60:.0f}min")
            flag = "yes" if r["nested"] else "**NO — leaked**"
            L.append(f"| {r['tag']} | {flag} | {r['modality']} | {r['backbone']} | "
                     f"**{_fmt(r['oof_f1'])}** | {_fmt(r['oof_p'])} | {_fmt(r['oof_r'])} | "
                     f"{_fmt(r['oof_auprc'])} | {_fmt(r['oof_auroc'])} | {_fmt(r['oof_ece'])} | "
                     f"{_fmt(r['derived'])} | {secs} |")
        L.append("")

    legacy = [r for r in rows if not r["nested"]]
    if legacy:
        L += ["> **Runs marked `NO — leaked` predate nested CV.** Their OOF figures are "
              "optimistic: the epoch was selected on the outer validation fold, and "
              "thresholds/calibrators were fitted on the pooled OOF they then scored. "
              "Measured cost on the 9-class config: **0.489 → 0.413 (−0.076)**. Do not "
              "quote them.", ""]

    if locked:
        hit = next((r for r in rows if r["tag"] == locked), None)
        if hit is None:
            L += [f"> `--locked {locked}` did not match any run.", ""]
        elif not hit["test_evaluated"]:
            L += [f"> Candidate **{locked}** is locked but has no test evaluation yet. "
                  f"Run `python -m src.cv_fast --eval-test --tag {locked}` once.", ""]
        else:
            L += ["## Final report — locked candidate", "",
                  f"Selected on OOF alone, then evaluated **once** on the held-out "
                  f"92-clip test set.", "",
                  "| candidate | scope | OOF F1 | **test F1 (final, unbiased)** |",
                  "|---|---|---|---|",
                  f"| {hit['tag']} | {hit['scope']} | {_fmt(hit['oof_f1'])} | "
                  f"**{_fmt(hit['test'])}** |", ""]
    else:
        n_eval = sum(1 for r in rows if r["test_evaluated"])
        L += [f"> Test column withheld — no candidate locked yet. "
              f"({n_eval} run(s) have touched the test set; pass `--locked <tag>` only "
              f"after selecting on OOF.)", ""]

    (reports / "model_comparison.md").write_text("\n".join(L))
    (reports / "model_comparison.json").write_text(json.dumps(rows, indent=2, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--locked", default=None,
                    help="tag of the candidate already selected on OOF; only then is "
                         "its test number shown")
    a = ap.parse_args()
    main(a.config, a.locked)
