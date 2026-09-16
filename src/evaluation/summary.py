"""Compare runs — ranked by OUT-OF-FOLD performance only.

Two rules this module enforces, both of which the previous version violated:

1. **The 92-clip test set is never a selection signal.** The old summary sorted rows
   by `test_ensemble_metrics.macro_f1`, which is selection on the held-out set. Here
   the test column is hidden unless `--locked <tag>` names the candidate that was
   already chosen on OOF, and even then it is shown for that row alone.
2. **Aggregate seeds before ranking.** MPS/CUDA are nondeterministic run to run
   by roughly +/-0.02 macro-F1. Ranking individual runs lets the luckiest seed of a
   mediocre config top the table -- `w5_2` scored 0.598 while its own three-seed
   mean was 0.568 +/- 0.021, tied with baseline. Runs are grouped into configs
   (trailing seed suffix stripped) and ranked on the MEAN; single-seed configs are
   marked provisional because one run cannot be compared against a multi-seed mean.
3. **Never rank across class scopes.** A 4-family macro-F1 and a 9-behavior macro-F1
   are different exams. Rows are grouped by scope and ranked only within a group.

Runs predating nested CV are flagged: their OOF figures were inflated by a two-part
leak (epoch chosen on the outer fold, thresholds/calibrators fitted on the pooled OOF
they scored), measured at −0.076 macro-F1 on the 9-class configuration.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from src.config import load_config


def _fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return "-" if v != v else f"{v:.{nd}f}"
    return str(v)


#: `w5_1`, `w5_2`, `seed_av_3`, `lora_A_s2` -> one config each.
_SEED_SUFFIX = __import__("re").compile(r"_(?:s|seed)?\d+$")


def config_name(tag: str) -> str:
    base = _SEED_SUFFIX.sub("", tag)
    return base or tag


def collect(reports):
    rows = []
    for f in sorted(reports.glob("*_cv.json")):
        d = json.loads(f.read_text())
        if d.get("diagnostic"):
            # single-fold runs cover 1/k of the pool; ranking them beside full
            # k-fold runs would compare different amounts of evidence
            continue
        oof = d.get("oof_metrics") or {}
        nested = bool(d.get("nested_cv", False))
        rows.append({
            "tag": d.get("tag", f.stem),
            "config": config_name(d.get("tag", f.stem)),
            "seed": d.get("seed"),
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

    by_scope = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_scope[r["scope"]][r["config"]].append(r)

    def agg(runs, key):
        vals = [r[key] for r in runs if isinstance(r[key], float) and r[key] == r[key]]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    L = ["# AutiLens - configuration comparison (**ranked by out-of-fold, not test**)", "",
         "Selection uses outer-OOF performance only. The 92-clip held-out test set is "
         "reporting-only and is shown for the locked candidate alone, after selection.",
         "", "**Seeds are averaged before ranking.** A single run cannot be compared "
         "against a multi-seed mean: run-to-run variance is about +/-0.02 macro-F1, so "
         "the luckiest seed of an average config will outscore the honest mean of a "
         "better one. Configs with one seed are marked *provisional*.",
         "", "Rows are grouped by class scope; scopes are never ranked against each other.", ""]

    for scope in sorted(by_scope):
        groups = []
        for cfg_name, runs in by_scope[scope].items():
            f1_m, f1_s = agg(runs, "oof_f1")
            groups.append({"config": cfg_name, "n": len(runs), "runs": runs,
                           "f1": f1_m, "f1_std": f1_s,
                           "p": agg(runs, "oof_p")[0], "r": agg(runs, "oof_r")[0],
                           "auprc": agg(runs, "oof_auprc")[0],
                           "auroc": agg(runs, "oof_auroc")[0],
                           "ece": agg(runs, "oof_ece")[0],
                           "nested": all(x["nested"] for x in runs),
                           "modality": runs[0]["modality"], "backbone": runs[0]["backbone"],
                           "secs": agg(runs, "secs")[0]})
        groups.sort(key=lambda g: -(g["f1"] if g["f1"] is not None else -1))
        L += [f"## scope: {scope}", "",
              "| config | seeds | nested CV | modality | backbone | **OOF F1** | OOF P | OOF R | OOF AUPRC | OOF AUROC | OOF ECE | time |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for g in groups:
            f1 = ("-" if g["f1"] is None else
                  (f"{g['f1']:.3f} ± {g['f1_std']:.3f}" if g["n"] > 1 else f"{g['f1']:.3f}"))
            seeds = f"{g['n']}" + ("" if g["n"] > 1 else " *(provisional)*")
            secs = "-" if g["secs"] is None else (f"{g['secs']:.0f}s" if g["secs"] < 120
                                                  else f"{g['secs'] / 60:.0f}min")
            flag = "yes" if g["nested"] else "**NO — leaked**"
            L.append(f"| {g['config']} | {seeds} | {flag} | {g['modality']} | {g['backbone']} | "
                     f"**{f1}** | {_fmt(g['p'])} | {_fmt(g['r'])} | {_fmt(g['auprc'])} | "
                     f"{_fmt(g['auroc'])} | {_fmt(g['ece'])} | {secs} |")
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
