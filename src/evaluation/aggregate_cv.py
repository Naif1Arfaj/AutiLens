"""Aggregate *_cv.json runs into reports/cv_summary.{json,md}."""
from __future__ import annotations

import json

from src.config import load_config


def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    reports = cfg.resolve_path("reports_dir")
    rows = [json.loads(f.read_text()) for f in sorted(reports.glob("*_cv.json"))]
    rows.sort(key=lambda r: -r["test_ensemble_metrics"]["macro_auprc"])
    (reports / "cv_summary.json").write_text(json.dumps(rows, indent=2, default=float))

    L = ["# AutiLens AI - cross-validated results", "",
         "Subject-disjoint k-fold on pooled train+val; official test split held out. "
         "Ensemble = mean of the k fold models. TTA (h-flip) on. Thresholds tuned on out-of-fold predictions.", "",
         "| config | modality | backbone | folds | OOF macro-F1 | test single macro-F1 | test **ensemble** macro-F1 | ens. macro-AUPRC | ens. macro-AUROC | ens. ECE |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        e = r["test_ensemble_metrics"]
        L.append(
            f"| {r['tag']} | {r['modality']} | {r['backbone']} | {r['folds']} "
            f"| {r['oof_macro_f1']:.3f} "
            f"| {r['test_single_model_macro_f1']['mean']:.3f} ± {r['test_single_model_macro_f1']['std']:.3f} "
            f"| {e['macro_f1']:.3f} | {e['macro_auprc']:.3f} | {e['macro_auroc']:.3f} | {e['ece']:.3f} |"
        )
    L += ["", "_train+val pool = 148 usable clips. Absolute scores stay modest; the "
          "cross-validated ensemble numbers are the ones to trust for the "
          "vision-only vs. audio vs. vision+audio comparison (PDF §10)._"]
    (reports / "cv_summary.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
