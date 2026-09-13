"""Collect per-model test metrics into an ablation summary table."""
from __future__ import annotations

import json
from pathlib import Path

from src.config import load_config

ORDER = ["vision_mean", "vision_lstm", "vision_transformer", "audio_transformer", "av_transformer"]


def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    reports = cfg.resolve_path("reports_dir")
    rows = []
    for f in sorted(reports.glob("*_test_metrics.json")):
        d = json.loads(f.read_text())
        m = d["metrics"]
        rows.append({
            "model": f.name.replace("_test_metrics.json", ""),
            "modality": d.get("modality"),
            "macro_f1": m["macro_f1"],
            "macro_auprc": m["macro_auprc"],
            "macro_auroc": m["macro_auroc"],
            "micro_f1": m["micro_f1"],
            "ece": m["ece"],
            "n_samples": m["n_samples"],
        })
    rank = {n: i for i, n in enumerate(ORDER)}
    rows.sort(key=lambda r: rank.get(r["model"], 99))

    (reports / "ablation_summary.json").write_text(json.dumps(rows, indent=2, default=float))

    hdr = "| model | modality | macro-F1 | macro-AUPRC | macro-AUROC | micro-F1 | ECE |"
    sep = "|" + "---|" * 7
    lines = ["# AutiLens AI - ablation study (test split)", "",
             f"Test clips: {rows[0]['n_samples'] if rows else 'n/a'}  |  "
             "subject-disjoint splits  |  per-class thresholds tuned on validation.", "",
             hdr, sep]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['modality']} | {r['macro_f1']:.3f} | {r['macro_auprc']:.3f} "
            f"| {r['macro_auroc']:.3f} | {r['micro_f1']:.3f} | {r['ece']:.3f} |"
        )
    lines += ["", "_Small research dataset (train=121 clips): absolute scores are low and "
              "high-variance; read this as a relative modality/temporal comparison, not a benchmark._"]
    (reports / "ablation_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
