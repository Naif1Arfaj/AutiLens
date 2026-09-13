"""Deterministic (template) screening report from model outputs (PDF section 11).

No LLM: the narrative is assembled from the structured prediction only, so it
cannot invent observations that the model did not produce.
"""
from __future__ import annotations

from src.inference import Prediction


def _fmt_window(w: list[float]) -> str:
    def mmss(s: float) -> str:
        return f"{int(s // 60):02d}:{s % 60:04.1f}"
    return f"{mmss(w[0])}-{mmss(w[1])}"


def build_report(pred: Prediction) -> dict:
    detected = [b for b in pred.behaviors if b["detected"]]
    top = pred.behaviors[0] if pred.behaviors else None
    ev_by_label = {e["label"]: e for e in pred.evidence}

    lines: list[str] = ["AutiLens AI - Behavioral Screening Report", ""]
    lines.append(f"Modalities used: {pred.modality}")
    lines.append("")
    lines.append("Detected behaviors (model probability >= tuned threshold):")
    if detected:
        for b in detected:
            frag = f"  - {b['label']}: {b['probability'] * 100:.0f}%"
            if b["label"] in ev_by_label:
                frag += f"  [evidence around {_fmt_window(ev_by_label[b['label']]['window_s'])}]"
            lines.append(frag)
    else:
        lines.append("  - None above threshold.")

    lines += ["", "All class probabilities:"]
    for b in pred.behaviors:
        lines.append(f"  - {b['label']}: {b['probability'] * 100:.0f}%")

    lines += ["", "Interpretation:"]
    if detected:
        names = ", ".join(b["label"] for b in detected)
        lines.append(
            f"  The model recognised patterns it associates with the dataset "
            f"categories: {names}. Confidence is limited by clip quality, "
            f"occlusion, and the small research dataset used for training."
        )
    else:
        lines.append(
            "  The model did not recognise any annotated behavior above its "
            "operating threshold in this clip."
        )
    if top:
        lines.append(f"  Highest-scoring category overall: {top['label']} ({top['probability'] * 100:.0f}%).")

    lines += ["", "Limitations:",
              "  - Trained on the AV-ASD research dataset with subject-disjoint splits; "
              "generalisation to other populations, cameras and settings is unverified.",
              "  - Behavior labels are observational, not clinical constructs.",
              "  - Evidence windows are approximate saliency, not verified events.",
              "", "Recommended next steps:",
              "  - Treat as a prompt for structured human observation, not a decision.",
              "  - Seek assessment by qualified professionals for any diagnostic question.",
              "", "IMPORTANT: " + pred.disclaimer]

    return {"text": "\n".join(lines),
            "detected_behaviors": [b["label"] for b in detected],
            "disclaimer": pred.disclaimer}
