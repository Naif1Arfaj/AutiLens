# Serving multi-class checkpoints

Everything trained in this repo is **multi-label**: 9 independent sigmoids over
the behavior list in `configs/default.yaml`, each with its own threshold tuned on
out-of-fold predictions. Any number of behaviors can fire on one clip.

A **multi-class** checkpoint is the other shape: a softmax over mutually
exclusive classes, exactly one of which wins per clip. The app supports both and
picks the view from `Prediction.task` — you do not need to touch the UI.

## What the `.pt` must contain

`torch.load(path)` should return a dict with:

| key | required | purpose |
|---|---|---|
| `folds` | yes | `list[state_dict]` — one entry per fold head. A single model is a list of one. This is what marks the file as a fast-pipeline ensemble. |
| `task` | **yes** | `"multiclass"`. Also accepted: `"multi-class"`, `"single-label"`, `"categorical"`. |
| `classes` | **yes** | `list[str]` of class names, in output-unit order. `class_names` also accepted. |
| `cfg` | yes | at minimum `{"model": {"modality": "vision"\|"audio"\|"av"}, "features": {...}, "windows": {...}}` — merged over `configs/default.yaml`. |
| `thresholds` | no | **must be absent.** A softmax head has nothing to threshold, and its presence is what the detector reads as "multi-label". |

Detection order is in `detect_task()` ([`src/inference_fast.py`](../src/inference_fast.py)):
explicit `task` → `cfg.model.task` → a `classes` list with no `thresholds` →
otherwise multi-label. Setting `task` explicitly is strongly preferred; the
inference rule is a fallback, not a contract.

The head is rebuilt with `FeatureHead`, so the classifier's output dimension must
equal `len(classes)`. `FeatureHead` sizes itself from `cfg["labels"]`, which the
predictor overwrites with your class list before loading the state dict.

## Getting it ranked and served

Selection is by measured accuracy, not filename — see `src/model_registry.py`.
Drop `models/<tag>.pt` next to a `reports/<tag>_cv.json` containing:

```json
{
  "tag": "<tag>", "task": "multiclass",
  "modality": "vision", "backbone": "swin3d_t",
  "folds": 5, "calibrated": false,
  "classes": ["...", "..."],
  "test_ensemble_metrics": { "accuracy": 0.71, "ece": 0.04 }
}
```

Selection ranks on the **out-of-fold** score, not the test score — choosing on the
test set would leak it. Include `oof_macro_f1` (or `oof_metrics.accuracy` for a
multi-class head) and set `nested_cv: true` if the run used nested CV, since
nested-CV runs are ranked ahead of pre-nested ones. `test_ensemble_metrics` is
still read, but for display only. With two
multi-class checkpoints present, the better-scoring one is served and both appear
in the UI's candidate table.

A checkpoint with no report is still listed, flagged "not yet evaluated", and
sorted last — visible, but never auto-served over a model with real numbers.

To pin one regardless of score: `AUTILENS_CKPT=models/<tag>.pt`.

## Verifying before the weights arrive

```bash
python3 scripts/check_result_shapes.py   # both head shapes, no backbone needed
python3 -m src.model_registry            # the live ranking
```

## What changes in the UI

| | multi-label | multi-class |
|---|---|---|
| Result view | ranked behavior rows, threshold tick per bar | single verdict card + distribution bar |
| Detected | every behavior over its threshold | the argmax class only |
| Evidence | one window per detected behavior | one window, for the winning class |
| `to_dict()` | `top_class` absent | `top_class: {label, probability}` added |

`behaviors`, `evidence`, `modality` and `disclaimer` keep their exact previous
meaning in both cases, so existing API clients are unaffected.
