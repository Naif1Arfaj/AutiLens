# Impact Card — AutiLens AI

| | |
|---|---|
| **Beneficiary** | Behavioral therapists, early-intervention specialists, and researchers reviewing video for autism-related behavior screening |
| **Problem** | Manually reviewing full video recordings to tag behaviors is slow, and tagging consistency varies between reviewers |
| **Solution** | A computer-vision model flags 10 defined behaviors with confidence + approximate timestamps; a language model turns that into a plain-language, non-diagnostic screening report |
| **Target user & task** | A specialist doing a first-pass review of a short video clip, deciding what deserves closer attention |
| **Value delivered** | Faster first-pass triage of video review; consistent, confidence-scored, timestamped flags instead of a blank recording to watch in full |
| **Expected impact indicator** | Reviewer time-per-clip (baseline: full clip playback + manual notes) vs. time to review AI-flagged windows + generated report — to be measured in a pilot, not claimed here |
| **What it is NOT** | A diagnostic tool. It does not diagnose autism, and its output must not be treated as a final decision (see `README.md` Ethics & Privacy) |
| **Current maturity** | Research prototype trained on 335 clips (AV-ASD dataset, full 427-clip download, subject-disjoint splits); modest, honestly-reported accuracy (see `reports/cv_summary.md`) — a demonstration of feasibility, not a production-ready screening system |
| **Key limitation** | Small training set limits reliability per class; see `reports/*_examples.md` for concrete failure cases |
| **Responsible-use guardrails** | Non-diagnostic disclaimer on every report; LLM constrained to only the CV model's structured output (no invented clinical claims); no facial identity used; consent checkbox required in the demo UI |
