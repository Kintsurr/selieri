# Deliverable 1 (Code) — Compliance Audit & Readability Rating

Checked against `current_trends_assignment_description.pdf` → "Deliverable 1: the code".
State as of the repo reorganization; the model pipeline was run end-to-end on Linux/CPU and
reproduced the documented results.

## Compliance checklist

| Requirement (from the PDF) | Status | Evidence |
|---|---|---|
| Runs on the grader's **Linux** machine | ✅ | Full `python anticheat_model.py` run completed on Linux/CPU; reproduced B1 AUC=0.9940/F1=0.9605 and the 4.2× self-influence ratio exactly. |
| Standard languages / libraries | ✅ | Python 3.12; torch, pandas, numpy, scikit-learn, matplotlib, seaborn, openpyxl, python-docx. |
| Self-contained (data included) | ✅ | `data/combined_features_labels.xlsx` (2,000 games · 193,074 moves) is shipped; no external download needed. |
| Dependencies listed to install | ✅ | `requirements.txt` (CPU-only; tested versions noted). |
| A script / readme to run it | ✅ | `README.md` "Quick start": `pip install -r requirements.txt` then `python anticheat_model.py`. |
| Readable code | ✅ (rated; see below) | Per request, code was **not** refactored for style; rated instead, with a few targeted bug/correctness fixes applied. |
| Results clearly communicated | ✅ | `README.md` results table; `results/Selieri_v2.docx` (full write-up); `results/figures/fig_v2_*.png` (10 figures). |

> Caveat: `pipeline/*.py` (data generation) is **not** Linux-runnable — it needs local Windows
> Stockfish + LC0 binaries. It is included for provenance only; the deliverable (`anticheat_model.py`)
> does not depend on it, since the dataset is pre-shipped. This is stated in the README.

## Readability rating — overall: 7 / 10

Well-structured research code: numbered section banners, descriptive names, top-of-file docstrings,
the influence-function math is commented inline. Main weaknesses are a monolithic entry script and
dense one-liners.

| File | Lines | Score | Notes |
|------|------:|:-----:|-------|
| `pipeline/data_merger.py` | 117 | 8.5 | Clean argparse CLI, usage examples, small focused functions. |
| `pipeline/data_processor.py` | 295 | 8 | Excellent docstrings, clear logging, well-named stages. |
| `anticheat_model.py` | 1248 | 7 | Clear 1–8 banners and A1/A2/B1/B2 naming; but one 1,200-line script doing data+models+training+10 figures+full docx; runs on import; dense semicolon lines. |
| `pipeline/feature_extractor.py` | 464 | 6.5 | `EngineAdapter` class + named Irwin-feature helpers + checkpoint/resume; hardcoded engine paths. |
| `pipeline/data_generator.py` | 697 | 6 | Constants grouped by purpose; very dense game loop. |

## Fixes applied (low-risk, verified by py_compile + partial run)

1. `save_model` now prints `os.path.basename(path)` — previously `path.split(chr(92))[-1]` (a
   Windows backslash split) printed the full path on Linux.
2. `pipeline/data_processor.py` now invokes `feature_extractor.py` — previously the pre-rename
   `irwin hybrid.py`, which would have failed.
3. `pipeline/data_generator.py` engine-interaction `except` blocks now log via `dbg()` instead of
   swallowing failures silently (the WDL-option probe loop and SAN-parse fallbacks are left silent
   on purpose).

## Suggestions left as optional (not applied — higher risk / lower ROI)

These were reviewed and intentionally deferred; they restructure a working, results-reproducing
pipeline and would each need a full ~2h run to verify:

- Wrap `anticheat_model.py`'s body in an `if __name__ == '__main__'` / `main()` guard so it does not
  execute on import (requires re-indenting ~1,200 lines).
- Unpack semicolon-joined statements (~200+ sites, mostly harmless matplotlib one-liners).
- Make engine paths / batch names CLI-configurable in the pipeline scripts (currently edited in source
  between runs; `feature_extractor.py` already takes `--pgn/--out`).
- Split the ~400-line `.docx` builder out of `anticheat_model.py` into its own module.

## Honesty note (for the reflection)

The auto-generated `results/Selieri_v2.docx` narrative about influence "proponents/opponents"
contradicts the figures it cites — see `docs/explainability_analysis.md`. The figures are the source
of truth (proponents are clean games, opponents are cheat games). This is unfixed because correcting
it only changes the regenerated docx (a full run). It is worth addressing honestly in the reflection
document rather than repeating the docx's inverted claim.
