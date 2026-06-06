# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Selieri is a chess-cheating-detection study built as a VUB "Current Trends" course
assignment (`docs/current_trends_assignment_description.pdf`). Two deliverables are graded:

1. **Code** — must run on the grader's **Linux** machine, be self-contained (data included),
   list its dependencies, and ship a run script/readme.
2. **Reflection document** — max 2 pages: why the topic, why the approach, main results, what
   was learnt. Graded on being *critical-minded and honest*.

Keep both in mind when changing anything: code must stay Linux-runnable and the honesty bar
matters (see "Known integrity issue" below).

## Running the model pipeline

```bash
pip install -r requirements.txt
python anticheat_model.py          # or: python3 anticheat_model.py
```

`anticheat_model.py` (repo root) is the **only script meant to run on Linux** and the only one
needed to reproduce all results. It reads `data/combined_features_labels.xlsx`, runs entirely on
**CPU**, trains all four models, computes the influence analysis, and writes outputs under
`results/` (`results/models/*.pt`, `results/figures/fig_v2_*.png`, `results/Selieri_v2.docx`).
Those output dirs are auto-created via `os.makedirs`. There is no test suite and no separate
train/eval entry point — the single script does everything, fixed `SEED = 42`.

A full run is ~1.5–2h on CPU; the Koh & Liang self-influence stage (Step 3/3, LiSSA over 1,400
games) dominates. To smoke-test a change without a full run, launch it and confirm it reaches
"Ep 1" of model A1 (data load + model build + one training epoch), then kill it — epoch-1 numbers
are deterministic (`1.0683 / 0.7969 / 0.256 / 0.373`).

Verified reproduced results: A1 0.9839/0.8896, A2 0.9670/0.8433, **B1 0.9940/0.9605**,
B2 0.9719/0.9115; self-influence ratio 4.2×.

## The four-model design (core architecture)

All four share one BiLSTM (`anticheat_model.py`): 2 layers, hidden=128, bidirectional →
256-dim/timestep → `Linear(256→64)→ReLU→Dropout→Linear(64→1)`. They differ on two axes:

|        | SF + Maia (18 feat) | SF-only control (9 feat) |
|--------|---------------------|--------------------------|
| **Per-move** (logit per timestep, masked BCE, game verdict via `max(move_prob)`) | A1 | A2 |
| **Per-game** (masked mean-pool → one logit, balanced BCE) | B1 | B2 |

The 18 features are 9 Stockfish-D20 metrics + the same 9 from LC0/Maia-D20
(Rank, CPL, AdvWP, BestWP, WCL, Ambiguity05, difNextBest, difNextWorst, Sharpness). The thesis
("Cheater Zone") is that engine-assisted moves show **low SF rank + high LC0 rank** simultaneously,
so the dual-engine contrast is the discriminative signal — that's why A2/B2 (SF-only) are controls.

All four models use the **same stratified train/val/test index split** (1400/300/300) so results
are directly comparable; per-move and per-game use separate `StandardScaler`s.

The explainability section implements **Koh & Liang (2017) influence functions** via the **LiSSA**
inverse-Hessian-vector product, restricted to the classifier head (16,513 params) for CPU
tractability. Convention here: `I = -∇L(z_test)ᵀ H⁻¹ ∇L(z_train)`, **negative = proponent**
(opposite to TracIn's sign, which the docx cites — flag this if editing the writeup).

## Repo layout

```
anticheat_model.py   # the deliverable (reads data/, writes results/)
requirements.txt
data/                # combined_features_labels.xlsx (2,000 games · 193,074 moves)
results/             # generated: models/  figures/  Selieri_v2.docx
docs/                # explainability_analysis.md, assignment PDF
papers/              # the two reference papers
pipeline/            # Windows-origin data-generation scripts (see below)
```

## Data-generation pipeline (Windows-origin, NOT runnable here)

`pipeline/` produced `data/combined_features_labels.xlsx` on Windows with local Stockfish + LC0
binaries; the scripts hardcode `stockfish\stockfish.exe` / `Lc0\lc0.exe` and per-run batch
filenames, and the engines + `sim_games/` are absent. **Do not assume these run on the grading
machine** — only the shipped dataset is needed. Stages:

```
data_generator.py     simulate Maia-vs-Maia games, inject Stockfish "cheat" moves → sim_games/batchN.pgn + .xlsx
feature_extractor.py  per-move SF+LC0 features at depths 10/15/20, one JSON-list row per game (has --pgn/--out)
data_processor.py     orchestrator: discovers batches, calls the extractor, merges, loops every 30 min
data_merger.py        `merge` (features↔labels) then `combine` (stack batches) → combined_features_labels.xlsx
```

Data-alignment gotcha used throughout: feature files are **1-based `GameIndex`**, label files are
**0-based `game_id`**; joins are always `game_id = GameIndex - 1`.

## Fixes already applied (don't re-flag)

- `save_model` prints `os.path.basename(path)` (was `path.split(chr(92))[-1]`, a Windows-only split).
- `data_processor.py` references `feature_extractor.py` (was the pre-rename `irwin hybrid.py`).
- `data_generator.py` engine-interaction `except` blocks log via `dbg()` (the WDL-option probe loop
  and SAN-parse fallbacks are intentionally left silent).

Deliberately **not** done (owner declined as high-risk / low-ROI): wrapping the script in a
`main()` guard, unpacking semicolon-joined lines, splitting out the docx builder, and making engine
paths CLI-configurable. Don't reintroduce these without asking.

## Known integrity issue (relevant to the honest reflection)

`docs/explainability_analysis.md` documents a real discrepancy: the influence figures
(`results/figures/fig_v2_proponents.png`) show **proponents = all clean games, opponents = all
cheat games**, but the generated `results/Selieri_v2.docx` prose (Abstract, §6.2, Conclusion) claims
the opposite — a hallucinated narrative from a fixed template (`anticheat_model.py:~1122-1128`) never
reconciled with the live run. The self-influence result (cheat games ~4.2× higher magnitude) is
sound; the proponent/opponent claims are not. This was intentionally left unfixed (fixing it only
changes the regenerated docx, which costs a full ~2h run). When wrapping up the writeup or
reflection, treat the figures as ground truth and do not repeat the docx's inverted story.
