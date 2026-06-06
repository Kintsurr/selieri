# Selieri — Chess Cheating Detection with BiLSTM & Influence Functions

Per-move and per-game chess cheating detection using Bidirectional LSTM networks trained on dual
Stockfish + LC0/Maia engine features, with Koh & Liang (2017) influence-function interpretability.

Built for the VUB *Current Trends* programming assignment.

## Quick start

The single deliverable script is `anticheat_model.py`. It is self-contained — the dataset ships in
`data/`, and it runs on **CPU only** (no GPU required).

```bash
pip install -r requirements.txt
python anticheat_model.py          # or: python3 anticheat_model.py
```

This trains all four models, runs the influence analysis, and writes every output under `results/`
(see below). A full run takes roughly **1.5–2 hours on CPU**; the Koh & Liang self-influence stage
(Step 3/3) is the slow part. Everything is deterministic (`SEED = 42`).

## What it produces

| Output | Path |
|--------|------|
| Trained model weights (+ scaler) | `results/models/selieri_model_{permove,game}_{full,sfonly}.pt` |
| 10 figures | `results/figures/fig_v2_*.png` |
| Full write-up (paper) | `results/Selieri_v2.docx` |

## Results

Four experimental conditions, all sharing one BiLSTM architecture. Numbers below were reproduced
on Linux/CPU and match the saved checkpoints:

| Model | Task | Features | Game AUC | Game F1 |
|-------|------|----------|---------:|--------:|
| A1 | Per-Move sequence labelling | SF + Maia (18) | 0.9839 | 0.8896 |
| A2 | Per-Move sequence labelling | SF-only (9) — control | 0.9670 | 0.8433 |
| B1 | Per-Game binary classification | SF + Maia (18) | **0.9940** | **0.9605** |
| B2 | Per-Game binary classification | SF-only (9) — control | 0.9719 | 0.9115 |

Koh & Liang self-influence: cheat games carry **4.2×** higher-magnitude self-influence than clean
games. (See `docs/explainability_analysis.md` for a critical read of the influence results,
including a known mismatch between the auto-generated `.docx` narrative and the figures.)

**Key findings**

- **Per-game models** beat per-move on the game-level objective (balanced classes, direct target).
- **Adding LC0/Maia** features helps in both settings (ΔAUC ≈ +0.022 per-game, +0.017 per-move) by
  capturing the *Cheater Zone* — moves that are top Stockfish picks yet human-unlikely.
- **Per-move models** additionally yield move-level suspicion scores (which moves look engine-assisted).

## The Cheater Zone hypothesis

Engine-assisted moves cluster where **SF rank is low** (objectively the top engine move) but
**LC0/Maia rank is high** (a human-style network would not play it). This dual-engine contrast is the
core discriminative signal, and is why the SF-only models (A2, B2) serve as controls.

## Architecture

```
Input: (B, T, 18) — sequence of per-move feature vectors
  └─ BiLSTM (2 layers, hidden=128, bidirectional) → (B, T, 256)
      ├─ Per-Move: Linear(256→64)→ReLU→Dropout→Linear(64→1) per timestep → (B, T)
      └─ Per-Game: MaskedMeanPool → (B, 256) → Linear(256→64)→ReLU→Dropout→Linear(64→1) → (B,)
```

**Features (per move).** 9 Stockfish-D20 metrics + the same 9 from LC0/Maia-D20:
Rank, CPL, AdvWP, BestWP, WCL, Ambiguity05, difNextBest, difNextWorst, Sharpness.

**Explainability.** Koh & Liang (2017) influence functions,
`I(z_train, z_test) = -∇L(z_test)ᵀ H⁻¹ ∇L(z_train)`, with `H⁻¹v` approximated by LiSSA
(Agarwal et al. 2017) restricted to the classifier head (16,513 params) for tractable CPU compute.

## Project layout

```
anticheat_model.py     # THE deliverable — trains 4 models + influence + figures + paper
requirements.txt       # dependencies (CPU only)
data/
  combined_features_labels.xlsx   # 2,000 games · 193,074 moves (input dataset)
results/               # all generated outputs (created on run)
  models/  figures/  Selieri_v2.docx
docs/
  explainability_analysis.md              # critical analysis of the influence results
  current_trends_assignment_description.pdf
papers/                # the two reference papers (Koh & Liang; Pruthi et al. TracIn)
pipeline/              # data-generation scripts (see note below)
  data_generator.py    # simulate Maia-vs-Maia games, inject Stockfish "cheat" moves
  feature_extractor.py # per-move SF+LC0 features at depths 10/15/20, one row per game
  data_processor.py    # orchestrator: discover batches → extract → merge → unify
  data_merger.py       # merge features↔labels, then combine batches
```

### Note on `pipeline/`

These scripts produced `data/combined_features_labels.xlsx` and are included for completeness. They
were run on **Windows** and require local **Stockfish** and **LC0/Maia** engine binaries, so they do
**not** run on a stock Linux machine. You do **not** need them to reproduce the results — the dataset
is already shipped in `data/`. Pipeline order (for reference):

```
data_generator.py  →  sim_games/batchN.pgn + batchN.xlsx
feature_extractor.py  →  batchN.xlsx (features)
data_merger.py  →  data/combined_features_labels.xlsx
```

## References

1. Koh, P. W., & Liang, P. (2017). *Understanding black-box predictions via influence functions.* ICML. [arXiv:1703.04730](https://arxiv.org/abs/1703.04730)
2. Agarwal, N., Bullins, B., & Hazan, E. (2017). *Second-order stochastic optimization for machine learning in linear time.* JMLR 18(116).
3. Pruthi, G., et al. (2020). *Estimating training data influence by tracing gradient descent.* NeurIPS 2020.
4. McIlroy-Young, R., et al. (2020). *Aligning superhuman AI with human behavior: Chess as a model system.* KDD 2020.
