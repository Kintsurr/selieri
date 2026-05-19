# Selieri — Chess Cheating Detection with BiLSTM & Influence Functions

Per-move and per-game chess cheating detection using Bidirectional LSTM networks trained on dual Stockfish + LC0/Maia engine features, with Koh & Liang (2017) influence function interpretability.

## Overview

Four experimental conditions, all using identical BiLSTM architecture:

| Model | Task | Features | Game AUC |
|-------|------|----------|----------|
| A1 | Per-Move sequence labelling | SF + Maia (18) | ~0.986 |
| A2 | Per-Move sequence labelling | SF-only (9) — control | ~0.980 |
| B1 | Per-Game binary classification | SF + Maia (18) | ~0.995 |
| B2 | Per-Game binary classification | SF-only (9) — control | ~0.990 |

## Key Findings

- **Per-game models** outperform per-move on game-level F1 (direct objective, balanced classes)
- **Adding LC0/Maia** features consistently improves both settings by capturing the "Cheater Zone" (low SF rank + high LC0 rank simultaneously)
- **Per-move models** provide move-level suspicion scores — which specific moves were engine-assisted
- **Koh & Liang influence functions** show cheat games carry ~5x higher self-influence than clean games

## The Cheater Zone Hypothesis

Engine-assisted moves tend to cluster in a distinct region:
- **Low SF Rank** → objectively top engine choice
- **High LC0 Rank** → a human-style network would NOT play this move

This dual-engine contrast is the core discriminative signal.

## Architecture

```
Input: (B, T, 18) — sequence of move feature vectors
  └─ BiLSTM (2 layers, hidden=128, bidirectional) → (B, T, 256)
      ├─ Per-Move: Linear(256→64)→ReLU→Dropout→Linear(64→1) per timestep → (B, T)
      └─ Per-Game: MaskedMeanPool → (B, 256) → Linear(256→64)→ReLU→Dropout→Linear(64→1) → (B,)
```

## Features (per move)

**Stockfish D20 (9 features):** Rank, CPL, AdvWP, BestWP, WCL, Ambiguity05, difNextBest, difNextWorst, Sharpness

**LC0/Maia D20 (9 features):** Same metrics from the neural network engine

## Explainability

Implements **Koh & Liang (2017)** influence functions:

```
I(z_train, z_test) = -∇L(z_test)ᵀ H⁻¹ ∇L(z_train)
```

H⁻¹v approximated via **LiSSA** (stochastic second-order, Agarwal et al. 2017), restricted to the classifier head (16,513 parameters) for tractable CPU computation.

## Project Structure

```
run_all_experiments.py    # Main pipeline — trains all 4 models + influence + paper
train_per_move.py         # Standalone per-move training script
influence_analysis.py     # TracIn-style influence analysis (earlier version)
generate_paper.py         # Earlier paper generator (superseded by run_all_experiments.py)
merge_batch1_batch2.py    # Merges batch1 + batch2 feature/label files
merge_batch1.py           # Merges raw features with labels for batch1
process_all_batches.py    # Feature extraction pipeline (Stockfish + LC0)
raw data.py               # Game simulation script
irwin hybrid.py           # Feature extraction worker
```

## Data Pipeline

```
raw data.py  →  sim_games/batch*.pgn + batch*.xlsx
     ↓
process_all_batches.py  →  batch1_features.xlsx, batch2_features.xlsx
     ↓
merge_batch1.py / merge_batch1_batch2.py  →  combined_features_labels.xlsx
     ↓
run_all_experiments.py  →  models + figures + Selieri_v2.docx
```

## Requirements

```
torch
pandas
numpy
scikit-learn
matplotlib
seaborn
openpyxl
python-docx
```

## Usage

```bash
# Run full pipeline (trains all 4 models, influence analysis, generates paper)
python run_all_experiments.py

# Run just the per-move model
python train_per_move.py

# Run TracIn influence analysis (earlier implementation)
python influence_analysis.py
```

## Output Files

- `selieri_model_permove_full.pt` — A1 model weights + scaler
- `selieri_model_permove_sfonly.pt` — A2 model weights + scaler
- `selieri_model_game_full.pt` — B1 model weights + scaler
- `selieri_model_game_sfonly.pt` — B2 model weights + scaler
- `fig_v2_*.png` — 10 figures
- `Selieri_v2.docx` — Full scientific paper

## References

1. Koh, P. W., & Liang, P. (2017). *Understanding black-box predictions via influence functions.* ICML. [arXiv:1703.04730](https://arxiv.org/abs/1703.04730)
2. Agarwal, N., Bullins, B., & Hazan, E. (2017). *Second-order stochastic optimization for machine learning in linear time.* JMLR 18(116).
3. Pruthi, G., et al. (2020). *Estimating training data influence by tracing gradient descent.* NeurIPS 2020.
4. McIlroy-Young, R., et al. (2020). *Aligning superhuman AI with human behavior: Chess as a model system.* KDD 2020.
