# Selieri — Analysis of Learning & Explainability Results

**Scope:** review of `anticheat_model.py`, the four trained `.pt` models, the 10
`results/figures/fig_v2_*.png` figures, and `results/Selieri_v2.docx`, compared against the two
reference papers:

- **[KL17]** Koh & Liang (2017), *Understanding Black-box Predictions via Influence Functions* — `1703.04730v3.pdf`
- **[TracIn]** Pruthi, Liu, Kale & Sundararajan (2020), *Estimating Training Data Influence by Tracing Gradient Descent* — `2002.08484v3.pdf`

---

## 1. Learning results

Four BiLSTM models (2-layer, hidden=128, bidirectional) on 2,000 simulated games
(193,074 moves, ~8.7% cheat moves, ~10.5:1 class imbalance, 50% cheat games).
Stratified split 1400/300/300. Metrics below are read from the saved checkpoints
(authoritative) and match the docx tables.

| Model | Features | Game Acc | Game Prec | Game Rec | Game F1 | Game AUC |
|-------|----------|---------:|----------:|---------:|--------:|---------:|
| A1 Per-Move | SF+Maia (18) | 0.8767 | 0.8054 | 0.9933 | 0.8896 | 0.9839 |
| A2 Per-Move | SF-only (9)  | 0.8167 | 0.7363 | 0.9867 | 0.8433 | 0.9670 |
| B1 Per-Game | SF+Maia (18) | 0.9600 | 0.9481 | 0.9733 | 0.9605 | 0.9940 |
| B2 Per-Game | SF-only (9)  | 0.9100 | 0.8968 | 0.9267 | 0.9115 | 0.9719 |

Move-level (per-move models only):

| Model | Move Acc | Move Prec | Move Rec | Move F1 | Move AUC |
|-------|---------:|----------:|--------:|--------:|---------:|
| A1 SF+Maia | 0.8395 | 0.3501 | 0.9841 | 0.5165 | 0.9308 |
| A2 SF-only | 0.8210 | 0.3246 | 0.9769 | 0.4873 | 0.9247 |

**Reading of the learning results (all internally consistent and real):**

- **Per-game beats per-move at game level.** B1 is the strongest model
  (AUC 0.9940, F1 0.9605). This is expected: per-game models optimise the
  balanced game label directly (pos_weight≈1.0), while per-move models fight a
  10.4:1 imbalance and recover a game verdict via `max(move_prob)`.
- **SF+Maia > SF-only in every setting.** ΔAUC = +0.0221 (per-game),
  +0.0169 (per-move); ΔF1 = +0.049 / +0.046. The LC0/Maia "human-likeness"
  channel adds genuine, if modest, signal — consistent with the Cheater-Zone
  hypothesis.
- **Move-level precision is low (0.32–0.35) with very high recall (0.97–0.98).**
  The per-move classifier flags many moves; the game verdict survives only
  because one true-positive move per cheat game is enough under max-pooling.
  This is a real property of the imbalance + threshold, not an error, but it
  means "per-move suspicion scores" are noisy at the individual-move level.
- **No leakage red flags** in the metrics themselves; numbers are plausible for
  *simulated* data and the docx correctly lists "simulated data" as the first
  limitation.

---

## 2. Explainability results (Koh & Liang influence functions, applied to A1)

The pipeline computes three things on the classifier head only (16,513 params),
using LiSSA for the inverse-Hessian-vector product:

1. **Per-train-game gradients** (1400 games).
2. **Influence matrix** `I(z_train, z_test) = -∇L(z_test)ᵀ H⁻¹ ∇L(z_train)` for
   the 50 most suspicious test games. Sign convention adopted: **negative =
   proponent (helpful/supports the suspicion), positive = opponent (harmful).**
3. **Self-influence** `I(z,z)` for every training game.

### What the figures actually show

- **`fig_v2_proponents.png`** — Top-10 proponents are **G632, G832, G898, G625,
  G1780, G870, G728, G1542, G827, G556 — every one labelled `(clean)`.**
  Top-10 opponents are **G1447, G1107, G1005, G465, G1466, G145, G66, G278,
  G166, G1006 — every one labelled `(cheat)`.**
  → **Proponents = 0/10 cheat; Opponents = 10/10 cheat.**
- **`fig_v2_selfinf.png`** — mean self-influence: cheat = **−0.2543**,
  clean = **−0.0610** → **4.2× higher in magnitude** for cheat games.
  (Left panel is squashed because a few outlier games have huge negative
  self-influence, stretching the x-axis to ≈ −50; a cosmetic/clip issue, not a
  correctness one.)
- **`fig_v2_heatmap.png`** — one test-game row (rank ≈10) is strongly negative
  across nearly all training games, and the two rightmost (highest-variance)
  training columns flip sharply positive. Consistent with the matrix range
  reported.

### Interpretation

- The **self-influence result is sound and well-supported.** Cheat games being
  rare and high-gradient → higher self-influence magnitude is exactly the
  behaviour [KL17 §5.4] and [TracIn §4.1] describe for atypical/hard examples.
  The 4.2× number is real and the qualitative claim is defensible. (Caveat:
  "higher self-influence" means higher *magnitude*; the signed values are
  −0.25 vs −0.06, i.e. cheat is *more negative*.)
- The **proponent/opponent result is the opposite of what the model decisions
  story would predict.** Under the adopted convention, the games that *support*
  the model's suspicion of cheat-predicted test games are all **clean**, and the
  games that *oppose* it are all **cheat**. This is a striking, real result that
  deserved analysis — instead the prose papers over it (see §4).

---

## 3. Fidelity to the reference papers

### Method vs [KL17] — faithful

- Influence formula `I = -∇L(z_test)ᵀ H⁻¹ ∇L(z)` matches [KL17] Eq. (2) exactly,
  including the **negative=helpful** sign of `I_up,loss`.
  (`anticheat_model.py:571`, `:466`.)
- **LiSSA recursion** `estimate = v + (1-damping)·estimate − HVP/scale`
  (`anticheat_model.py:506-526`) is the standard Agarwal-et-al. estimator that
  [KL17] use for stochastic `H⁻¹v`.
- **Damping = 0.01** addresses non-PSD Hessians from a non-converged / non-convex
  model — exactly the fix [KL17 §4.2] prescribes ("add a damping term λ if H has
  negative eigenvalues").
- **Restricting to the last layer** (classifier head) to keep the Hessian
  tractable is the same shortcut [KL17] use ("we work with parameters in the
  last layer") and that [TracIn] calls "cherry-picking layers."
- **Self-influence to rank/score training points** is the evaluation both papers
  use ([KL17 §5.4], [TracIn §4.1]).

→ The implementation is a legitimate, recognizable Koh & Liang influence-function
analysis. Nothing in the *method* is fabricated.

### Terminology vs [TracIn] — borrowed, with a sign caveat worth flagging

- "**Proponents / opponents**" is **[TracIn]'s** terminology (Remark 3.2), not
  Koh & Liang's. The docx introduces it inside the K&L section (§2.2, §6) and
  cites Pruthi as ref [3], which is fine, but a reader checking the TracIn paper
  will hit a **sign clash**: [TracIn] defines **positive = proponent**, whereas
  this work (correctly, for K&L's `I_up,loss`) uses **negative = proponent**.
  The two are opposite because TracIn scores *loss reduction* while K&L's
  `I_up,loss` scores *loss change*. The docx never explains this, so the chosen
  convention looks arbitrary unless the reader knows both papers.

### Results realism vs the papers

- The papers validate influence estimates against leave-one-out retraining and
  report them as approximate/noisy ([KL17 §4], [TracIn] Fig. 2). This project
  does **no** such validation and runs LiSSA with modest depth (100, and 50 for
  self-influence) — so the influence numbers should be read as indicative, not
  precise. The docx's §7.4 does acknowledge "LiSSA introduces approximation
  variance," which is appropriately cautious.

---

## 4. Did the AI hallucinate in the explainability part?

**Yes — in the *narrative interpretation*, not in the computation.** The numbers
and figures are real and correctly computed; the prose that explains them in
`Selieri_v2.docx §6.2` asserts conclusions that are **contradicted by the very
figures it references.**

**Evidence — docx §6.2 says:**

> "0/10 top proponents are labelled cheat games, **confirming the model
> generalises from real cheat patterns rather than spurious correlations.**
> **Opponents tend to be clean games** with unusually engine-like moves
> (forced lines, endgame precision) that confuse the model."

**What `fig_v2_proponents.png` actually shows:**

- Proponents: **0/10 cheat (all clean).**
- Opponents: **10/10 cheat (all cheat).**

So:

1. **"0/10 … confirming the model generalises from real cheat patterns" is a
   logical non-sequitur.** 0/10 cheat proponents is *evidence against* that
   conclusion, not for it. The sentence is a canned template
   (`anticheat_model.py:1122-1128`, `f'{n_cheat_props}/10 ... confirming ...'`)
   written assuming `n_cheat_props` would be high; it was never reconciled with
   the computed value of 0.
2. **"Opponents tend to be clean games" is factually false** — the figure shows
   all 10 opponents are *cheat* games. This is a direct contradiction of the
   plotted data, i.e. a hallucinated description.

This same unsupported story propagates to the **Abstract** and **Conclusion**,
which claim the model "grounds its decisions in **prototypical cheat games**"
with high self-influence. The self-influence half is fine, but the
"grounds decisions in cheat games" half is contradicted by the proponent result
(the games it leans on are clean).

**Not hallucinated (important to be fair):**

- The metrics tables, ΔAUC/ΔF1 values, epoch counts, and self-influence ratio
  (4.2×) are all real, computed live, and consistent with the checkpoints.
- The influence-function *method* is faithfully implemented per [KL17].
- The self-influence finding (cheat games higher-magnitude self-influence) is
  genuine and theory-consistent.

**Root cause:** the explainability conclusions were written as fixed narrative
templates encoding the *expected* outcome (proponents=cheat, opponents=clean),
and were not regenerated/checked against the actual run, which came out inverted.
The figures are the source of truth; the prose drifted from them.

---

## 5. Recommendations

1. **Rewrite docx §6.2, the Abstract, and §9** to match the figures: proponents
   are clean, opponents are cheat. Then *explain* the inversion rather than deny
   it — e.g. investigate whether (a) the "most suspicious" test set is dominated
   by false positives, (b) the masked, `pos_weight`-scaled per-move loss flips
   gradient alignment, or (c) the adopted sign convention should be inverted to
   match intent. Any of these is a publishable observation; the current text is
   not.
2. **State the sign convention explicitly** and note it is opposite to TracIn's,
   since both papers are cited.
3. **Add a sanity check / LOO spot-check** for a handful of influence scores, as
   both reference papers do, before drawing conclusions.
4. **Fix the self-influence left panel** (clip/winsorize outliers) so the
   distributions are visible, and clarify "higher self-influence" = higher
   magnitude (signed values are negative).
5. Treat the 4.2× self-influence result as the *solid* explainability finding;
   treat the proponent/opponent claims as currently *unreliable* pending rewrite.
