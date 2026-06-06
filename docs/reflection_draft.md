# Reflection — Selieri: Chess Cheating Detection  *(DRAFT)*

> **This is a working draft, not the final hand-in.** Rewrite it in your own words and trim to the
> **2-page** limit. Bracketed `[…]` notes are instructions/choices for you to resolve, not final text.
> Source material: `results/Selieri_v2.docx` (the auto-generated paper) and
> `docs/explainability_analysis.md` (our critical review of the results).
>
> *Author(s): [your name(s)] · Course: Current Trends · Topic: [link to the explainability lecture]*

---

## 1. Why we picked this topic

Online chess has exploded, and so has engine-assisted cheating — it is a live, unsolved detection
problem with real stakes (titles, prize money, banned accounts) and imperfect commercial solutions
(Lichess' Irwin, Chess.com's internal tools). That made it a good fit for a *Current Trends*
project: a concrete, data-rich anomaly-detection task where we could **build something ourselves**
rather than just read about it.

What pushed us from "a classifier" to "this project" was the **explainability lecture**. Anti-cheat
is exactly the kind of high-stakes decision where a bare yes/no verdict is not enough — an accusation
needs to be *justifiable*. So we deliberately chose a topic where we could apply the explainability
techniques from class (influence functions) to our own model and ask not just *"is this game
cheated?"* but *"why does the model think so, and which evidence does it lean on?"*

`[DRAFT NOTE: add one personal sentence — e.g. you play online chess / saw a cheating scandal — the
rubric rewards honest, personal motivation.]`

## 2. Why we approached it the way we did

- **Dual-engine features (the "Cheater Zone").** Our core hypothesis: an engine-assisted move is one
  that is *objectively top* (low **Stockfish** rank) yet *human-unlikely* (high **LC0/Maia** rank,
  since Maia is trained to imitate human play; McIlroy-Young et al., 2020). The contrast between a
  raw-strength engine and a human-like engine is the signal. We encoded 9 metrics per engine
  (Rank, CPL, win-probability measures, sharpness, …) at depth 20 → 18 features per move.
- **A controlled four-model design.** To test whether the dual-engine idea actually adds value, we
  built a 2×2: per-move vs per-game prediction, each with the full 18 features vs a Stockfish-only
  9-feature **control**. All four share one BiLSTM and the *same* train/val/test split so the
  comparisons are clean.
- **A sequence model (BiLSTM).** Cheating is contextual — a single strong move is not suspicious, a
  *pattern* across a game is. A bidirectional LSTM lets each move be judged with both past and future
  context.
- **Explainability via influence functions.** We implemented **Koh & Liang (2017)** influence
  functions, using the **LiSSA** inverse-Hessian estimator (Agarwal et al., 2017) and restricting it
  to the classifier head (16,513 params) so it runs on CPU. We borrowed the **proponents/opponents**
  framing from **Pruthi et al. (2020, TracIn)**. This is the part directly seeded by the lecture: we
  wanted a principled way to point at the *training games* that most drive a given verdict.

## 3. Main results

`[Numbers pulled from results/Selieri_v2.docx / the saved checkpoints; reproduced on Linux/CPU.]`

| Model | Task | Features | Game AUC | Game F1 |
|-------|------|----------|---------:|--------:|
| A1 | Per-move | SF + Maia (18) | 0.9839 | 0.8896 |
| A2 | Per-move | SF-only (9) — control | 0.9670 | 0.8433 |
| **B1** | **Per-game** | **SF + Maia (18)** | **0.9940** | **0.9605** |
| B2 | Per-game | SF-only (9) — control | 0.9719 | 0.9115 |

- **The dual-engine features help, consistently.** Adding Maia/LC0 improved game AUC by ~+0.022
  (per-game) and ~+0.017 (per-move) over the Stockfish-only controls — modest but consistent support
  for the Cheater-Zone hypothesis.
- **Per-game beats per-move** at the game-level objective (B1 best), because it optimises the balanced
  game label directly; per-move models fight a ~10:1 move-level class imbalance and recover a verdict
  via max-pooling (high recall, low move-level precision).
- **Explainability — the solid finding:** cheat games have **~4.2× higher self-influence** than clean
  games. This matches the theory (Koh & Liang §5.4; TracIn): rare, atypical, high-gradient examples
  are the most "influential," and injected engine moves are exactly that.

## 4. What we learnt

- **Applying the lecture's explainability tools for real is harder than the slides suggest.** Getting
  influence functions to run meant practical compromises: restrict to the last layer for tractability,
  add a damping term because the Hessian of a non-converged, non-convex model isn't positive-definite,
  and accept that LiSSA gives *approximate* scores we did not validate against leave-one-out retraining.
- **The most valuable lesson was about honesty in explanations.** Our auto-generated write-up
  (`Selieri_v2.docx`) confidently states that the influence "proponents" are cheat games and
  "opponents" are clean — but our **own figures show the exact opposite** (proponents are all *clean*
  games, opponents all *cheat*). The prose was a templated narrative that assumed the expected result
  and was never reconciled with the actual run (see `docs/explainability_analysis.md`). Catching this
  taught us the real point of the explainability lecture: an explanation you don't verify against
  ground truth can be confidently wrong — and an *anti-cheat* tool that fabricates its justification
  is worse than one that just outputs a score. `[This is a strong, honest reflection point — keep it.]`
- **Modelling trade-offs:** per-move vs per-game is a genuine design choice (interpretable move-level
  scores vs. better game-level accuracy), and class imbalance shapes everything downstream.
- **Engineering reality:** the data was generated on Windows with local Stockfish/LC0 engines; making
  the analysis reproducible and self-contained on Linux (shipping the dataset, pinning dependencies,
  one run command) was its own lesson in "works on my machine" → "works on the grader's machine."

## 5. Status & next steps  *(this is an ongoing project)*

This is a **work in progress**, and the current numbers are interim:
- Results are on **simulated** games (Maia-vs-Maia with injected Stockfish moves); real-world
  validation on confirmed cheating cases is the obvious gap.
- A third 1,000-game batch is collected but not yet processed.
- Explainability to revisit: reconcile/justify the proponent–opponent **sign convention** (it is
  opposite to TracIn's), regenerate the write-up so the prose matches the figures, and spot-check a
  few influence scores against leave-one-out retraining.
- Possible extensions: multi-depth features (depths 10/15/20), attention pooling for interpretable
  move weights, and an A1+B1 ensemble (move-level scorer + game-level verifier).

---

### References followed
- Koh, P. W., & Liang, P. (2017). *Understanding black-box predictions via influence functions.* ICML. arXiv:1703.04730.
- Pruthi, G., Liu, F., Kale, S., & Sundararajan, M. (2020). *Estimating training data influence by tracing gradient descent (TracIn).* NeurIPS.
- Agarwal, N., Bullins, B., & Hazan, E. (2017). *Second-order stochastic optimization for ML in linear time (LiSSA).* JMLR 18(116).
- McIlroy-Young, R., et al. (2020). *Aligning superhuman AI with human behavior: Chess as a model system (Maia).* KDD.

`[LENGTH CHECK: aim for ~2 pages. If over, cut §2 bullets and §5 to a couple of lines — keep §4's
honesty paragraph, it's the differentiator the rubric (critical-minded, honest) rewards.]`
