# M1: calibrated anchor for online RL (2026-10-03)

## Why

Online RL here regresses Q(s, a) only for the action actually taken (pure Monte Carlo kyoku reward), and the anchor
(`patches/anchor_train.patch`) holds every legal action's Q at a frozen model's Q. Rollout labels show that the
model's Q margins are 10–30× the real value differences (regression slope of the rollout difference on the model's
Q difference, tpt 408k, single-wall labels with seed-cluster CIs):

| decision kind | pairs | slope |
|---|---|---|
| ordinary discard → tile-efficiency tile | 464k | 0.032 [0.028, 0.036] |
| riichi → dama | 116k | 0.052 [0.045, 0.058] |
| dama → riichi | 200k | 0.052 [0.043, 0.061] |
| fold → push | 867k | 0.063 [0.061, 0.065] |
| push → fold | 265k | 0.099 [0.095, 0.103] |

The ranking is informative (the binned means are monotone), but the scale is inflated, a leftover of offline CQL.
Since the taken action's Q is calibrated (Stage 0), Monte Carlo updates can move it by a point or two, while the
untaken actions sit 18–55 pt below it (median model margins) and the anchor keeps them there. The greedy policy can
therefore never flip, which is consistent with continued online RL not improving (408k → 448k: −0.30 at 100k) and
with targeted exploration (Stage 1) moving the untaken side by only 1–3 pt in 40k steps.

## What

The anchor target becomes `q* − c · (q* − Q_anchor(s, ·))`, with `q*` the anchor model's best legal Q and
`c = [anchor] margin_scale`. Scaling about the best action keeps the argmax and the whole ordering of the anchor,
so play does not change at step 0; it only puts the margins on the measured scale, where the taken action's
Monte Carlo evidence can overturn a choice. `c = 0.05` is the middle of the measured slopes.

## Run (single-variable against the historical 408k → 448k continuation)

Same as that run (start tpt 408k, anchor cprime_final, λ = 0.1, 7 workers, pool base×4 / v4×2 / C392k×1,
ε 0.005, temperature 0.1, 40k steps), plus `margin_scale = 0.05` in `[anchor]`, no `[explore]`.

## Pre-registered reading

- Mechanism (on a fixed state set from the A3 label games, 40k-step snapshot vs 408k):
  the median top-2 margin shrinks; the greedy action differs from 408k on more decisions than Stage 1 448k did
  (otherwise the lock was not the binding constraint); the flips relative to 408k on the step20 low-noise test
  pairs (1000 games, label averaged over 8 walls) gain ≥ 0 pt/game.
- Strength: 100k duplicate pairs vs 408k ≥ 0 passes, ≤ −0.3 stops; also paired against the historical 448k.

## Results (2026-10-03 – 10-05): did not pass, line closed

All runs continue from 408k for 40k steps. Strength is in pt per hanchan with a 95% seed-group CI.

| run | readout at the end | 100k vs v4 | − 408k | − historical 448k |
|---|---|---|---|---|
| historical continuation (control) | margin 22.2 pt, greedy ≠ 408k 2.49% | −0.93 | −0.30 | — |
| c = 0.05 | margin 2.5 pt, greedy ≠ 408k 11%; test_play −13 / −12 at 412k / 416k, stopped by the collapse guard at 416,800 | — | — | — |
| `margin_level = 'trainee'`, c = 0.05 (pilot) | test_play −53 at 412k: untaken actions lose their level constraint (design error) | — | — | — |
| c = 0.3 (5060 Ti, 4 workers) | margin 7.8 pt, greedy ≠ 408k 4.34%, low-noise flip gain +0.62 ± 0.64 | −1.038 ± 0.460 | −0.401 ± 0.547 | −0.106 ± 0.546 |
| c = 0.3, `margin_states = 'pushfold_riichi'` (trx, historical topology) | margin 17.0 pt, greedy ≠ 408k 3.08%, flip gain +0.40 ± 0.76 | −0.996 ± 0.455 | −0.359 ± 0.500 | −0.064 ± 0.514 |

- **Pre-registered gate.** Both full runs end at or below the −0.3 stop line against 408k.
- **Against the plain continuation** (same machine and config, one variable), both are 0 ± 0.5.
- **Mechanism.** The unlocked c = 0.3 drifts the way the plain continuation already does, only faster: calls
  26.15% (v4 30.55%, 408k 30.61%, historical 448k 27.32%), win rate −0.85 pp. The push/fold-only version keeps calls
  (30.16%) but riichis less (−1.24 pp).
- **Not an opponent-pool mismatch.** Played 1v3 against 3 × rl1_best (40k games each), c = 0.3 − 408k is
  −1.09 ± 0.86 (z −2.48), so the global compression hurts against both opponents.
- **Reading.** Every 40k continuation from 408k (historical, Stage 1 exploration, c = 0.3, push/fold-only) lands
  −0.13 to −0.40 against 408k, and every intervention lands 0 ± 0.5 against the plain continuation. The
  "vs 408k ≥ 0" gate is biased against all of them, since 408k is shared by every pair and continuations regress
  from it. Future online-RL interventions should use a same-length control continuation as the reference. Effects
  below ~0.6 pt are not resolvable at 100k.
