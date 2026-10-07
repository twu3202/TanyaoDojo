# Offline base with suit × dragon symmetry augmentation (2026-10-05)

## Why

The online-RL improvement line is closed: targeted exploration (Stage 1), the globally calibrated anchor (M1, c = 0.3)
and the push/fold + riichi-only version (M1-sel) all land at 0 ± 0.5 against the plain 408k → 448k continuation at
100k games. The project goes back to the offline base, the other end of the pipeline. Of the offline levers, only
the LR-schedule fix has helped so far (v11). Doubling the data, widening with defence features (v5 / v18) and
capacity (B1, 256×54) were all ≈ 0. None of our offline configs uses augmentation (`enable_augmentation = false`
everywhere, and Mortal's own augmentation is only the man ↔ pin swap).

**Prior, stated before the run.** The case for a gain is modest:
- v18 (twice the data, older years) did not beat v5, and the step21 learning curve said label count is not the
  bottleneck for the A3 decisions.
- The symmetry check D1 found our models already nearly equivariant: under a symmetry the greedy action changes on
  ~2% of decisions, mostly near-ties.

What this run tests is whether enforced equivariance, plus seeing every position under varied tile labels, lifts
the base. Recomputed with the dora rule (below), the factor is 18, not the 36 quoted on 10-03.

## What

`[dataset] symmetry_augmentation = true` relabels each log by a uniform random one of 18 symmetries before libriichi
parses it (`symaug.py`). Everything else is re-encoded by libriichi from the relabelled game: observation, action
labels, masks. One epoch still has the same 1,289,661 logs and the same number of samples.

- **Suits.** All 6 permutations of m / p / s, red fives included. The dora successor (9 → 1) stays inside the suit.
- **Dragons.** Only the 3 cyclic rotations P → F → C → P. Swapping two dragons is not a symmetry, because the dora
  indicator relation is directed (indicator P makes F dora, F makes C, C makes P); a swap would give dora to the
  wrong tile. Number reversal (1 ↔ 9) fails for the same reason.
- **Winds.** Untouched.
- **Ryuuiisou.** The one rule still broken (souzu + hatsu only). It is too rare to matter.

## Check (`symaug_check.py`, 300 random 2019–2026 logs, 195,047 decisions × 17 non-trivial symmetries)

- Identity through `load_log` is bit-identical to `load_gz_log_files`.
- `relabel` inverts.
- Every string value in the logs is a tile, a wind or `?`, so no tile string escapes the regex.
- Player ids, sample counts, kyoku indices, dones, gamma flags, GRP features and final scores are identical.
- Actions and masks equal the originals under the action permutation.
- Observations: 882 of 1012 rows are exactly the original rows with tile columns permuted. The other 130 rows (red
  fives, per-suit rows) are re-encoded by libriichi, so they are correct by construction.
- Load cost per log: plain 237 ms, augmented 235 ms (the relabel is negligible next to the parse).

## Run (single variable against v11)

`configs/local_train_symaug.toml` is v11 item for item:
- same 2019–2026 houou logs, same file index (verified identical);
- pts [6, 4, 2, 0], CQL 5, 192×40, batch 512;
- LR cosine 1e-4 → 1e-5 over 1.21M steps, one epoch;
- plus `symmetry_augmentation = true`.

Code runs from a separate tree, `Mortal_symaug/mortal` (train.py = anchor version ba587de4 + this patch;
dataloader 29e7a08d + this patch), so `Mortal/mortal` is untouched. `num_workers` is 8, against v11's 6; this only
affects throughput.

The run starts 2026-10-05 15:37 on the 5060 Ti (about 9 steps/s). `symaug_run.sh` restarts after a crash and stops
at 1.22M steps (the v11 best point), keeping `symaug_final.pth`. `symaug_chain.sh` then runs the evaluation.

## Second arm (trx, from 2026-10-05 22:43): 256×54

`configs/local_train_symaug256.toml` is B1 (the v11 recipe at 256×54) item for item, plus the augmentation. It runs
on trx from the same code tree (every file has the same md5 as the WSL tree) at about 13 steps/s;
`symaug256_chain.sh` evaluates it on the trx station. With v11 and B1 already measured at 100k, the two arms make
a 2×2 design: augmentation on or off × 192×40 or 256×54.

- **Each arm against its own unaugmented twin.** symaug against v11, symaug256 against B1, with the gates below.
- **Augmentation main effect.** The mean of the two arm differences. Its CI is about √2 narrower than one arm's,
  if the two arms do not interact.
- **Interaction.** The difference between the two arm differences: does augmentation help the larger model more?

## Pre-registered reading (same gates as B1)

- **12k stop gate.** 12k duplicate vs v4 (s10000 4k + s14000 8k), paired with the arm's twin (v11 or B1). A point
  estimate < 0 stops that arm here.
- **100k decision.** Otherwise run 100k paired with the twin. A difference ≥ +0.5 with z ≥ 2 passes, and that base
  goes on to online RL with the tpt recipe. If neither arm passes, symmetry augmentation is ruled out for the
  offline base.
- **Mechanism readouts**, reported with every result:
  - diag_gap / place / bust against v4 at the same games;
  - the EV map on s14000 8k;
  - the D1 symmetry check on symaug against v11. Greedy changes under a symmetry should fall well below v11's;
    if they do not, the augmentation did not take.

## Results (2026-10-07): ruled out for the offline base

symaug reached 1.22M steps at 06:52 (`symaug_final`, md5 679b6618). Duplicate vs v4, paired with v11 at the same
seeds (pt/hanchan, 95% CI over seed groups):

| games | symaug − v11 | z | symaug | v11 |
|---|---|---|---|---|
| 12k (s10000 4k + s14000 8k) | +1.80 ± 1.69 | 2.10 | −1.98 | −3.79 |
| 100k (s10000–34999) | **+0.004 ± 0.597** | 0.01 | −2.494 | −2.498 |

- The 12k gate passed, so the run extended to 100k. At 100k the difference is zero: the 12k lead was v11 drawing
  poorly on s10000–15999, as with earlier 12k highlights. The per-segment differences (s11000 −0.82, s16000 −0.66,
  s20000 −0.07, s25000 +0.07) show no trend.
- Against B1 (256×54): +0.19 ± 0.61. B1 − v11 at the same 100k: −0.18 ± 0.62.
- **D1 symmetry readout** (300 games, 45,253 decisions; identity + 5 suit + 2 dragon rotations):

| | symaug | v11 |
|---|---|---|
| greedy changes, suit permutations | 2.11% | 2.26% |
| greedy changes, dragon rotations | 1.44% | 1.94% |
| symmetrized greedy ≠ plain greedy | 1.08% | 1.27% |
| median top-2 Q margin | 1.89 | 1.87 |

  The augmentation took (every violation went down), but v11 was already almost equivariant. The remaining changes
  are near-ties, a few hundredths of the median top-2 margin, so there was little to fix.

**Reading.** 100k is below +0.5 with z 0.01, so symaug fails the pre-registered decision, and symmetry augmentation
is ruled out for the offline base. The 256×54 arm stopped at 640k steps on 2026-10-06 when trx was needed, and was
cancelled on 2026-10-07 after this result, so the 2×2 design was not completed: with a zero main effect at 192×40
and B1 ≈ v11, the interaction is not worth three more days of GPU time. The 640k checkpoint is kept.
