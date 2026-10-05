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

The run starts 2026-10-05 15:37 on the 5060 Ti and moves to trx when trx is free. `symaug_run.sh` restarts after a
crash and stops at 1.22M steps (the v11 best point), keeping `symaug_final.pth`.

## Pre-registered reading (same gates as B1)

- **12k stop gate.** 12k duplicate vs v4 (s10000 4k + s14000 8k), paired with v11. A point estimate < 0 stops here.
- **100k decision.** Otherwise run 100k paired with v11. symaug − v11 ≥ +0.5 with z ≥ 2 passes, and the base goes
  on to online RL with the tpt recipe. Anything else rules out symmetry augmentation for the offline base.
- **Mechanism readouts**, reported with every result:
  - diag_gap / place / bust against v4 at the same games;
  - the EV map on s14000 8k;
  - the D1 symmetry check on symaug against v11. Greedy changes under a symmetry should fall well below v11's;
    if they do not, the augmentation did not take.
