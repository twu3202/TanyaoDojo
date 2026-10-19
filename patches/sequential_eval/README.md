# Sequential evaluation: the same decisions for about half the games

2026-10-19. Tool: `jax_rl/mjai_bot/seqtest.py`. The simulation lives in a local research directory: it reads our
evaluation dumps, and the method is below.

## Method

A truncated Wald SPRT on duplicate seed groups:

- **Unit.** One seed group is four rotations of one wall. The per-group value is its mean placement points, or,
  for a comparison, the per-group difference of two models on the same seeds.
- **Test.** H0: δ ≤ δ0 against H1: δ ≥ δ1. The variance is the running plug-in estimate.
- **Order and looks.** Groups are taken in ascending seed order from 10000, fixed before any game is played. The test
  looks every 1,000 groups (4k games) and caps at 25,000 groups (the sacred 100k).
- **At the cap.** An undecided test reports the fixed-sample result.
- **Scope.** A stop is a decision, not an effect size: the mean at a stop is biased towards the bound it crossed.
  Records stay on the fixed 100k.

The noise comes from our own data. 13 models have all 25,000 groups on seeds 10000–34999. From them:

- 11 paired comparisons (each centred on its own mean) give a per-group SD of 45.1, so a paired 100k has a CI of
  ±0.56.
- Absolute runs against v4 give an SD of 36.9, so ±0.46.

Each design × true effect below is 2,000 simulated runs drawn i.i.d. from the paired pool.

## Results: P(promote) / mean games played, by true effect of candidate − control

| design | −1.0 | −0.5 | 0 | +0.25 | +0.5 | +0.75 | +1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| current: 12k gate (≥ 0), then 100k, promote at ≥ +0.5 and z ≥ 2 | 0.00 / 23k | 0.00 / 35k | 0.02 / 55k | 0.11 / 66k | 0.34 / 77k | 0.65 / 85k | 0.84 / 90k |
| fixed 100k, same promote rule | 0.00 / 100k | 0.00 / 100k | 0.02 / 100k | 0.14 / 100k | 0.40 / 100k | 0.74 / 100k | 0.94 / 100k |
| SPRT 0 vs +0.5, α 0.05, β 0.2 | 0.00 / 24k | 0.00 / 37k | 0.02 / 69k | 0.09 / 82k | 0.31 / 86k | 0.65 / 76k | 0.89 / 62k |
| SPRT 0 vs +1.0, α 0.05, β 0.2 | 0.00 / 12k | 0.00 / 17k | 0.04 / 29k | 0.13 / 39k | 0.33 / 48k | 0.60 / 45k | 0.83 / 40k |
| **SPRT 0 vs +1.0, α 0.025, β 0.2** | 0.00 / 12k | 0.00 / 17k | **0.02 / 30k** | 0.08 / 42k | **0.26 / 54k** | 0.54 / 54k | **0.80 / 49k** |

- SPRT 0 vs +1.0 makes the current procedure's decisions at about half the games.
- At α 0.025 its false-promote rate under the null (2%) equals the current one, for slightly less power at +0.5.
- An SPRT aimed at +0.5 does not help: within 100k it ends undecided 35–59% of the time near the bounds.
- Nothing settles a 0.5-pt question inside 100k, sequential or fixed; that takes about 230k games.

## Replay of the 11 real comparisons (ascending seed order; the current rule replayed on the same order)

| comparison | 100k | current | SPRT 0 vs +1.0, α 0.025 |
| --- | --- | --- | --- |
| q25 − v11 | +0.55 (z 1.80) | drop at 100k | undecided at 100k → fixed result |
| q25one − v11 | +0.45 (z 1.47) | drop at 100k | undecided at 100k → fixed result |
| B1 − v11 | −0.18 | drop at 100k | drop at 40k |
| symaug − v11 | +0.00 | drop at 100k | drop at 40k |
| v11long − v11 | −1.26 | drop at 12k | drop at 12k |
| q25one − q25 | −0.10 | (reported at 100k) | drop at 48k |
| Stage 1 448k − 408k | −0.13 | drop at 100k | drop at 4k |
| M1-sel 448k − 408k | −0.36 | drop at 100k | drop at 24k |
| M1 c0.3 448k − 408k | −0.40 | drop at 12k | drop at 8k |
| historical 448k − 408k | −0.30 | drop at 100k | drop at 32k |
| v4p 448k − 408k | −0.23 | drop at 100k | drop at 28k |

The decisions are the same, and the games played fall from 924k to 436k (−53%).

At α 0.05 the historical 448k − 408k pair would have been promoted at 12k. Its first 3,000 groups ran
+2.11 ± 1.42, a fluctuation of about 3σ in a comparison that ends at −0.30. That one false promote is why α 0.025 is
the proposal.

## Proposal (needs approval before it replaces any rule)

- **Screening comparisons.** Candidate vs control, e.g. a new base vs v11, or an online snapshot vs 408k:
  - SPRT 0 vs +1.0, α 0.025, β 0.2;
  - 1,000-group segments in ascending seed order from 10000, cap 25,000 groups;
  - undecided at the cap → the fixed 100k CI, read as today.
  - A promote means "worth a confirmation", not a result.
- **Unchanged.** Records against v4 and the project target (≥ +0.01 at 100k) stay on the fixed sacred 100k, as do
  runs already pre-registered with a fixed design (the online transfer test).
- **Usage.** In a chain, run `seqtest.py` after each 1,000-group segment:
  - exit 0 = promote, 1 = drop;
  - 3 = play the next segment;
  - 4 = cap reached undecided;
  - 2 = bad input.

  Only a gap-free prefix of the fixed order counts. Quote the dump globs, so the script expands them, not the shell.
