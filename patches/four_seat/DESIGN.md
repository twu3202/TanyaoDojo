# Four-seat self-play on the main line: design draft

Draft, 2026-11-09. Prepared while the critic diagnostic runs, at the user's choice ("prepare both, run the diagnostic
first"). **Superseded by the pre-registration in [README.md](README.md) (11-10).** The check there found that the
arena's four rotations are mostly the same game under self-play, so the "4x / ~3.25x data per game" figures below do not
hold for the existing arena; the run keeps one rotation per seed.

## What changes

- **Now (1v3):** each worker plays the trainee against three copies of one fixed opponent from the pool (base, base,
  v4, snapshot), and only the trainee's samples train. Each seed is played four times, with the trainee rotating
  through the seats (files `_a` to `_d`).
- **Four-seat:** the trainee's current parameters sit in all four seats, and all four seats' samples train.
- At the same game cost, a game gives four times the samples. The opponents are the current policy, not a fixed one.

## What others do (background research, 11-09)

- **Mortal:** 1v3 against a frozen copy of the starting model, trainee samples only, since the first public commit.
  No public reason was found. Equim said such details "don't matter" when asked about the baseline (Discussion #38).
  He advises against reusing samples (#64), and says online training is "extremely sensitive to every hyperparameter"
  (#33).
- **DouZero:** one network per position, self-play, every position's data used, no opponent pool.
- **PerfectDou:** each position learns from its own data against the latest opponent model; it notes "no convergence
  guarantees for three-player games".
- **IJCAI four-player mahjong competitions:** the top teams used PPO or IMPALA with self-play of the latest model. The
  organisers name RL instability and measuring progress as open problems.
- **ACH (Tencent):** its slides say self-play with single-agent RL does not converge because the target moves.
- **Known risks:**
  - a moving target;
  - overfitting to one's own play (AlphaGo sampled past versions to prevent it);
  - strategy cycles (the reason for AlphaStar's league);
  - correlated samples within a game;
  - losing the fixed reference (OpenAI Five plays 80% against itself and 20% against past versions).

## Why the expected gain is smaller than I wrote on 11-05

- **More data per game is mostly freshness.** The freshness test (10-29) gave every sample one pass instead of 1.8 and
  did not help: F160k − A160k = −0.56 ± 0.89 ([online_freshness](../online_freshness/README.md)).
- **What is left is the change of opponents.** We evaluate 1v3 against v4. Today a quarter of the training games put
  v4 in the opponent seats. Pure self-play removes that, so the change could cut either way on our metric.
- **Prior:** about 10–15% that a four-seat stage beats the control A160k by the screening rule. On 11-05 I wrote that
  it was the cheapest change with possibly the largest gain; the freshness result was already in then, and I did not
  weigh it.

## Engineering found so far

- **No Rust change is needed for a first probe.** In the worker, the champion engine becomes a second engine on the
  trainee's own parameters, named `trainee`. The trainer's `player_names = ['trainee']` then loads all four seats.
- **The four rotations become near-duplicates.** With the same policy in every seat, the files `_a` to `_d` of a seed
  are four rollouts of one deal, differing only by sampling at T 0.1 (about 5% of decisions).
  - Playing one game per seed needs a small arena change in libriichi and a rebuild. WSL on win5060 has no Rust
    toolchain, so installing it would need the user's go.
  - A CPU check can measure how far the rotations diverge.
- **The server counts files, not samples.** A four-seat file holds four times the samples, so `capacity` and the
  drain need resizing to keep the trainer's samples per drain comparable.
- **Our own code:**
  - `run_selfplay_m1.sh` gives each worker one opponent model;
  - `player.py` `TrainPlayer` builds the champion engine.
  Both would change behind a config flag that is off by default.

## Variants to choose from

| | Seats | v4 exposure | Data per game | Risk |
| --- | --- | --- | --- | --- |
| S1 pure | all four trainee, on every worker | none | 4x (rotations near-duplicate) | moving target, no fixed reference |
| S2 mixed, like OpenAI Five | 3 workers four-seat, 1 worker 1v3 against v4 | 25% of games, ~8% of samples | ~3.25x | milder |
| S3 two seats | trainee in two seats, the pool in the other two (the `two_vs_two` arena) | as now | 2x | the smallest change |

**Draft recommendation:** S2, from A80k against the control A160k, with the usual SPRT and mechanism readouts. Every
other knob stays as A80k → A160k: the anchor, the reward and the temperature.

## Open before pre-registration

1. Variant (S1 / S2 / S3), and whether the user wants this run at all at a 10–15% prior.
2. The capacity and drain size under four times the samples per file.
3. Whether to change the arena for one game per seed (needs a Rust toolchain on WSL, or a build on trx).
4. The CPU checks:
   - all four seats load;
   - the champion engine follows the trainee's parameter updates;
   - how far the rotations diverge.
