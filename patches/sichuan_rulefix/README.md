# Corrected-rule Sichuan baseline (2026-11-15)

Authorized by the user's instruction to continue training on their RTX 5060 Ti. This supersedes the earlier
no-training hold for this bounded job. The variant system has priority; no riichi continuation or rental is started.

## Question

Can the corrected response-priority engine train stable, reproducible specialist baselines suitable for the
multi-variant system? This is not a single-variable estimate of the historical rule bug's effect.

## Frozen design

- Two from-scratch runs, each 50,331,648 environment steps (384 blocks). Never restore a serialized old environment.
- Environment seeds 20261115 / 20261116; parameter-initialization seeds 101 / 102.
- 128 channels x 6 blocks, original 22 observation planes, 1024 environments, 32 steps per update,
  4 updates per compiled block, minibatch 4096, 1 epoch, Adam 3e-4, gamma 1, GAE lambda 0.95,
  clip 0.2, entropy 0.01, value weight 0.5, grad norm 0.5, shaping 0.5, dual-clip disabled.
- JAX memory fraction 0.75. No simultaneous training or guest GPU work. All new code and outputs live in a
  distinct run directory; the deployed legacy tree and weights are unchanged.
- Save each block atomically; retain snapshots every 64 blocks (8,388,608 steps). Non-finite metrics stop training
  before saving a bad checkpoint. Four-hour hard timeout per training seed; full chain bounded to 24 hours.
- Logging every four blocks. No checkpoint selection by training loss or noisy intermediate evaluations.

## Gates and evaluation

1. Verify source manifest, JAX GPU device and free GPU / no foreign booking markers.
2. Run the independent response-priority regressions, the reference tests and a 100-game compiled differential
   test on the deployment environment. Run a 12-block GPU smoke training; it is discarded, not a selected model.
3. Run both full training seeds from fresh initialization, sequentially.
4. Corrected-engine duplicate 1v3 on seeds 10000..12999, two fixed segments (1000 + 2000 groups = 12,000 games).
   All four seat rotations must finish; shadow mask/current-player mismatches and action fallbacks must be zero.
   Raw group arrays and JSON reports are stored under the new run only, never the historical evaluator folder.
5. Compare each final seed against L1 and against the frozen historical P2 b2272 (298M) checkpoint.
   Also measure that frozen checkpoint against L1 on the corrected engine, enabling same-wall paired differences.
6. Report each seed separately, group-level sample SD / sqrt(n) CI, and paired deltas against the corrected-engine
   frozen baseline. A positive CI versus L1 is the minimum competence gate, not a claim of superiority to history.
   A stronger-recipe claim requires both seed point estimates positive in paired L1 comparisons and direct H2H,
   then a separately budgeted fixed 100k confirmation. No 4k record claims, no extrapolation to riichi pt units.

## Boundaries and departures

- New independent parameter seeds fix the old trainer's hard-coded PRNGKey(1) for every run. The new `init_seed`
  option defaults to 1, preserving behavior of existing commands.
- Atomic saving and finite-metric checks are operational safeguards, not learning-algorithm changes.
- This changes engine trajectories and uses the current trainer (including earlier GAE fixes), so a historical
  score difference cannot be attributed solely to the priority correction.
- CPU/GPU regression results and startup throughput will be recorded before calling the launch successful.
- A STOP file in the run directory stops the current run-owned process group at the next five-second controller
  check (TERM, then KILL after 60 seconds if necessary), preserves the last complete checkpoint, and prevents
  later stages. The controller and current-stage PIDs are recorded; never broad pkill. No persistent watcher
  or OS configuration change is installed.


## Startup adjustment before formal training

The first isolated deployment (v1) passed all 11 CUDA targeted/vmap regressions, but its serial 100-game CUDA
reference differential spent over eight minutes on device/host round trips at roughly 2% GPU utilization.
It was deliberately terminated before any smoke or formal training; no mismatch verdict was returned.
Deployment v2 keeps the CUDA targeted/vmap regressions and runs the random full-game differential on the same
host's CPU JAX backend. This changes validation scheduling only, not the training or evaluation design.


## Launch receipt

Deployment v2 passed 11 CUDA targeted/vmap regressions, 100 full-game CPU JAX differential games, and a
1,572,864-step GPU smoke. Formal seed 101 started on 2026-11-15 at 17:31 CST. At 17:33 it had logged 1,048,576
steps; GPU utilization was 98%, memory 6593/8151 MiB. The atomically saved 1,517,342-parameter checkpoint was
read back and all arrays were finite. Both training seeds and all fixed evaluations later finished.


## Fixed evaluation on the response-priority-only engine (2026-11-16)

Both 50,331,648-step runs finished on November 15, at 18:32 and 19:33 respectively (about 61 minutes each).
Five complete fixed evaluations used 3,000 groups and 12,000 games each, with zero shadow mismatches/fallbacks.

| Comparison | Mean ± 95% group CI |
| --- | --- |
| Frozen legacy P2 b2272 vs L1 | +2.024 ± 0.114 |
| Seed 101 vs L1 | +1.900 ± 0.118 |
| Seed 101 vs frozen legacy, direct 1v3 | −0.327 ± 0.114 |
| Seed 101 minus legacy, paired on L1 games | −0.124 ± 0.123 |
| Seed 102 vs L1 | +1.845 ± 0.116 |
| Seed 102 vs frozen legacy, direct 1v3 | −0.388 ± 0.113 |
| Seed 102 minus legacy, paired on L1 games | −0.179 ± 0.122 |

## Draw-turn correction and warm adaptation (2026-11-18)

The engine now explicitly distinguishes a draw from the forced discard after a pung. On the corrected engine,
frozen P2 scored +0.437 ± 0.090 against L1. Two warm-start PPO adaptations, each 16,777,216 steps with fresh Adam
and a 1e-4 learning rate, scored +0.188 ± 0.084 and +0.131 ± 0.083 in direct 1v3 against frozen P2. Their paired
gains against L1 were +0.200 ± 0.095 and +0.148 ± 0.093. Each result is a 12,000-game fixed-panel readout in
Sichuan settlement points. The checkpoints are candidates for independent confirmation.

The batch evaluator retains Python reference play and checks JAX legality, active player, draw state and final
score. `--verify-serial` compares complete trajectories. Run from the project root with suitable local checkpoints.

```bash
PYTHONPATH=. python patches/sichuan_rulefix/eval_fixed.py CANDIDATE OPPONENT --batch-groups 32 --output runs/sc_eval/NAME
```
