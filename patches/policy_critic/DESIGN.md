# Separate policy and critic: design draft for the structural operator (a)

Draft, 2026-11-03. Not pre-registered yet; no GPU used. Prepared while the anchor-schedule test runs, at the user's
choice. Borrows the PPO recipe of verl (actor / critic, dual-clip, critic warm-up, KL to a reference policy, the
rollout-correction diagnostics), implemented inside our own Mortal trainer.

## Why a different operator

- Mortal's online step is Monte Carlo Q regression on the action taken, and the policy is argmax Q. Every fix so
  far kept that coupling.
- Its errors sit on the untaken side: Stage 0 measured +49 pt there, left from offline CQL and frozen after rl1.
  D2 measured Q gaps inflated 10–30×. Because the policy is argmax Q, those errors decide the play.
- A policy gradient uses only what happened: advantage = return − V(s). The untaken side's Q never enters it.
- **Precedent:** Suphx's RL stage used policy gradients with importance sampling and entropy regularisation.

## Offline readouts that set the design (11-03, CPU, logs of the reward arm)

**Policy sharpness.** softmax(Q/T) from the Q values workers log in `meta`, trainee seat, 400 games. Only actions
that become events carry `meta`; passes and agari decisions do not.

| Decision | Median top-1 − top-2 gap | Share of gaps < 0.1 pt | T 0.1: mean p(top 1) | T 0.1: share p(top 1) < 0.9 | T 0.1: entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Discard (45,846) | 1.12 pt | 7.8% | 0.947 | 16.4% | 0.125 nats |
| Call taken (1,909) | 1.68 pt | 4.8% | 0.969 | 9.4% | 0.072 nats |
| Riichi (806) | 1.75 pt | 4.7% | 0.970 | 8.7% | 0.070 nats |

- **Sampling at T 0.1 explores where it matters.** About 5% of decisions leave the greedy action, almost all of
  them close calls. That is the targeted exploration branch B wanted, without a separate mechanism.

**Staleness.** Policy KL between control snapshots on 9,014 trainee decisions:

| Pair (A160k line) | KL at T 0.1, mean / p99 | Greedy agreement |
| --- | ---: | ---: |
| s160k ‖ s152k (8k batches) | 0.0082 / 0.160 nats | 98.47% |
| s152k ‖ s144k | 0.0066 / 0.127 | 98.48% |
| s160k ‖ s144k (16k batches) | 0.0088 / 0.153 | 98.46% |

- **Workers lag far less than PPO's trust region.** Workers run 400–800 batches behind the trainer, against 8,000
  here. So the behavior policy can be approximated by the last-submitted parameters; tagging every game with its
  parameter version is not needed for a probe.
- **The Q line jitters rather than drifts.** The KL does not grow from 8k to 16k batches. About 1.5% of decisions
  flip back and forth without a direction, which fits ten flat main-line stages.

## Design

- **Policy.** A Mortal network (Brain + DQN head), initialised from the base. Its output o(s, ·) is used as logits:
  π(a|s) = softmax(o(s, ·) / T) over legal actions, T = 0.1, so π starts at softmax(Q_base / 0.1).
  - **Workers sample π exactly with existing code:** `boltzmann_epsilon = 1.0`, `boltzmann_temp = 0.1`,
    `top_p = 1.0`.
  - **Evaluation stays greedy, argmax o:** the same `run_eval.py` path and the same checkpoint keys (`mortal`,
    `current_dqn`). Every existing readout works unchanged.
- **Critic.** A small V head (1024 → 256 → 1) on the policy's features `phi`, which are detached first. It is
  regressed on the same Monte Carlo return the Q loss uses now (`q_target_mc`), so it learns the value of the
  behavior policy and touches no untaken action.
  - **Why not a separate Brain (changed 11-03).** A second trunk with its own forward and backward does not fit
    beside the policy, the reference and behavior copies and four workers on the 8 GB card.
  - **Still decoupled.** No critic gradient reaches the policy. The trunk was trained as a Q network, so its features
    already carry value. A wrong V only adds variance to the advantage, not bias.
- **Behavior policy, checked in code (11-03).**
  - The trainee's engine in the workers runs without the rule-based agari guard (`MortalEngine` default `False`;
    only the opponents' baseline engine has it on).
  - With `boltzmann_epsilon = 1` the engine samples `Categorical(logits = q / T)` over legal actions, which is π
    exactly.
  - The remaining mismatch is precision (workers use AMP) and parameter lag.
- **Losses**, per batch of trainee decisions:
  - Policy: PPO clip (ε 0.2) with dual-clip (c 3) on the advantage G − V(s), normalised per batch. The ratio is
    π_θ / π_beh, where π_beh comes from a frozen copy of the last-submitted parameters (one extra forward per batch).
  - KL to the anchor policy softmax(Q_base / T): k3 estimator, coefficient β. It replaces the Q-space λ anchor.
  - Entropy bonus: off at first. Exploration comes from T. Turn it on only if entropy collapses.
  - Critic: MSE(V, G).
  - Kept: next-rank auxiliary loss, gradient accumulation 8, grad clip 1.0, frozen BN.
- **Critic warm-up.** The first N batches (proposal 8,000, about 1 hour) train only V. The policy stays at the base,
  and the workers already sample at T 0.1, so V learns the right behavior policy.
- **Logged every save, in the style of verl.**
  - Behavior mismatch: KL(π_beh ‖ π_θ), ratio mean / max, the fraction clipped at each bound.
  - Policy health: entropy, KL to the anchor policy, greedy agreement with the base.
  - Critic: explained variance of V.
- **Not borrowed.** Group baselines over duplicate walls: training games are not dealt in duplicate, and the
  kyoku-level alignment needs work. Its variance gain is to be estimated offline first.

## Probe (to pre-register once the code passes its checks)

- **Start.** A80k, against the existing controls at 160k (A160k, plus the R and F arms; W200k when it is in).
  The alternative start, tpt 408k, would go straight for the record but has no same-budget control.
- **Budget.** 8,000 warm-up batches plus 72,000 policy-gradient batches to 160k. The extra forward makes this about
  10–12 GPU hours, within R3's 12.
- **Screening.** P160k − A160k by the usual SPRT (0 vs +1, capped at 16,000 groups).
- **Mechanism readouts.**
  - Did the policy move where Q is pessimistic? Greedy agreement with the base by decision bucket, push / fold and
    riichi rates, and the attack / defence split.
- **Stop rules.**
  - The collapse guard, as in every online run.
  - KL to the anchor above a cap (proposal 0.05 nats) for two saves in a row stops the run.

## Before any GPU time (CPU checks)

1. The trainer's π equals the worker's sampling distribution for the same parameters: the engine path with ε 1,
   checked on logged states.
2. One batch: the losses are finite, and the gradients reach the policy and the critic only where intended.
3. A 200-batch smoke run on the CPU with a small batch: the diagnostics are logged, and the checkpoint loads in
   `run_eval.py`.
4. Choose β so that the KL to the anchor at λ 0.2's typical policy movement (greedy agreement about 98.5% per 8k
   batches) costs about as much as the current anchor loss.

## Files

- **`train_pg.py`** (written 11-03, deployed as `Mortal/mortal/train_pg.py`). A fork of `train.py`'s online branch; the
  production trainer is untouched. It ships as `train_pg.patch`, a diff from the deployed `train.py` (build 67bce509)
  to `train_pg.py`.
  - Config: a `[pg]` section with `temp`, `clip`, `clip_c`, `kl_coef`, `ent_coef`, `value_coef`, `value_hidden` and
    `policy_start_step` (absolute).
  - At start it checks that every `[train_play.*]` profile samples π exactly.
  - Diagnostics, logged at every save in the trainer log and TensorBoard: entropy, KL to the reference, KL from
    behavior, ratio mean / max, the clip fractions (high, low, dual), greedy agreement with the reference, the
    advantage SD, and the critic's explained variance.
  - Check mode (`PG_CHECK_FILES`, `PG_CHECK_STEPS`): no server.
- **`pg_check.sh`** runs the CPU checks.
- **Planned:**
  - a `TRAINER` switch in `run_selfplay_m1.sh`, defaulting to `train.py`, to be added between chains;
  - chain scripts in the style of `xfer_*`, reusing `xfer_eval.sh`, `seqtest.py` and the pairing code.
