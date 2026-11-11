# Separate policy and critic: does a policy-gradient stage beat the Q stage from the same start?

Pre-registered 2026-11-03, before the run. 5060 Ti. Prepared alongside the anchor-schedule test, at the user's
choice. It runs only if that test is dropped; if it is promoted, the main line keeps the GPU and this waits for the
user.

The design and its offline readouts are in [DESIGN.md](DESIGN.md). The decision to write the trainer ourselves,
copying verl's PPO pieces rather than porting verl or vime, is in
`research/2026-11-03_rl_infra_survey` (gitignored).

## Question

- Every online stage so far has been Monte Carlo Q regression with the policy taken as argmax Q.
- Its errors sit on the untaken side (Stage 0: +49 pt; D2: Q gaps inflated 10–30 times), and argmax Q acts on
  them.
- The replay of the historical line has stayed level at about −1.9 against v4 for ten stages. Neither the reward nor,
  if this runs, the anchor schedule explained the historical jump.
- **The question:** from the same start (A80k) and with the same budget (80,000 batches), does a policy-gradient
  stage, whose advantage uses only the actions taken, beat the Q stage A160k?

## Design

**Arm P** (`runs/xfer_p1`, port 5590) starts from A80k, with A80k as the 'base' opponents and v4 in the pool.

- **Trainer:** `train_pg.py` instead of `train.py`.
- **Config:** it must equal v11s2's apart from three things:
  - the workers sample π = softmax(Q / 0.1) exactly (`boltzmann_epsilon` 1.0 instead of 0.005);
  - the Q-space anchor is off (`weight` 0);
  - a `[pg]` section is added.

**Schedule and losses**

| Batches | What trains |
| --- | --- |
| 80,000 → 88,000 | The critic only (V head on the detached policy features). The policy and the workers stay at A80k. |
| 88,000 → 96,000 | The policy as well, with its learning rate ramped from 0 to 1e-5 (fresh Adam). |
| 96,000 → 160,000 | Policy and critic at full learning rate. |

- **Policy loss:**
  - PPO clip 0.2 with dual clip 3 on the batch-normalised advantage, return − V(s);
  - the ratio is taken against the last parameters submitted to the workers;
  - the next-rank auxiliary loss is kept.
- **KL to A80k:** exact over the legal actions at T 0.1. The coefficient adapts toward a target of 0.02 nats (initial
  1.0, horizon 2,000 batches).
  - For scale: the Q line's snapshots 8,000 batches apart differ by 0.008 nats and about 1.5% of greedy decisions.
- **Entropy bonus:** off. Exploration comes from sampling at T 0.1.

**Stop rules**

- The collapse guard, as in every online run.
- A watcher in the chain stops the arm if KL to the base exceeds 0.05 nats at two saves in a row.

## Readouts

- **Trajectory, deciding nothing:** 12k of P120k and P160k.
- **Screening:** P160k − A160k by the SPRT (0 vs +1, α 0.025, β 0.2, looks per 1,000 groups from seed 10000),
  capped at the 16,000 groups A160k has.
- **Also reported:** P160k − A80k (against the control's +0.28 ± 0.52), P160k − tpt 408k, P160k vs v4.
- **Mechanism readouts**, as for every evaluation: call / win / deal-in / riichi rates, the attack and defence split,
  bust and 4th rates.
- **PG diagnostics over the run**, from the trainer log: KL to the base, KL from the behavior policy, entropy, clip
  fractions, greedy agreement with the base, and the critic's explained variance.
  - These tell "the policy did not move" apart from "it moved and got worse".

## Verdicts

- **Promote.** A policy-gradient stage is a lever the Q stages are not. Next: a longer PG run, and the same operator
  from tpt 408k against its own Q continuation.
- **Drop.** At this budget, separating the policy from the critic does not beat the Q stage.
  - If the diagnostics show the policy barely moved (KL far below target, greedy agreement above 99%), the probe was
    too timid. One rerun at a higher KL target is allowed.
  - Otherwise (b), the sparse search at push / fold and riichi points, is next.
- **Undecided at the cap.** Read the fixed 16,000 groups: ≥ +0.5 with z ≥ 2 counts as a promote, ≤ 0 as a drop.
- **Prior.** About 20% for a promote.
- **Cost.** About 11 hours of training (one extra forward per batch for the behavior copy, against the control's
  Q-anchor forward), plus 1–3 hours of evaluation.

## Result (2026-11-04): dropped

The arm trained 80,000 batches without a restart (11-04 02:49–12:15), on 22.9M fresh trainee samples (reuse 1.78).
Neither the collapse guard nor the KL watcher fired. The evaluation chain ended at 13:41.

The SPRT stopped for H0 at the second look:

| Look | Groups | Mean | LLR (bounds −1.58, +3.47) |
| --- | ---: | ---: | ---: |
| 1 | 1,000 | −1.50 ± 2.46 | −1.26 |
| 2 (stop) | 2,000 | −1.13 ± 1.71 | −2.14 |

Pairings on the groups each snapshot played (P160k: seeds 10000–11999 and 14000–15999; the 12k rows: seeds
10000–10999 and 14000–15999):

| Comparison | Groups | Difference | z |
| --- | ---: | ---: | ---: |
| **P160k − A160k** | 4,000 | **−1.61 ± 1.25** | −2.52 |
| P160k − A80k (the start) | 4,000 | −1.08 ± 1.24 | −1.71 |
| A160k − A80k (control, all its seeds) | 16,000 | +0.28 ± 0.52 | +1.05 |
| P120k − A80k (12k) | 3,000 | −1.06 ± 1.43 | −1.45 |
| P160k − P120k (12k) | 3,000 | −0.15 ± 1.53 | −0.19 |
| P160k − tpt 408k | 4,000 | −2.61 ± 1.52 | −3.36 |
| P160k vs v4 | 4,000 | −2.97 ± 1.16 | −5.02 |
| P120k vs v4 (12k) | 3,000 | −2.47 ± 1.35 | −3.58 |

The policy-gradient stage lost about 1.1 to its own start by 120k and did not recover. Every other stage-2 variant
from A80k stayed within ±0.3 of it.

**Mechanism.** The 12k rows use the same 12,000 games per model; the last two rows use the 16,000 games of P160k's
seeds. All are against the v4 seats in the same games.

| | call rate (v4) | win, pp | deal-in, pp | riichi, pp | net / kyoku (attack / defence) | bust us / v4 | 4th, pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A80k (start), 12k | 31.0 (30.6) | −0.47 | +0.23 | +0.16 | −23 (−6 / −17) | 6.83 / 6.46% | +1.11 |
| P120k, 12k | 31.4 (30.6) | −0.51 | +0.45 | −0.02 | −50 (−19 / −31) | 7.06 / 6.53% | +1.63 |
| P160k, 12k | 30.7 (30.6) | −0.57 | +0.43 | +0.10 | −62 (−31 / −31) | 7.17 / 6.45% | +1.78 |
| A160k (control), 12k | 30.4 (30.6) | −0.52 | +0.05 | +0.08 | −13 (−8 / −5) | 6.98 / 6.63% | +0.39 |
| P160k, 16k | 30.7 (30.6) | −0.52 | +0.43 | +0.09 | −59 (−27 / −31) | 7.13 / 6.49% | +2.07 |
| A160k (control), 16k | 30.4 (30.6) | −0.51 | +0.09 | +0.13 | −17 (−11 / −7) | 7.00 / 6.70% | +0.82 |

- **Both sides got worse.** Deal-in rose 0.2 pp over the start, and defence went from −17 to −31 points per kyoku.
  Attack went from −6 to −31, without a change in win or riichi rate. 4th place rose 0.5–0.7 pp.
- **Calls** rose by 0.4 pp at 120k and fell back by 160k.
- **Position entering South 4 and converting it** (avg_pt, the two sum to avg_pt), on the 16,000 games:

  | | entering South 4 | converting it |
  | --- | ---: | ---: |
  | P160k | −1.60 | −1.37 |
  | A160k | −0.71 | −0.65 |

  The loss is spread over the whole game, not concentrated at the end.

**PG diagnostics over the run** (trainer log, every 200 batches):

| Batches | KL to A80k | KL from behavior | entropy | greedy agreement with A80k | explained variance |
| --- | ---: | ---: | ---: | ---: | ---: |
| 88,200 (policy starts) | 0.00001 | 0.00001 | 0.089 | 99.95% | ~0.03 |
| 98,200 | 0.018 | 0.004 | 0.080 | 97.7% | ~0.08 |
| 128,200 | 0.022 | 0.005 | 0.075 | 97.7% | ~0.08 |
| 160,000 | 0.024 | 0.006 | 0.072 | 97.6% | 0.095 |

- **The policy moved.** KL sat at the 0.02 target from about 98k on, and about 2.4% of greedy decisions changed. So
  the result is "it moved and got worse", and the too-timid rerun does not apply.
- **The critic explained under 10% of the return variance** throughout. The advantage was therefore mostly noise.
  This is the likeliest reason the policy drifted in a harmful direction: a hypothesis, not tested here.
- **The ratio diagnostic was spoiled by a few samples.** ratio_max sat at the code's clamp e^20 (4.85 × 10^8) from
  about 118k on, and ratio_mean spiked to 10^4 at times.
  - These are samples that the behavior copy (the last submitted parameters) rates as almost impossible.
  - Under both PPO clips such samples carry no gradient. The clipped fractions stayed under 1% on each side, so they
    were wasted rather than harmful.
  - Logging the sampling probability in the workers, as vime does, would remove them. Not done here.

**Verdict, by the rule set in advance:** drop. At this budget, separating the policy from the critic does not beat
the Q stage. The policy moved as far as intended, so the rerun allowed for a too-timid probe does not apply. Next by
the rule is (b), the sparse search at push / fold and riichi points, which is not prepared yet.

### What this probe did not test (added 2026-11-05)

A comparison with conventional actor-critic designs and with Suphx, ACH, PerfectDou, DouZero and Mortal
(`research/2026-11-05_rl_design_audit`, gitignored) found that this probe departed from them in five ways that bear
on the verdict:

- **The critic was a small head on detached policy features.** No source uses a permanently detached value head.
  Conventional designs give the critic its own network or let value gradients train the shared trunk. PPG's
  detached variant works only through an auxiliary phase that trains the shared features on the value loss.
  - The reason was the 8 GB card: a second trunk was judged not to fit. This was not measured; about 2.4 GB were
    free.
  - The user was not told of this compromise before the run.
- **The critic saw public information only.** In imperfect-information games the usual and most effective choice is
  a privileged critic that also sees hidden state: PerfectDou's value net sees the opponents' hands, and AlphaStar's
  ablation gives 82% against 22%. As a baseline it adds no bias.
- **Behavior probabilities were reconstructed**, not logged at sampling time as IMPALA, ACH and verl do. Hence the
  ratios pinned at e^20.
- **There was no entropy bonus and no return normalization.**
- **Four workers** produced about 7,200 games per drain, with samples reused 1.8 times.

So the drop says that PG with a weak, public-only critic and reconstructed behavior probabilities does not beat the
Q stage at this budget. It does not say that policy gradients cannot. The prior from others is still sobering:
- Mortal's author tried PPO for months before v1 and never beat the supervised baseline.
- Suphx needed 1.5M games and 44 GPUs per agent, adaptive entropy and rejection of large importance weights.

A cheap check comes before any rerun. Fit three critics offline on existing games and compare their explained
variance: public-only and separate, shared with gradients, and privileged.

## Checks done before any GPU time (CPU, 2026-11-03, `pg_check.sh`)

**Warm-up only, 64 batches of 32.**
- The policy (`mortal`, `current_dqn`) equals the base bit for bit.
- Only the value head moved.

**64 warm-up then 136 PG batches**, first version, without the learning-rate ramp:

| Diagnostic | Value |
| --- | --- |
| Mean ratio | 0.9994 |
| KL to the base | 0.011 nats |
| KL from behavior | 0.0027 |
| Entropy | 0.091 |
| Clipped high / low | 0.16% / 0.46% |
| Greedy agreement with the base | 98.35% |

All diagnostics were finite. The state loads in `run_eval.load_champion` and answers a `react_batch`.

**Finding: a fresh Adam moves fast.** 17 optimizer steps already flipped about 1.8% of greedy decisions, as many
as the Q line flips in 1,000. Hence the learning-rate ramp and the adaptive KL.

**Re-check with the learning-rate ramp (`lr_warmup` 64 in the check) and adaptive KL (target 0.02).**
- Warm-up: the policy learning rate stays 0 and the policy stays bit-identical.
- After 136 PG batches: KL to the base 0.0055 (half the first version), greedy agreement 98.8%, mean ratio 0.9991.
- KL was below target, so `kl_coef` adapted 1.0 → 0.986 and was saved in the state's `pg` entry.

**Deployed 11-03 12:53.** `pg_chain.sh` is waiting behind the anchor-schedule and X1 chains.
- `xfer_helpers.sh.pg` sits beside the live copy; the two differ only in the guard's trainer line.
- The `run_selfplay_m1.sh` sed was dry-run on a copy: one line changed, syntax OK.

## Tool changes

- **`train_pg.py`** is new. The production `train.py` is untouched.
- **`xfer_helpers.sh` guard and `run_selfplay_m1.sh`** launch `${TRAINER:-train.py}`. Without this, a guard restart of
  a PG arm would silently continue its state with Q learning.
  - The chain installs both by an atomic `mv` while nothing runs.
  - The default stays `train.py`.
- **`pg_free_watch.sh`** (11-04) covers lending after the chain. `gpu_share.sh` lends the card only at a boundary, and
  nothing is queued after this chain.
  - When the chain ends, the watcher writes FREE (idle) to `GPU_STATUS_mortal.txt`.
  - It runs `gpu_share.sh` once a booking is active, until 11-04 23:00.
