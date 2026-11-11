# Reward schedule: does the old training reward give the online stage the historical C' jump?

Pre-registered 2026-11-02, before the run. 5060 Ti. Chosen by the user after the default main-line continuation
levelled off.

## Where the replay stands

The v11 line replayed on the 5060 Ti is level after 12 stages:

- Stages 1–3 gained +0.68, +0.28 and about 0.
- Nine more re-anchored stages (`mainline_next.sh`, ml1–ml9, 240k → 960k) swung between −1.35 and −2.94 against v4,
  averaging about −1.95, with no trend. Each was dropped by the SPRT against tpt 408k.
- Ideas ruled out:
  - sample freshness ([online_freshness](../online_freshness/README.md));
  - a v4-sized network ([online_capacity](../online_capacity/README.md));
  - a snapshot ensemble of the 408k family ([snapshot_ensemble](../snapshot_ensemble/README.md)).

## What still differs from the historical line

The historical C' stage started from rl1_best, whose strength was similar to A80k (−1.96 against −1.82 at 100k). It
gained about +1.5 to +1.75 on the 12k, where the replay's stage 2 gained +0.28. Two differences in recipe remain
untested:

1. **The training reward.** C' and the λ 0.1 stretch after it trained on the old placement points [6, 4, 2, 0]. The
   eval-proportional [2.390, 1.195, 0, −3.585] came only at 368k → 408k, where it added +0.25 ± 0.48. The replay has
   used the eval-proportional reward from its first step.
   - Mortal's reward is a change in expected placement points between kyoku, so only the shape of `pts` matters.
   - Both vectors have the same standard deviation.
   - The old one spaces the four places evenly. The new one puts most of the weight on avoiding 4th.
2. **The anchor schedule.** C' ran about 40 hours at λ 0.5, during which the policy barely moved, before 9 hours at
   λ 0.2 made the jump. The replay goes straight to λ 0.2.

This test takes the first.

## Design

- **Arm R** (`runs/xfer_v11r`) replays the stage-2 control (A80k → A160k, `runs/xfer_v11s2`) with one change:
  `pts = [6, 4, 2, 0]`.
  - The chain checks that the generated config differs from v11s2's only in that line, apart from paths and port.
  - Everything else is the same: the start A80k, re-anchored to A80k at λ 0.2; A80k as the 'base' opponents and v4
    in the pool; the default sample reuse; 80,000 batches to 160k.
  - `test_play` and the collapse guard read points in the evaluation's [90, 45, 0, −135] whatever the training
    reward is, so the guard is unchanged.
- **Readouts.**
  - Trajectory, deciding nothing: 12k of R120k and R160k.
  - Screening: R160k − A160k by the SPRT (0 vs +1, α 0.025, β 0.2, looks per 1,000 groups from seed 10000), capped
    at the 16,000 groups A160k has.
  - Also reported: R160k − A80k against the control's +0.28 ± 0.52; R160k − tpt 408k; R160k vs v4; the mechanism
    readouts.

## Verdicts

- **Promote.** The reward schedule is a lever: the historical line climbed on the old reward. The old-reward line
  then continues by itself through `mainline_next.sh`, in re-anchored 80k stages with the old reward, each screened
  against 408k. Replaying the switch to the eval-proportional reward at the end is the step after that.
- **Drop.** The reward is not why the replay's stages gain less. That leaves the λ 0.5 phase as the last
  single-variable difference. The chain ends.
- **Undecided at the cap.** Read the fixed 16,000 groups: ≥ +0.5 with z ≥ 2 counts as a promote, ≤ 0 as a drop.
- **Cost.** About 9 hours of training and 1–4 hours of evaluation.

## Result (2026-11-03): dropped

The SPRT stopped for H0 at 4,000 groups (mean −0.26 ± 1.11, LLR −2.36 below −1.58), with no gain at any look
(+0.61 at 2,000, +0.35 at 3,000). Pairings on the 6,000 groups played (seeds 10000–15999):

| Comparison | Groups | Difference |
| --- | ---: | ---: |
| **R160k − A160k** | 6,000 | **−0.40 ± 0.91** |
| R160k − A80k | 6,000 | +0.21 ± 0.87 |
| A160k − A80k (control, same seeds) | 16,000 | +0.28 ± 0.52 |
| R160k − tpt 408k | 6,000 | −0.75 ± 1.24 |
| R160k vs v4 | 6,000 | −1.63 ± 0.94 |
| R120k vs v4 (12k) | 3,000 | −1.86 ± 1.34 |

Sample reuse was 1.77, the same as the control's.

**Mechanism.** These are the same 24,000 games per model, measured against the v4 seats in the same games.

| | call rate (v4) | win, pp | deal-in, pp | riichi, pp | net / kyoku (attack / defence) | bust us / v4 | 4th, pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A80k (start) | 30.9 (30.6) | −0.46 | +0.31 | +0.12 | −32 (−9 / −23) | 6.95 / 6.56% | +1.31 |
| A160k (eval-proportional reward) | 30.4 (30.6) | −0.51 | +0.09 | +0.14 | −15 (−9 / −7) | 6.92 / 6.67% | +0.63 |
| R160k (old reward) | 31.1 (30.6) | −0.30 | +0.41 | +0.17 | −31 (−2 / −29) | 6.93 / 6.67% | +1.21 |

- **The old reward bought attack and lost defence.** From A80k, R160k closed the attack gap (−9 → −2 points per
  kyoku), but deal-ins rose to +0.41 pp and the defence deficit widened to −29. The control learned defence instead
  (deal-in +0.09, defence −7) and halved the excess 4th places. That fits the shapes: the old reward spaces the
  places evenly, the eval-proportional one puts most of the weight on avoiding 4th.
- **So the reward is not why the historical line climbed.** It changes where the stage gains, not how much. The
  last single-variable difference, the λ 0.5 phase, is tested next ([online_anchor_schedule](../online_anchor_schedule/README.md)).

## Tool changes

- `xfer_setup.sh` takes `PTS`, and its manifest prints `pts`.
- `mainline.txt` takes an optional fourth field, the `pts` of the line.
- `rew_chain.sh` skips the setup when the arm is already set up. Its first start (11-02 21:02) stopped falsely: the
  config check read the configs through the wrong argument, compared two empty files and stopped. The check was
  fixed, it finds only the pts line, and the chain restarted at 22:35 from the untouched base.
