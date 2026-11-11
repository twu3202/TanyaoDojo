# Control rerun: how much of the stage-2 control's gain is the recipe?

Pre-registered 2026-11-05, before the run. 5060 Ti. Chosen by the user after the policy-gradient probe was dropped,
to run while (b) is designed on the CPU.

## Why

Four stage-2 variants from A80k were each screened against one control run, A160k (`runs/xfer_v11s2`), and all four
were dropped ([online_anchor_schedule](../online_anchor_schedule/README.md#the-control-may-be-a-lucky-draw)):

| Arm | vs A80k |
| --- | ---: |
| Fresh samples, F160k | +0.05 ± 0.84 |
| Old reward, R160k | +0.21 ± 0.87 |
| Warmed anchor, W200k | −0.10 ± 0.94 |
| Policy gradient, P160k | −1.08 ± 1.24 |
| **Control, A160k** | **+0.28 ± 0.52** |

- The control is the only arm that learned defence.
- Each arm is a single training run, so training variance is in none of these CIs. Part of the control's lead may be
  its draw.

## Design

- **Arm C** (`runs/xfer_v11c2`, port 5595) is A160k trained again.
  - Same start (the A80k archive, md5 e3dde87b), same code (train.py 67bce509 and the rest as in v11s2's manifest).
  - The chain checks that the generated config equals v11s2's, apart from paths and port.
  - 80,000 batches, 80k → 160k, as v11s2.
  - Nothing seeds the workers or the trainer, so the two runs differ only by their draw.
- **Evaluation:** the fixed 16,000 groups A160k has (seeds 10000–25999), 12k first. There is no sequential stop:
  this is an estimate, not a screen.
- **Cost:** about 9 hours of training and 3 hours of evaluation.

## Readouts

- **Primary:** C160k − A80k on the 16,000 groups, beside A160k − A80k (+0.28 ± 0.52) on the same groups.
- **Spread between the two runs:** C160k − A160k. Both evaluation CIs are in its CI; the training spread is what is
  left over.
- **Pooled control:** the mean of the two runs per group, minus A80k.
- **The four dropped variants**, re-read against C160k and against the pooled control on each variant's own groups.
- **Mechanism readouts:** 12k for A80k, A160k and C160k; the 16,000 groups for A160k and C160k. The question is
  whether C160k learns defence too (deal-in and defence per kyoku against A80k's).

## Reading

- **The recipe.** C160k − A80k ≥ +0.2 and C160k learns defence: the stage-2 gain is the recipe, and the variants'
  losses against the control stand as measured.
- **The draw.** C160k − A80k ≤ 0: most of A160k's +0.28 was its draw.
  - The variants are re-read against the pooled control.
  - "Only the control learned defence" is revised.
- **In between:** unresolved at this size; the pooled estimate is reported.
- **Not reopened:** no dropped variant is promoted by the re-read; the screen asked for +1. A variant ≥ +0.5 above
  the pooled control with z ≥ 2 is flagged for a full retest.
- **After the chain:** nothing is queued. The card is marked FREE, and (b) gets it when its design is approved.

## Files

`ctl_chain.sh` runs the whole test (log `runs/xfer_tools/ctl_chain.log`, markers `CTL_CHAIN_DONE` /
`CTL_CHAIN_STOPPED`).
- It rewrites `GPU_STATUS_mortal.txt` at each phase.
- After the evaluation, it lends the idle card to bookings for a day, as `pg_free_watch.sh` did.

## Result (2026-11-06): the control's gain is the recipe; its "defence" was a 12k reading

The arm trained 80,000 batches without a restart (11-05 14:13–23:09), on 22.4M fresh trainee samples (reuse 1.83).
The evaluation of the 16,000 groups ended at 02:26 on 11-06.

| Comparison | 12k (3,000 groups) | 16,000 groups | z (16k) |
| --- | ---: | ---: | ---: |
| **C160k − A80k** | −0.14 ± 1.20 | **+0.38 ± 0.53** | +1.40 |
| A160k − A80k (the first run) | | +0.28 ± 0.52 | +1.05 |
| **C160k − A160k** (the two runs) | −0.82 ± 1.27 | **+0.10 ± 0.55** | +0.35 |
| pooled control − A80k | +0.27 ± 1.03 | +0.33 ± 0.44 | +1.45 |
| C160k − tpt 408k | | −1.15 ± 0.74 | −3.02 |
| C160k vs v4 | −1.55 ± 1.32 | −1.67 ± 0.57 | −5.80 |

**Training spread.** The two runs differ by +0.10 ± 0.55. This is one pair at one stage, so it bounds the spread only
loosely, but it shows no sign of a training spread large against the evaluation CIs.

**The dropped variants against the pooled control**, on each variant's own groups:

| Variant | Groups | − C160k | − pooled control | z (pooled) |
| --- | ---: | ---: | ---: | ---: |
| Fresh samples, F160k | 6,000 | −0.08 ± 0.92 | −0.32 ± 0.78 | −0.80 |
| Old reward, R160k | 6,000 | +0.08 ± 0.92 | −0.16 ± 0.80 | −0.39 |
| Warmed anchor, W200k | 5,000 | −0.24 ± 1.01 | −0.35 ± 0.87 | −0.79 |
| Policy gradient, P160k | 4,000 | −1.09 ± 1.26 | −1.35 ± 1.13 | −2.34 |

- F, R and W cannot be told from the control (all within 0.4 of it). None is ≥ +0.5, so none is flagged for a retest.
- P is worse than the pooled control (z −2.3).

**Mechanism.** These are against the v4 seats in the same games.

| | games | call rate (v4) | win, pp | deal-in, pp | riichi, pp | net / kyoku (attack / defence) | bust us / v4 | 4th, pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A80k (start), 12k | 12,000 | 31.0 (30.6) | −0.47 | +0.23 | +0.16 | −23 (−6 / −17) | 6.83 / 6.46% | +1.11 |
| A160k, 12k | 12,000 | 30.4 (30.6) | −0.52 | +0.05 | +0.08 | −13 (−8 / −5) | 6.98 / 6.63% | +0.39 |
| C160k, 12k | 12,000 | 30.5 (30.6) | −0.55 | +0.24 | −0.16 | −35 (−17 / −18) | 6.88 / 6.82% | +1.26 |
| A160k, 16,000 groups | 64,000 | 30.3 (30.5) | −0.55 | +0.20 | +0.09 | −27 (−14 / −13) | 7.04 / 6.72% | +1.20 |
| C160k, 16,000 groups | 64,000 | 30.4 (30.5) | −0.53 | +0.33 | −0.19 | −35 (−14 / −21) | 6.98 / 6.73% | +1.15 |

- **"Only the control learned defence" is withdrawn.** It rested on A160k's 12k (deal-in +0.05, defence −5). On
  64,000 games A160k reads +0.20 and −13, close to the start's 12k (+0.23, −17). C160k shows no defence gain either.
- The previous four READMEs used that reading to describe the variants ("the control was the only arm to learn
  defence"). It was a 12k effect.

**Reading, by the rules set in advance**
- C160k − A80k is ≥ +0.2, but C160k does not learn defence. So the first branch is met for strength only.
- The draw branch (≤ 0) is clearly not met.
- **What it settles:**
  - The stage-2 control's gain is reproducible: +0.28 and +0.38 from two runs; pooled +0.33 ± 0.44, not
    significant alone.
  - The single recipe differences (fresh samples, old reward, warmed anchor) changed nothing measurable.
  - A training draw alone is an unlikely explanation for a +1.5 jump like C′'s, though one pair cannot rule it out.
- **12k misread twice here:** C160k − A80k read −0.14 at 12k and +0.38 on 16,000 groups, and the defence story came
  from 12k.

**After the chain:** the card is FREE.
- The eval-only re-check of the C′ jump (P0 in `research/2026-11-05_progress_review`) is the next GPU job.
- The chain lends the idle card to bookings until 11-07 02:27.
