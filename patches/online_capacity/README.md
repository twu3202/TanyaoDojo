# Online capacity test: does a v4-sized network go further with the same online recipe?

Pre-registered 2026-10-25, before the run. 5060 Ti. Chosen by the user after the
[early-end gap study](../early_end_gap/README.md) closed without a fixable situation.

## Question

Capacity has been tested only offline. B1, the v11 recipe at v4's size (256 × 54, 23.8M parameters), landed at v11's
level: −0.18 ± 0.62 at 100k. The [online transfer test](../online_transfer/README.md) then showed two things:

- one online stage of 80k batches moves v11 by +0.68;
- it erases a better offline base's +0.55.

So the online stage, not the base, sets where a lineage ends. Our 192 × 40 lineage stops 0.6–0.7 below v4. v4 itself
is 256 × 54, and the same recipe still lifts it (+0.12, A²). Capacity is the one large lever not yet tested where it
would matter, in the online stage.

## Design

Arm C replays the online transfer test's recipe from B1 and is compared with arm A, the v11 arm already played.

| | arm A (done 10-20) | arm C |
| --- | --- | --- |
| start | v11, 192 × 40 | B1 final (`runs/b1_256x54/b1_final.pth`, md5 a7d220e8, 1,222,800 steps), 256 × 54 |
| offline vs v4 at 100k | −2.50 ± 0.46 | −2.68 ± 0.46 |
| anchor | own start, frozen, λ 0.2 | own start, frozen, λ 0.2 |
| opponents (4 GPU workers) | base, v4, base, base | base, v4, base, base |

Everything else is identical: the same config template, `train.py` 67bce509 and helpers, 80,000 batches, snapshots
every 8,000, MINBUF 4,000. `xfer_setup.sh` now writes the base's own `[resnet]` size into the arm's config; that size
also builds the trainee, its anchor and the workers' copy. For v11 the config is unchanged.

## Readouts and rules

1. **Trajectory, deciding nothing:** 12k (s10000 + s14000) of the 40k and 80k snapshots.
2. **Screening: C80k − A80k** by the screening rule adopted 10-21
   ([sequential_eval](../sequential_eval/README.md)):
   - SPRT 0 vs +1.0, α 0.025, β 0.2;
   - looks every 1,000 groups in ascending seed order from 10000, cap 25,000 groups.

   The verdicts:
   - **Drop:** capacity does not help the online stage enough to matter. The line stops here.
   - **Promote:** the rest of the 100k is played anyway as the confirmation. If the 100k gives C80k − A80k ≥ +0.5
     with z ≥ 2, capacity is the lever, and arm C becomes the candidate for the later online stages.
   - **Undecided at the cap:** read the fixed 100k as usual: ≥ +0.5 and z ≥ 2 passes, ≤ 0 is excluded, in between is
     unresolved.
3. **Also reported** on everything played:
   - C80k − B1 offline (the online gain);
   - the interaction with the offline difference;
   - C80k − tpt 408k;
   - the mechanism readouts (diag_gap / place / bust).
   - Sample reuse (`xfer_reuse.py`). With a 2.4× larger network, both the trainer and the workers are slower, and
     the ratio decides reuse.

## Known differences and risks

- **Memory.** The card has 8 GB. The 192 × 40 arms used 5.0 GB. The trainer, the anchor and the workers' trainee copy
  all grow 2.4× in compute. If the first trainer start runs out of memory, the fallback is 3 workers, reported as such.
- **Time.** About 2.4× the arm A training time, so roughly 22 hours for 80k batches. Then the 12k, then up to 100k for
  screening, at about 4.8 hours per 100k.
- **Device.** B1's offline dumps come from the trx evaluation station: GPU fp32 on another card. They are used only
  for the secondary gain row. The screening pair (C80k, A80k) is all on the 5060 Ti.
- **Training variance.** One training seed per arm, as in the transfer test.

## Files

- `cap_chain.sh` queues the whole test (log `runs/xfer_tools/cap_chain.log`).
- It reuses `patches/online_transfer/` (`xfer_setup.sh`, `xfer_arm.sh`, `xfer_eval.sh` with its new `seg` mode,
  `xfer_pair.py`) and `jax_rl/mjai_bot/seqtest.py`.

## Stage 2 (pre-registered 2026-10-25, queued behind stage 1 at the user's request)

The GPU would otherwise sit idle after stage 1's screening, so stage 2 is queued now. It runs whatever stage 1's
verdict is.

- **What it does.** `cap2_chain.sh` continues arm C from its 80k snapshot to 160k batches in `runs/xfer_b1s2`. The
  snapshot is a full online state, so Adam, the step count and best_perf all carry on.
- **One recipe change: the 80k snapshot becomes the base.** The arm is re-anchored to its own 80k snapshot (λ 0.2),
  and the same snapshot replaces the offline base as the 'base' opponents; v4 stays in the pool. Anchoring to the
  current best is what carried the historical v11 line's later gains, from C' (07-23) to 408k: about +1.2 at 100k
  beyond its first stage.
- **Readouts.**
  - 12k of the 120k and 160k snapshots (trajectory).
  - Screening C160k − tpt 408k by the same SPRT (0 vs +1, α 0.025, β 0.2, looks per 1,000 groups, cap 100k). A promote
    is confirmed by the full 100k.
  - Paired readouts on everything played: C160k − 408k, C160k − C80k (the stage gain), C160k − A80k, and C160k vs v4.
- **Reading.**
  - C160k − 408k ≥ +0.5 with z ≥ 2 at 100k: the big line is our new best own model. Its 100k against v4 is then the
    record check for the project target.
  - A drop means it is not decisively better than 408k at 160k. The stage gain then says whether it is still
    climbing, which decides whether a stage 3 is worth its GPU-day.

## Stage 1 result (2026-10-26): dropped; the larger network lost ground in its online stage

Arm C trained 11:48 on 10-25 to 04:52 on 10-26, about 1.24 batches/s, using 7.2 of the card's 8 GB with all 4
workers. There was no restart and no collapse-guard stop. Sample reuse was 1.84, the same as the 192 × 40 arms
(1.82).

The SPRT dropped C80k − A80k at 3,000 groups: −1.48 ± 1.76, with LLR −2.45 below the −1.58 bound.

| 12k (s10000 + s14000), pt vs v4 | offline start | 40k | 80k |
| --- | ---: | ---: | ---: |
| arm A, v11 (192 × 40) | −3.79 | −2.58 | −1.41 |
| arm C, B1 (256 × 54) | −3.32 | −2.70 | **−3.68** |
| C − A, paired | +0.47 ± 1.82 | −0.12 ± 1.76 | **−2.27 ± 1.75 (z −2.55)** |

| mechanism, 12k vs v4 | deal-in | riichi | net / kyoku (attack / defence) | bust us / v4 | 4th, pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| A 40k → 80k | +0.48 → +0.23 pp | +0.36 → +0.16 pp | −56 → −23 (−6 / −17) | 7.24 → 6.83 / 6.5% | +1.90 → +1.11 |
| C 40k → 80k | +0.57 → +0.86 pp | +0.58 → +0.81 pp | −64 → −77 (−22 / −55) | 7.27 → 7.59 / 6.5% | +2.04 → +2.61 |

- **The two arms move in opposite directions between 40k and 80k.** v11 becomes more defensive: fewer deal-ins and
  riichi, and a smaller bust excess. B1 riichis and deals in more, and its defence loses 18 points per kyoku. At 80k
  it is below its own offline start on the 12k (−3.68 against −3.32).
- **Verdict by the pre-set rule: dropped.** With this recipe, a v4-sized network does not get further in the online
  stage; on this evidence it gets less far.
- **The recipe, not capacity itself, may be at fault.** The recipe was tuned for 192 × 40: constant lr 1e-5, anchor
  λ 0.2, a fresh Adam. One arm and one training seed cannot separate the two, so the result is a verdict on this
  recipe at this size.
- Stage 2 started from C80k at 06:56 on 10-26, as queued.

## Stage 2 control (pre-registered 2026-10-26, queued behind stage 2 at the user's request)

`cap3_chain.sh` continues arm A (v11, 192 × 40) from its 80k snapshot to 160k in `runs/xfer_v11s2`. It is set up
exactly as arm C's stage 2: re-anchored to its own 80k snapshot (λ 0.2), with that snapshot also as the 'base'
opponents and v4 in the pool. It answers two questions.

1. **Capacity at equal steps.** C160k − A160k on every seed both have played.
2. **Does the re-anchored stage reproduce the historical later-stage gain on this machine?** The v11 line went from
   −1.96 (rl1_best) to −0.64 (408k) at 100k after its first stage.
   - Screening: A160k − A80k by the same SPRT (0 vs +1, α 0.025, β 0.2, looks per 1,000 groups). A80k already has the
     full 100k.
   - A promote is confirmed by the full 100k. If that shows a gain of ≥ +0.5 with z ≥ 2, the stage is a reproducible
     baseline for testing recipe changes on this machine.
   - A160k − tpt 408k is reported alongside.

## Stage 2 result (2026-10-27): dropped at the first look; capacity at equal steps stays about −2.3

Arm C's stage 2 trained from 06:56 to 23:19 on 10-26. Reuse was 1.90, with no restart and no guard stop. The SPRT of
C160k − tpt 408k stopped for H0 at the first look: −3.61 ± 3.14 on 1,000 groups, LLR −1.60.

The control (arm A's stage 2) trained from 00:53 to 09:37 on 10-27 and is being screened now. On the 12k
(s10000 + s14000, 3,000 groups):

| pt vs v4, or paired difference | 80k | 120k | 160k |
| --- | ---: | ---: | ---: |
| arm C, B1 256 × 54 | −3.68 | −2.61 | −3.05 |
| arm A, v11 192 × 40 | −1.41 | −2.36 | −0.73 |
| **C − A at equal steps** | **−2.27 ± 1.75** | | **−2.32 ± 1.73 (z −2.63)** |
| stage gain C160k − C80k | | | +0.63 ± 1.28 |
| stage gain A160k − A80k | | | +0.68 ± 1.21 |
| C160k − tpt 408k | | | −2.85 ± 1.79 |
| A160k − tpt 408k | | | −0.53 ± 1.72 |

- **Both arms gain about the same in stage 2.** Re-anchoring to the 80k snapshot moves each by about +0.65 on the 12k.
  Neither gain is significant.
- **The size gap does not close.** C − A is −2.27 at 80k and −2.32 at 160k. With this recipe, the v4-sized network
  stays about 2.3 points behind at equal steps, on the same walls, twice in a row.
- **The capacity line closes here for this recipe.** Whether a recipe retuned for 256 × 54 does better, for example
  with a smaller lr, is a separate question.

## v11 line, stage 3 (pre-registered 2026-10-27, queued behind the control at the user's request)

`cap4_chain.sh` continues arm A from 160k to 240k in `runs/xfer_v11s3`. It re-anchors again, to A160k, which also
becomes the 'base' opponents; v4 stays in the pool. This is the main line now: on the 12k, A160k is already close to
408k (−0.53 ± 1.72).

- **Readouts.** 12k of the 200k and 240k snapshots, then screening of A240k − tpt 408k by the same SPRT (0 vs +1,
  α 0.025, β 0.2, looks per 1,000 groups, cap 100k). A promote is confirmed by the full 100k.
- **Reading.** A240k − 408k ≥ +0.5 with z ≥ 2 at 100k makes A240k the best own model, reached with a recipe
  reproduced on this machine; its 100k against v4 is then the record check. A drop leaves 408k as the best; A240k −
  A160k then says whether repeated re-anchoring still climbs.

## Control and stage 3 results (2026-10-27): re-anchoring adds little on this machine; the line levels off below 408k

**Stage 2 control (A160k − A80k).** The SPRT dropped at 16,000 groups: +0.28 ± 0.52, LLR −3.13. The 12k reading of
+0.68 had again run high. Over those 16,000 groups, A160k is −1.77 ± 0.57 against v4 and −1.25 ± 0.74 (z −3.29)
against 408k.

**Stage 3 (A240k, re-anchored to A160k).** It trained from 13:34 to 22:26 on 10-27, with reuse 1.82. The SPRT of
A240k − 408k dropped at 2,000 groups (−2.26 ± 2.14). On 4,000 groups:

| v11 line on the 5060 Ti | vs v4 | gain over the previous stage |
| --- | ---: | ---: |
| A80k (stage 1, own-base anchor), 100k | −1.82 ± 0.46 | +0.68 ± 0.45 over v11 |
| A160k (re-anchored to A80k), 16k groups | −1.77 ± 0.57 | +0.28 ± 0.52 |
| A240k (re-anchored to A160k), 4k groups | −1.69 ± 1.14 | −0.33 ± 1.04 |
| tpt 408k, for reference | −0.64 ± 0.46 | |

- **The stage gains shrink: +0.68, +0.28, about 0.** The line levels off around −1.7 to −1.8 against v4, about 1.1
  below 408k.
- **The historical v11 line went further, but its recipe was not the same.** It ran on trx with 7–12 workers, so each
  sample was reused about 1.3 times against 1.8 here. Over its 408k steps it also trained on about 2.4 times the fresh
  games. Its path also changed the anchor λ (0.5 → 0.2 → 0.1) and switched the reward definition. Re-anchoring alone,
  at this machine's sample reuse, does not reproduce it.
- **Closed:** the capacity question for this recipe (−2.3 at equal steps), and the re-anchored replay of the v11 line
  on this machine.

## The default continuation, stages 4–12 (2026-10-29 to 11-02): no trend

From 10-29 13:19, `patches/online_transfer/mainline_next.sh` continued the line on its own in re-anchored 80k-batch
stages: each re-anchored to the previous snapshot, with default reuse (about 1.8) and the eval-proportional reward.
Every stage was screened against tpt 408k by the SPRT, and every one was dropped within 2,000 groups. The readings
are on 3,000–6,000 groups (12k + SPRT segments), so each is ±1.1–1.5.

| Snapshot | vs v4 | − previous snapshot | − tpt 408k |
| --- | ---: | ---: | ---: |
| A240k (stage 3) | −1.69 | | |
| ml1 320k | −1.88 | −0.19 | −1.52 |
| ml2 400k | −1.43 | +0.80 | −0.56 |
| ml3 480k | −1.68 | −0.60 | −1.32 |
| ml4 560k | −1.35 | +0.33 | −0.99 |
| ml5 640k | −2.94 | −1.59 (z −3.12) | −2.58 |
| ml6 720k | −1.94 | +1.00 | −1.58 |
| ml7 800k | −2.82 | −0.52 | −2.63 |
| ml8 880k | −1.45 | +1.43 (z +2.45) | −1.09 |
| ml9 960k | −2.07 | −0.62 | −1.71 |
| ml10 1,040k | −1.50 | +0.57 | −1.14 |

- **800k more batches did not move the line.** It swings between −1.35 and −2.94 and averages about −1.90, level
  with A80k's −1.82.
- **Stopped at the user's choice (11-02)** after ml10 (960k → 1,040k; dropped at 2,000 groups like the others).
  The next test is the training reward ([online_reward_schedule](../online_reward_schedule/README.md)).
