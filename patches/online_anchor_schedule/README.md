# Anchor schedule: does a λ 0.5 phase before λ 0.2 give the online stage the historical C' jump?

Pre-registered 2026-11-03, before the run. 5060 Ti. Chosen by the user after the reward test was dropped.

## Where the replay stands

- The reward is ruled out ([online_reward_schedule](../online_reward_schedule/README.md)): stage 2 replayed with
  the old reward [6, 4, 2, 0] gave R160k − A160k = −0.40 ± 0.91 on 6,000 groups, dropped by the SPRT.
- Ten re-anchored main-line stages (ml1–ml10) averaged about −1.91 against v4, with no trend.
- The last recipe difference to the historical line that can be tested on its own is the anchor schedule.

## The historical schedule

C' started from rl1_best and was anchored to it (RESULTS.md, 12k, paired):

| Phase | Hours | 12k vs v4 | Gain |
| --- | ---: | ---: | --- |
| λ 0.5 (C392k) | about 36 | −2.31 | vs rl1_best +0.19 ± 1.13 (z 0.33) |
| then λ 0.2 (C' final) | 9 | −0.75 | vs C392k +1.56 ± 1.29 (z 2.37) |

- At λ 0.5 the policy barely moved: riichi choice 44.32% → 44.36%, bucket agreement with rl1_best within 0.1 pp.
- The jump came after relaxing to 0.2. The replay goes straight to λ 0.2 and gained +0.28 ± 0.52 in the same stage
  (A160k − A80k).
- **The question:** does a phase at λ 0.5, which leaves the policy where it is, prepare the network so that relaxing
  to 0.2 gains more than starting at 0.2?

## Design

- **Arm W** (`runs/xfer_v11w`) starts from A80k, anchored to A80k, like the control A160k (`runs/xfer_v11s2`).
  - **Phase 1:** 80,000 batches at λ 0.5 (80k → 160k), about 9 hours. The chain checks that the generated config
    differs from v11s2's only in the weight line, apart from paths and port.
  - **Phase 2:** the weight line set back to v11s2's λ 0.2, so the config equals the control's. State = the 160k
    archive, with its Adam state, as when the historical trainer re-read the changed config. 40,000 batches
    (160k → 200k), about 4.5 hours.
  - **Unchanged:** the eval-proportional reward, A80k as the 'base' opponents and v4 in the pool, default sample reuse.
- **Phase lengths.** The historical phases were about 36 and 9 hours on the server. The λ 0.5 phase is scaled down
  to the control's 80k batches; the λ 0.2 phase keeps about the historical 40–47k batches (9 hours at 0.69–0.80 s
  per batch).
- **W200k has 40k more batches than A160k.** Neither history (C392k ≈ rl1_best) nor the replay (ten flat main-line
  stages) shows a gain from training time alone, so the extra batches are not counted as an advantage.
- **Readouts.**
  - Trajectory, deciding nothing: 12k of W160k and W200k.
  - Screening: W200k − A160k by the SPRT (0 vs +1, α 0.025, β 0.2, looks per 1,000 groups from seed 10000), capped
    at the 16,000 groups A160k has.
  - Also reported: W160k − A80k (the λ 0.5 phase; historically +0.19) and W200k − W160k (the relaxation;
    historically +1.56), W200k − tpt 408k, W200k vs v4, the mechanism readouts.

## Verdicts

- **Promote.** The schedule is a lever. The warmed line continues by itself through `mainline_next.sh` from W200k,
  in re-anchored 80k stages at λ 0.2, each screened against 408k.
- **Drop.** No single recipe difference explains the historical jump. Stop rebuilding the historical path and turn
  to a structurally different improvement operator: (a) separate policy and critic, being prepared in parallel.
- **Undecided at the cap.** Read the fixed 16,000 groups: ≥ +0.5 with z ≥ 2 counts as a promote, ≤ 0 as a drop.
- **Prior.** About 25% for a promote (self-review of 11-02).
- **Cost.** About 13.5 hours of training and 1–4 hours of evaluation.

## Result (2026-11-04): dropped

Both phases ran without a trainer restart:
- **Phase 1** (λ 0.5): 11-03 10:20–19:27, sample reuse 1.76.
- **Phase 2** (λ 0.2): 19:27–00:44, reuse 1.80.

The SPRT stopped for H0 at 3,000 groups, at 02:21:

| Look | Groups | Mean | LLR (bounds −1.58, +3.47) |
| --- | ---: | ---: | ---: |
| 1 | 1,000 | −0.95 ± 2.20 | −1.15 |
| 2 | 2,000 | −0.46 ± 1.53 | −1.58 |
| 3 (stop) | 3,000 | −0.25 ± 1.26 | −1.81 |

The pairings below use the 5,000 groups W200k played (seeds 10000–12999 and 14000–15999). The 12k rows use seeds
10000–10999 and 14000–15999.

| Comparison | Groups | Difference | History (C') |
| --- | ---: | ---: | ---: |
| **W200k − A160k** | 5,000 | **−0.46 ± 0.99** | |
| W200k − A80k | 5,000 | −0.10 ± 0.94 | |
| A160k − A80k (control, same seeds) | 5,000 | +0.36 ± 0.94 | |
| A160k − A80k (control, all its seeds) | 16,000 | +0.28 ± 0.52 | |
| λ 0.5 phase: W160k − A80k (12k) | 3,000 | −0.35 ± 1.06 | +0.19 ± 1.13 |
| relaxation: W200k − W160k (12k) | 3,000 | +0.20 ± 1.24 | +1.56 ± 1.29 |
| W200k − tpt 408k | 5,000 | −1.10 ± 1.33 | |
| W200k vs v4 | 5,000 | −1.82 ± 1.02 | |
| W160k vs v4 (12k) | 3,000 | −1.76 ± 1.33 | |

**The historical sequence did not replay.**
- The λ 0.5 phase did not move the strength, as in history.
- The relaxation, worth +1.56 in C', gave +0.20 here.

**Mechanism.** These are the same 12,000 games per model (the 12k seeds), measured against the v4 seats in the same
games.

| | call rate (v4) | win, pp | deal-in, pp | riichi, pp | net / kyoku (attack / defence) | bust us / v4 | 4th, pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A80k (start) | 31.0 (30.6) | −0.47 | +0.23 | +0.16 | −23 (−6 / −17) | 6.83 / 6.46% | +1.11 |
| W160k (after λ 0.5) | 30.6 (30.5) | −0.58 | +0.15 | +0.17 | −27 (−14 / −13) | 6.95 / 6.69% | +1.41 |
| W200k (after relaxing to λ 0.2) | 30.5 (30.6) | −0.48 | +0.20 | +0.37 | −24 (−8 / −16) | 6.95 / 6.46% | +1.16 |
| A160k (control, λ 0.2 throughout) | 30.4 (30.6) | −0.52 | +0.05 | +0.08 | −13 (−8 / −5) | 6.98 / 6.63% | +0.39 |

On the 20,000 games of the SPRT seeds the picture is the same:

| | deal-in, pp | defence / kyoku | 4th, pp |
| --- | ---: | ---: | ---: |
| W200k | +0.23 | −16 | +1.24 |
| A160k | +0.09 | −7 | +0.75 |

**Reading the mechanism**
- **λ 0.5 phase:** it traded attack for defence (−6 / −17 → −14 / −13).
- **Relaxation:** it won the attack back and pushed riichi to +0.37 pp, the most of any arm. Defence slipped back to
  −16.
- **Net:** W200k ends with the start's profile (−24 against −23 points per kyoku), while the control learned defence.

**The control may be a lucky draw**

*Update 2026-11-06: it was not. A second run of the control ([online_control_rerun](../online_control_rerun/README.md))
gained +0.38 ± 0.53 over A80k against the first run's +0.28, and the two runs differ by +0.10 ± 0.55. The control's
"defence" below was a 12k reading: on 64,000 games A160k reads deal-in +0.20 and defence −13, like its start.*

There is one training seed per arm. Against A80k, on each variant's own groups, the four stage-2 arms from A80k
stand as follows:

| Arm | vs A80k |
| --- | ---: |
| Fresh samples, F160k | +0.05 ± 0.84 |
| Old reward, R160k | +0.21 ± 0.87 |
| Warmed anchor, W200k | −0.10 ± 0.94 |
| Control, A160k | +0.28 ± 0.52 |

- The control is the only one of the four that learned defence. Part of its lead over every variant may be its
  training seed.
- That rescues no variant: none comes near the +1 the screen looks for.
- A second seed of the control would settle how much of A160k's +0.28 is the recipe. That costs about 11 hours.

**Verdict, by the rule set in advance:** no single recipe difference explains the historical jump.
- The rebuild of the historical path stops here.
- The structurally different operator (a), separate policy and critic
  ([policy_critic](../policy_critic/README.md)), started at 02:49 on 11-04.

## Tool changes

- `xfer_setup.sh` takes `ANCHOR` (the `[anchor]` weight), and its manifest prints it.
