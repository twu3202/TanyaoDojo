# P2 short: does an unanchored first stage grow rl1_best's profile, and can anchored training then repair it?

Pre-registered 2026-11-06, before the run. 5060 Ti. Chosen by the user after the C′ re-check held.

## Why

The C′ re-check ([cprime_recheck](../cprime_recheck/README.md)) confirmed the historical jump on new walls: C′ final
− C392k = +0.98 ± 0.63 (z 3.1). Its mechanism readouts suggest why the replay line never reproduced it:

| | attack / defence per kyoku | riichi, pp | deal-in, pp | 4th, pp |
| --- | ---: | ---: | ---: | ---: |
| rl1_best (unanchored first stage) | +26 / −60 | +1.93 | +0.87 | +2.5 |
| C′ final (after relaxing to λ 0.2) | +12 / −30 | +0.58 | +0.31 | +1.4 |
| A80k (replay start, anchored from its first batch) | −8 / −21 | +0.10 | +0.33 | +1.4 |

- rl1_best and A80k are equally strong (−0.19 ± 0.74) but play differently.
- C′'s relaxation repaired rl1_best's leak and kept most of its attack.
- **Hypothesis:** the jump is the repair of an aggressive, leaky profile that only an unanchored first stage grows.
  The replay line was anchored from its first batch and never had it, so its relaxations had nothing to repair.

## Design

### Phase 1: arm u1 (`runs/xfer_u1`, port 5596)

- **Start:** v11 (md5 2ae5581c, the same file rl1 started from).
- **Recipe:** July's B1 recipe, which produced rl1_best. The config equals the replay control v11s2's except in exactly
  five lines; the chain checks this:

  | Line | Phase 1 | v11s2 |
  | --- | --- | --- |
  | Anchor | weight 0 (off) | 0.2 |
  | LR peak / final | 5e-6 | 1e-5 |
  | Boltzmann temperature | 0.05 | 0.1 |
  | Training reward | [6, 4, 2, 0] | evaluation-proportional |

- **Length:** 64,000 batches. rl1_best was taken at 60,000; the archive grid is 8,000.

### Gate (set in advance)

U64k is measured on the 16,000 groups. It must meet all three:

- attack ≥ +10 points per kyoku against the v4 seats (rl1_best +26, A80k −8);
- riichi ≥ +1.0 pp (rl1_best +1.93, A80k +0.10);
- U64k − A80k ≥ −0.5, i.e. not much weaker than the replay start.

Met: phase 2 starts by itself. Not met: the chain ends, and the hypothesis loses its first leg.

### Phase 2: arm u1a (`runs/xfer_u1a`, port 5597)

- **Start and anchor:** U64k, anchored to itself.
- **Recipe:** exactly the replay control's (config = v11s2's). Only the start differs from A160k.
- **Length:** 80,000 batches (64k → 144k), then the 16,000 groups.
- **The learning-rate trap.** `train.py` restores the scheduler's own peak/final from an online state, which would
  keep phase 1's 5e-6. So the chain sets them to 1e-5 in the copied state.pth and logs it.

### Readouts

| Phase | Reported |
| --- | --- |
| 1 | 12k of U32k (trajectory); U64k vs v4; U64k − A80k, − rl1_best, − C′ final, − tpt 408k; the mechanism of U64k beside rl1_best's and A80k's (100k) |
| 2 | U144k − U64k (the anchored stage) beside the pooled control's +0.33 over A80k; U144k − pooled control, − A160k, − C′ final, − tpt 408k; mechanism |

### Reading phase 2

| Result | Reading |
| --- | --- |
| U144k − U64k ≥ +0.7 with z ≥ 2 | The anchored stage repairs an aggressive start, as C′ did. A path to try toward the goal, for the user to choose |
| ≤ +0.33 (no better than the control's stage) | The start is not the missing ingredient |
| In between | Undecided at this size |

**Prior:** about 35% that phase 1 meets the gate, and about 40% that phase 2 then reads ≥ +0.7.

**Cost:** phase 1 about 7.5 h of training and 4 h of evaluation; phase 2 about 9 h and 3 h.

## Departures from the July run (process rule R8, told to the user before the run)

1. **Topology.** Four GPU workers share the 8 GB card. rl1's July run was on trx, and its worker count was not
   recorded.
   - The replay arms see about 1.8 games per batch, against about 2.6 for rl1 (155k games over 60k batches).
   - The July trainer was also slowed by a single-process loader (about 1.45 batches/s), which kept its data fresher.
   - The alternative was trx, idle but with vLLM-Omni priority. The user chose the 5060 Ti.
2. **Snapshot selection.** rl1_best was best.pth, the maximum of many noisy test_play readings. Here the snapshot is
   fixed in advance at 64k. This avoids the selection bias but may miss a transient peak: rl1's line fell about 2 pt
   by 310k games.
3. **Opponent pool.** July used base/v4/snap. Snap is the trainee's state at launch, which equals v11 if the run never
   restarted. Here the pool is base, v4, base, base.
4. **Code.** `train.py` 67bce509, the anchor patch with weight 0, behaves as July's warm-start-fixed trainer.

## Files

`u1_chain.sh` writes the log `runs/xfer_tools/u1_chain.log`, with the markers `U1_CHAIN_DONE` / `U1_CHAIN_STOPPED`.
- Results: `u1_phase1.txt`, `u1_mech1.txt`, `u1_phase2.txt`, `u1_mech2.txt`.
- It rewrites `GPU_STATUS_mortal.txt` at each phase.
- It lends the idle card for a day after it ends.

## Result, phase 1 (2026-11-06): the gate is not met

Phase 1 trained 13:04–19:42: 64,000 batches on 18.4 M fresh trainee samples, reused 1.79 times. Its evaluation ended
at 23:35, an hour ahead of plan.

| Pairing | Groups | Difference | z |
| --- | ---: | ---: | ---: |
| U64k vs v4 | 16,000 | −1.65 ± 0.58 | |
| U32k vs v4 (trajectory) | 3,000 (12k) | −1.84 ± 1.31 | |
| U64k − A80k (the replay start) | 16,000 | +0.40 ± 0.75 | +1.05 |
| U64k − rl1_best | 16,000 | +0.59 ± 0.70 | +1.63 |
| U64k − pooled control (A160k, C160k; 160k anchored batches from v11) | 16,000 | +0.07 ± 0.69 | +0.21 |
| U64k − C′ final | 16,000 | −0.73 ± 0.70 | −2.04 |
| U64k − tpt 408k | 16,000 | −1.12 ± 0.72 | −3.05 |

**Mechanism**, against the v4 seats in the same games. U64k and rl1_best are 64,000 games each, A80k 100,000.

| | call rate (v4) | win, pp | deal-in, pp | riichi, pp | net / kyoku (attack / defence) | bust us / v4 | 4th, pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **U64k** | 28.6 (30.5) | −0.24 | +0.43 | **+0.88** | −35 (**+0** / −35) | 7.36 / 6.67% | +1.63 |
| rl1_best | 29.1 (30.5) | −0.06 | +0.87 | +1.93 | −33 (+26 / −60) | 7.69 / 6.58% | +2.53 |
| A80k | 30.8 (30.6) | −0.44 | +0.33 | +0.10 | −29 (−8 / −21) | 7.02 / 6.63% | +1.42 |

**Gate**, by the rule set in advance:

| Criterion | Threshold | U64k | |
| --- | --- | ---: | --- |
| Attack against the v4 seats | ≥ +10 per kyoku | +0.4 | not met |
| Riichi rate | ≥ +1.0 pp | +0.88 | not met |
| U64k − A80k | ≥ −0.5 | +0.40 | met |

**Verdict: not met.** No phase 2 ran. The hypothesis loses its first leg: July's recipe in this pipeline, from the
same v11, did not grow rl1_best's profile in 64,000 batches. The prior was about 35%.

**Reading.**
- **The unanchored stage is not worse.** After 64k batches U64k is level with the anchored control after 160k
  (+0.07 ± 0.69). Neither it nor its strength gain over the two starts is significant.
- **It is not rl1_best-like.** Its profile sits between the two starts:
  - riichi rose to +0.88 pp (A80k +0.10, rl1_best +1.93);
  - it calls less than v4 (28.6% against 30.5%);
  - its attack is level with v4 (0, against rl1_best's +26);
  - its defence is −35, between A80k's −21 and rl1_best's −60.
- **Candidate reasons**, all from the R8 list above:
  1. **Snapshot selection.** rl1_best was best.pth, the best of many noisy test_play readings. In this run the same
     rule picked U40k: test_play +3.3 at 40k on 800 games, SE about 3 pt. U40k was never evaluated.
  2. **Topology and freshness.** 1.8 games per batch here against about 2.6 in July.
  3. **A training draw.** July ran once. Strength spread is small (+0.10 ± 0.55 in the control rerun), but a profile
     may vary more than strength does.
  4. **Opponent pool:** base, v4, base, base here against July's base/v4/snap.

## Follow-up: the phase 1 trajectory (evaluation only, started 2026-11-06 23:42)

This is exploratory and post hoc. It reopens no gate and starts no training. `u1_traj.sh` (log
`runs/xfer_tools/u1_traj.log`, markers `U1_TRAJ_DONE` / `U1_TRAJ_STOPPED`, results `u1_traj.txt`) does three things:
- checks that best.pth and the U40k archive carry the same weights;
- plays the 12k of U8k, U16k, U24k, U40k, U48k and U56k (U32k and U64k exist), paired on the same 3,000 groups, with
  the mechanism of each;
- extends U40k to the 16,000 groups.

It takes about 6.5 hours, then the card is lent for a day.

**Reading, set before the results:**

| Result | Reading | Next |
| --- | --- | --- |
| U40k on 16,000 groups meets both profile thresholds (attack ≥ +10, riichi ≥ +1.0 pp) | Snapshot selection can explain the gate failure: the profile appears transiently | Phase 2 from U40k, for the user to choose. Not automatic, because the snapshot was chosen after the gate |
| No snapshot on 12k reaches both thresholds | The profile did not appear in this run at all, and selection is ruled out | The repair leg can only be tested from rl1_best itself, i.e. P1 |
| Otherwise | Undecided | Reported as is |

### Result of the follow-up (2026-11-07 06:03): the profile appears and fades, and July's selection rule would have caught it

The check passed: best.pth and the U40k archive carry identical weights.

**Trajectory on the 12k groups** (3,000 groups and 12,000 games each, paired; mechanism against the v4 seats):

| Snapshot | vs v4 | attack / defence | riichi, pp | deal-in, pp | 4th, pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| U8k | −1.58 ± 1.30 | −9 / −19 | +0.37 | +0.28 | +1.29 |
| U16k | −0.98 ± 1.30 | −20 / −16 | −0.38 | +0.13 | +0.82 |
| U24k | −1.36 ± 1.30 | −25 / +5 | −0.27 | −0.23 | +1.09 |
| U32k | −1.84 ± 1.31 | +17 / −35 | +1.19 | +0.44 | +2.01 |
| **U40k** (best.pth) | −2.09 ± 1.30 | +22 / −39 | +1.31 | +0.45 | +2.12 |
| U48k | −1.71 ± 1.32 | **+41** / −55 | **+1.79** | +0.84 | +2.19 |
| U56k | −1.68 ± 1.32 | +28 / −47 | +1.19 | +0.67 | +1.67 |
| U64k | −1.31 ± 1.33 | +5 / −32 | +0.91 | +0.30 | +1.47 |
| rl1_best, same games | | +21 / −55 | +1.92 | +0.81 | +2.60 |
| A80k, same games | | −6 / −17 | +0.16 | +0.23 | +1.11 |

On these groups every snapshot is within noise of U64k: the differences run from −0.78 to +0.33, each ±1.5 to 1.7.

**U40k on all 16,000 groups:**

| Pairing | Difference | z |
| --- | ---: | ---: |
| U40k vs v4 | −1.61 ± 0.57 | |
| U40k − U64k | +0.04 ± 0.66 | +0.13 |
| U40k − A80k | +0.44 ± 0.73 | +1.19 |
| U40k − rl1_best | +0.63 ± 0.70 | +1.77 |
| U40k − C′ final | −0.69 ± 0.69 | −1.96 |

Mechanism of U40k on 64,000 games:
- call rate 30.1 (v4 30.4); win −0.13 pp; deal-in +0.67 pp; riichi **+1.40 pp**;
- net −26 per kyoku, made of attack **+20** and defence −46;
- bust 7.40 / 6.71%; 4th +1.67 pp.

**Reading, by the rule set before the results: U40k meets both profile thresholds** (attack +20 ≥ +10, riichi +1.40 ≥
+1.0).
- **The profile does appear.** The unanchored stage does grow an aggressive, leaky profile:
  - it forms between 24k and 32k, peaks near 48k (stronger than rl1_best on the same games) and fades by 64k;
  - strength stays flat throughout.
- **The gate was missed through timing.** The fixed 64k snapshot came on the way down. July's selection rule picked
  a snapshot inside the window, as it did for rl1_best.
- **U40k is a milder rl1_best:** attack +20 against +26, riichi +1.40 against +1.93, defence −46 against −60. It is
  slightly stronger (+0.63, z 1.8, not significant).
- **The first leg of the hypothesis holds, with a caveat:** the profile is transient. Phase 2 from U40k is now a
  proposal for the user. It was not started, because the snapshot was chosen after the gate.
- The user had already chosen the next run: the C′ step replayed from C392k ([cprime_replay](../cprime_replay/README.md)),
  started 11-07 10:26.
