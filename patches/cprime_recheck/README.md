# P0: was the historical C′ jump real?

Pre-registered 2026-11-06, before the run. 5060 Ti, evaluation only. Proposed in the 11-05 whole review
(`research/2026-11-05_progress_review`, gitignored).

Started at 02:4x on 11-06 on an idle card, before the user had approved it. It trains nothing and changes no
direction. Stop it with `kill $(cat runs/xfer_tools/p0_chain.pid)`.

## Why

- **The gap sits on C′.** The replay line sits about 1.2 below tpt 408k, and almost all of that gap rests on one
  historical step: C′, rl1_best → C′ final (07-22 to 07-24 on trx).
- **C′ is a 12k reading.** It was measured only on 12k, re-evaluated on 09-12:

  | Pairing | Difference | z |
  | --- | ---: | ---: |
  | C′ final − C392k (relaxing λ 0.5 → 0.2) | +1.56 ± 1.29 | 2.37 |
  | C′ final − rl1_best | +1.75 ± 1.33 | 2.58 |

- **12k has misled us before.** It overstated effects 1.5 to 3.5 times three times, and the control rerun found two
  more misreadings.
- **Four rebuilds failed.** Four single-variable rebuilds on the replay line (reward, anchor schedule and two others)
  never reproduced the jump.
- **Training draws are unlikely to explain it.** The control rerun put one stage's training spread at
  +0.10 ± 0.55.
- **So the question is whether the jump itself holds on new walls.** The three snapshots still exist, so this needs
  evaluation only.

## Design

- **Models.** These are the historical checkpoints, copied unchanged into `runs/xfer_hist/archive`:

  | Model | Step | Source |
  | --- | ---: | --- |
  | rl1_best | 60,000 | `weights_backup/rl1_best_155k.pth` |
  | C392k (λ 0.5) | 215,800 | `runs/cprime_C392k_0723_1921.pth` |
  | C′ final (λ 0.2) | 278,600 | `runs/cprime_final_0724_0939.pth` |

- **Harness.** The current one for all three: `xfer_eval.sh`, GPU fp32, AMP off, 4 processes, the sacred duplicate
  protocol against mortal_v4. No pairing mixes devices.
- **Decision groups:** the 13,000 groups that played no part in finding the jump. These are seeds 11000–13999 and
  16000–25999.
- **Report only:** the 12k groups of 09-12 (10000–10999, 14000–15999), where the jump was found.
- **Order.** Segment by segment, all three models per segment, so that any finished part is fully paired.

## Verdict

The primary is C′ final − C392k (the relaxation phase) on the decision groups.

| Reading | Outcome | Next |
| --- | --- | --- |
| ≥ +0.7 with z ≥ 2 | **The jump holds** | The faithful C′ replay (P1) becomes the next proposal, for the user to approve: from rl1_best with the July config, ideally on trx with 7 workers |
| ≤ +0.3 | **The jump fails** | It was mostly a 12k high draw. Rebuilding the historical path stops |
| In between | Undecided at this size | Seeds 26000–34999 are added once (22,000 decision groups) and the same lines read again. Still in between: undecided |

**Also reported:**
- C392k − rl1_best (the λ 0.5 phase, historically +0.19) and C′ final − rl1_best.
- All three against v4.
- rl1_best − A80k: the starts of the two lines.
- C′ final − tpt 408k.
- The mechanism readouts.
- A loading check against the 09-12 dumps of the same models.

**Prior:** about 40% that the jump holds.

**Cost:** 48,000 groups, about 9 hours; with the extension 75,000 groups, about 14 hours.

## Files

`p0_chain.sh` writes the log `runs/xfer_tools/p0_chain.log`, with the markers `P0_CHAIN_DONE` / `P0_CHAIN_STOPPED`.
Its results go to `p0_final.txt` and `p0_mech.txt`.
- It rewrites `GPU_STATUS_mortal.txt`, and its idle check also waits for a guest's hold.
- Afterwards it lends the idle card to bookings for a day.

## Result (2026-11-06): the jump holds

The chain ran 02:44–12:33 on 11-06. The verdict needed no extension.

| Pairing | 13,000 new groups | z | old 12k of 09-12 | all 16,000 |
| --- | ---: | ---: | ---: | ---: |
| **C′ final − C392k** (relaxing λ 0.5 → 0.2) | **+0.98 ± 0.63** | **+3.09** | +1.52 ± 1.29 | +1.08 ± 0.56 |
| C392k − rl1_best (the λ 0.5 phase) | +0.24 ± 0.54 | +0.89 | +0.19 ± 1.13 | +0.23 ± 0.48 |
| C′ final − rl1_best | +1.23 ± 0.63 | +3.82 | +1.70 ± 1.33 | +1.32 ± 0.57 |
| C′ final vs v4 | −0.95 ± 0.63 | | −0.78 ± 1.31 | −0.92 ± 0.57 |
| rl1_best vs v4 | −2.18 ± 0.64 | | −2.48 ± 1.35 | −2.24 ± 0.58 |

The C′ final row above is the comparison itself. The rows below are on all common groups:

| Pairing | Groups | Difference | z |
| --- | ---: | ---: | ---: |
| rl1_best − A80k (the starts of the two lines) | 16,000 | −0.19 ± 0.74 | −0.49 |
| C′ final − tpt 408k | 16,000 | −0.39 ± 0.59 | −1.29 |

**Loading check.** On the groups the 09-12 dumps share (7,000 for C′ final, 2,000 for C392k), today's per-group means
equal them exactly. So the copies and the harness reproduce the old games.

**Verdict, by the rule set in advance: the jump holds.**
- +0.98 with z 3.1 on groups that played no part in finding it.
- The 12k overstated it by about half (+1.52), but it is real. It is the largest measured gain of any single step in
  the project.
- The λ 0.5 phase gave nothing measurable (+0.24 ± 0.54), as in history.
- **Nothing after C′ is measurable either.** C′ final is level with tpt 408k (−0.39 ± 0.59). The later steps
  (λ 0.1, the reward change, the opponent pool) added nothing at this precision.

**Mechanism.** These are 64,000 games per model, against the v4 seats in the same games. A80k is its 100k from the
transfer test.

| | call rate (v4) | win, pp | deal-in, pp | riichi, pp | net / kyoku (attack / defence) | bust us / v4 | 4th, pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rl1_best | 29.1 (30.5) | −0.06 | +0.87 | +1.93 | −33 (+26 / −60) | 7.69 / 6.58% | +2.53 |
| C392k (λ 0.5) | 29.1 (30.4) | +0.01 | +0.78 | +1.98 | −32 (+25 / −57) | 7.65 / 6.66% | +2.35 |
| **C′ final (λ 0.2)** | 30.0 (30.4) | −0.02 | +0.31 | +0.58 | −18 (+12 / −30) | 7.44 / 6.72% | +1.41 |
| A80k (replay start), 100k | 30.8 (30.6) | −0.44 | +0.33 | +0.10 | −29 (−8 / −21) | 7.02 / 6.63% | +1.42 |

**Reading the mechanism.** This is a hypothesis; it has not been tested.

- **rl1_best and A80k are equally strong** (−0.19 ± 0.74) **but play differently.**
  - rl1_best came from an unanchored first stage. It attacks (+26 per kyoku, win rate level with v4) and leaks:
    riichi +1.93 pp, deal-in +0.87 pp, defence −60, 4th +2.5 pp.
  - A80k was anchored to v11 from its first batch. It is passive: attack −8, riichi +0.10, defence −21.
- **The relaxation fixed the leak and kept most of the attack.** Riichi went +1.98 → +0.58, deal-in +0.78 → +0.31,
  defence −57 → −30 and 4th +2.35 → +1.41, while attack stayed at +12.
- **So the jump looks like the repair of a defect** that rl1_best had and A80k never developed. Relaxing the anchor
  on the replay line had nothing to repair, which fits the four failed rebuilds.
- **The ingredient the replay line never had is an unanchored first stage.** Every replay arm was anchored from its
  first batch. This is item 8 in `research/2026-11-05_rl_design_audit`.

**Next, for the user to approve (nothing is queued):**
- **P1, faithful C′ replay.** From rl1_best's full state with the July config, about 25 hours of training. It asks
  whether the operator reproduces the jump from the same start on our hardware.
- **P2, the whole path from v11.** An unanchored first stage like rl1, then the C′ schedule on it. It asks whether
  our pipeline can make an rl1_best-like start itself, and then repair it.
