# Sample freshness: does training each fresh sample once take the online stage further?

Pre-registered 2026-10-28, before the run. 5060 Ti. Chosen by the user after the
[online capacity test](../online_capacity/README.md) closed.

## Question

On the 5060 Ti, the v11 line's online stages gained +0.68, +0.28 and about 0, and the line levelled off around −1.7
against v4. The historical line on trx went on to 408k (−0.64). The largest difference between the two machines is
how often each fresh sample is trained on:

- **5060 Ti:** 4 self-play workers; about 1.8 passes per sample.
- **trx:** 7–12 workers; about 1.3 passes per sample, so more fresh games per update.

If staleness or overfitting to recent games holds the 5060 Ti line back, the same stage with fresh-only data should
gain more. That would also say that the main line belongs on trx's workers.

## Design

- **Arm F.** It replays the stage-2 control (A80k → A160k, `runs/xfer_v11s2`) with one change: `num_epochs = 1`.
  The trainer makes one pass over each drain of new games and then waits for the next, so each fresh sample is used
  once. While it waits, the workers fetch the new parameters, so play is also more on-policy.
- **Everything else is identical to the control:**
  - start A80k, a full online state;
  - re-anchored to A80k at λ 0.2, with A80k as the 'base' opponents and v4 in the pool;
  - 80,000 batches, to 160k.
- **Cost.** About the same number of updates, but with about 1.8 times the fresh games, so roughly 14–17 hours
  (generation-bound), plus evaluation.
- **Tool changes.**
  - `xfer_setup.sh` takes `EPOCHS`.
  - The snapshot archiver now scans every logged step count, not only the last line: with `num_epochs = 1` the
    trainer logs a new line every few hundred batches.

## Readouts and rules

1. **Trajectory, deciding nothing:** 12k of F120k and F160k.
2. **Screening: F160k − A160k** by the SPRT (0 vs +1, α 0.025, β 0.2, looks per 1,000 groups). It is capped at the
   16,000 groups A160k has played (seeds 10000–25999). Verdicts:
   - **Promote:** freshness is a lever. Recommend moving the main line onto trx's workers (the user decides).
   - **Drop:** sample reuse is not what holds the 5060 Ti line back.
   - **Undecided at the cap:** read the fixed 16,000-group difference. A result of ≥ +0.5 with z ≥ 2 counts as a
     promote. ≤ 0 counts as a drop.
3. **Also reported:**
   - F160k − A80k, against the control's +0.28 ± 0.52 on the same seeds;
   - F160k − tpt 408k;
   - F160k vs v4;
   - per-drain reuse from `xfer_reuse.py`, which should be about 1.0.

## Result (2026-10-29): dropped; fresh-only training learns less, not more

**Training.** Arm F trained from 10:20 to 22:48 on 10-28.

- It ran 80,000 batches, which is 40.96M samples drawn from 46.32M fresh samples. Overall reuse was 0.88.
- Within each drain, reuse was about 0.97.
- The rest of the gap comes from the drains that land on a 4,000-step test boundary: the trainer exits there after a
  few batches, so those games are never trained on. About 12% of fresh games were lost this way.

**Screening.** The SPRT of F160k − A160k dropped at 4,000 groups: −0.09 ± 1.08, LLR −1.94 against a lower bound of
−1.58. On the 6,000 groups played (seeds 10000–15999):

| | Result |
| --- | ---: |
| F160k vs v4 | −1.79 ± 0.93 |
| F160k − A160k (same stage at reuse 1.8) | −0.56 ± 0.89 (z −1.23) |
| F160k − A80k (the stage's gain) | +0.05 ± 0.84; the control gained +0.28 ± 0.52 over 16,000 groups |
| F160k − tpt 408k | −0.91 ± 1.23 |
| F120k vs v4 (12k) | −1.51 ± 1.32 |

**Mechanism.** These are the same 24,000 games per model, measured against the v4 seats in the same games.

| | call rate (v4) | win, pp | deal-in, pp | riichi, pp | net / kyoku (attack / defence) | bust us / v4 | 4th, pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A80k (start) | 30.9 (30.6) | −0.46 | +0.31 | +0.12 | −32 (−9 / −23) | 6.95 / 6.56% | +1.31 |
| A160k (reuse 1.8) | 30.4 (30.6) | −0.51 | +0.09 | +0.14 | −15 (−9 / −7) | 6.93 / 6.67% | +0.63 |
| F160k (reuse 0.9) | 31.4 (30.5) | −0.24 | +0.31 | −0.06 | −27 (−5 / −22) | 6.72 / 6.59% | +1.23 |
| tpt 408k | 30.7 (30.5) | −0.09 | −0.14 | −0.85 | −21 (−16 / −5) | 6.91 / 6.51% | +0.89 |

- **The reused control learned defence in this stage; the fresh arm did not.** From A80k to A160k, deal-ins fell from
  +0.31 to +0.09 pp and the defence deficit from −23 to −7 points per kyoku. F160k kept A80k's deal-in rate and
  defence (−22), and it calls and wins a little more.
- **Sample reuse is not what holds the 5060 Ti line back.** The readouts suggest the extra passes over each drain are
  where part of the stage's defence is learned, and fresher data alone does not make up for them. The strength
  difference itself (−0.56, z −1.23) is not significant.
- **This does not argue for moving the main line onto more trx workers.** The gap to the historical line has to sit
  in its other differences: the λ schedule (0.5 → 0.2 → 0.1) and the reward change.

## Interruption (2026-10-29)

- **What happened.** At 00:21 the WSL distro went down in the middle of segment s12000:
  - D:, which holds the WSL disk (`D:\wsl\Ubuntu-22.04\ext4.vhdx`), had filled up to 9.6 MB free;
  - systemd crashed;
  - after that the distro would not start (`Wsl/Service/CreateInstance/E_FAIL`).
- **State at the crash.** Training and the 12k were already done, and the segments s10000, s11000 and s14000 had
  finished.
- **Recovery.**
  - 32 GB was moved from D: to C:, and the WSL keep-alive task was restarted.
  - `fresh_chain.sh` became re-runnable: it skips setup and training once the arm is trained, and `xfer_eval.sh`
    skips finished segments.
  - The chain resumed at 12:07. The unfinished s12000 was replayed from scratch.
- **Guard added.** The chain scripts and `mainline_next.sh` (per segment and per stage) and `xfer_arm.sh` (per
  minute) now stop cleanly when D: has less than 10 GB free.
- **Cost.** About 11.5 hours of idle GPU. No result was lost.

## Default follow-on (the user's standing choice, 2026-10-28)

When the chain ends, `patches/online_transfer/mainline_next.sh` starts and keeps going until stopped:

- **Which line.** It continues F160k (with `num_epochs = 1`) if F was promoted, otherwise A240k.
- **Each stage.**
  - 80k batches, re-anchored to the latest snapshot.
  - 12k at +40k and +80k.
  - SPRT against tpt 408k.
  - The pointer then advances to the new snapshot.
- **To stop it:** `touch runs/xfer_tools/MAINLINE_STOP` (after the current stage), or kill `mainline.pid` before a
  stage starts.
