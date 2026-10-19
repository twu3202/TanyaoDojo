# Online transfer test: does a better offline base stay better through online RL?

Pre-registered 2026-10-19, before the run. 5060 Ti, queued behind the q12one evaluation.

## Question

Data selection gives the best offline base so far: q25 is +0.55 ± 0.60 over v11 at 100k (q25one +0.45 ± 0.60; the
two share v11 and the walls, correlation 0.53, so they are not two replications). Our best model, tpt 408k
(−0.64 ± 0.46 vs mortal_v4), was grown from v11 by online RL. Nothing so far says whether an offline gain survives
online RL. It could add, shrink or vanish, for example if online RL fixes the same defects that selection fixed.

Most of the historical online gain came early. The C' phase (07-23, anchored to v11 at λ 0.2) gained +1.56 at 12k in
about 40k batches. Everything after it, from C' final to 408k and beyond, moved less than the noise. So this test
replays that early phase from two bases with one fixed recipe. It does not rerun the whole multi-stage history.

## Design

| | arm A | arm B |
| --- | --- | --- |
| start | v11 (`runs/symaug/v11_best.pth`, md5 2ae5581c) | q12one if its 100k vs v4 is above q25's on the same 25,000 groups, else q25 (`q25_final.pth`, b190d4c2) |
| anchor | v11, frozen, λ 0.2 | the arm's own start, frozen, λ 0.2 |
| opponents (4 GPU workers) | v11, v4, v11, v11 | base, v4, base, base |

Everything else is the tpt 408k recipe as run by M1 on the 5060 Ti (`configs/online_selfplay_xfer.toml`):

- evaluation-proportional pts, opt_step_every 8, submit 400, constant lr 1e-5;
- boltzmann ε 0.005 / temperature 0.1, frozen BN, grad clip 1.0;
- `train.py` 67bce509 (anchor patch; M1's margin_scale left at its default 1.0, i.e. the original anchor).

A warm start from an offline state resets steps and best_perf and builds a fresh Adam. The trainer starts only after
4,000 files are in the buffer.

- **Budget.** 80,000 batches = 10,000 Adam updates per arm, snapshots every 8,000. About 11 hours per arm at the
  ~2 batches/s M1 measured here.
- **Readouts.**
  - 12k (s10000 + s14000) of the 40k and 80k snapshots; this traces the trajectory and decides nothing.
  - 100k of both 80k snapshots, with diag_gap / diag_place / diag_bust.
  - Per-drain sample reuse (`xfer_reuse.py`).
  - Pairings on the same 25,000 groups (`xfer_pair.py`, which refuses incomplete or conflicting coverage).
- **Cost.** About 22 h of training, 4 h of 12k and 15 h of 100k: roughly 41 hours of the 5060 Ti.

## What is read, and how

1. **Recipe check: A80k − v11.** If arm A gains less than +0.5, this recipe did not reproduce an online jump on this
   machine. The transfer question then stays unanswered and the result is reported as such, not as "no transfer".
2. **Primary: B80k − A80k** at 100k, with its CI. The offline difference on the same seeds (B base − v11) and the
   interaction (online difference − offline difference) are computed from the four per-group vectors jointly.
3. **Reading.**
   - B80k − A80k ≥ +0.3: the offline gain is largely kept.
   - ≤ 0: it is erased by online RL.
   - In between: unresolved at this size.

   These are pre-set reading rules for a point estimate with a ±0.6 CI, not significance tests.
4. **Next step.** If arm B ends as the better online model, it is the candidate to continue long toward the target
   (≥ +0.01 vs v4 at 100k). Both arms are also reported against tpt 408k.

## Known differences from the historical runs

- The topology is 4 workers here against 7–12 on trx. Fresh trainee data per drain is about 5,600 games here against
  9,600, so each sample is reused about 2.2 times here against 1.3 there. This is measured from the trainer logs at
  about 165 trainee samples per game, and `xfer_reuse.py` counts it exactly for these arms.
- The historical C' jump trained with pts [6,4,2,0]; this recipe uses evaluation-proportional pts, as everything since
  09-14 has.
- The opponent pool is base-conditioned. Arm B trains against a stronger frozen opponent than arm A does, so the test
  compares two pipelines, not two initial weights alone.
- One training seed per arm, so training variance is not in the CI.

## Files

- `xfer_chain.sh` queues the whole test. A `HOLD` file in `runs/xfer_tools` holds the start, and `chain.pid` cancels it.
- `xfer_setup.sh` prepares one arm, `xfer_arm.sh` trains it, and `xfer_eval.sh` evaluates one snapshot.
- `xfer_helpers.sh` holds the archiver, the trainer guard and the stop script; the collapse guard is
  `xfer_collapse_guard.py`.
- `xfer_reuse.py` counts sample reuse and `xfer_pair.py` computes the pairings.

The launcher is `scripts/run_selfplay_m1.sh` (WSL), run with `SP` and `CFG` set. `group_ci.py --strict` comes from
this repository.
