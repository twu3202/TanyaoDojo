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
- `gpu_share.sh` (added 11-03) lends the card to another session's short test, but only at a boundary:
  `xfer_arm.sh` calls it before it starts an arm and `xfer_eval.sh` before each evaluation call, so a running arm is
  never shared. It also never holds while a trainer, worker or eval process is running.
  - A guest books by creating `C:\Users\asus\GPU_BOOK_<name>.txt` and marks its use with `GPU_BUSY_<name>.txt`.
  - A line `not_before=YYYY-MM-DD HH:MM` in the booking defers it until then.
  - It can ask for a deterministic co-located load (4 × 400 bench_eval rounds) with `GPU_LOAD_please.txt`.
  - The hold lapses after 45 minutes without a BUSY file and ends after 90 minutes in any case.
  - Without a booking the script returns at once.

The launcher is `scripts/run_selfplay_m1.sh` (WSL), run with `SP` and `CFG` set. `group_ci.py --strict` comes from
this repository.

## Result (2026-10-21): the offline gain does not survive the first online stage

Both arms ran 80,000 batches without a trainer restart. The collapse guard never fired.

- **Arm v11:** 07:57–16:45 on 10-20, s80000 infer md5 104014f8.
- **Arm q25:** 18:14 on 10-20 to 02:51 on 10-21, s80000 infer md5 144ffc6b.
- **Sample reuse:** 1.82 in both arms. Each drain after the first was 7,200 files; the first drains reused 2.31 and
  3.21.
- **Evaluation:** with the 4-process layout. The last 100k finished at 13:13 on 10-21.

All rows are paired on the same groups, in pt per hanchan with 95% CIs.

| comparison | 100k | z (100k) | 12k at 80k | 12k at 40k |
| --- | ---: | ---: | ---: | ---: |
| v11 online 80k vs v4 | −1.82 ± 0.46 | | −1.41 | −2.58 |
| q25 online 80k vs v4 | −1.82 ± 0.45 | | −1.25 | −2.02 |
| **recipe: A80k − v11** | **+0.68 ± 0.45** | **+2.99** | +2.38 | +1.21 |
| gain of arm B: B80k − q25 | +0.13 ± 0.43 | +0.58 | +0.87 | +0.10 |
| **primary: B80k − A80k** | **−0.00 ± 0.60** | **−0.01** | +0.16 | +0.56 |
| offline: q25 − v11 | +0.55 ± 0.60 | +1.80 | +1.67 | |
| interaction (online − offline difference) | −0.55 ± 0.62 | −1.74 | −1.50 | −1.11 |
| A80k − tpt 408k | −1.18 ± 0.60 | −3.86 | | |
| B80k − tpt 408k | −1.18 ± 0.61 | −3.80 | | |

**Reading, by the rules set in advance**

1. **The recipe check passes.** Arm A gained +0.68 (z 3.0), at least the +0.5 required. The test therefore answers
   its question.
2. **Primary result: erased.** The primary difference is −0.00, and the rule for ≤ 0 is "erased by online RL".
   - The CI runs from −0.60 to +0.59, so it still contains the whole offline +0.55. "Erased" is the point reading;
     it does not show that the gain is zero.
   - The interaction is −0.55 (z −1.74).
3. **The two arms converge.** Both end at −1.82. The online phase moved v11 by +0.68 and q25 by only +0.13.
4. **The 12k overstated the recipe 3.5 times** (+2.38 against +0.68 at 100k). Its two segments are v11's worst:
   −3.79 at 12k against −2.50 at 100k. This is the failure the 12k rule warns about.

**Mechanism** at 100k, relative to the v4 seats in the same games:

| | avg_pt | call rate (v4) | win | deal-in | riichi | net / kyoku (attack / defence) | 4th, pp | bust us / v4 | position entering South 4 / converting it |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q25 offline | −1.95 | 31.10 (30.50) | −0.16 | +0.37 | +0.39 | −32 (−6 / −26) | +1.46 | 7.17 / 6.58 | −1.04 / −0.91 |
| v11 online 80k | −1.82 | 30.79 (30.55) | −0.44 | +0.33 | +0.10 | −29 (−8 / −21) | +1.42 | 7.02 / 6.63 | −0.89 / −0.93 |
| q25 online 80k | −1.82 | 30.73 (30.45) | −0.24 | +0.37 | +0.66 | −27 (−4 / −23) | +1.43 | 7.12 / 6.58 | −0.98 / −0.84 |
| tpt 408k | −0.64 | 30.61 (30.49) | −0.12 | −0.17 | −0.84 | −18 (−15 / −3) | +0.68 | 6.95 / 6.67 | +0.49 / −1.13 |

- **One profile at −1.8 to −1.9.** q25 offline and both online arms look alike:
  - deal-in +0.33 to +0.37 pp;
  - defence −21 to −26 points per kyoku;
  - 4th place +1.4 pp;
  - −0.9 to −1.0 pt of avg_pt already lost in the position entering South 4.
- **tpt 408k differs in exactly these places:**
  - deal-in −0.17 pp;
  - defence −3;
  - 4th place +0.68 pp;
  - +0.49 pt gained in the position entering South 4.
- **The remaining 1.2 points to 408k are defence and early-game position.** The online phase changed neither, in
  either arm.

**Correction to the pre-registration**

The Question section says the C' phase was "anchored to v11 at λ 0.2". It was not. C' started from rl1_best:
v11 plus about 155k games of unanchored online RL, −1.96 at 100k in July (unpaired). It was then anchored to rl1_best,
at λ 0.5 for about 36 hours and at 0.2 for about 9 hours.

By data volume, the phase replayed here is the first online stage, rl1. That stage used about 155k games; here each
arm trained on about 141k fresh games (22.5M trainee samples at about 160 per game). The result matches that stage:
rl1_best was +0.55 over v11 at 100k (unpaired), and arm A is +0.68 (paired). The later stages, from C' to 408k, are
worth another +1.18 at 100k and were not replayed.

**So the test answers a narrower question than intended.** Through the first online stage, q25's offline advantage
does not survive as a point estimate. Whether a base advantage reappears after the later stages is still open.
