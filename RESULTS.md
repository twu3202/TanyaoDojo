# Evaluation results log

## Phase 1 baseline: v1 vs Mortal v4 (100k-game duplicate 1v3)

Method: fixed wall seeds (seed_key=20260711), the challenger rotates through all 4 seats to cancel out luck,
jun_pt=[90,45,0,-135], champion = a local copy of the full Mortal v4 weights (256ch×54blk, the finished product of offline + online RL).

| checkpoint | Steps | avg_rank | avg_pt (95% CI) | Verdict | Eval hardware |
|---|---|---|---|---|---|
| **v1_best** | 540k | **2.5434 ± 0.007** | **-3.99 ± 0.53** | Slightly below v4 | AutoDL 4090 |
| v1_final | 1.21M (full epoch) | 2.5613 ± 0.007 | -5.10 ± 0.53 | Slightly below v4 | Local RTX 6000 Ada |

4th-place rate: best 27.27% / final 27.60% (an even split would be 25%) — the main source of lost points is finishing 4th too often.

### Key findings

1. **v1_best (540k steps) is significantly stronger than v1_final (1.21M steps, a full epoch)**, by about 1.1pt, with
   non-overlapping confidence intervals (best [-4.53,-3.46] vs final [-5.64,-4.57]). **Training the full epoch made it weaker.**
   - The 1000-game test_play during training is far too noisy (it bounced between -2 and -9 the whole way) to reflect this
     difference; it took 100k games to bring it out.
   - Suspected cause: the current LR schedule is a **constant 1e-4 with no decay** (warm_up=max_steps=1000 in the config,
     peak=final=1e-4), so the policy oscillates/degrades late in a long run; adding cosine decay to a lower final LR would very likely recover it.
   - **Phase 2 to-dos**: ① add proper LR decay; ② take checkpoint selection seriously (best was picked by test_play,
     but test_play is too noisy; it should be replaced by periodic larger-scale evaluations).

2. **-4pt is a reasonable gap for "half-finished vs finished product"**: v1 only did offline CQL (Phase 1),
   while v4 is the finished product of offline + online RL. A purely offline model trailing an online-fine-tuned model by
   4pt is within the normal range (official Mortal releases differ by ~1.2pt per generation). **Online RL (Phase 2b) is exactly the means to close this gap.**

3. **Pipeline validated** ✅: trained from scratch on our own data and our own environment, a model approaching v4, with no blow-ups.
   Phase 1 goal achieved; ready for Phase 2.

### v1 deliverable

**v1_best (540k steps) is adopted as the Phase 1 deliverable** — it is the stronger of the two, -4pt from v4.
Backups: `weights_backup/v1_best.pth` (deliverable), `weights_backup/v1_final.pth` (kept for comparison).

---

## Phase 2 Track B: v5 defense features vs Mortal v4 (100k-game duplicate 1v3)

Method as above (seed_key=20260711, the challenger rotates through 4 seats, jun_pt=[90,45,0,-135],
champion = local Mortal v4). challenger = v5 best.pth (360k steps, frozen).
v5 = the version-4 base + 10 defense channels (per-opponent deal-in rate / genbutsu / suji) + cosine LR decay + 256-wide conv + the most recent 8 years of data (1.29M games).

| checkpoint | Steps | avg_rank | avg_pt (95% CI) | Verdict |
|---|---|---|---|---|
| **v5_best** | 360k | **2.5279 ± 0.007** | **-2.67 ± 0.53** | Below v4, but significantly better than v1_best |

4th-place rate: 26.57% (1st 24.67% / 2nd 24.45% / 3rd 24.31% / 4th 26.57%) — points are still lost mainly to finishing 4th,
but clearly less than v1_best's 27.27%.

### Key findings

1. **The defense features work**: v5 (-2.67pt) improves on Phase 1's v1_best (-3.99pt) by ~1.3pt, with non-overlapping
   confidence intervals (v5 [-3.20,-2.14] vs v1_best [-4.53,-3.46]). Defense-side solver features + the LR fix + 256 width
   together pulled the offline base about a third of the way toward v4. The narrower 4th-place rate supports the "fewer deal-ins → fewer last places" mechanism.
2. **Still not past v4**: the whole 95% CI is < 0, confidently below v4. The remaining ~2.7pt is **the structural offline-vs-online gap**
   — the ceiling of a purely offline model, which only Track C (RL self-play fine-tuning) can close.
3. **The 1000-game in-training evaluation cannot be trusted**: midway through training, v5's test_play reported -0.045pt (nearly level with v4),
   an optimistic illusion; 100k duplicate games brought out the true value of -2.67pt. More confirmation that checkpoint selection must rely on large-scale evaluation.

### v5 deliverable

**v5_best (360k steps) is adopted as the Track B deliverable**: -2.67pt from v4, currently the strongest offline base.
Backup: `weights_backup/v5_best.pth`. **-2.67pt > the -5pt threshold → no rollback needed**; the next round,
**v18 (the v5 recipe × all 18 years, 2.51M games)**, started training automatically at 02:05 on 2026-07-16 to test whether more data can close in further on v4.
Track C RL self-play will warm-start from the strongest offline base (v18 or v5) to close the last ~2.7pt.

---

## Phase 2 Track A: v11, an LR-fix-only baseline, vs Mortal v4 (100k-game duplicate 1v3)

Method as above (seed_key=20260711, the same 100k walls, champion=mortal_v4).
challenger = v11 best.pth (the best at about 1.31M steps, frozen 2026-07-16 08:57).
v11 = **version 4 (no defense features) × 192 wide × cosine LR decay × the most recent 8 years of data** — its only substantive difference from v1 is the LR fix.
(v11 also drifted at the end of its epoch; after idling on the LR floor for 5.8h with no improvement it was stopped and evaluated by hand.)

| checkpoint | avg_rank | avg_pt (95% CI) | 4th-place rate | Verdict |
|---|---|---|---|---|
| **v11_best** | **2.5261 ± 0.007** | **-2.50 ± 0.53** | 26.47% | Below v4, statistically tied with v5 |

### Key findings (very informative)

1. **Most of the gain comes from the LR fix, not the defense features**: v11 (-2.50) vs v1_best (-3.99) = **+1.5pt, all of it
   contributed by cosine LR decay** (the same 8 years of data, the same 192 width, neither with defense features).
2. **v5 and v11 are statistically tied** (-2.67 vs -2.50, a 0.17pt difference, far smaller than the ±0.75 CI on the difference):
   **the combined net contribution of "defense features + 256 width" ≈ 0**. Together with community evidence that "backbone capacity matters little" (width ≈ 0),
   we infer that **the net contribution of the defense features is also ≈ 0** (explicit priors on per-tile deal-in rate / genbutsu / suji, which the model may already learn implicitly from the data).
3. **Direct implication for choosing the Track C base**: if v18 (18 years of data) also shows no significant gain, then v11 (192 wide, faster inference)
   is the better base candidate for self-play throughput; the defense features can be dropped (the user's preference can be met at zero cost).
4. v19 (a clean v4×256×18-year ablation) has essentially lost its purpose — v5≈v11 already shows the combined effect ≈ 0, so there is nothing left to decompose.

### v11 deliverable

Backup: `weights_backup/v11_best.pth`. Current top three offline: **v11 -2.50 ≈ v5 -2.67 ≪ v1_best -3.99**;
offline ceiling ≈ **-2.5pt**; the focus moves to closing the gap in Track C.

---

## Phase 2 wrap-up: v18, all 18 years of data, vs Mortal v4 (100k-game duplicate 1v3)

Method as above (seed_key=20260711, the same 100k walls, champion=mortal_v4).
challenger = **the v18 final weights mortal.pth (1,249,200 steps, annealing complete)**.
v18 = the v5 recipe (version-5 defense features + 256 wide + cosine LR decay) × **all 18 years, 2.51M games**.
Note: v18's best.pth is a ~600k-step mid-training checkpoint saved on a +4.0 noise spike in test_play on 07-16 (before annealing finished);
it cannot represent v18, so it was discarded and the final weights were evaluated instead (kept as `best_spike596k.pth`).

Process note: at 920,800 steps the server's GPU was yielded to another project, so training migrated to the local Windows machine
(RTX 5060 Ti 8GB, WSL) and resumed losslessly to completion (state / data / log counts verified item by item, original batch 1024,
steady state 3.8 steps/s); past max_steps it was stopped by hand as usual (pitfall #2). The evaluation also ran locally (~4.9h).

| checkpoint | Steps | avg_rank | avg_pt (95% CI) | 4th-place rate | Verdict |
|---|---|---|---|---|---|
| **v18_final** | 1.249M | 2.5359 ± 0.007 | **-3.02 ± 0.53** | 26.56% | Below v4, statistically tied with v5/v11 |

### Key findings

1. **Doubling the data (1.29M → 2.51M games) gives no gain**: v18 (-3.02) vs v5 with the same recipe on the recent 8 years (-2.67)
   differs by -0.35pt, within the CI; vs v11 (-2.50) by -0.52pt, with the CIs overlapping at the edge. The point estimate is even slightly negative —
   our guess is that style/level drift in the early (2009-2015) Phoenix-room logs dilutes the recent data; not pursued further.
2. **In-training test_play misled us for the third time**: the mean of the last 16 rounds, -1.59±0.58 (SEM), was clearly more optimistic than
   the 100k true value of -3.02 (bias from a small fixed seed set + noise). The iron rule stands: trust only 100k duplicate games.
3. **⭐ Track C base decision (per the branch pre-set in HANDOFF §4)**: v18 shows no gain → **base = v11**
   (-2.50, version 4 / 192 wide, inference ~50% faster than 256 wide, self-play throughput first). The defense features are dropped.

### Phase 2 overall conclusion

The offline phase is closed: **offline ceiling ≈ -2.5pt (v11)**; the marginal returns in all three directions — features, width, data — are ≈ 0,
and the remaining gap is the structural offline-vs-online gap. Everything now goes to **Track C self-play RL** (base v11, aiming to close 2.5pt and overtake v4).
v18 final weights archived: `weights_backup/v18_final_1249200.pth`.

---

## Track C's first RL gain: rl1_best vs Mortal v4 (100k-game duplicate 1v3, 2026-07-22)

Method as above (the same walls, seed_key=20260711). challenger = **rl1_best**: the peak weights captured by best.pth when the
v11 base, trained with the B1 recipe (online value regression, opt_step_every=8, LR 5e-6, opponent pool base/v4/snap), had played
**~155k self-play games** (the first output after train.py's best_perf fix).

| checkpoint | avg_rank | avg_pt (95% CI) | 1st-place rate | Verdict |
|---|---|---|---|---|
| v11 (RL starting point) | 2.5261 ± 0.007 | -2.50 ± 0.53 | 24.7% | Offline ceiling |
| **rl1_best** | **2.5088 ± 0.007** | **-1.96 ± 0.53** | **25.6%** | **First break through the offline ceiling** |

Reading: avg_pt +0.55 (the single-metric CIs overlap slightly, z≈1.4), but **the avg_rank CIs are fully separated**, the evaluation is paired on the same walls,
and the magnitude matches the community's value-regression benchmark (+0.6pt per 500k games, discussion #91) → **judged a real gain**.
This also confirmed that the value line's "gain window" is transient (by 310k games of B1 training, test_play had already fallen ~2pt from its peak),
so **the operating mode = peak extraction**: capture with best.pth → verify at 100k → bank it and rebase.

Next (from 2026-07-22 07:52): deploy **anchored value fine-tuning** (C', a local patch: the Q-values of all legal actions are anchored to BASE, λ=0.5,
temperature 0.1, LR 1e-5), rebased on rl1_best, aiming to push the next peak higher under the anchor's protection.
Backup: `weights_backup/rl1_best_155k.pth`. **Current position: -1.96, about 2pt left to v4.**

## Re-evaluating C' anchored value fine-tuning (2026-09-12)

The anchored line started from rl1_best (λ=0.5, relaxed to 0.2 from 07-24 01:00) was shut down back then without a duplicate evaluation. Re-evaluated under the same duplicate protocol
(12k = s1 seeds 10000-10999 on CPU fp32 + s2 14000-15999 on GPU fp32, paired within each segment on the same device; the same model's CPU/GPU paired difference is -0.10±0.35):

| Model | 12k avg_pt vs v4 | Paired |
|---|---|---|
| rl1_best | -2.50 ± 1.34 | — (100k: -1.96 ± 0.53) |
| C392k (λ=0.5, about 36 hours) | -2.31 | vs rl1_best +0.19 ± 1.13 (z=0.33) |
| **C' final (λ=0.2, 9 hours)** | **-0.75 ± 1.31** | **vs rl1_best +1.75 ± 1.33 (z=2.58)**; vs C392k +1.56 ± 1.29 (z=2.37) |

C' final vs C392k is +1.55 / +1.57 in the two segments, nearly identical: **the gain comes from relaxing the anchor from 0.5 to 0.2, not from training time**.
Mechanism (4000 games on the same v4 trajectories): riichi choice rate rl1_best 44.32% → C392k 44.36% → C' final 41.71% (v4 itself 39.75%);
attack/defense diagnosis (8000 games): deal-in rate relative to v4 +0.86pp → +0.15pp, defense -56 → -20 points per hand.

**Lessons**:
1. Back then the line was shut down after looking only at test_play (2000 games, random walls, GPU AMP, challenger without the agari guard), which has **no fixed offset** from the sacred protocol;
   it cannot be used to judge gains or losses, let alone to pick best.pth — best.pth is the maximum of 54 test_play readings, and on 4k duplicate games it was actually the worst (-2.78).
2. λ=0.5 "holding steady" was the anchor pinning the policy in place; λ=0 (the first night, A') collapses or drifts slowly; λ=0.2 is the only working point validated so far.

**Current position: -0.75 (12k), about 0.75pt left to v4; from 2026-09-12, training continues from C' final at λ=0.2.**

## Continuing C': what you anchor to decides whether it advances or retreats (2026-09-14)

Continuing from C' final (278,600 steps) at λ=0.2 with the anchor target still rl1_best, **the play retreated to rl1_best's within two hours**; switching the anchor target to C' final held it.
Mechanism readouts use two tools: `diag_disagree.py` (riichi choice rate over 4000 games of the same v4 trajectories, at 20,691 riichi-eligible decision points; v4 itself 39.01%)
and `diag_gap.py` (per-hand rates from the duplicate logs, relative to the v4 seats in the same games on the same walls).

| Snapshot | Anchor target | Steps | Riichi choice rate | 12k avg_pt vs v4 | 12k paired vs C' final | Per-hand riichi diff / deal-in diff / defense (s14000) |
|---|---|---|---|---|---|---|
| rl1_best | — | — | 43.77% | -2.50 ± 1.34 | -1.75 | +2.03 / +0.86 / -56 |
| C' final | rl1_best (λ 0.5→0.2) | 278,600 | 41.11% | -0.75 ± 1.31 | — | +0.81 / +0.15 / -20 |
| Continuation (4 workers) | rl1_best | 319,200 | 44.39% (@287,600) | -2.04 ± 1.34 | **-1.26 ± 1.37 (z=-1.81)** | +2.06 / +0.79 / -52 |
| Continuation (7 workers) | rl1_best | 286,600 | 43.01% | — | 4k -1.32 (z=-1.19) | — |
| **Continuation (7 workers)** | **C' final** | **322,000** | **39.93%** | **-0.56 ± 1.30** | **+0.22 ± 1.21 (z=0.36)** | +0.37 / +0.32 / -28 |

The two anchors paired directly at matched steps (12k): anchor C' final − anchor rl1_best = **+1.48 ± 1.42 (z=2.05)**, borderline, but in the same direction as the mechanism readouts;
in the 7-worker control that differs only in the anchor target (about 8,000 steps), riichi choice rate is 41.09% vs 43.01%.

**Mechanism**: with λ=0.2, the anchor term pulls Q toward the anchor target at every step. In July the run started near rl1_best, and the combined pull of the anchor and the online target pushed it to 41% riichi choice;
but for "anchor at rl1_best" that is not an equilibrium, so starting from C' final it gets pulled back. **C' final is a good snapshot, but "keep training anchored to rl1_best" is not a reproducible recipe**;
only anchoring to the current best snapshot holds. The 43k steps anchored to C' final were stable but did not improve, so from 2026-09-14 08:45 λ was relaxed to 0.1 on the same line (with the criterion recorded in advance).

**Process lesson**: for the first 26 hours of this continuation `--worker-device cuda:0` was not passed, so the workers fell back to generating on CPU and the trainer ran 13× slower (the script default has been changed);
meanwhile the 4k screening avg_pt values (-0.10 / -0.08 / -0.35 / -0.83) were all within noise — it was the per-hand riichi rate (+0.41 → +2.27pp) that caught the regression first.

## Value line λ=0.1: the play changed, the strength did not; the gap is located in bust games and the reward definition (2026-09-14)

On the line anchored to C' final, λ was relaxed from 0.2 to 0.1 for a full 40k steps (328,000 → 368,000). 12k (s10000 locally + s14000 on the server, GPU fp32;
the two cards' fp32 results match game by game on 98 of 100 decks):

| Snapshot | 12k avg_pt vs v4 | Paired vs C' final | Paired vs end of λ=0.2 (322,000) | Deal-in diff / win diff / call rate / offense / defense (s14000) |
|---|---|---|---|---|
| C' final | -0.78 ± 1.31 | — | — | +0.15 / +0.08 / 30.13 / +16 / -20 |
| End of λ=0.2, 322,000 | -0.56 ± 1.30 | +0.22 (z=0.36) | — | +0.32 / +0.34 / 30.48 / +26 / -28 |
| **λ=0.1, 368,000** | **-0.53 ± 1.32** | **+0.25 (z=0.38)** | **+0.03 (z=0.04)** | **-0.04 / -0.36 / 29.05 / +5 / -4** |

(C' final is on the GPU fp32 basis here; a separate 16k run on s16000-19999 gave -0.37 ± 1.14, about -0.55 over the combined 28k.)
Relaxing the anchor slid the play along an iso-strength line toward defense: the deal-in rate fell below v4's for the first time and the defense gap went -28 → -4, but the win rate and call rate fell with it, and strength did not change.

**Offense/defense point flow nearly matches v4, yet avg_pt still trails by 0.5 — the gap is in placement**. A new tool, `diag_place.py`, splits avg_pt by rank on entering South 4 into "entry position + South 4 conversion",
with games that end early because someone busts (about 22% of games) broken out separately; `diag_bust.py` measures the bust rate. Every snapshot enters South 4 ahead of v4 (+0.25 to +1.15), games played through South 4 roughly break even when summed by rank,
and **the loss is concentrated in games that end before reaching South 4** (-0.56 to -1.63 avg_pt per reading, the same sign in all 8 readings): our bust rate is +0.2 to +1.3pp higher, our 4th-place rate in early-ended games is +1.7 to +5.3pp higher,
and we take both more 1sts and more 4ths than v4. The gain from rl1_best → C' final also came mainly from this term (-1.14 → -0.61, s14000).

**Root-cause candidate: the training reward does not match the evaluation metric**. Every training config (offline and online) uses `pts = [6,4,2,0]`, with placement gaps of 2/2/2; evaluation uses [90,45,0,-135],
with gaps of 45/45/135 — training penalizes 4th place only a third as much as evaluation does, which matches exactly the "high-variance, busts often" play. From 2026-09-14 16:20, a branch from 368,000
changes only `pts = [2.390, 1.195, 0, -3.585]` (the evaluation's ratios, with the same standard deviation as the old values), `configs/online_selfplay_tenhoupt.toml`; at 408,000 steps it gets a 12k paired against the branch point.

**Throughput**: ever since C', the trainer had been stuck parsing in a single thread because of `[dataset] num_workers = 0` (1.45 batch/s, GPU utilization about 5%). With 8/12 processes, a 4000-step cycle
dropped from about 49 minutes to about 17, with every drain still a full 9,600 files; evaluation and the riichi-choice probe moved to the server and run in parallel (4k 33 → 6.6 minutes, probe 16 → 2.75 minutes).

## Training reward switched to the evaluation's placement points: every mechanism prediction hit on 8k (it does not hold at 100k; see the correction below), point estimate up to -0.20 (2026-09-14 → 09-16)

Branching from λ=0.1 at 368,000 steps and **changing only the placement points of the training reward**, [6,4,2,0] → [2.390, 1.195, 0, -3.585] (= the same ratios as the evaluation's [90,45,0,-135],
with the same standard deviation as the old values), with λ=0.1 / anchor C' final / opponent pool / loader all unchanged. After a full 40k steps (408,000), the 12k:

| | s10000 (4k) | s14000 (8k) | 12k combined |
|---|---|---|---|
| tpt 408,000 avg_pt vs v4 | -0.41 ± 2.25 | **-0.09 ± 1.64** | **-0.20 ± 1.33** |
| Paired vs branch point (λ=0.1, 368,000) | -0.35 (z=-0.28) | +0.68 (z=+0.76) | +0.33 ± 1.42 (z=0.46) |
| Paired vs C' final | +1.69 (z=1.38) | +0.03 (z=0.04) | +0.59 ± 1.39 (z=0.83) |

**-0.20 is this project's best 12k point estimate to date** (C' final -0.78, λ=0.1 -0.53, all on the GPU fp32 basis), but neither pairing is significant,
so by our rules **no record is claimed**. The criterion said "z < 2 and the mechanisms moved as predicted → run another 40k steps", and that is already running (448,000 steps gets the next 12k).

**Mechanism predictions (written down before the change) and results** (s14000 8k, relative to the v4 seats in the same games on the same walls):

| Metric | C' final | λ=0.1, 368,000 | **tpt 408,000** | Prediction |
|---|---|---|---|---|
| Bust rate (us/v4) | 6.89/6.54 | 6.89/6.57 | **6.53/6.49** | Gap → 0 ✅ |
| 4th place in games that end early | 26.3/24.6 | 26.5/24.5 | **25.7/24.8** | Gap narrows ✅ |
| 4th-place rate, pp | +0.87 | +1.30 | **+0.25** | Falls ✅ |
| 1st-place rate, pp | +1.37 | +1.15 | **+0.48** | May fall with it ✅ |
| Placement split: South 4 conversion | -0.43 | -1.02 | **-0.00** | Shrinks ✅ |
| Riichi rate diff / deal-in rate diff | +0.81 / +0.15 | +0.47 / -0.04 | **-0.77 / -0.17** | Not predicted |

The play went from the high-variance "more 1sts and more 4ths" to a steady style with a riichi rate **below** v4's, a bust rate level with v4's, and the call rate up to 30.77;
in the placement split, "South 4 conversion" closed from -1.02 to -0.00 — exactly the term that had been losing. **The mismatch between the reward definition and the evaluation metric is a real source of points**;
how much it is worth awaits the 12k at 448,000 and the extra s16000 run (combined with C' final into 28k).

**Two trainer crashes**: 09-14 16:51 and 09-16 16:12, both CUDA OOM while respawning a child process — without a cap, a single probe shard from the evaluation station could take 6-17GB.
The first time there was no guard, and training sat stopped for 48 hours before anyone noticed; now `scripts/trainer_guard.sh` restarts it automatically (the second crash lost only 7 minutes),
and evaluation processes are capped by `GPU_MEM_FRAC` (probe 0.12, eval 0.15) and must first pass `scripts/gpugate.sh`.

### The 100k milestone at 448,000 steps: −0.93 ± 0.46 — the 12k +0.56 was a high draw (2026-09-17)

The 12k at 448,000 steps was **+0.56 ± 1.30** (the project's first positive 12k); per the criterion "only a 100k point estimate ≥ +0.01 counts as parity", it was extended to 100k
(seeds 10000–34999; split between the local 5060Ti and the server's RTX 6000 Ada — the two cards' fp32 results match game by game on 98 of 100 decks):

| Segment | Games | avg_pt vs v4 |
|---|---|---|
| s10000–10999 | 4k | +1.61 ± 2.20 |
| s11000–13999 | 12k | −0.78 ± 1.31 |
| s14000–15999 | 8k | +0.04 ± 1.62 |
| s16000–29999 | 56k | −0.86 ± 0.61 |
| s30000–34999 | 20k | −2.11 ± 1.04 |
| **Total** | **100k** | **−0.93 ± 0.46** (95% CI [−1.39, −0.47]) |

**The criterion is not met, and the whole CI is below 0 — 448k is significantly weaker than v4.** The two 12k segments happened to be its two best.
Paired on the same 28k walls (s10000 + s14000 + s16000–19999): 448k vs C' final **+0.34 ± 0.92 (z=0.73)**, 448k vs 408k −0.13 (z=−0.26).
For reference: the value line's only previous 100k was rl1_best −1.96 ± 0.53 (July, unpaired); 448k is about 1.0 higher.

**100k mechanism readouts** (relative to the v4 seats in the same games; C' final and 408k are 28k):

| | avg_pt | Riichi diff | Deal-in diff | Win diff | Call rate | Offense / defense | 4th, pp | Bust us/v4 | "Ended before South 4" row |
|---|---|---|---|---|---|---|---|---|---|
| C' final (28k) | −0.55 | +0.58 | +0.32 | +0.03 | 30.03 | +17 / −32 | +1.26 | 7.33 / 6.72 | −0.81 |
| tpt 408k (28k) | −0.08 | −0.86 | −0.18 | +0.01 | 30.67 | −4 / −4 | +0.30 | 6.83 / 6.78 | −0.63 |
| **tpt 448k (100k)** | **−0.93** | +0.00 | −0.22 | **−0.72** | **27.32** | **−24** / +3 | +1.05 | 7.15 / 6.72 | **−0.92 ± 0.26** |

Reading:
1. **At 100k almost the entire gap sits in games that "end before reaching South 4"** (−0.92, of −0.93 overall); we still enter South 4 ahead (+0.26), and games played through South 4 are only −0.27 in total.
   This localization is significant at 100k and agrees with what we saw on 8k/16k on 9-14.
2. The reward-definition change pressed the bust gap down to +0.05pp and the 4th pp to +0.30 at 408k; **by 448k the play had kept sliding toward passivity**: call rate 30.7 → 27.3, win rate −0.72pp, offense −24,
   the bust gap back up to +0.43pp, 4th pp +1.05. **Mechanistically 408k is more balanced than 448k, but their strength cannot be told apart on 28k**.
3. Lesson: **a 12k's ±1.30 has no discriminating power near 0**. Every remaining effect in this project is under 1pt, and the differences between candidates are smaller still (0.1–0.5);
   questions like this can only be answered by 100k (±0.46) or by pairing on the same walls.

**Next step (criterion written first)**: extend 408k to 100k (paired with 448k on the same 100k walls, about 3 hours); if 408k is significantly better than 448k → switch the anchor to 408k and continue training at λ=0.1 with the Tenhou-ratio reward (iterating the recipe);
if they cannot be told apart → the case for anchoring to 408k rests on mechanism alone (calls and busts more balanced); training continues the same way, but every verdict is settled at 100k.

### The 100k at 408,000 steps, and a correction to "every mechanism prediction hit" (2026-09-17)

408k extended to 100k (on the same 100k walls as 448k): **−0.64 ± 0.46** (95% CI [−1.10, −0.18]), also significantly below v4.
Paired on the same 100k walls, 448k − 408k = **−0.30 ± 0.49 (z=−1.17)**: indistinguishable.

| 100k, relative to the v4 seats in the same games | avg_pt | Riichi diff | Deal-in diff | Win diff | Call rate | Offense / defense | 1st / 4th, pp | Bust us/v4 | South 4 entry position / conversion |
|---|---|---|---|---|---|---|---|---|---|
| tpt 408k | **−0.64** | −0.84 | −0.17 | −0.12 | 30.61 | −15 / −3 | −0.23 / +0.68 | 6.95 / 6.67 | +0.49 / −1.13 |
| tpt 448k | −0.93 | +0.00 | −0.22 | −0.72 | 27.32 | −24 / +3 | +0.12 / +1.05 | 7.15 / 6.72 | +0.26 / −1.19 |

**Correction**: the "every mechanism prediction hit" in the previous section used 8k (s14000) readings — at the time 408k read a bust gap of +0.04pp, 4th pp +0.25 and South 4 conversion −0.00.
At 100k these are **+0.28pp, +0.68 and −1.13**: still somewhat better in direction than C' final (28k: +0.61pp, +1.26, −0.96), but **nowhere near closed**.
The bust rate and placement-split readings on 8k are far noisier than I judged at the time; the claim that "all six hit" is withdrawn. **The gap is still concentrated in games that end early because someone busts**,
and the reward change's effect on it is not yet confirmed at 100k — s368000 (the branch point) is being extended to 100k, to be paired with 408k on the same walls.

**New experiment (from 09-17 08:40, criterion written first): switch the training opponent pool to v4**. Of the 310 self-play rounds after 09-16 23:18, only 52 (16.8%) were against v4;
the rest were against rl1_best (66%) and C392k (17%), both of which choose riichi about 43.8% of the time, versus 39.0% for v4, the evaluation opponent. This is the same kind of mismatch as "reward definition ≠ evaluation metric";
the way the play kept sliding toward passivity after the anchor was relaxed (calls 30.6 → 27.3) also fits "against aggressive opponents, folding more pays".
Branch from 408k with all 6 workers playing v4 (server capacity 4000 → 5000, keeping each drain at 9,600), everything else identical to the 408k→448k stretch.
At 448,000 steps, a 100k paired with tpt 448k on the same walls: z ≥ +2 → the opponent pool is a lever; mechanism predictions: call rate stays ≥ 30, win diff better than −0.72, bust gap < +0.43pp.

### The reward-definition experiment settled at 100k: +0.25 ± 0.48, not significant (2026-09-18)

The branch point s368000 (λ=0.1, old reward [6,4,2,0]) completed at 100k: **−0.89 ± 0.46**. Paired with the reward branch on the same 100k walls:

| Pairing (same 100k walls) | Difference | z |
|---|---|---|
| 408k (40k steps after the change) − s368000 | **+0.25 ± 0.48** | 1.01 |
| 448k (80k steps after the change) − s368000 | −0.05 ± 0.49 | −0.18 |

| 100k, relative to v4 in the same games | avg_pt | 1st / 4th, pp | Bust diff, pp | 4th in early-ended games, diff | South 4 conversion | Riichi diff | Call rate |
|---|---|---|---|---|---|---|---|
| s368000 | −0.89 | +0.81 / +1.35 | +0.45 | +2.4 | −1.19 | +0.52 | 28.8 |
| 408k | −0.64 | −0.23 / +0.68 | +0.28 | +1.9 | −1.13 | −0.84 | 30.6 |
| 448k | −0.93 | +0.12 / +1.05 | +0.43 | +2.1 | −1.19 | +0.00 | 27.3 |

**Conclusion**: switching the training reward to the evaluation's ratios does make the placement distribution less extreme at 100k (408k's 1st and 4th pp both fall),
but the loss in bust games (South 4 conversion −1.19 → −1.13) barely moves, the strength gain of +0.25 is not significant, and another 40k steps returns it to where it started.
**It is not the main cause of this 0.6–0.9 point gap**. It is kept as the default (consistent with evaluation, harmless), but is no longer counted as a validated lever.

### The opponent-pool experiment settled at 100k: +0.07 ± 0.51, also zero (2026-09-18)

Branching from 408k and changing only the self-play opponents, from "4× rl1_best + 1× C392k + 2× v4" to 6× v4 (everything else identical), run to 448,000 steps for a 100k:

| | 100k avg_pt vs v4 | Paired on the same 100k walls |
|---|---|---|
| v4 opponent pool, 448k | **−0.86 ± 0.46** | vs old-pool 448k **+0.07 ± 0.51 (z=0.27)**; vs old-pool 408k −0.23 (z=−0.92) |

**Criterion not met** (it required z ≥ +2). Mechanistically the opponent pool did have an effect; it just did not turn into points:

| 100k, relative to v4 in the same games | avg_pt | Riichi diff | Deal-in diff | Win diff | Call rate | Offense / defense | Bust diff | South 4 conversion |
|---|---|---|---|---|---|---|---|---|
| 408k (old pool) | −0.64 | −0.84 | −0.17 | −0.12 | 30.6 | −15 / −3 | +0.28 | −1.13 |
| 448k (old pool) | −0.93 | +0.00 | −0.22 | −0.72 | 27.3 | −24 / +3 | +0.43 | −1.19 |
| **448k (v4 pool)** | −0.86 | +0.30 | **+0.49** | **+0.24** | 30.6 | **+14 / −42** | +0.76 | −1.33 |

Playing against v4 pushed the play toward offense (win rate above v4's for the first time, offense −24 → +14), at the cost of defense −42 and a bust gap of +0.76; net effect +0.07.

**The common shape of this week's four levers**: λ 0.2→0.1, the anchor target, the reward definition, the opponent pool — each one clearly moved the play (riichi rate, call rate and the offense/defense split swung back and forth within ±25 points per hand),
**but none of them touched the loss in "games that end early because someone busts"** (South 4 conversion is −1.1 to −1.3 on every snapshot, CI ±0.26 at 100k).
Strength all falls in a band from −0.6 to −0.9, with no pairwise comparison significant. Conclusion: **the expected return from turning more knobs on this line is low**;
next, first measure "where we and v4 choose differently in low-score positions" (`diag_disagree.py --by-score`) to pin that 1.1 points of behavior onto specific decisions, then decide what to change.

### The gap located to "deal-ins cost more", and Plan A: anchored fine-tuning starting from the champion itself (2026-09-18 → 19)

With all four training knobs at zero on 100k, we switched to localizing first. 100k breakdown (tpt 408k, compared with the v4 seats in the same games):

| | Us | v4 per seat | Diff |
|---|---|---|---|
| Placement points per game, games played through South 4 (about 78%) | +0.28 | −0.09 | **+0.37** |
| Score before entering South 4 | 25,088 | 24,941 | **+147** |
| **Busted ourselves** | **5.86%** | **5.44%** | **+0.42pp** (100k, ±0.15) |
| Our placement points in games where someone else busts | +43.3 | +45.6 | −2.3 |

**We beat v4 in games played through to the end, and we lead entering South 4; nearly everything we lose is in "getting busted ourselves".** Breaking the busts down further by cause (per game, us/v4):
killed by a deal-in 4.98/4.86, by an opponent's tsumo 1.88/1.76, by noten payments 0.12/0.09 — **close to half of the excess comes from opponents' tsumo**, which push/fold cannot decide;
the score at the start of the fatal hand is the same on both sides (4,944/4,956), i.e. it is not "playing worse at low scores" but **dropping to low scores more often**.

One level up, the cause appears: **the deal-in rate is level with v4's, but each deal-in costs more**.

| Snapshot | Avg deal-in cost, us / v4 | Diff | Avg win value, us / v4 |
|---|---|---|---|
| C' final (28k) | 5,282 / 5,166 | **+116** | 6,816 / 6,745 |
| s368000 (100k) | 5,251 / 5,169 | +82 | 6,893 / 6,744 |
| tpt 408k (100k) | 5,227 / 5,137 | +90 | 6,702 / 6,736 |
| tpt 448k (100k) | 5,236 / 5,172 | +64 | 6,871 / 6,751 |
| v4p 448k (100k) | 5,262 / 5,138 | **+124** | 6,726 / 6,735 |

About 1.26 deal-ins per game × 90–125 points ≈ 110 points per game, which is the main source of the "net point-flow gap of −6 to −28 per hand", and it also fattens the lower tail of the score distribution → more busts.
**All five snapshots (across three branches, two reward definitions and two opponent pools) point the same way**: this is a stable behavioral difference, not an accident of one line.

**The capacity hypothesis is rejected**: v4 uses the same observation version as we do (version 4), but it is 256×54 (23.7M) while our entire line is 192×40 (10.8M).
Isolating capacity with our own offline models: v11 (192×40) average deal-in cost +48, v5 (256 wide + defense features) +102 (8k / 5.2k games, ±120) —
**a bigger network did not make deal-ins cheaper**, so a multi-day retraining of a large network is not worth starting.

**Plan A (from 2026-09-19 20:18, criterion written first)**: since the remaining gap is fine-grained defensive judgment and the training knobs cannot move it,
swap the base for the champion itself: **anchored value fine-tuning starting from mortal_v4, anchor = v4 itself, λ=0.2** (the only working point ever validated to improve),
opponent pool all v4, reward on the evaluation's scale, loader/test_play unchanged, config `configs/online_selfplay_v4base.toml`, archive prefix `v4b_`.
- This line's reading is the net increment: challenger = the fine-tuned model, champion = v4, so **avg_pt is directly "the improvement over v4"**.
- **Zero-point calibration**: v4 itself as the challenger against v4 should score 0.00 under duplicate symmetry (a local 4k verifies that the evaluation chain has no offset for the 256 model).
- Verdict: a 100k at 40k steps. A point estimate ≥ +0.01 meets the goal; only a CI lower bound > 0 counts as significantly beating v4.
- Stop-loss: riichi choice ≥43% or ≤35%, a deal-in diff ≥+0.8pp, or two consecutive 4k pairings with z ≤ −2.
- The cost of this choice: the nature of the conclusion changes from "our own stack beats v4" to "pushing v4 a little higher"; both will be reported as they are.

### Plan A settled at 100k: +0.10 ± 0.34 — the point estimate meets the bar, but cannot be separated from 0 (2026-09-20)

Starting from the champion mortal_v4 itself, anchor = v4, λ=0.2, opponent pool all v4, 40k steps, then a 100k (challenger = the fine-tuned model, champion = v4,
so **avg_pt is directly the net increment over v4**; zero point calibrated: v4 itself as the challenger against v4 = **−0.056 ± 0.091**, placement counts [1000,998,1001,1001]):

| Segment | Games | avg_pt vs v4 |
|---|---|---|
| s10000–10999 | 4,000 | −0.191 |
| s11000–13999 | 12,000 | +0.180 |
| s14000–19999 | 24,000 | +0.077 |
| s20000–22999 | 12,000 | +0.172 |
| s23000–34999 | 48,000 | +0.101 |
| **Total** | **100,000** | **+0.102 ± 0.337** (95% CI [−0.236, +0.439]) |

**The criterion we set (a 100k point estimate ≥ +0.01) is met; but the CI contains 0, so we cannot claim it is statistically stronger than v4.**
The 4k screens at each checkpoint (8k/16k/24k/32k/40k steps) read −0.08 / −0.69 / −0.69 / −0.61 / −0.14, then 48k +0.59 and 56k −0.77 —
**no upward trend at any point**, and the 100k +0.10 falls, consistently with them, in the "indistinguishable from v4" range.

**This magnitude cannot be measured**: moving the lower end of +0.10's CI off 0 would take about 1.1M games (a 100k already takes 5 hours with both machines in parallel),
i.e. about 55 GPU-hours. The project recorded long ago that "effects under 1pt are unaffordable to measure", and +0.1 is another order of magnitude smaller than that.

**Mechanism (100k): the excess busts disappeared**, which in turn validates the localization in the previous section:

| 100k | avg_pt | Bust us/v4 | 4th, pp | Placement split, entry/conversion | Avg deal-in cost | Offense / defense |
|---|---|---|---|---|---|---|
| **v4b 40k (base = v4)** | **+0.10** | **6.78 / 6.76 (+0.02)** | −0.11 | −0.01 / +0.11 | 5,147 | −8 / +7 |
| 408k (own base, v11) | −0.64 | 6.95 / 6.67 (+0.28) | +0.68 | +0.49 / −1.13 | 5,227 | −15 / −3 |

"Busts more than v4, deals in for more than v4" is **a property of our own base lineage**, and it disappears with v4's weights. On top of v4, our RL pushed the play slightly more conservative
(riichi −0.33pp, win −0.15pp, offense −8, defense +7), net effect +0.10, not significant.

**The nature of the conclusion must be stated clearly**: this line is "v4 + our RL", not "our own stack beats v4". The best result of our own base lineage is still **100k −0.64** (408,000 steps).
**The honest summary**: our anchored value fine-tuning, applied to a base that is already at the frontier, adds at most about 0.1 points, and cannot be distinguished from 0 at sample sizes we can afford.

### 160k more steps were no better: the 28k extension at s200000 = +0.36 ± 0.63 (2026-09-20)

Plan A stopped at **204,000 steps** (the GPU was yielded to another process). Among the checkpoint screens, `v4b_s200000` read **+1.40 at 4k (paired z=1.76)**,
the brightest checkpoint on this line. Following **the criterion fixed before running** (4k does not count; only if the 28k is still ≥ +0.5 does it go to 100k), it was extended locally:

| Segment | Games | avg_pt vs v4 |
|---|---|---|
| s10000–10999 | 4,000 | +1.395 ± 1.609 |
| s11000–13999 | 12,000 | −0.026 ± 0.953 |
| s20000–22999 | 12,000 | +0.401 ± 0.981 |
| **Total** | **28,000** | **+0.360 ± 0.630** (95% CI [−0.270, +0.990]) |

The 4k highlight regressed **for the third time** (previously 4k +2.67 → back to zero, and 12k +0.56 → 100k −0.93). +0.36 ± 0.63 is fully compatible with the 40k-step checkpoint's
100k +0.102 ± 0.337, so **there is no evidence that 160k more steps made it stronger**; the +0.5 criterion is not met, so as agreed in advance it **does not go to 100k**.

### A²: measuring Plan A's +0.10 to significance — +0.123 ± 0.101 over 1.1M games (2026-09-20 → 22)

No more training, only more games. The criterion was fixed before the run: challenger `v4b_s40000`; beyond the existing seeds 10000–34999, add 10 blocks × 100k games;
**at each interim look every 200k games, only stop-loss is judged** (stop if the cumulative point estimate < 0), and **the success test is done only once, at the 1.1M-game endpoint** (cumulative 95% CI lower bound > 0),
to avoid inflating the false-positive rate by looking repeatedly.

| Cumulative games | avg_pt vs v4 |
|---|---|
| 500k | +0.185 ± 0.151 |
| 700k | +0.129 ± 0.127 |
| 900k | +0.146 ± 0.112 |
| **1.1M (final verdict)** | **+0.1234 ± 0.1014** (95% CI [+0.022, +0.225]) |

"Start from the champion and fine-tune it with our online RL" does beat the champion, but only by +0.12. That is "v4 + our RL" and **does not count toward the own-pipeline goal**;
our own pipeline's best is still 100k −0.64 (tpt 408k).

## Pricing our own lineage's gap per decision (Step 0, 2026-09-21 → 25)

Counting dangerous tiles (`diag_dealin.py` / `diag_pushfold.py` / `diag_compound.py`) can split out pieces such as "pushing too much" and "running out of genbutsu early",
but cannot put a price on them. So we switched to **counterfactual rollouts**: the walls in the libriichi arena are determined by the seed recorded in the log, so any evaluation log can be branched on the **true wall**
at any decision point, replacing our action with v4's and letting v4 play on; a telescoping identity splits "our actual result − v4 playing our seat the whole way" exactly into the individual disagreement points
(the tooling lives in a separate evaluation module, [riichi-eval](https://github.com/twu3202/riichi-eval), AGPL-3.0). tpt 408k, 100k games, 1.32M disagreement points (hand horizon, priced by the GRP):

| Decision type | pt/hanchan |
|---|---|
| Against a riichi, "we push, v4 folds" | **−0.148 ± 0.075** |
| "We riichi, v4 does not" | −0.117 ± 0.094 |
| Tile choice when staying dama at a riichi-eligible point | −0.063 ± 0.053 |
| Choice among genbutsu | −0.053 ± 0.038 |
| Pon | −0.072 ± 0.102 |
| "We fold, v4 pushes" | +0.013 ± 0.063 |
| Ordinary tile choice (72% of disagreements) | +0.043 ± 0.344 |
| **Total** | **−0.433 ± 0.404** (measured −0.637 ± 0.458) |

**Push/fold and riichi judgment account for about half; ordinary tile choice is zero**. On a **raw-point** basis the same category points the opposite way (not declaring riichi earns more raw points but loses more placement points),
so raw-point diagnostics would point in the wrong direction. Lineage comparison (same tool): the offline v11 pushes and folds symmetrically (0.30 times per game each), while rl1_best pushes 0.56 times per game at −0.36 —
**the excess pushing was created by online RL**; v4b, started from v4, is not negative in any category (negative control).

Two targeted fixes both failed:

- **Flipping by a threshold on our own Q margin**: no threshold makes money. But at the spots where "v4 would also fold", folding just this one tile and then playing on as usual
  gives **+0.161 ± 0.142** (44k games, z=2.2); folding all the way never makes money. The value lies in "don't push this tile yet", and our Q cannot find these spots.
- **Fine-tuning on rollout-priced action pairs (A3 v1, `a3_finetune.py`)**: 60k games generated with a new seed key, 1.44M labelled pairs;
  12k paired λ=4 −0.91 ± 1.57, λ=16 −0.26 ± 1.40, independent hold-out +0.015 ± 1.00 → closed.
  Single-rollout labels are too noisy (an SD of about 15 points per pair against a signal of about 0.4), and the fine-tune scrambled the relative order among genbutsu.

## B1: the offline base scaled up to v4's size — capacity ruled out (2026-09-24 → 29)

The only known structural difference between v4 and us is backbone size (256×54 vs 192×40). B1 = the v11 recipe unchanged, with only the size changed
(`configs/local_train_b1_256x54.toml`), stopped at 1.22M steps, the same as v11's best checkpoint. Criterion written in advance: only a 100k paired vs v11 of ≥ +0.5 with z ≥ 2 moves on to online RL.

| Reading | B1 − v11 (paired on the same walls) |
|---|---|
| Preview (1.086M checkpoint, 24k) | −1.51 ± 1.27 |
| 12k | +0.65 ± 1.81 (z=0.71, passes the stop-loss gate) |
| **100k** | **−0.182 ± 0.617** (z=−0.58); B1 itself −2.680 ± 0.464 |
| 100k with variance reduction | −0.317 ± 0.564 |

Criterion not met. Of the four offline levers, more data (v18), more width plus defense features (v5) and more capacity (B1) are all about 0; only the original LR fix worked.
**The offline recipe tops out at about −2.5**. The 12k +0.65 was pulled back by the 100k — one more case of "a small-sample highlight regresses".

## Does the GRP in the training target bias it? (2026-09-29 → 30, interim reading)

Mortal-style training truncates the Q target at the end of each hand and values the hand's end with the GRP (a "position → expected final placement" model trained on human games),
without looking any further ahead; if the GRP is off, it biases the target directly. Two checks:

- **Calibration audit** (tpt 408k, 100k games, 4.23M "seat × hand start" samples): accurate overall (residuals ≲ 1.3 points in every bin), but with two systematic biases,
  of the same shape for our seats and for v4's seats: entering South 4 in 4th place, the actual outcome is 5–7 points worse than the GRP predicts; dropping below 8,000 in the East round, about 5 points worse.
  On a table of strong bots, placements are harder to turn around than on a human table.
- **Decision-level check**: the same rollouts computed both "rolled out to game end" and "truncated at hand end + GRP", paired point by point (47k games, 626k points):
  across all decisions the difference is **−0.000 ± 0.045** points per point, and "we push, v4 folds" +0.06 ± 0.66 per point — **no truncation bias is visible at the decision level**;
  the gap pricing above stands, and the problem does not lie in the reward definition.

## Mahjax line: a clean rerun after fixing two bugs, and locating the gap (2026-09-11)

With both defects fixed — the constant wall RNG and the per-seat GAE reset at episode boundaries — we reran from the uncontaminated BC base g186 (100k -4.66) with **hyperparameters identical, character for character, to the contaminated line**
(qc_clean), to decide whether "the bugs were the culprit behind the plateau":

| Snapshot | Steps | 12k avg_pt vs v4 | vs base -4.66 |
|---|---|---|---|
| qc_clean b64 | 514M | -3.03 | +1.63 (z=1.96) |
| qc_clean b384 | 849M | -3.38 ± 1.30 | +1.29 (z=1.77) |
| qc_clean6 | 1.398B | -3.95 ± 1.33 | +0.71 |
| Control: contaminated line b1408 | 1.477B | -4.04 ± 1.54 | +0.62 |

At equal compute, clean vs contaminated is **+0.09 ± 1.91**: both bugs are real, but they are **not the cause of the plateau**; beyond 500M steps, more steps buy nothing.

Gap localization (`jax_rl/mjai_bot/diag_gap.py`, pure parsing of the duplicate logs; the challenger rotates through all four seats, so seat and tile luck cancel; qc_clean6, 12k vs 3×v4):

| Metric | agent | v4 | Diff |
|---|---|---|---|
| Win rate | 20.98% | 21.72% | -0.75pp |
| Deal-in rate | 12.99% | 12.83% | +0.16pp |
| Riichi rate | 16.62% | 19.03% | **-2.41pp** |
| Call rate | 29.36% | 30.63% | -1.27pp |
| Average win value | 6,479 | 6,695 | -216 |

Net per-hand point-flow gap -109 points = **offense -95 + defense -13**. Defense is already level with v4; the gap is almost entirely offensive, and its shape is "too passive";
discard efficiency was measured long ago and has no headroom (shanten-optimal 96.4%). Compare the value line: rl1_best's riichi rate is 2.03pp **higher** than v4's and its deal-in rate 0.86pp higher,
and C' final pulled these back to +0.81 / +0.15pp — the two lines made opposite errors on the same axis (riichi / push-fold judgment), and the closer to v4, the stronger.
