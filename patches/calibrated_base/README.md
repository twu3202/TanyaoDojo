# Calibrated offline base (Cal-QL): does a base whose Q is not pessimized on untaken actions transfer better online?

Pre-registered 2026-11-10, while the model-soup chain was running and before any Cal-QL step was trained. Queued
on the 5060 Ti after the soup chain. Chosen by me under the user's 11-10 instruction to decide 5060 Ti work without
asking.

- **If the soups win** (patches/model_soup), I create `runs/xfer_tools/CQ_SKIP` and the card goes to "K draws + soup"
  instead.
- **Otherwise** this runs by itself.

## Why

- **The Q gaps are overstated.** Our models' Q gaps between actions are 10-30 times their rollout-measured size (D2,
  10-03). The error sits on the untaken side, +49 pt, and dates from the offline base (Stage 0, 09-30).
- **The offline loss explains it.** Mortal's offline conservative term (CQL, min_q_weight 5) pushes every untaken
  action's Q down until a temperature-1 softmax of Q matches the behaviour policy. A gap then encodes how rarely
  humans chose an action, not how much worse it is. The online loss is Monte Carlo on taken actions only, and the
  anchor holds the untaken values, so these gaps never close: pessimistic lock-in.
- **This file measured it.** v11's median gap between its best and second-best legal action is 1.87 Q units,
  4.8 where a call is available. On the [6,4,2,0] reward that is about a placement step.
- **The literature says it matters.** Reincarnating RL (2206.01626) found that CQL-pretrained discrete agents
  fine-tune worse online the higher the CQL weight. Cal-QL (Nakamoto et al. 2023, 2303.05479) fixes this by never
  pushing an action's Q below the behaviour policy's value. This was on the 10-19 list of offline-to-online ideas
  still to test.
- **What is new here.** M1 (10-03) compressed the gaps at the anchor of the 408k line and came out at −0.1 against
  its plain continuation. It did not change the base itself, and it did not test the first online stage, which is
  where this pipeline gains most (+0.68).

## Design

- **Offline fine-tune.**
  - Start: v11 (`runs/symaug/v11_best.pth`, md5 2ae5581c, 1,220,000 steps), with its optimizer state and LR at
    its 1e-5 floor.
  - 40,000 steps of v11's offline recipe (config = v11long's, which has the same data and recipe), with one change:
    `[cql] calibrated = true`. The conservative term becomes logsumexp(max(Q, G)) − Q(data action), where G is the
    data's Monte Carlo return (Cal-QL with the logged return as the behaviour value).
  - Snapshots at 1.23M, 1.24M, 1.25M and 1.26M.
- **Gate: did the calibration happen?**
  - The median top-2 Q gap on 40 Houou 2026 files (about 24,000 decisions; `gap_readout.py`) must fall to at most
    0.5 × v11's.
  - If not, the chain stops: Cal-QL does not calibrate our Q, and the online stage would test nothing new.
- **Online stage.**
  - The XFER recipe from the calibrated base: anchor λ 0.2 to the base, pool base, v4, base, base, 80,000 batches.
  - The config is checked identical to A80k's (`xfer_v11`) up to paths and port.
- **Code.**
  - `train.py` (md5 5df8e75e) adds the flag, off by default: the deployed build differs from 67bce509 only in the
    flag and the Cal-QL branch (`calql.patch`, the only file shipped for it). The chain unit-checks that no gradient reaches actions whose Q is
    below the return.
  - `gap_readout.py`, `cq_chain.sh` (`runs/xfer_tools/cq_chain.sh`, log `cq_chain.log`, markers `CQ_CHAIN_DONE` /
    `CQ_CHAIN_STOPPED` / `CQ_CHAIN_SKIPPED`, results `cq_gap_offline.txt`, `cq_seq.txt`, `cq_final.txt`,
    `cq_mech.txt`, `cq_gap_online.txt`).

## Evaluation and verdict

- **Order.** 12k of CQ80k, then SPRT CQ80k − A80k: 0 vs +1, α 0.025, β 0.2, a look per 1,000 groups from seed 10000,
  capped at 16,000 groups.
- **Promote:** the calibrated base goes on. Its second stage gets the control recipe, compared with the pooled
  control.
- **Drop:** pessimistic lock-in from the offline base is not what holds the v11 line back. The offline-to-online
  handoff closes as a lever.
- **Undecided at the cap:** the fixed 16k is reported.
- **Also reported:**
  - CQ80k against v4, A160k, C160k, U64k and 408k;
  - the offline trainer's test games against v4 (1,000 games at 1.24M and 1.26M);
  - mechanism readouts (call, win, deal-in, riichi, attack and defence, 4th place);
  - the online Q-gap and flip readout of CQ80k against A80k;
  - sample reuse.
- **Priors:** gate met ~75%; promote ~7%. The XFER test saw a better offline base (q25, +0.55) end level with v11
  after one online stage, so base changes tend to wash out.

## Departures (R8)

| | Convention | Here | Why, and the cost |
| --- | --- | --- | --- |
| 1 | CQL as Mortal ships it | Cal-QL bound on the conservative term | The variable under test |
| 2 | An offline base trains a full epoch from scratch | 40,000 fine-tune steps from v11 | Calibration acts on the Q head's scale and should settle in thousands of steps; the gate checks it happened. A from-scratch base costs about 46 hours. Cost: the base keeps v11's features |
| 3 | A matched control base (v11 plus 40,000 plain steps) | None: the online control is A80k from v11 itself | v11's LR is already at its floor after a full epoch, so 40,000 more plain steps should change little; the gap readout shows how much Cal-QL moved. Cost: a small confound between "Cal-QL" and "40,000 more offline steps" |
| 4 | Cal-QL estimates the behaviour value V(s) | The logged return G of the data action, one sample | Unbiased for V(s) when actions follow the behaviour policy. It is noisy (about 24 pt per decision), so the bound is soft: untaken actions settle a little below V |
| 5 | Screens: SPRT against the variant's own control | Against A80k, a different start (v11 vs the calibrated base) | The question is whether base and stage together beat the plain pipeline's first stage |

## Addendum (11-10 20:40, before the gate result): a large-gap gate if the median gate fails

Written after one look at the data: a 10-file CPU readout of the 1.23M snapshot (10,000 of the 40,000 steps).

| | median top-2 gap | riichi available | call available | discard-only | p90 | agrees with human | greedy differs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v11 | 1.868 | 1.644 | 4.837 | 1.429 | 6.561 | 81.48% | |
| Cal-QL 1.23M | 1.608 (0.86×) | 1.429 | 2.319 (0.48×) | 1.339 (0.94×) | 3.211 (0.49×) | 81.09% | 5.44% |

- **What it shows.** The large gaps have already halved, both where a call is available and in the 90th percentile.
  The median is three-quarters discard-only decisions, and those have barely moved. That fits departure 4: with a
  one-sample return as the behaviour value, untaken actions settle about one noise width below V, so near-equal
  discards keep a gap of that size. The median gate was a poor measure of pessimistic lock-in, and I should have
  seen that when I wrote it.
- **The change.** If the median gate fails at 1.26M on the 40-file readout, the chain is relaunched with a
  large-gap gate: the call-available median gap and the p90 gap must both be ≤ 0.5 × v11's. If they are, the online
  stage runs. If not, the chain stops as registered. (`cq_cont.sh` waits for "gate not met" and relaunches
  `cq_chain.sh` with `CQ_GATE=large`. The chain resumes: it skips the offline step and recomputes the readout.)
- **Why this does not bias the verdict.** The gate only saves compute. The online result is screened by the same
  SPRT against A80k either way, and nothing about that test depends on why the arm was run.
- **What changes in the reading.** A drop would then mean that halving the base's large gaps does not help the first
  online stage. It would not say that a fully calibrated base fails. The prior for promote stays about 5–7%.
- Departure 6: the gate is changed after one look at partial data. The cost is a forking path on the
  compute decision only, stated here.

## Offline result and gate (11-10 21:31)

40 Houou 2026 files, 25,411 decisions with at least two legal actions; GPU.

| | median top-2 gap | riichi available | call available | discard-only | p90 | agrees with human | greedy differs from v11 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v11 (1.22M) | 1.806 | 1.678 | 4.789 | 1.415 | 6.649 | 81.17% | |
| 1.23M | 1.583 | 1.510 | 2.361 | 1.348 | 3.245 | 80.90% | 5.36% |
| 1.24M | 1.571 | 1.563 | 2.137 | 1.374 | 3.041 | 81.06% | 5.67% |
| 1.25M | 1.557 | 1.690 | 2.116 | 1.342 | 2.961 | 80.90% | 5.88% |
| **1.26M (final)** | **1.586 (0.88×)** | 1.595 | **2.071 (0.43×)** | 1.385 (0.98×) | **2.964 (0.45×)** | 81.07% | 5.72% |

- **The median gate failed** (0.878, needs ≤ 0.5). **The large-gap gate from the addendum passed**: the
  call-available median is 0.43× and the p90 is 0.45×. The chain was relaunched at 21:31 and goes to the online stage.
- Cal-QL acted within the first 10,000 steps and then plateaued. Where a call is available the gap more than
  halved; on discard-only decisions it did not move. Agreement with the logged human action is unchanged (81.1%), and
  5.7% of greedy choices differ from v11's.
- The offline trainer's own test games against v4 (1,000 games each) read −1.8 and +0.99, so the base is not broken.

## Online result (11-10 22:45): the arm collapsed within 8,000 batches and was stopped; the lever closes

The collapse guard stopped the arm. It needs two test_play readings against v4 below −8 in a row, and the run gave
**−12.94 at 4,000 batches and −17.72 at 8,000**. A80k's own run read +0.51 and −0.62 at the same points. With no 80k
snapshot there is no SPRT; the verdict is **drop**.

**Diagnostic evaluation** (evaluation only, 23:38): 1,000 groups each on seeds 10000-10999, sacred protocol,
GPU fp32. "A8k" is A80k's own run at 8,000 batches.

| | Groups | Difference | z |
| --- | ---: | ---: | ---: |
| CQ8k vs v4 | 1,000 | −13.05 ± 2.56 | −10.00 |
| A8k vs v4 | 1,000 | −3.43 ± 2.31 | −2.92 |
| CQ base vs v4 | 1,000 | −4.13 ± 2.31 | −3.50 |
| v11 vs v4 | 1,000 | −6.51 ± 2.32 | −5.51 |
| **CQ8k − A8k** | 1,000 | **−9.62 ± 3.36** | −5.60 |
| CQ8k − CQ base | 1,000 | −8.92 ± 3.34 | −5.24 |
| A8k − v11 | 1,000 | +3.08 ± 2.13 | +2.84 |
| CQ base − v11 | 1,000 | +2.39 ± 2.95 | +1.59 |

**Mechanism readouts** (4,000 games each; relative to the v4 seats; call rate with v4's in brackets)

| Model | Call | Win | Deal-in | Riichi | Net (attack / defence) | 4th |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| v11 | 30.45 (30.56) | −1.10 | +0.96 | +0.64 | −102 (−41 / −61) | +4.80 |
| CQ base | 31.62 (30.46) | −0.30 | +0.65 | +0.78 | −61 (−4 / −57) | +3.40 |
| A8k | 30.49 (30.59) | −0.88 | +0.40 | −0.21 | −77 (−37 / −40) | +2.23 |
| **CQ8k** | 35.03 (30.88) | **−4.21** | +0.60 | **−4.87** | **−316 (−293 / −23)** | **+6.93** |
| A80k | 30.98 (30.49) | −0.45 | +0.24 | −0.09 | −13 (+3 / −16) | +1.90 |

**Q gaps and Q levels on logged Houou decisions** (gaps on 40 files, `gap_readout.py`; levels on 10 files,
`q_levels.py`)

| Model | median top-2 gap (call available) | p90 | agrees with human | greedy differs from v11 | mean Q: greedy / human / second / rest |
| --- | --- | ---: | ---: | ---: | --- |
| v11 | 1.806 (4.789) | 6.649 | 81.17% | | +0.17 / −0.04 / −2.47 / −5.93 |
| CQ base | 1.586 (2.071) | 2.964 | 81.07% | 5.72% | **+1.94** / +1.76 / +0.34 / −1.01 |
| A8k | 1.611 (4.873) | 6.763 | 81.12% | 1.55% | −0.01 / −0.19 / −2.57 / −5.98 |
| **CQ8k** | **0.138** (0.762) | 1.213 | **58.14%** | **43.95%** | **+0.15** / +0.08 / −0.21 / −1.08 |
| A80k | 1.547 (4.734) | 6.677 | 81.11% | 1.80% | |

**What happened**

- **Offline, Cal-QL did two things.** It lifted the untaken actions by about 5 units (rest −5.9 → −1.0, second-best
  −2.5 → +0.3), as intended. It also pushed the greedy, data-like action about 1.8 above the level the returns support
  (+0.17 → +1.94).
  - The second effect comes from the conservative term's `−Q(data action)`. In plain CQL the logsumexp balances it
    because every untaken action is pushed down. Under Cal-QL the untaken actions sit below the return and carry no
    gradient, so the data action is pushed up until its own softmax weight balances the push. For about 10 legal
    actions and min_q_weight 5 that is roughly +1.6 above the return.
- **Online, the Monte Carlo loss pulls taken actions to the return level** (about 0 here). It does this in both lines:
  v11's greedy +0.17 → −0.01 by 8,000 batches, the Cal-QL base's +1.94 → +0.15. The untaken actions do not move
  (anchored, and never taken).
  - In v11 the greedy action lands 2.5 above the second-best. In the Cal-QL base it lands on it (+0.15 vs −0.21).
    Gaps collapse (median 1.59 → 0.14), choices flip (44% of decisions differ from v11; agreement with the human
    81% → 58%), and the newly chosen actions are pulled to the return in turn.
  - The policy unravels: win −4.2 pp, riichi −4.9 pp, attack −293 pt per kyoku, 4th +6.9 pp.
- **The reading.** The order of actions in our offline Q is held by the conservative term: by pushing untaken actions
  far down (plain CQL), or by lifting the data action (Cal-QL). Monte Carlo returns alone cannot hold it, because the
  true gaps are about 1 pt against a 24 pt return SD (D2/D3, 10-03). Pessimistic lock-in is what lets plain CQL
  survive the first online stage: the online loss can only reorder actions where a taken action's return falls below
  untaken values that sit 2-5 units down.
  - This matches M1 c0.05 (10-03), which compressed the anchor's gaps and also collapsed.
- **Verdict as pre-registered: drop.** Pessimistic lock-in from the offline base is not something to remove under this
  recipe: removing it removes the policy. The offline-to-online handoff closes as a lever.
- **Exploratory: the Cal-QL base offline.** It read +2.39 ± 2.95 over v11 on the first 1,000 groups. With 2,000
  more groups (seeds 14000-15999, 3,000 in all, 11-11 00:30): CQ base vs v4 −3.48 ± 1.33, v11 −4.12 ± 1.33, **CQ base −
  v11 +0.63 ± 1.70** (z 0.73). Offline, Cal-QL is about level with v11.
- Limits:
  - A Cal-QL variant with a much smaller conservative weight would not lift the data action as far. But its order
    would be held even less, so I do not expect a different outcome.
  - The arm ran 8,000 of 80,000 batches. The Cal-QL paper reports recovery after an initial dip. Here the guard
    stopped a run at −17.7 against v4, with its Q gaps at 0.14, so recovery would mean relearning the policy from
    Monte Carlo returns alone.

## Cost

About 1.7 hours offline and 9 hours online, then up to 3 hours of evaluation; about 14 GPU hours in all. The chain
marks the card BUSY, serves bookings at the arm start and at each evaluation call, and lends the card for a day after
it ends.

## Glossary

| 中文 | English | Meaning here |
| --- | --- | --- |
| 保守 Q 学习 | conservative Q-learning (CQL) | the offline loss term that pushes untaken actions' Q down |
| 校准的保守 Q 学习 | calibrated CQL (Cal-QL) | the same term, but never below the behaviour policy's value |
| 悲观锁死 | pessimistic lock-in | untaken actions' Q stuck so low that online updates on taken actions cannot flip a choice |
| 行为策略 | behaviour policy | the policy that produced the data (Houou humans) |
| 蒙特卡洛回报 | Monte Carlo return | the realized kyoku reward, the training target |
| 前二差 | top-2 gap | best minus second-best legal action's Q |
| 离线基座 | offline base | the model trained on human logs before online RL |
| 序贯检验 | sequential test (SPRT) | stops the evaluation as soon as the evidence decides |
| 崩塌哨兵 | collapse guard | stops an online run after two test_play readings below −8 against v4 in a row |
| 数据动作 | data action | the action logged in the human game record |
| 贪心动作 | greedy action | the action with the highest Q, the one the model plays |
| 在线遗忘 | unlearning (offline-to-online dip) | the drop in strength when an offline base starts online training |
