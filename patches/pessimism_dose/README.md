# Pessimism dose: does a v11 base with a smaller conservative weight gain more in its first online stage?

Pre-registered 2026-11-10 23:50, after the Cal-QL arm collapsed (patches/calibrated_base, "Online result") and
before any step of this arm was trained. Queued on the 5060 Ti after the Cal-QL diagnostics. Chosen by me under the
user's 11-10 instruction to decide 5060 Ti work without asking.

## Why

- **What the Cal-QL collapse showed.** The order of actions in our offline Q is held by the conservative (CQL) term.
  - Plain CQL pushes untaken actions far down: in v11 the second-best action sits 2.5 below the greedy one and the
    rest 6 below. The greedy action sits at about the level the returns support (+0.17, against −0.01 after 8,000
    online batches).
  - Cal-QL instead lifted the data action 1.8 above that level. Online Monte Carlo training removed the lift within
    8,000 batches, and the policy came apart.
- **What the first online stage does.** It reorders a few decisions (1.6% by 8,000 batches, 1.8% by 80,000) where a
  taken action's returns fall below untaken values that sit 2-6 units down. That stage is the only one in this
  pipeline that gains reliably: +3.1 ± 2.1 over v11 by 8,000 batches on seeds 10000-10999, +0.68 at 100k by 80,000.
- **The question.** If the untaken actions sat closer to the greedy one, with no lift, could the first online stage
  reorder more of the decisions it gets wrong?
  - The control is plain CQL at the same recipe, so the only change is how far down the untaken actions sit.
  - The risk is the other side of the same coin: more decisions within reach of return noise (24 pt SD per decision
    against true gaps of about 1 pt, D2/D3).
  - M1 c0.3 compressed the anchor's gaps on the 408k line and came out −0.40 against 408k. That was a late stage,
    where nothing has gained since. This test is on the first stage, where the gains are.

## Design

- **Offline fine-tune.**
  - Start: v11 (`runs/symaug/v11_best.pth`, md5 2ae5581c, 1,220,000 steps), its optimizer state, LR at its 1e-5
    floor (checked: the Cal-QL fine-tune logged 1e-5 throughout).
  - 40,000 steps of v11's offline recipe (config = v11long's, as for Cal-QL), with one change: `[cql] min_q_weight`
    5 → 1. Plain CQL, no Cal-QL line (the chain checks both).
  - Snapshots at 1.23M, 1.24M, 1.25M and 1.26M.
- **Gate.** Both must hold, else the chain stops:
  - the p90 top-2 Q gap on 40 Houou 2026 files ≤ 0.7 × v11's (the large gaps really shrink; Cal-QL reached 0.45);
  - the greedy action's mean Q on 10 files ≤ v11's + 0.5 (no lift: Cal-QL's was +1.77).
- **Online stage.**
  - The XFER recipe from the new base, exactly as A80k: anchor λ 0.2 to the base, pool base, v4, base, base,
    80,000 batches, port 5603.
  - The config is checked identical to `xfer_v11`'s up to paths and port. The collapse guard is unchanged.
- **Code.**
  - `a1_chain.sh` (`runs/xfer_tools/a1_chain.sh`, log `a1_chain.log`, markers `A1_CHAIN_DONE` / `A1_CHAIN_STOPPED` /
    `A1_CHAIN_SKIPPED`, skip file `A1_SKIP`). It was generated from `cq_chain.sh`, with the Cal-QL parts removed and
    the gate replaced.
  - `gap_readout.py` and `q_levels.py` come from patches/calibrated_base.
  - `train.py` is the deployed 5df8e75e, whose only change from 67bce509 is the Cal-QL flag, off by default.

## Evaluation and verdict

- **Order.** 12k of A1_80k, then SPRT A1_80k − A80k: 0 vs +1, α 0.025, β 0.2, a look per 1,000 groups from seed
  10000, capped at 16,000 groups.
- **Promote:** base pessimism is a lever for the first stage. Next: the control's second stage from A1_80k against
  the pooled control, and a second dose (weight 2 or 0.5, whichever side the readouts point to).
- **Drop:** lowering the base's pessimism does not let the first online stage fix more. The offline conservative
  weight is set aside as a lever, in both directions of the handoff (Cal-QL and dose).
- **Collapse guard fires:** drop. The diagnostic evaluation as for Cal-QL (1,000 groups each, Q gaps and levels)
  is run.
- **Undecided at the cap:** the fixed 16k is reported.
- **Also reported:**
  - A1_80k against v4, A160k, C160k, U64k and 408k;
  - offline test games against v4;
  - mechanism readouts (call, win, deal-in, riichi, attack and defence, 4th place);
  - the online Q-gap, flip and Q-level readouts of A1_80k against A80k;
  - sample reuse.
- **Priors:** gate met ~60% (a weight change might move the large gaps slowly at a 1e-5 LR); promote ~5%. The base
  rate of single-variable screens since 10-19 is 0 of 24.

## Departures (R8)

| | Convention | Here | Why, and the cost |
| --- | --- | --- | --- |
| 1 | An offline base trains a full epoch at its own conservative weight | 40,000 fine-tune steps from v11 at weight 1 | Cal-QL moved the gaps within 10,000 steps at the same LR, so 40,000 should settle; the gate checks it. Cost: the base keeps v11's features |
| 2 | A matched control base (v11 plus 40,000 steps at weight 5) | None: the control is A80k from v11 | v11's LR is at its floor after a full epoch; the readouts show how far weight 1 moved the gaps and levels. Cost: a small confound with "40,000 more offline steps" |
| 3 | One dose per test | Weight 1 only (a 5× change) | Make a large effect first; a second dose only after a promote |
| 4 | Screens against the variant's own control | SPRT against A80k, a different start | As for Cal-QL: the question is whether base and stage together beat the plain first stage |

## Cost

About 1.7 hours offline and 9 hours online, then up to 3 hours of evaluation; about 14 GPU hours in all. The chain
marks the card BUSY, serves bookings at the arm start and at each evaluation call, and lends the card for a day after
it ends.

## Glossary

| 中文 | English | Meaning here |
| --- | --- | --- |
| 悲观程度 | pessimism | how far the untaken actions' Q sits below the greedy action's |
| 保守权重 | conservative weight (min_q_weight) | the CQL term's weight in the offline loss; v11 used 5 |
| 剂量 | dose | the value of the changed setting |
| 微调 | fine-tune | continued offline training from v11's state, at its 1e-5 LR floor |
| 新平衡 | new balance | where the fine-tune's gaps stop moving (within 10,000 steps in round one) |
| 抬升 | lift | the data action's Q pushed above the level its returns support (Cal-QL's failure mode) |
| 首段 | first online stage | the 80,000 online batches from an offline base (A80k from v11) |
| 序贯检验 | sequential test (SPRT) | stops the evaluation as soon as the evidence decides |

## Result of the first dose (11-11 02:07): gate not met, nothing went online

Weight 1 shrank the typical gap within 10,000 steps and then stopped moving. The large gaps barely moved, so the
pre-registered gate failed and the chain ended without an online stage. This is not a verdict on the question: the
arm was never built.

Q-gap readout on the 40 Houou 2026 files (25,411 decisions with at least two legal actions; top-2 gap in Q units):

| Model | Steps | Median | Riichi available | Call available | Discard-only | p90 | Agrees with the human | Greedy differs from v11 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v11 | 1,220,000 | 1.806 | 1.678 | 4.789 | 1.415 | 6.649 | 81.17% | — |
| weight 1 | 1,230,400 | 1.213 | 1.115 | 3.877 | 0.916 | 5.733 | 81.12% | 5.75% |
| weight 1 | 1,240,000 | 1.244 | 1.199 | 4.219 | 0.925 | 5.988 | 81.11% | 5.69% |
| weight 1 | 1,250,000 | 1.196 | 1.176 | 4.147 | 0.886 | 5.911 | 81.21% | 5.64% |
| weight 1, final | 1,260,000 | 1.294 (0.72×) | 1.190 | 3.862 (0.81×) | 0.982 (0.69×) | 5.858 (**0.88×**) | 81.14% | 5.85% |

Q levels on 10 files (mean Q of the greedy, human and second-best actions, and the mean of the rest):

| Model | Greedy | Human | Second | Rest | Calls: greedy / second / rest |
| --- | ---: | ---: | ---: | ---: | --- |
| v11 | +0.172 | −0.035 | −2.471 | −5.930 | +0.024 / −4.723 / −5.003 |
| weight 1, final | +0.113 | −0.042 | −2.042 | −5.225 | +0.027 / −3.903 / −4.161 |

- **Gate:** p90 ratio 0.881 (needed ≤ 0.7): not met. Greedy mean Q −0.059 against v11 (needed ≤ +0.5): met, so
  there was no lift.
- **Offline test games against v4** (the trainer's own two runs, 1,000 games each): −2.66 and −1.85. The base is
  not broken.
- **Reading.**
  - The conservative term pushes each action down in proportion to its softmax weight. Actions 4-6 units below the
    greedy one get almost no push at weight 5 either.
  - So lowering the weight stops a push those actions hardly feel. Nothing pulls them up except features they share
    with actions taken elsewhere.
  - The fine-tune found that new balance within 10,000 steps: the second-best action rose 0.43 and the rest 0.70.
    The large gaps v11 learned early in its epoch stay.

## Second round (pre-registered 11-11 02:20, before any step): does any fine-tune dose reach the gate?

**Question.** Can a fine-tune of v11 build the arm the first round asked for at all? Round one gave no verdict, so
the question is open.

**Design.**
- **Two offline fine-tunes from v11**, everything as round one except two things:
  - Doses: weight 0.2, then weight 0. At weight 0 the conservative term is off and the fine-tune is plain Monte
    Carlo regression on the human action.
  - Length: 20,000 steps each (1.22M → 1.24M), with a snapshot at 1.23M. Round one settled within 10,000 steps.
  - Runs `runs/v11a02` and `runs/v11a00`. Each config is checked to differ from v11long's only in the weight line.
- **Gate, unchanged, for each dose's final snapshot:** p90 ≤ 0.7× v11's on the 40 files, greedy mean Q ≤ v11's +
  0.5 on 10 files, and the trainer's test games against v4 not below −8.
- **Which base goes online:** the largest passing weight (0.2 before 0), so the smallest change from v11 that
  reaches the gate.
- **Online stage and evaluation:** exactly as round one (A80k's recipe, port 5603), with the same verdict rules.
  The arm is `v11a02` or `v11a00`.
- **Neither passes:** fine-tuning cannot build the arm. Only a base trained from scratch at a lower weight could
  (about 46 hours on the 5060 Ti for 1.22M steps, plus the stage). That is not run by default; it goes to the 11-16
  review.
- **Code:** `a2_chain.sh` (`runs/xfer_tools/a2_chain.sh`, log `a2_chain.log`, markers `A2_CHAIN_DONE` /
  `A2_CHAIN_STOPPED`, results `a2_*.txt`). It was generated from `a1_chain.sh`, with the offline step made a function
  run once per dose.
- **Priors:** weight 0.2 passes ~20% and weight 0 ~35% (a gate pass from either ~40%). Promote ~3%.
- **Cost:** about 1.7 hours offline. If a dose passes, 9 hours online and up to 3 hours of evaluation.

**Departures (R8), in addition to 1-4.**

| | Convention | Here | Why, and the cost |
| --- | --- | --- | --- |
| 5 | A second dose only after a promote (this README) | A second and third dose after a gate failure | Round one built no arm, so the question has no answer yet. Cost: three doses read on the same 40 files. The gate is unchanged and was fixed before round one |
| 6 | 40,000 fine-tune steps | 20,000 | Round one's gaps settled within 10,000 steps; the 1.23M snapshot shows whether these doses settle too. Cost: a slow drift would be missed |
| 7 | An offline base keeps a conservative term | Weight 0 removes it | It is the end of the dose range. The Cal-QL result says the action order rests on this term, so weight 0 may reorder decisions offline. Cost: a riskier base; the readouts report flips against v11 and the collapse guard is unchanged |

## Second-round offline result (11-11 03:59): weight 0.2 passes the gate by a hair and goes online; weight 0 loses the action order

| Model | Steps | Median | Riichi available | Call available | Discard-only | p90 | Agrees with the human | Greedy differs from v11 | Test games vs v4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v11 (weight 5) | 1,220,000 | 1.806 | 1.678 | 4.789 | 1.415 | 6.649 | 81.17% | — | |
| weight 1, final | 1,260,000 | 1.294 | 1.190 | 3.862 | 0.982 | 5.858 (0.88×) | 81.14% | 5.85% | −2.66, −1.85 |
| weight 0.2 | 1,230,400 | 0.605 | 0.579 | 2.792 | 0.425 | 4.769 | 80.80% | 6.14% | |
| weight 0.2, final | 1,240,000 | 0.525 (0.29×) | 0.508 | 2.810 (0.59×) | 0.363 (0.26×) | 4.618 (**0.695×**) | 80.48% | 7.23% | −4.77 |
| weight 0 | 1,230,400 | 0.046 | 0.074 | 0.278 | 0.032 | 0.431 | 67.84% | 28.46% | |
| weight 0, final | 1,240,000 | 0.030 | 0.046 | 0.152 | 0.022 | 0.241 (0.04×) | **66.40%** | **29.92%** | **−16.83** |

| Q levels, 10 files | Greedy | Human | Second | Rest | Calls: greedy / second / rest |
| --- | ---: | ---: | ---: | ---: | --- |
| v11 (weight 5) | +0.172 | −0.035 | −2.471 | −5.930 | +0.024 / −4.723 / −5.003 |
| weight 1 | +0.113 | −0.042 | −2.042 | −5.225 | +0.027 / −3.903 / −4.161 |
| weight 0.2 | +0.066 | −0.001 | −1.337 | −3.843 | +0.006 / −3.118 / −3.313 |
| weight 0 | −0.012 | −0.023 | −0.104 | −0.332 | −0.031 / −0.217 / −0.234 |

- **Gate.**
  - Weight 0.2 passes: p90 0.695 (≤ 0.7), greedy −0.106 against v11 (no lift), test games −4.77.
  - Weight 0 also meets the gap and lift rules, but fails the test-game rule (−16.83 < −8). Under the rule fixed
    before the run, the largest passing weight goes online: weight 0.2, arm `v11a02`, from 03:59.
- **The pass is by a hair.** p90 0.695 against a 0.7 line, and the trainer's 1,000 test games read −4.77 (SE about
  2.5) against round one's −2.66 / −1.85. The online result will say whether the base itself costs strength; the
  12k and the SPRT measure the arm, not this readout.
- **Reading.**
  - The dose response is steep near zero. The untaken actions sit at −5.9 (weight 5), −5.2 (1), −3.8 (0.2) and
    −0.3 (0).
  - **Weight 0 is the Cal-QL mechanism shown offline.** With no conservative term, Monte Carlo regression on the
    human action pulled every action to the return level through shared features within 10,000 steps.
    - The gaps vanished: median 0.03, against 1.8.
    - 30% of greedy choices changed, agreement with the human fell from 81% to 66%, and test games read −16.8.
    - So the action order in our offline Q is held by the conservative term itself, not only in the online stage.
  - Round one's "lowering the weight cannot lift the far actions" was wrong in degree. Shared features do lift them,
    but only when the push is nearly gone.

## Online result (11-11 14:09): dropped by SPRT at 5,000 groups

The weight-0.2 arm `v11a02` trained all 80,000 batches (03:59 → 12:55) without tripping the collapse guard. Its
trainer test games against v4 averaged −2.53 over 19 readings (lowest −5.85; A80k's own run averaged −0.70). Sample
reuse 1.76 (A80k 1.82).

**SPRT, A2_80k − A80k** (0 vs +1, α 0.025, β 0.2, a look per 1,000 groups from seed 10000):

| Look | Groups | Mean ± 95% | LLR (bounds −1.58, +3.47) |
| --- | ---: | ---: | ---: |
| 1 | 1,000 | +0.03 ± 3.00 | −0.20 |
| 2 | 2,000 | +0.97 ± 2.08 | +0.41 |
| 3 | 3,000 | +0.44 ± 1.72 | −0.08 |
| 4 | 5,000 | −0.37 ± 1.33 | **−1.88: stop, H0** |

**Pairings** (6,000 groups, seeds 10000-15999, the whole of what was played; GPU fp32):

| Pair | Groups | Mean ± 95% | z |
| --- | ---: | ---: | ---: |
| A2_80k vs v4 | 6,000 | −2.42 ± 0.94 | −5.05 |
| A2_80k − A80k | 6,000 | −0.58 ± 1.21 | −0.93 |
| A2_80k − A160k | 6,000 | −1.19 ± 1.21 | −1.92 |
| A2_80k − C160k | 6,000 | −0.71 ± 1.22 | −1.13 |
| A2_80k − U64k | 6,000 | −0.59 ± 1.25 | −0.92 |
| A2_80k − 408k | 6,000 | −1.54 ± 1.25 | −2.42 |
| 12k (s10000 + s14000): A2_80k vs v4 | 3,000 | −2.43 ± 1.33 | −3.56 |
| 12k: A2_80k − A80k | 3,000 | −1.02 ± 1.71 | −1.17 |

A80k reads −1.84 against v4 on the same 6,000 groups. By segment, A2_80k − A80k: +0.03 (10000-10999), +1.90
(11000-11999), −0.62 (12000-12999), −1.70 (13000-13999), −1.54 (14000-15999).

**Mechanism readouts** (challenger against the v4 seats; per kyoku; 6,000 groups is under R5's 10,000, so the
profile is suggestive):

| Model | Games | Call (v4) | Win | Deal-in | Riichi | Net (attack / defence) | Bust (v4) | 4th |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- | ---: |
| A80k | 100,000 | 30.79 (30.55) | −0.44 | +0.33 | +0.10 | −29 (−8 / −21) | 7.02 (6.63) | +1.42 |
| A2_80k | 24,000 | 29.82 (30.67) | −1.19 | +0.11 | −0.70 | −45 (−39 / −6) | 7.19 (6.62) | +1.56 |

**Online Q readouts** (gap readout on the 40 Houou 2026 files, Q levels on 10, as offline):

| Model | Median gap | Riichi | Call | Discard-only | p90 | Agrees with the human | Greedy differs from A80k | Mean Q: greedy / human / second / rest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v11 (offline) | 1.806 | 1.678 | 4.789 | 1.415 | 6.649 | 81.17% | | +0.172 / −0.035 / −2.471 / −5.930 |
| A80k | 1.547 | 1.520 | 4.734 | 1.120 | 6.677 | 81.11% | — | +0.014 / −0.155 / −2.496 / −5.926 |
| weight 0.2 (offline) | 0.525 | 0.508 | 2.810 | 0.363 | 4.618 | 80.48% | | +0.066 / −0.001 / −1.337 / −3.843 |
| A2_80k | 0.455 | 0.437 | 2.953 | 0.290 | 4.698 | 80.23% | 7.63% | −0.005 / −0.063 / −1.387 / −3.864 |

**Verdict: drop**, by the rule fixed before the run. Lowering the base's pessimism does not let the first online
stage fix more. The offline conservative weight is set aside as a lever, in both directions of the handoff: lifting
the data action (Cal-QL) collapses, and pushing the untaken actions less (this test) is no better and probably worse.
The prior for a promote was ~3%; the base rate of single-variable screens since 10-19 is now 0 of 25.

**Exploratory, not pre-registered (11-11 14:52): base or stage?** A 12k of the weight-0.2 base itself
(`a2_diag.sh`), paired on the 12k seeds (s10000 + s14000, 3,000 groups) with v11's existing 12k from the Cal-QL
diagnostics and with both 80k snapshots:

| Pair, 3,000 groups | Mean ± 95% | z |
| --- | ---: | ---: |
| v11 vs v4 | −4.12 ± 1.33 | −6.07 |
| weight-0.2 base vs v4 | −3.39 ± 1.35 | −4.90 |
| A80k vs v4 | −1.41 ± 1.32 | −2.10 |
| A2_80k vs v4 | −2.43 ± 1.33 | −3.56 |
| weight-0.2 base − v11 | +0.73 ± 1.75 | +0.82 |
| stage gain, control: A80k − v11 | +2.71 ± 1.27 | +4.17 |
| stage gain, dose: A2_80k − weight-0.2 base | +0.96 ± 1.56 | +1.21 |
| difference of the stage gains | **−1.75 ± 1.99** | −1.72 |

| 12k profile (per kyoku, against the v4 seats) | Call (v4) | Win | Deal-in | Riichi | Net (attack / defence) | 4th |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| v11 | 30.49 (30.62) | −0.83 | +0.75 | +0.65 | −73 (−27 / −46) | +2.93 |
| weight-0.2 base | 31.67 (30.68) | −1.06 | +0.63 | −0.77 | −66 (−35 / −31) | +2.47 |
| A80k | 30.99 (30.56) | −0.47 | +0.23 | +0.16 | −23 (−6 / −17) | +1.11 |
| A2_80k | 29.85 (30.61) | −1.23 | +0.04 | −0.64 | −45 (−46 / +1) | +1.38 |

The 12k seeds read harder than the whole set (v11 −4.12 here, −2.50 at 100k), and 12k overstates stage gains (the
control's is +2.71 here, +0.68 at 100k). The difference of the two stage gains is the reading that matters, and it is
within noise at 3,000 groups.

**Reading.**
- **The lower pessimism survived the online stage.** The online stage left the gaps and the untaken actions'
  levels where the fine-tune put them (median 0.525 → 0.455, rest −3.84 → −3.86). Online Monte Carlo training only
  moves the actions that get taken, so nothing pushes the untaken ones back down.
- **The base was not the cost; the stage gained less.** The fine-tuned base reads no weaker than v11 (+0.73 ± 1.75).
  The first stage gained less from it than from v11, by 1.75 ± 1.99 (z −1.7). The point estimate runs against the
  hypothesis: from a less pessimistic base the first stage fixes less, not more. It is suggestive only.
- **The passive riichi came with the fine-tune.** Offline, riichi fell from +0.65 to −0.77 pp against v4 and calls
  rose 1.2 pp, with 7.2% of greedy choices changed from v11. This is why A2_80k differs from A80k in 7.6% of decisions
  while A80k's stage moved only 1.8% of v11's.
- **Both stages taught defence; only the control's also taught attack** (R5: suggestive). Defence improved by about
  30 points per kyoku in both (−46 → −17, −31 → +1). Attack improved by 21 from v11 (−27 → −6) and fell by 11 from
  the weight-0.2 base (−35 → −46). From v11, which over-riichis (+0.65), the stage moved riichi toward v4's rate
  (+0.16) and wins rose 0.36 pp. From the weight-0.2 base, which already under-riichis (−0.77), riichi barely moved
  (−0.64) and wins fell 0.17 pp. Defence is what the first stage learns from either start. It added attack only where
  the base's riichi errors were on the side the stage corrects.
- The Mahjax line's plateau had the same shape (riichi −2.4 pp, attack −95 / defence −13). Across lines, the
  further a model sits from v4 on the riichi and push-fold axis, the weaker it is, whichever side it errs on.
