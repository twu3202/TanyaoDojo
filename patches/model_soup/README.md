# Model soups: does averaging the weights of independent training draws beat the draws themselves?

Pre-registered 2026-11-10, before any soup game was played. Evaluation only, on the 5060 Ti. Chosen by me under the
user's 11-10 instruction to decide 5060 Ti work without asking; this file records the reasoning and the departures.

## Why

- Every online variant since 10-19 has failed its screen; the four-seat probe (patches/four_seat) was the latest.
  The control recipe's own gain is real (+0.33 over A80k pooled, reproduced twice), and so is its draw-to-draw noise.
- The diagnostics say our online updates are noise-dominated: the Monte Carlo target's SD is 24 pt per decision
  against true action gaps of about 1 pt (D2/D3, 10-03), and the policy jitters in place (structural operator (a)).
- If a stage's change is "shared improvement + independent noise", a uniform weight average of several draws from
  the same start (a model soup, Wortsman et al. 2022; WARM/WARP for RLHF policies) keeps the shared part and shrinks
  the noise. It costs nothing at play time.
- We already own the ingredients: five 160k arms trained from A80k and four 448k continuations of tpt 408k.

## The weight-space check (11-10, before the games)

`soup.py` reports each draw's change from its start (delta) and how the deltas align:

| | ingredients | |delta| / |start| | mean pairwise cosine | |mean delta| / mean |delta| |
| --- | --- | --- | --- | --- |
| v11 line, start A80k | A160k, C160k, F160k, R160k, FS160k | 0.0106-0.0128 | +0.39 | 0.71 (independent: 0.45) |
| 408k line, start tpt 408k | historical 448k, Stage 1 448k, M1 c0.3 448k, M1-sel 448k | 0.0050-0.0069 | +0.15 | 0.60 (independent: 0.50) |

So the draws share a direction but most of each delta is their own: the v11 soup removes about half of the
independent part, the 408k soup most of it. Whether that independent part costs strength is what the games decide.

## Design

- **SV** (`runs/xfer_soupv`, tag `xfer_soupv_s160000`, md5 16bf32b1): the uniform average of A160k, C160k, F160k,
  R160k and FS160k. Floating tensors are averaged, BatchNorm running statistics included; integer buffers come from
  the first ingredient. 16,000 groups, the pooled control's seeds (10000-25999).
- **ST** (`runs/xfer_soupt`, tag `xfer_soupt_s448000`, md5 81292e9e): the uniform average of the four 40,000-batch
  continuations of tpt 408k. 25,000 groups, the 100k protocol (seeds 10000-34999), so it can stand as a record.
- Sacred protocol unchanged: duplicate 1v3 against mortal_v4, seed_key 20260711, GPU fp32, the same engine and
  `xfer_eval.sh` as every evaluation since 10-21.
- Guard: each soup's first call is 1,000 groups; under -8 against v4 the soup is called broken and not evaluated
  further.
- Code: `soup.py` (builds the soups, md5 aeb49d12), `soup_pair.py` (paired readouts), `soup_chain.sh`
  (`runs/xfer_tools/soup_chain.sh`, log `soup_chain.log`, markers `SOUP_CHAIN_DONE` / `SOUP_CHAIN_STOPPED`, results
  `soup_final.txt`, `soup_mech.txt`).

## Evaluation and verdict

| Question | Primary readout | Yes | No |
| --- | --- | --- | --- |
| Does a soup beat the recipe on the v11 line? | SV - pooled control (mean of A160k and C160k per group), 16,000 groups | >= +0.5 and z >= 2 | <= 0 |
| Is ST a new own-pipeline best? | ST - tpt 408k, 25,000 groups (100k) | z >= 2 (about +0.5) | otherwise |
| Does averaging recover the draw noise? | soup - mean of its ingredients' outcomes per group: SV on the 5,000 groups all five have, ST on 25,000 | both >= +0.3 | both <= 0 |

- Between "yes" and "no" on the first question: reported as a lean; the third question then decides.
- **If averaging helps:** next on the card is "K draws + soup" per stage (two draws per stage at first: the 5060 Ti,
  and trx when it is free at low priority), starting from the better soup.
- **If ST is a record:** it becomes the start for further stages, and its 100k against v4 goes into RESULTS.
- **If neither:** the soup lever closes. The next 5060 Ti job is chosen and pre-registered separately.
- Also reported: each soup against v4, against each ingredient, against A80k / 408k / C' final; mechanism readouts
  (call rate, win, deal-in, riichi, attack and defence per kyoku, bust, 4th place) on all games played.
- Priors: first question ~10%, ST record ~5%, averaging helps on both lines ~20%.

## Departures (R8)

| | Convention | Here | Why, and the cost |
| --- | --- | --- | --- |
| 1 | Evaluate trained snapshots | Evaluate weight averages that were never trained | The variable under test; no training compute |
| 2 | Screens use SPRT 0 vs +1 per 1,000 groups | Fixed 16,000 and 25,000 groups | The effect in question is +0.3-0.5. SPRT would stop near 3,000 groups at H0 without telling 0 from +0.4. Cost: 8 GPU hours instead of about 2 |
| 3 | BatchNorm statistics come from training | Averaged with the weights, not recomputed | Recomputing needs a pass over fresh self-play data. The ingredients are within ~1% of their start in weight norm, so the averaged statistics should fit; a broken soup would show at the first 1,000 groups |
| 4 | Soups of identical recipes | Ingredients include variants (freshness, reward, four-seat; M1, M1-sel, Stage 1) | These are the draws we have; training more control draws costs 9 h each. The v11 primary is against the pooled control, which is conservative for the soup: three of the five ingredients read below it |
| 5 | One machine per comparison | ST's ingredients were trained on two machines and M1-sel was evaluated on trx (GPU fp32, 98% identical games to the 5060 Ti) | Pairings mix the two machines' GPU fp32 dumps, as the 100k comparisons of October did |

## Cost

About 8 GPU hours: SV 64,000 games (3.1 h), ST 100,000 games (4.8 h), at 5.78 games/s. The chain marks the card
BUSY, serves bookings at each evaluation call (at most ~60 min apart), and lends the card for a day after it ends.

## Glossary

| 中文 | English | Meaning here |
| --- | --- | --- |
| 模型汤 | model soup | the uniform average of the weights of several models fine-tuned from one start |
| 权重平均 | weight averaging | averaging parameters, not predictions |
| 训练抽签 | training draw | one run of a recipe; runs differ by self-play randomness |
| 增量 | delta | an ingredient's weights minus its start's |
| 余弦相似度 | cosine similarity | the alignment of two deltas; 0 = independent, 1 = the same direction |
| 合并对照 | pooled control | the mean of the two control runs (A160k, C160k) per seed group |
| 批归一化 | BatchNorm | the network's normalization layers, whose running statistics are averaged too |
| 配对 | paired | compared on the same seed groups |

## Result (11-10 19:50): neither soup beats the recipe; the soup lever closes

Both soups played their full pre-registered sizes (the guard passed: SV −1.56, ST +0.16 against v4 on the first
1,000 groups). Paired, GPU fp32, the sacred protocol.

**v11 line, SV (five 160k arms from A80k), 16,000 groups**

| SV minus | Groups | Difference | z |
| --- | ---: | ---: | ---: |
| v4 | 16,000 | −1.94 ± 0.57 | −6.68 |
| **pooled control (A160k, C160k), primary** | 16,000 | **−0.21 ± 0.40** | −1.05 |
| mean of its 5 ingredients | 5,000 | −0.31 ± 0.59 | −1.02 |
| A160k / C160k | 16,000 | −0.17 ± 0.48 / −0.26 ± 0.49 | −0.67 / −1.06 |
| F160k / R160k / FS160k | 6,000 / 6,000 / 5,000 | −0.29 / −0.45 / +0.03 | −0.75 / −1.12 / +0.07 |
| A80k (the start) | 16,000 | +0.11 ± 0.44 | +0.51 |
| tpt 408k | 16,000 | −1.41 ± 0.74 | −3.71 |

The mean of the five ingredients reads −0.13 ± 0.39 against the pooled control (5,000 groups).

**408k line, ST (four 448k continuations of tpt 408k), 25,000 groups (100k games)**

| ST minus | Groups | Difference | z |
| --- | ---: | ---: | ---: |
| v4 | 25,000 | −0.68 ± 0.46 | −2.88 |
| **tpt 408k, primary** | 25,000 | **−0.04 ± 0.45** | −0.17 |
| mean of its 4 ingredients | 25,000 | +0.26 ± 0.34 | +1.47 |
| historical 448k / Stage 1 448k | 25,000 | +0.26 ± 0.46 / +0.09 ± 0.45 | +1.10 / +0.39 |
| M1 c0.3 448k / M1-sel 448k | 25,000 | +0.36 ± 0.51 / +0.32 ± 0.47 | +1.38 / +1.35 |
| C′ final | 16,000 | +0.54 ± 0.57 | +1.87 |

The mean of the four ingredients reads −0.30 ± 0.39 against 408k (25,000 groups). For comparison, 408k itself reads
−0.64 ± 0.46 against v4 on the same 100k walls.

**Verdicts, as pre-registered**

| Question | Readout | Verdict |
| --- | --- | --- |
| Does a soup beat the recipe on the v11 line? | SV − pooled control −0.21 ± 0.40 | **No** (≤ 0) |
| Is ST a new own-pipeline best? | ST − 408k −0.04 ± 0.45 | **No** |
| Does averaging recover the draw noise? | SV −0.31 ± 0.59, ST +0.26 ± 0.34 | **Neither** (opposite signs) |

So the soup lever closes, and the card goes to the calibrated base (patches/calibrated_base), which was queued for
this outcome.

**Mechanism readouts** (all games played; relative to the v4 seats in the same games: call rate (v4's), win / deal-in
/ riichi pp, attack / defence pt per kyoku, bust, 4th place pp)

| Model | Games | Call | Win | Deal-in | Riichi | Net (attack / defence) | 4th |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: |
| A160k | 64,000 | 30.34 (30.53) | −0.55 | +0.20 | +0.09 | −27 (−14 / −13) | +1.20 |
| C160k | 64,000 | 30.36 (30.54) | −0.53 | +0.33 | −0.19 | −35 (−14 / −21) | +1.15 |
| **SV** | 64,000 | 30.76 (30.50) | −0.43 | +0.33 | −0.05 | −32 (−11 / −21) | +1.33 |
| tpt 408k | 100,000 | 30.61 (30.49) | −0.12 | −0.17 | −0.84 | −18 (−15 / −3) | +0.68 |
| historical 448k | 100,000 | 27.32 (30.55) | −0.72 | −0.22 | +0.00 | −20 (−24 / +3) | +1.05 |
| Stage 1 448k | 100,000 | 30.79 (30.48) | +0.03 | +0.05 | −0.30 | −22 (−6 / −16) | +0.93 |
| M1 c0.3 448k | 100,000 | 26.15 (30.55) | −0.85 | −0.21 | +0.22 | −23 (−29 / +6) | +1.06 |
| M1-sel 448k | 100,000 | 30.16 (30.52) | −0.06 | −0.25 | −1.24 | −28 (−29 / +1) | +0.74 |
| **ST** | 100,000 | 28.85 (30.52) | −0.26 | −0.15 | +0.22 | −10 (−7 / −3) | +0.84 |

**Reading**

- **408k line:** every continuation drifted below its start (their mean −0.30), each in its own direction (call rate
  26–31%, riichi −1.24 to +0.22 pp). The soup cancels the drift and lands back on 408k (−0.04), with a profile between
  its ingredients. That fits "continuations = 408k + noise that costs about 0.3": averaging removes the noise but finds
  no shared improvement, because 40,000 batches from 408k had none to find.
- **v11 line:** here the stage does gain (A80k → pooled control +0.33), but the soup keeps only +0.11 of it and reads
  below both control runs. So the gain is not a shared direction plus removable noise. Each run's gain seems tied to
  its own weights, and averaging partly cancels it. SV plays like the arms, deal-in included (+0.33 pp, 4th +1.33 pp).
- **Limits:**
  - Three of SV's five ingredients are variants that read below the control. A soup of control draws only was not
    tested, since each would cost 9 GPU hours. The 408k soup had no such excuse and still found nothing above its
    start.
  - The BatchNorm statistics were averaged, not recomputed (departure 3). The guard and the profiles show no sign of
    a broken soup.
