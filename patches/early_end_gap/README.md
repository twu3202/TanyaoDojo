# Early-end gap: where tpt 408k's last 0.64 points go

Pre-registered 2026-10-21, after the [online transfer test](../online_transfer/README.md). The offline-base line is
stopped: q25's offline +0.55 did not survive the first online stage.

## What is known

- **tpt 408k is −0.64 ± 0.46 against v4 at 100k.** `diag_place` splits that into three parts:
  - +0.49 from the position entering South 4;
  - −0.86 from games that end before South 4 because someone is busted (22% of games);
  - −0.27 from converting the games that reach South 4.

  448k showed the same split on 09-14.
- **Every own model has the early-end deficit.** At 100k it is −0.64 for v11 + 80k online and −0.58 for q25 + 80k
  online. In those games we average −3 to −5 placement points per game below the v4 seats. Our bust rate is 6.95%
  against v4's 6.67%.
- **The EV map has never been sliced by score situation.** The game-horizon EV map of 408k (100k games, 1.32M
  disagreement points, `riichi-eval/runs/evmap_tpt408_game` on the 5060 Ti) prices every decision type: push against
  riichi −0.15, riichi judgement −0.18, ordinary tile choice 0.

## Step 1 (CPU, existing data): slice the EV map by the situation at the start of the kyoku

`riichi-eval/tools/ev_context.py` groups the decisions by information known before they were made:

- own score: < 8k, 8–16k, 16–30k, ≥ 30k;
- whether an opponent is below 8k;
- the round: East, South 1–3, South 4+;
- our rank.

The per-decision contributions telescope exactly to the realized gap, so each slice is an unbiased share of it. The
slicing never conditions on how the game ended. That would select on outcomes our decisions caused.

**Reading.** A slice carries part of the gap if its contribution is ≤ −0.15 pt per hanchan with z ≤ −2. The
decision families within it say what kind of decision is wrong.

## Step 2 (GPU, 5 hours): 100k fresh 408k games on a diagnostic key

`eeg_gen.sh` plays 408k against v4, duplicate, on seed key 20261021. These walls are independent of the sacred ones,
and the games never enter a record. It runs right after the throughput round, so the 5060 Ti does not sit idle.

- **Replication.** 408k's level and the early-end split (`diag_place`) on independent walls.
- **Raw material.** The slices that Step 1 flags get a state-filtered EV map on these games. Pricing a subset of
  decisions exactly needs one more rollout per decision, from the next disagreement point of the same game
  (riichi-eval `evaluate(select=...)`, added 10-21).

## Then

- **If a slice carries the gap:** design an intervention for that situation. Accept it on the EV map of that slice
  first, then screen it against 408k with the sequential test
  ([sequential_eval](../sequential_eval/README.md), the screening rule from 2026-10-21).
- **If no slice does:** the early-end deficit is spread over many situations. Report that and stop.

## Step 1 result (2026-10-21): the East carries the gap, and near-bust decisions do not

This slices the 408k game-horizon EV map: 100k games, 1.32M disagreement points. The contributions sum to −0.643,
against the logged −0.637. Each row is pt per hanchan with a 95% CI over seed groups. The game horizon prices each
decision exactly to the end of the game; the GRP column prices the same rollouts at the end of the kyoku.

| decisions in kyoku that start … | per game | game horizon | kyoku horizon + GRP |
| --- | ---: | ---: | ---: |
| all | 13.23 | −0.64 ± 0.46 | −0.43 ± 0.40 |
| with someone below 8k, whole game | 2.00 | −0.06 ± 0.21 | −0.08 ± 0.13 |
| **in the East** | 6.87 | **−0.62 ± 0.44** | **−0.31 ± 0.24** |
| in South 1–3 | 4.80 | +0.17 ± 0.42 | +0.06 ± 0.24 |
| in South 4+ | 1.55 | −0.19 ± 0.23 | −0.18 ± 0.21 |
| **East, someone below 8k** | 0.47 | **−0.21 ± 0.13** | **−0.11 ± 0.06** |
| South 1–3, someone below 8k | 1.11 | +0.10 ± 0.17 | +0.01 ± 0.09 |
| South 4+, someone below 8k | 0.43 | +0.06 ± 0.08 | +0.02 ± 0.07 |
| East, nobody below 8k | 6.40 | −0.41 ± 0.44 | −0.20 ± 0.24 |
| East, we are not the dealer | 5.25 | −0.43 ± 0.47 | −0.30 ± 0.21 |
| East, we are the dealer | 1.62 | −0.19 ± 0.35 | −0.01 ± 0.14 |

1. **The study's starting hypothesis fails over the whole game.** It was that decisions made close to a bust cause
   the early-end deficit. Decisions in kyoku that start with someone below 8k add up to only −0.06 ± 0.21. The games
   that end early are mostly set up by decisions made before anyone is low.
2. **The East carries the gap.** It holds −0.62 ± 0.44 (z −2.75) of the −0.64. By the pre-set rule (≤ −0.15 with
   z ≤ −2), the East is the slice.
3. **Within the East, no decision family is significant on its own:**
   - push against riichi −0.20 ± 0.23;
   - ordinary tile choice −0.18 ± 0.51;
   - fold against riichi −0.11 ± 0.16;
   - calls −0.10 ± 0.35.
4. **Sharpest cell: the East with someone below 8k.** It costs −0.21 ± 0.13 (z −3.2) on the game horizon and
   −0.11 ± 0.06 (z −3.7) priced by GRP, from only 0.47 decisions per game. Its riichi decisions are −0.06 ± 0.05.
   The same situation in the South is not negative. This cell was picked after looking at about 40, so it needs fresh
   walls.
5. **No sign that GRP misprices the East at this size.** The East's game horizon minus its GRP valuation is
   −0.31 ± 0.41 (z −1.5). That would have pointed at the kyoku-end GRP that also shapes Mortal's training targets, but at
   this size the training target is not implicated.

The slicing tool is `riichi-eval/tools/ev_context.py`; the outputs are in `runs/evctx_tpt408_*` on the 5060 Ti.

## Step 2b (pre-registered 2026-10-21, before any fresh game is priced)

`eeg_evmap.sh` runs right after the generation. It prices the fresh 100k games (key 20261021) on the game horizon,
with the GRP valuation alongside, restricted to:

- **(a) the East with someone below 8k** (`--only-phase East --only-below 8000`), about 4 hours. It replicates if the
  contribution is ≤ −0.10 with z ≤ −2.
- **(b) the whole East** (`--only-phase East`), about a day and a half. It replicates if ≤ −0.30 with z ≤ −2.

**How a subset is priced.** Each decision in the subset also rolls out the next disagreement of its game, so every
delta in the subset is exact. On 16 games at CPU fp32, all 234 deltas matched the full map.

**Next.** With both maps, the decision families within the East are read on 200k games, and the intervention is
designed for whichever replicates. If neither does, Step 1's East reading was noise and the study reports that.

## Step 2 result so far (2026-10-22)

**Fresh walls replicate the level and the early-end split.** Generation took 18:08–23:10 on 10-21 (5.5 games/s).

| tpt 408k vs v4, 100k each | sacred walls (key 20260711) | fresh walls (key 20261021) |
| --- | ---: | ---: |
| avg_pt | −0.64 ± 0.46 | −0.70 ± 0.46 |
| position entering South 4 / converting it | +0.49 / −1.13 | +0.39 / −1.09 |
| games ending before South 4 (22%) | −0.86 | −0.80 |
| deal-in / riichi vs v4 | −0.17 / −0.84 pp | −0.14 / −0.86 pp |
| net per kyoku (attack / defence) | −18 (−15 / −3) | −16 (−15 / −1) |
| bust us / v4 | 6.95 / 6.67% | 6.98 / 6.71% |

**(a) The East with someone below 8k does not replicate.** The run took 10.9 hours, not the 4 estimated.

| East, someone below 8k (0.47 decisions per game) | game horizon | kyoku horizon + GRP |
| --- | ---: | ---: |
| Step 1, sacred walls (picked from about 40 cells) | −0.21 ± 0.13 | −0.11 ± 0.06 |
| **Step 2, fresh walls** | **−0.12 ± 0.13 (z −1.8)** | **+0.01 ± 0.06** |

- **The pre-set rule fails.** It needed ≤ −0.10 with z ≤ −2. The point estimate clears −0.10, but z is only −1.8.
- **The GRP valuation drops to zero**, from −0.11 (z −3.7). The cell's Step 1 reading was mostly selection.
- On the game horizon a smaller cost may remain, but it does not reach the bar.

**(b) The whole East does not replicate either.** The run took 10:03 on 10-22 to 06:56 on 10-24 (44.8 hours).

| decisions in East kyoku | game horizon | kyoku horizon + GRP |
| --- | ---: | ---: |
| Step 1, sacred walls | −0.62 ± 0.44 | −0.31 ± 0.24 |
| **Step 2, fresh walls** | **+0.14 ± 0.44** | **−0.06 ± 0.24** |
| both, 200k | −0.24 ± 0.31 | −0.18 ± 0.17 |

The pre-set rule needed ≤ −0.30 with z ≤ −2, and the fresh walls give the opposite sign. A game-horizon slice of half
the decisions swings by ±0.44 between two independent 100k sets, so Step 1's East reading was noise.

## Result (2026-10-25): no situation carries the early-end deficit; one decision family recurs

The study closes by its own rule: neither slice replicates, so the early-end deficit is spread over many situations.
The deficit itself is real and replicates: −0.86 and −0.80 in games that end before South 4, on two independent
100k sets.

One reading holds on both sets. It was named in advance as the next thing to read on 200k: the decision families
within the East.

| East, pt per hanchan | sacred | fresh | both, 200k |
| --- | ---: | ---: | ---: |
| **push against riichi**, game horizon | −0.20 ± 0.23 | −0.21 ± 0.23 | **−0.20 ± 0.16 (z −2.5)** |
| push against riichi, GRP | −0.05 ± 0.08 | −0.10 ± 0.08 | −0.08 ± 0.05 (z −2.8) |
| fold against riichi, game horizon | −0.11 ± 0.16 | +0.04 ± 0.16 | −0.03 ± 0.11 |
| riichi decisions, game horizon | −0.04 ± 0.23 | −0.08 ± 0.23 | −0.06 ± 0.16 |
| ordinary tile choice, game horizon | −0.18 ± 0.51 | +0.34 ± 0.51 | +0.08 ± 0.36 |

This is the push defect the 09-23 EV map already priced over the whole game: we push and v4 folds, −0.15 by GRP.
The new information is that on the game horizon it costs about −0.2 in the East alone. Three earlier interventions
aimed at it, and none lifted 408k: the one-step fold probe A1/A1b, the A3 fine-tune, and Stage 1's targeted
exploration. This study adds a location, not a new lever.
