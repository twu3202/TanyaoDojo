# Data quality: train the offline base on better games only (2026-10-11, pre-registration)

## Why

More data has not helped the offline base, and worse data hurt it:

| attempt | what was added | result |
|---|---|---|
| v18 | Tenhou houou 2009–2018 (data doubled) | ≈ 0 |
| A3 step21 learning curve | 25 → 100% of labels | quantity is not the bottleneck |
| symaug | 18 exact symmetries of every log | 100k +0.004 ± 0.597 |
| mjs (patches/majsoul_data) | 1.34M Mahjong Soul games, 93% jade | 12k −1.38 ± 1.77, stopped; more calls, +2 pp deal-in |

The user's question: is selecting the data more important than adding it? Proposal: score every game, keep the
better ones, and train only on those.

**The user's hypothesis.** Games where the scores stay close are higher quality. Games that end lopsided (three high
and one low, or one high and three low) are lower quality.

## Two concerns with filtering on the score itself

1. **Luck.** A lopsided finish can come from one dealer mangan run or one bad deal-in. That says little about how
   well the four players played. If close vs lopsided is mostly luck, filtering on it selects luck, not skill.
2. **Outcome-conditioned targets.** Mortal learns its Q values from each game's final placement. Keeping only games
   that ended close removes, by construction, games where a lead held or a deficit stayed. The base would then learn
   that leads evaporate and deficits recover. The filter would also remove most big-lead / big-deficit situations
   from training, although they occur in every evaluation.

So a filter based on a game's own result has to show first that it picks better **players**, measured on decisions
made **before** the result exists.

## Candidate metrics (`jax_rl/mjai_bot/game_metrics.py`, one pass over all 2,512,424 houou logs, no model)

- **M1, closeness** (the hypothesis), from the game's own scores:
  - final spread (1st − 4th);
  - patterns: 1st − 2nd (one high), 3rd − 4th (one low);
  - mean and max spread at kyoku starts;
  - busted.
- **M2, player strength** (free of the game's own outcome): for each player name, the average placement over all of
  their **other** games in the pool, shrunk toward 2.5 by an empirical-Bayes prior. A game's score is the mean of
  its four players.
- **Era.** The year, for reference: v18 said old years add nothing, and v11 uses 2019–2026.

## Yardstick (`jax_rl/mjai_bot/quality_sample.py`)

Agreement between the human action and mortal_v4's greedy action on non-forced decisions, all four seats. This is
the measure that ordered houou > throne > jade (82.1 / 79.7 / 76.5) before mjs.
- **e1 (primary).** Decisions in East 1, honba 0. Every game starts this kyoku at 25,000 each, so a difference
  between groups of games is a difference in the players, not in the situation, and it is measured before the game's
  result exists.
- **bal (secondary).** Kyoku that start with a spread < 10,000, excluding the last kyoku.
- **all.** Every decision; this measure mixes in the situation.

**Sample.** 12,000 games drawn uniformly at random from the pool, seed 20261011. One inference pass, one row per
game, so every metric is checked against the same numbers.

v4 is a yardstick for choosing the filter only. It is never a training signal: the arm's targets stay the human
games' own outcomes.

## Pre-registered rules (fixed before the sample is run)

1. **Is it quality?** A metric qualifies if the e1 agreement of its best quartile minus its worst quartile is
   ≥ 1.0 pp, with a game-clustered 95% CI that excludes 0. The hypothesis is tested as M1 = final spread (close vs
   lopsided). M1 patterns and M2 are tested the same way.
2. **Which metric selects.**
   - If M2 qualifies, M2 selects: it carries no outcome conditioning.
   - If only M1 qualifies, M1 selects, and the outcome-conditioning caveat goes with the result.
   - If neither qualifies, the study stops here and reports the table.
3. **Selection.**
   - Pool: all houou 2009–2026.
   - Keep the 1,289,661 best games by the selecting metric. That is v11's count, so the amount of data is equal
     and only which games are used changes.
4. **Is the difference big enough to train?**
   - Measure the selected set's e1 agreement minus v11's set (2019–2026) on the same sample.
   - Train only if this is ≥ +1.0 pp.
   - Calibration from mjs (the arm's training mix was about −2.5 pp in agreement, −1.4 pt at 12k) puts +1 pp
     near +0.5 pt. That is the smallest effect a 100k evaluation resolves (CI ±0.6). Below it, the arm could not
     answer, so it is not run.
5. **The arm.**
   - v11's recipe item for item: 192×40, pts [6, 4, 2, 0], CQL 5, batch 512, cosine 1e-4 → 1e-5 over 1.21M steps,
     stop at 1.22M.
   - Only the file list differs.
   - On the 5060 Ti after v11long.
   - Gates as B1: 12k paired with v11 < 0 stops; otherwise 100k, and ≥ +0.5 with z ≥ 2 passes.
   - Mechanism readouts as always: deal-in, calls, riichi, attack / defence, placements.

Logs, the per-game table (it holds player names) and every derived list stay on our machines (research/ is not
committed).

## v2 (2026-10-11, before any yardstick result): stakes-weighted regret replaces East-1 agreement

The user's objection: East 1 is mostly tile efficiency, while the decisions that move placement, and so the reward,
come late in the game. Measuring only East 1 controls the situation but looks where skill matters least. v2 keeps the
situation control and puts the weight where the stakes are. The v1 run (East-1 agreement on the 5060 Ti) was stopped
after 1,000 games, unread; it was also slowing training. v2 runs on the Mac (same libriichi encoding: 20 test games
give identical decision counts and 13,098 / 13,099 identical greedy actions).

**Yardstick v2** (`jax_rl/mjai_bot/quality_regret.py`). For each non-forced decision and each judge:
- **regret** = Q_judge(best legal action) − Q_judge(human action).
  - It is about 0 for near-ties (most efficiency choices).
  - It is large where the choice matters (push / fold, placement fights).
  - So it weights decisions by their stakes without hand-picking kyoku.
- **excess regret** = regret − the mean regret of the decision's situation stratum. The stratum is phase (E1 … S4,
  West) × all-last × the player's rank at kyoku start × the gap to the player above × the gap to the player below
  (buckets 0 / < 4k / < 8k / < 16k / ≥ 16k) × decision kind. Stratum means come from the houou sample.
  - This takes out the situation mix. A lopsided game has different late-game situations from a close one, and
    those must not count as worse play.
- **Judges.** mortal_v4 (strongest available), and C' λ0.2 (`cprime_final_0724_0939.pth`, our own best). A result
  counts only if it holds under both, so no conclusion rests on v4 alone. Judges are never a training signal.
- **Sample.**
  - The same 12,000 random houou games.
  - Plus 1,000 throne and 1,000 jade games (random, seed 20261011) for calibration.
  - Calibration compares kyoku without a kan and not all-last, for every source, because repaired jade logs lose
    those.

**Calibration.** J = jade − houou mean excess regret. In mjs, about half the training decisions came from jade, and
the arm lost ~1.4 pt at 12k. So moving a training set by J/2 ≈ 1.4 pt, and J/5 ≈ 0.5 pt, the smallest effect 100k
resolves. Rough, but it fixes the scale before looking.

**Rules v2** (replace rules 1 and 4; rules 2, 3 and 5 stand):
1. **Is it quality?**
   - Worst quartile minus best quartile in mean excess regret ≥ J/3, with a game-clustered 95% CI excluding 0.
   - It must hold under both judges.
   - Late-game decisions (S3, S4, all-last) are reported separately as a secondary view.
   - The hypothesis is tested as M1 final spread and as the M1 patterns (one high / one low). M2 is tested the same
     way. M3 (per-player mean excess regret over their other sample games) is exploratory. Players appear only a
     few times in 12,000 games, so M3 here only estimates how much players differ. If they differ enough, a
     player-balanced run to score most players is the next step, and its cost goes to the user first.
4. **Is the difference big enough to train?** The selected set minus v11's set in mean excess regret must be
   ≤ −J/5 under both judges.

Primary numbers are all decisions with excess regret. v1's East-1 agreement is still reported as a secondary
measure.

## v3 (2026-10-11, before any result): what the AI says to avoid

The user relayed a professional player's view of AI like ours: what the AI recommends is not necessarily the best
move, but what it says to avoid must not be played. The judges' top choice is unreliable among near-equal moves,
but a clear "avoid" is reliable. v2 regret still charges a human for picking a different reasonable move. v3 builds
the asymmetry in. The v2 run was stopped at 600 games, unread, and restarted with two extra fields per judge: its
greedy action, and `xother` = Q_j(own best) − Q_j(the other judge's best).

**Tolerance band.**
- δ_j = the 95th percentile of `xother_j` over houou decisions where the two judges disagree.
- This is how much judge j thinks a strong player's different choice can cost. Anything within δ_j of the best is
  "a move a strong player would make", not an error.

**Per-decision variants** (each also stratum-adjusted, as in v2):
- `reg_j`: v2 regret, for comparison;
- `tol_j` = max(0, regret_j − δ_j): only the loss beyond the band;
- `blunder` = [regret_v4 > δ_v4] and [regret_cp > δ_cp]: both judges independently say avoid;
- `sev` = blunder × tol_v4: how bad such a consensus blunder is.

**Which variant becomes primary (fixed now, decided by data, not by taste).** The variant that best predicts real
point loss.
- Unit: one seat in one kyoku, houou only.
- Predictor: the seat-kyoku's summed excess value.
- Outcome: its point change in that kyoku (next kyoku's start, or the final score for the last kyoku), minus the
  mean change of seat-kyokus in the same stratum (phase × all-last × rank × gap buckets).
- The variant with the most negative z wins: more error, more points lost.
- This directly tests the pro's claim that "avoid" moves really lose.
- Secondary: whether a seat's East-round value predicts its South-round point change (does the measure describe the
  player, beyond one kyoku).

The rules of v2 then apply unchanged, with J recomputed for the primary variant (and per judge where the variant is
per judge). The consensus variants need both judges by construction.

## v4 (2026-10-11, before any result): beyond AI-vs-AI disagreement

The user's point: v4 and C' share one AI logic, so their disagreement measures how much two AIs differ, not how
much strong humans and AI differ. The pro's remark was about the latter: on certain tiles, humans and AI choose
differently, and the AI is not always right.

The first look already shows the gap. On 46 test games:
- v4 / C' disagree on 9.7% of decisions, and v4 / the human-school judge on 8.6%;
- humans differ from every judge on about 22%.

AI-vs-AI spread is less than half of human-vs-AI spread, so a band built from it treats much of human style as
error. Two extensions; the v3 run was restarted at 15:23 with both, unread.

**1. A third, human-school judge (`hu` = v11_best, md5 2ae5581c).** The offline base trained only on houou games
with v11's recipe.
- The consensus blunder now needs all three judges: v4, C' and hu each find the human move worse than their own
  best by more than their band.
- The band δ_j = the 95th percentile of judge j's cost of the other judges' choices, where they disagree.
- hu learned from strong humans, so a move it is happy with is a human-school move and is not counted.

**2. Disagreement types, arbitrated by outcomes.**
- Every decision where the human differs from the AI consensus (v4 and C' agree on the best action) is typed by
  (human action class, AI action class) × game phase (East, South 1–2, South 3–4 / all-last / West). Classes:
  discard, riichi, chi, pon, kan, agari, abortive draw, pass. Examples: riichi vs AI dama, call vs AI pass, a
  different discard.
- For each type, the analysis reports:
  - how often it occurs;
  - the human deviation rate where the AI chose that class (a type that strong humans do often is a school of
    play, not a slip);
  - the judges' regret;
  - **the realised cost**: a regression of each seat-kyoku's stratum-adjusted point change on its counts of each
    type. The coefficient is the point change linked to one such deviation.
- Types whose realised cost is ≥ 0 or not distinguishable from 0 are treated as style. Their band is unlimited and
  they are not counted.
- Types with a significantly negative cost (z ≤ −2) are confirmed "avoid".
- The outcome regression is confounded (deviations happen in particular hands). The decisive check is a real-wall
  rollout (riichi-eval). From a sample of each type's decision points, play the human's action and the AI's
  action on the same wall, continue both with one policy, and compare. That needs GPU time and comes after the
  5060 Ti is free; it is proposed, not part of this run.

**Primary metric v4.** The v3 rule picks the primary variant among reg, tol, blunder (now three judges) and sev,
and **a fifth: `typed`**. `typed` = tol_v4 counted only on deviations of confirmed-avoid types (consensus
decisions only). The rest of the rules stand.

## v5 (2026-10-11, before any result): decisions carry their own weight

The user's point: decisions are not equally important.
- Opening discards (which wind to cut first) have an efficiency answer. Even houou players do not bother with it,
  because it barely moves the game.
- The first call, the riichi decision and push / fold against a riichi matter a lot.
- When one player has run away from the other three, that leader's discards matter much less.

v5 weighs both, without hand-set numbers.

**Decision context** (`jax_rl/mjai_bot/quality_context.py`, no model, same decisions in the same order as the
scoring run):
- turn number and shanten, from the loader;
- opponents in riichi, and the player's melds so far, by replaying the log with libriichi's PlayerState. The replay
  is aligned to the loader by legal-action signature; on 46 test games every row matches and 0.09% do not align.
- Categories:
  - `open`: discard, turn ≤ 6, no riichi on the table, shanten ≥ 2;
  - `first_call`: a call decision with no meld yet;
  - `call`: a later call decision;
  - `riichi`: the riichi decision;
  - `defence`: discard with an opponent in riichi;
  - `agari`;
  - `mid`: other discards;
  - `other`.

**Placement leverage** (`jax_rl/mjai_bot/placement_leverage.py`, all 2.51M games, no model).
- In each situation stratum (phase × all-last × rank × gap above × gap below), the slope of the final pt
  ([90, 45, 0, −135]) on the points won or lost in that kyoku.
- A runaway leader's kyoku barely moves its final pt; a close all-last fight moves it a lot.
- w_place = the stratum's leverage / the mean leverage.

**Category weight, from outcomes.**
- Regress each seat-kyoku's pt-relevant loss on its summed tol_v4 per category, on even games. The pt-relevant loss
  is the stratum-adjusted point change × that stratum's leverage.
- w_cat = max(0, −coefficient) / its mean. A category whose AI-judged error does not cost real pt gets weight 0;
  the user's expectation is that `open` will.

**New variants:**
- `wtol` = w_place × w_cat × tol_v4;
- `wsev` = w_place × w_cat × sev (three-judge consensus blunders only).

**Validation outcome for choosing the primary variant.** It becomes the pt-relevant loss, i.e. leverage × adjusted
point change, for every variant: the reward is final pt, not points. The raw-points version is still reported. Odd
games are used, since categories and types are fitted on even ones. The rest of the rules stand.

## Interim (2026-10-11 22:20): 5,000 houou games, 3.12M decisions, calibration still running

The Mac slept 17:13–21:18, so at the planned check only shards 0–9 of each worker were done. The calibration games
(throne, jade) sit at the end of each list and were moved to run next. Everything below is on houou only; the
rule-1 / rule-4 thresholds wait for J.

**v5.1 amendment, made after seeing these interim numbers (flagged as such).** As first fitted, the type and category
regressions compared seat-kyokus *with* a decision of some kind against all others. A riichi decision only exists in
a tenpai hand, so "riichi errors" looked profitable, and the same held for agari. Both regressions now control for the
opportunity:
- types: counts of AI-consensus decisions per (AI class, phase group);
- categories: decision counts per category.
A type's coefficient then compares deviating with following the AI in the same kind of spot. The numbers below are
v5.1; v5 is kept in `research/.../interim/regret_v5_interim_5000.txt`.

**Human-vs-AI disagreement types** (points per deviation in that kyoku, even games, opportunity-controlled):

| type | deviations | human deviation rate | cost | z | reading |
|---|---|---|---|---|---|
| dama where the AI riichis (E / S1–2 / S3–4) | 5,670 | 15–19% | −650 / −837 / −1,059 | −6.9 / −5.5 / −6.1 | AI right |
| pon / chi where the AI passes | 20,255 | 1.4–2.6% | −204 … −474 | −2.0 … −6.7 | AI right |
| a different discard (E / S1–2 / S3–4) | 431,913 | ~20% | −3 / −4 / +19 | −0.4 / −0.4 / +1.6 | style: no cost |
| pass where the AI pons / chis | 10,656 | 10–14% | +237 … +661 | +2.1 … +5.1 | humans did better |
| riichi where the AI stays dama, S3–4 | 673 | 0.2% | +1,002 | +3.8 | humans did better |

- 92% of human-vs-AI disagreements are a different discard, and they cost nothing measurable in the kyoku. This
  matches the professional's remark: on tile choice, the AI's preference is not the answer.
- Where the AI says "avoid", the data agree: calling when the AI passes, and staying dama when the AI riichis, both
  lose points.
- Declining the AI's calls looks better than following them. That would mean the AI over-calls, but it needs the
  rollout check: the outcome regression can still be confounded by hand quality.

**Category weights** (pt-relevant loss per unit of tol_v4, opportunity-controlled):
- open 0, mid 0, first_call 0, call 0;
- defence 2.2 (z −4.7);
- riichi 2.8 (z −2.3);
- agari 44 (z −8.2; small regrets, large cost).

Opening discards get weight 0, as the user expected. The first call does not: once the call decision is controlled
for, AI-judged error on it does not predict extra loss.

**Primary variant.** Plain regret (`reg`) predicts kyoku pt loss best (v4 z −28, C' z −30). The tolerance and
weighted variants score lower (−17 … −23).

Caveat found here: this criterion also rewards metrics that track a hard hand, where any play loses points. The
cleaner within-game check (a seat's East-round value vs its South-round change) is weak for every variant
(|z| ≤ 2.2). Over many games, regret clearly tracks the player: see M2.

**Rule 1 on the primary variant** (worst minus best quartile of mean excess regret):
- **M1, closeness (the user's hypothesis): no difference.**
  - Final spread: v4 +0.0003 ± 0.0036, C' −0.0004 ± 0.0014.
  - The one-high / one-low patterns and the mean in-game spread are also at 0.
  - On late-game decisions, close games have lower *raw* regret (v4 −0.011 ± 0.006), but after the situation
    adjustment the difference is +0.0003 ± 0.006. Lopsided games have different late-game situations, not worse
    play.
- **M2, player strength: a clear difference.**
  - v4 +0.034 ± 0.003 (z +19), C' +0.008 ± 0.001 (z +11).
  - Late game: z +11 / +5.
  - East-1 agreement +1.4 ± 0.5 pp.
  - Whether this clears J/3 waits for calibration.
- **M3, per player.** The between-player SD of true mean excess regret is ≈ 0.052 (v4) / 0.015 (C'), from 2,333
  players with ≥ 3 seats. Players differ more than M2's quartiles separate them.

## Calibrated reading (2026-10-11 23:05): 5,000 houou + 1,000 throne + 1,000 jade games

**Calibration** (excess regret, kyoku without a kan and not all-last):

| | v4 | C' |
|---|---|---|
| throne − houou | +0.012 ± 0.004 | +0.001 ± 0.002 |
| **jade − houou = J** | **+0.090 ± 0.004** | **+0.021 ± 0.002** |

- Throne sits close to houou and jade far below, the same order as the agreement probe and the mjs result.
- Thresholds:
  - rule 1, J/3: 0.030 (v4) / 0.0071 (C');
  - rule 4, J/5: 0.018 (v4) / 0.0042 (C').

**Rule 1.**
- **M1, closeness: fails** on every form. Final spread, one-high, one-low and mean in-game spread all differ by
  ≈ 0 under both judges.
- **M2, player strength: passes.** +0.034 ≥ 0.030 for v4 and +0.0079 ≥ 0.0071 for C', both CIs excluding 0.
- M2-weakest fails (v4 +0.021).

**Rule 4 (M2 selects).** The 1,289,661 games with the best M2, minus v11's set:
- v4 −0.0032 ± 0.0026 (needed ≤ −0.018);
- C' −0.0006 ± 0.0010 (needed ≤ −0.0042).

**Fails.** M2 does find better players, but a placement record is too noisy a label (between-player SD 0.026 ranks)
for a half-of-the-pool selection to move the training set. The shift is about 1/28 of J, roughly 0.1 pt by the mjs
calibration, so **the M2 arm is not trained**, as pre-registered.

**M3, per-player regret (exploratory, rule 1 v2).**
- Players differ a lot: the between-player SD of true mean excess regret is ≈ 0.052 (v4), 58% of J.
- Projection, not a result:
  - With every player scored on about 25 games (reliability ≈ 0.9), the best half of games by their four players'
    regret would sit about 0.02 (v4) better than the pool average.
  - That is right at the rule-4 line (0.018), so about +0.5 pt by the calibration: borderline.
- A smaller, stricter selection would separate more but use fewer distinct games. For example, the best quarter
  would be about 0.033 better. It needs its own pre-registration, because rule 3 fixes the count.
- Either way the next step is the player-balanced scoring run. Its cost goes to the user first.

The 12,000-game sample continues (shards 10–23); final numbers will replace these when it is done. With these
margins they cannot reverse the rule-4 verdict for M2.

## v6 (2026-10-11 23:35, before the player run): select by each player's own regret (M3)

The user offered trx, which is free, for the player-balanced scoring.

**Player-balanced sample.**
- Pool: all houou games except the 12,000-game random sample, which stays held out for evaluation.
- Eligible: player names with ≥ 25 games in that pool.
- Games are drawn in a fixed random order (seed 20261011) until every eligible player has 25 scored seats.
  - First pass: games where ≥ 3 of the four players still need seats.
  - Then ≥ 2, then ≥ 1.
- Same three judges, same scoring code (`quality_regret.py`), on trx CUDA. The Mac (MPS) agreed with CUDA on
  13,098 / 13,099 greedy actions.

**Player score.**
- Excess v4 regret (`v4_reg`, the primary variant) uses the stratum means of the 12,000-game sample.
- A player's score is the mean over their seats in the player sample, shrunk toward 0 with an empirical-Bayes
  prior k = seat variance / between-player variance.
- A game's score is the mean of its four players' scores. Unscored players count as 0, the population mean.
- C' is not used to score players. It stays an independent check of the selection.

**Rules.**
- **Rule 1 (M3).** On the held-out 12,000 games, the worst quartile minus the best quartile of game score must be
  ≥ J/3 under both judges (0.030 / 0.0071), with a CI excluding 0.
- **Rule 4 (M3).**
  - Select the best 1,289,661 games of the full pool by game score (rule 3: v11's count).
  - On the held-out sample, the selected set minus v11's set must be ≤ −J/5 under both judges (−0.018 / −0.0042).
  - Train only if both hold.
- **Reported only.** The best 25% of the pool (628k games), trained for the same 1.21M steps. It trades distinct
  games for quality; data quantity has not been the bottleneck (v18, step21). Running it would need its own
  decision.

**Arm, if rule 4 passes.** v11's recipe item for item, only the file list changes, gates as B1. It runs on the 5060 Ti
after v11long, or on trx if the user prefers.


## Final (2026-10-12): 12,000-game sample complete; M3 verdict

**Sample.** All 12,000 houou + 2,000 Mahjong Soul games are scored (8,730,949 decisions). Shards 0–12 and 24–27 were
scored on the Mac (MPS), 13–23 on trx (CUDA). The two devices agree on 13,098 / 13,099 greedy actions. Final numbers
(`regret_v51_final.txt`) confirm the interim reading:

| | v4 | C' |
|---|---|---|
| J = jade − houou | +0.094 ± 0.004 | +0.022 ± 0.002 |
| throne − houou | +0.016 ± 0.004 | +0.003 ± 0.001 |
| M1 final spread, worst − best quartile | +0.0002 ± 0.0024 | −0.0002 ± 0.0009 |
| M1 one-low / one-high / mean spread | all within ±0.002, z ≤ 1.3 | all within ±0.001 |
| M2 player strength, worst − best quartile | **+0.037 ± 0.002 (z 31)**, passes J/3 | **+0.0088 ± 0.0009**, passes |
| M2 rule 4: best 1,289,661 − v11's set | −0.001 ± 0.002 (needs ≤ −0.019), fails | +0.000 ± 0.001, fails |

**The user's hypothesis (late-game decisions, close vs lopsided).**
- Raw regret: −0.009 ± 0.004.
- Situation-adjusted: +0.002 ± 0.004.

Lopsided games have different late-game situations, not worse play.

**Disagreement types and category weights** keep their interim shape, with tighter errors.

AI right, the human deviation costs points:
- dama where the AI riichis: −618 / −842 / −902 points (z −7 … −10);
- calling where the AI passes: −212 … −431 (z −3 … −11);
- skipping the AI's kan: −464 … −831.

Different discard choice, 1.03M deviations, ≈ 0:
- E −9 ± 5, S1–2 +0.5 ± 8, S3–4 +26 ± 8.

Humans did better than the AI:
- passing where the AI calls: +233 … +535 (z +3 … +9);
- riichi where the AI stays dama, South: +466 / +759.

Category weights:
- open / mid / first_call / call: 0;
- defence 2.3 (z −8.7);
- riichi 2.2 (z −3.1);
- agari 44 (z −12).

**M3, players scored by their own regret** (`quality_player_select.py`, `m3_result.txt`):
- Player sample: 130,955 games, 523,820 seats, 12,514 players.
- Seat SD 0.077; between-player SD 0.067, about 0.7 J. Reliability at 25 seats: 0.95.
- **Rule 1 passes by a wide margin.** Worst minus best quartile of game score on the held-out 12,000 games:
  - v4 +0.079 ± 0.002 (z 81, 0.84 J);
  - C' +0.022 ± 0.001 (z 55, 1.0 J).
  - The worst quarter of houou games, by who is playing, is about as far from the best quarter as jade is from
    houou.
- **Rule 4 fails narrowly.** Best 1,289,661 games minus v11's set:
  - v4 −0.0139 ± 0.0015 (needs ≤ −0.0188);
  - C' −0.0040 ± 0.0006 (needs ≤ −0.0045).
  - **As pre-registered, the M3 arm with v11's count is not trained.**
- **Best 25%, reported only.** Minus v11's set:
  - v4 −0.0286 ± 0.0017 (0.30 J);
  - C' −0.0081 ± 0.0007 (0.36 J).
  - Both are well past the J/5 line. By the rough mjs calibration that is on the order of +0.8–1.0 pt.
  - It holds 628,106 games, 73.5% of them from 2019–2026. Training it for v11's 1.21M steps means about twice
    as many passes over half as many distinct games.
  - Whether to run it is the user's decision.

**Caveats for any arm built on this.**
- The judges are AIs, and recent players (who learned from AI tools) agree with them more. Part of "low regret" may
  be AI-likeness rather than skill. C' and the human-school judge agree with v4 at r ≈ 0.84 per game, but all three
  are AIs.
- The calibration from mjs is one-directional. Worse data hurt by about J/2 ≈ 1.4 pt; that better data helps
  symmetrically is the assumption the arm would test.

## v7 (2026-10-12): the best-25% arm (q25), user decision

After the final numbers, the user chose to try the best 25%. It is not the pre-registered rule-3 arm, which failed
rule 4 narrowly. It is the reported-only selection that passed J/5 under both judges, run by explicit decision.

- **Data.** `select_m3_q25.txt`: 628,106 houou games 2009–2026 with the best mean player score. 461,758 of them are
  in v11's list. The file index is generated from the list and sorted in reverse, as v11's.
- **Recipe.** v11 item for item: 192×40, pts [6, 4, 2, 0], CQL 5, batch 512, cosine 1e-4 → 1e-5 over 1.21M steps,
  stop at 1.22M. The only change is the file list (`configs/local_train_q25.toml`). One pass over 628k games was
  taken to be about 590k steps, so `num_epochs = 3` and the run is cut at 1.22M: about two passes over half as many
  distinct games as v11. *(Corrected 2026-10-16: a sample scan gives 658.7 samples per game, so one pass is about
  808k steps and q25 made about 1.5 passes. The 590k figure assumed v11's 1.21M steps were one pass; they are about
  0.74 of one. See v9.)*
- **Code.** Mortal_symaug with `symmetry_augmentation = false`, i.e. the original `load_gz_log_files` path that v11
  used.
- **Gates (as B1).** 12k paired with v11 < 0 stops; otherwise 100k, and q25 − v11 ≥ +0.5 with z ≥ 2 passes. Mechanism
  readouts with every eval. `q25_chain.sh` is the mjs chain with the tag changed.
- **Where.** On the 5060 Ti, started automatically by `q25_after_v11long.sh` once the v11long evaluation chain is
  done, so the GPU is never shared.
- **Where it actually ran (2026-10-13).** The trx GPU came free first, so the run started there at 04:32
  (`q25_trx_wait.sh`: waits for the GPU and for another session's lock files; `q25_chain_trx.sh` is the same chain on
  the evaluation station), with the 5060 Ti handover held by `HOLD_TRX`. At 11:34 trx was needed back and the run was
  stopped at 322,000 steps; the stop landed in the middle of a checkpoint write and truncated `mortal.pth`, so the run
  continues on the 5060 Ti from trx's `best.pth` (100,000 steps, full optimizer state), after the v11long chain. As
  with every other restart, the data epoch is reshuffled; the recipe and the gates are unchanged.

What a pass would mean: selecting who is playing, by their own regret, moves an offline base where adding data,
capacity or symmetry did not. A fail with these margins would say the regret-based quality is not what limits the
base, or that it is partly AI-likeness.

## v8 (2026-10-13, before any rollout): pricing the disagreement types by rollout

The type costs above come from a regression: how many points the human's seat made in the kyoku, against how often
the human deviated from the AI in each type, with situation and opportunity controls. A regression on realised
points can still carry whatever made the human choose differently. Rolling the decision out prices it directly.

**Method (`jax_rl/mjai_bot/quality_rollout.py`).**
- At each disagreement in a real Tenhou game, roll out both the human's action and the AI's, from the same state,
  to the end of the kyoku. mortal_v4 plays all four seats after the branch. This is a one-step deviation: only the
  decision differs, and the continuation is the same policy in both branches.
- Value: GRP expected placement pt for the deciding seat (primary), raw kyoku points (secondary, the regression's
  unit).
- Walls. A Tenhou log has no libriichi seed, so the kyoku's wall is rebuilt from the log: hands, logged draws and
  revealed indicators in place, never-seen tiles at random (libriichi rollout patch: `wall` in the spec; riichi_eval
  `log_wall`). Each decision is rolled out on that wall (k = 0) and on K = 8 re-deals of every tile still hidden at
  the branch (k = 1..8), shared by both branches. Given the four hands the hidden order is uniform, so averaging
  over re-deals averages over the wall exactly. d = V(AI) − V(human), averaged over k = 1..8; k = 0 is reported
  as a check (it must agree within noise).
- Positive control, done before any pricing: every kyoku of 30 houou games replayed from the log on its rebuilt
  wall ends exactly as logged (336 / 336). With the hidden tiles re-dealt the event sequences are still identical
  (336 / 336); final scores differ only in riichi wins, where the ura indicators were re-dealt (41 / 41).
- "AI" = mortal_v4 and C' λ0.2 greedy agree on the action class (call / pass / riichi / discard); the alternative
  is v4's action. Call decisions with a ron available are excluded.

**Kinds and what the regression predicts** (final regression, raw points per deviation; d = V(AI) − V(human)):

| kind | human | AI | regression | predicted sign of d |
| --- | --- | --- | --- | --- |
| pass2call | passes a chi / pon | calls | human +233…+535 (AI over-calls) | d < 0 |
| call2pass | calls | passes | human −212…−431 | d > 0 |
| dama2riichi | discards, riichi legal | riichi | human −618…−902 | d > 0 |
| riichi2dama | riichi | discards | human +466 / +759 in South | d < 0 |
| disc2disc | another tile kind | same tile kind (both AIs) | ≈ 0 (92% of disagreements, "style") | control |

For dama2riichi a third branch (v4's best discard without riichi) splits d into riichi-vs-dama and the tile choice.

**Built-in tilt, read through the control.** The continuation is v4, whose later play fits its own earlier choices,
so one-step rollouts lean toward the AI's action in every kind. disc2disc measures that lean where the regression
says the difference is about zero. A kind is read against it: a pass2call d below zero is evidence against the AI
despite the lean.

**Sample.** 4,000 houou games drawn with seed 20261013 from the whole 2009–2026 pool, excluding the 12,000-game
held-out sample and the 130,955-game player sample. Every disagreement of the four rich kinds is kept; disc2disc is
sampled at 1%. From a 20-game smoke run the yield is about 2.3 pass2call, 4.7 call2pass, 1.3 dama2riichi and 0.8
riichi2dama per game, so the rarest kind has about 3,000 decisions. The smoke run's spread (about 12 pt per decision
on two walls) puts the standard error near 0.2 pt for that kind.

**Reading, fixed now.** Per kind, over all phases, in GRP pt:
- **confirmed** if d has the predicted sign and the 95% CI excludes 0;
- **contradicted** if d has the opposite sign and the 95% CI excludes 0;
- **inconclusive** otherwise.

Raw points and the per-phase split (E / S12 / S34) are reported alongside but do not change the reading. For
riichi2dama the prediction was made for the South round; the South rows are reported with the same rule as a
secondary reading.

**Where.** On the 5060 Ti alongside the q25 run (inference only, capped with `GPU_MEM_FRAC`); results in
`research/2026-10-11_data_quality/rollout/`.

## v8 result (2026-10-13 15:19): rollouts reverse two of the regression's "human is better" types

3,991 games (9 of 4,000 unreadable), 36,583 disagreements priced, 0 rollout errors, 8 re-dealt walls + the logged
wall each. d = V(AI) − V(human), mean over the re-dealt walls, 95% CI over decisions. Primary unit GRP pt, all
phases; raw kyoku points (the regression's unit) alongside.

| kind | n | d, GRP pt | z | d, raw points | regression predicted | reading |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| pass2call | 9,751 | +0.92 ± 0.23 | +8.0 | +190 ± 39 | d < 0 (AI over-calls) | **contradicted**: the AI's call is better |
| call2pass | 15,926 | +0.62 ± 0.18 | +6.8 | +124 ± 30 | d > 0 | **confirmed** |
| dama2riichi | 4,465 | +2.38 ± 0.46 | +10.2 | +720 ± 81 | d > 0 | **confirmed** |
| riichi2dama | 2,995 | +0.66 ± 0.52 | +2.5 | −344 ± 103 | d < 0 (South) | **contradicted** on the primary reading |
| disc2disc (control) | 3,446 | +0.13 ± 0.27 | +1.0 | +16 ± 47 | ≈ 0 | ≈ 0 |

- **The continuation tilt is small.** The control, where the regression says the difference is style, comes out at
  +0.13 ± 0.27 pt. v4 continuing in both branches does not by itself manufacture the AI's margins above.
- **dama2riichi is about riichi, not the tile.** Split by the third branch: riichi vs v4's own dama discard
  +2.36 ± 0.44 pt; v4's dama discard vs the human's tile +0.02 ± 0.20. It grows late: E +2.09, S12 +2.06,
  S34 +3.77.
- **pass2call reverses the regression.** In every phase the AI's call is worth more (E +0.78, S12 +0.72,
  S34 +1.47 pt; raw +150…+275). The regression's "human is better when passing the AI's call" (+233…+535) was
  selection: humans pass in hands where the call would have gone less well anyway, which the regression could not
  see and the rollout removes.
- **riichi2dama splits by unit.** Over all phases the AI's dama is better in pt, driven by the East round
  (E +1.06 ± 0.65; S12 +0.62 ± 0.90; S34 −0.28 ± 1.47). In the South round the human's riichi does win more raw
  points (S12 −516 ± 185, S34 −761 ± 253, z −5.5 / −5.9), which is what the regression measured, but those points
  do not turn into placement pt. The pre-registered South secondary reading in pt is inconclusive.
- k = 0 (the logged wall) agrees with the re-dealt mean within noise in every kind (e.g. pass2call +0.92 vs +0.92,
  dama2riichi +2.08 vs +2.38).

What it means for the study.
- In all four rich kinds the AI's recommendation is the better one-step choice under the same continuation,
  including the two the regression had credited to the humans. The pro's remark holds in a stronger form for these
  kinds: what the AI recommends is better, not only what it says to avoid.
- Tile-choice disagreements (disc2disc, and the tile part of dama2riichi) are worth about nothing, which matches
  the regression's "style" reading of 92% of disagreements.
- The `typed` variant treated pass2call as style because the regression said humans gained; the rollouts say it is
  an error. q25 is selected on plain v4 regret, which already counts every deviation by its Q gap, so the arm is
  unaffected.
- Caveat: a one-step deviation under v4's continuation, valued at the end of the kyoku by GRP. It says which
  single choice is better given how v4 plays on, not what a human would make of the line afterwards.

Outputs: `research/2026-10-11_data_quality/rollout/` (`out/shard_*.jsonl`, `report.txt`; contains game paths, not
distributed).

## q25 result (2026-10-15): the best offline base so far, narrowly short of the pre-registered bar

Training finished at 13:50 on 2026-10-15 (1,220,000 steps; final infer md5 20f9be36). Two crashes on the 5060 Ti at
773,600 and 786,000 steps, both `assert masks[range(batch_size), actions].all()` (a logged action that is not legal:
a malformed log, which only q25's pre-2019 games can contain, since v11 used 2019+ only). `symaug_run.sh` resumed
from the last save each time, reshuffling the epoch, and the run completed.

| comparison | seeds | Δ pt / hanchan | z |
| --- | --- | ---: | ---: |
| q25 − v11 | 12k gate | +1.67 ± 1.72 | +1.90 |
| q25 − v11 | **100k** | **+0.55 ± 0.60** | **+1.80** |
| q25 − B1 | 100k | +0.73 ± 0.61 | +2.35 |
| q25 vs mortal_v4 | 100k | −1.95 ± 0.45 | |

- **Verdict (pre-registered: ≥ +0.5 and z ≥ 2): not passed.** The point estimate clears +0.5 but z is 1.80; the CI
  reaches −0.05.
- It is still the best offline base the project has made: v11 −2.50, B1 −2.68, symaug −2.49, v11long −3.76 against
  v4 on the same 100k. It is the first arm since v11 whose 100k estimate is above v11 by more than the noise of a
  single 12k.
- It got there with about 1.5 passes over half the games (first written as two; corrected in v9), which v11long
  suggests may be a handicap: v11long's 1.36 passes cost about 1.3 pt against v11's 0.74, though with a longer
  schedule.

Mechanism readouts (s14000 8k vs v4):

| | q25 | symaug | v11long | mjs |
| --- | ---: | ---: | ---: | ---: |
| call rate (v4 ≈ 30.6%) | 31.28% | 30.24% | 32.65% | 34.25% |
| deal-in rate vs v4 | +0.38 pp | +0.52 pp | +1.01 pp | +1.98 pp |
| win rate vs v4 | +0.03 pp | +0.02 pp | +0.32 pp | +0.29 pp |
| riichi rate vs v4 | +0.50 pp | +1.14 pp | +0.59 pp | +0.12 pp |
| net points per kyoku vs v4 (attack / defence) | −26 (+2 / −27) | +2 (+35 / −33) | −41 (+21 / −62) | −104 (+18 / −122) |
| 4th place vs v4 | +1.32 pp | +1.12 pp | +1.83 pp | +4.40 pp |
| busted | 6.80% (v4 6.44%) | 6.78% (v4 6.66%) | 7.03% (v4 6.29%) | 7.88% (v4 6.36%) |

q25 has the lowest deal-in excess of any arm and a call rate close to v4's: the selected players' defence is what
the base picked up.

## v9 (2026-10-16, before the run): q25 with one pass (q25one), user decision

After the q25 result the user chose to run q25 again with one pass ("先做 A").

- **Why.** q25 came out +0.55 ± 0.60 against v11 at 100k (z 1.80), just short of the bar. v11long showed that
  training on past the data costs: 1.36 passes over the same games were about 1.3 pt worse than v11's 0.74. q25 made
  about 1.5 passes. q25one asks how much of the selection gain the repetition was eating.
- **Pass length, measured.** `jax_rl/mjai_bot/scan_logs.py` reads files with the trainer's own reader. On 5,000 files
  drawn from q25's list (seed 20261016) it counts 658.7 samples per game (SD 175, SE 0.38%). That puts one pass over
  628,106 games at about 808,000 steps at batch 512. The 590k used for q25 was wrong (v7 corrected): it assumed that
  v11's 1.21M steps were one pass over 1.29M games, when they are about 0.74 of one. The same count puts v11long at
  1.36 passes and q25 at 1.51.
- **Recipe.** q25 item for item (`configs/local_train_q25one.toml`), except `num_epochs = 1`, cosine 1e-4 → 1e-5 over
  785,000 steps, stop at 795,000. That is 10,000 steps at the final LR, as in v11's 1.21M / 1.22M. The stop is 1.5%
  (about 4 SE) short of the estimated pass end, so the trainer never finishes the epoch and exits on its own.
- **Data.** q25's file list in q25's order. The Mortal_symaug loader now skips a seat's game when a logged action is
  marked illegal (`dataloader_skip_illegal.patch`). That is what crashed q25 twice, and every crash reshuffles the
  epoch, which would break "one pass". No such game was found in 6,777 scanned files, so the skip is a safeguard, not
  a filter.
- **Gates.** No 12k futility stop: the arm exists for the 100k pairing with q25. Pass: q25one − v11 ≥ +0.5 with
  z ≥ 2, the bar of every base arm. Reported with its CI: q25one − q25 on the same 100k (one pass against about 1.5).
  `q25one_chain.sh`.
- **Expectation, stated now.** If repetition was the handicap, q25one − q25 > 0. q25one also trains for 35% fewer
  steps than q25 and v11, so a shortfall could be from length too. The arm cannot split those two.
- **Where.** On the 5060 Ti, about 31 hours of training at 7.2 steps/s, then about 8.5 hours of evaluation.

**2026-10-17 04:20: q25one restarted from scratch.** The first attempt started at 20:29 on 10-16. It crashed at 250,800
steps on the first illegal-action game it met: the skip patch referenced `logging` inside `populate_buffer`, where an
older local `import logging` makes the name local to the whole function (UnboundLocalError). `symaug_run.sh` resumed
it, but a resume reshuffles the epoch and the arm exists to test exactly one pass, so that run was stopped (kept as
`runs/q25one_crash1`), the patch fixed (`import logging` in place, the scoping reproduced and tested), and q25one
restarted from step 0 with the same config. At least one such game is in the list, so the skip is not hypothetical.

## q25one result (2026-10-18): one pass is as good as 1.5, and neither clears the bar

The restarted run went through without a crash: it stopped at 795,200 steps at 04:52 on 2026-10-18 (final infer md5
2d110984). The loader skipped one seat's game with an illegal logged action (a 2012 log, 1 of 225 samples), which is
the case that crashed q25 and the first q25one.

| comparison | seeds | Δ pt / hanchan | z |
| --- | --- | ---: | ---: |
| q25one − v11 | 12k | +2.27 ± 1.75 | +2.54 |
| q25one − v11 | **100k** | **+0.45 ± 0.60** | **+1.47** |
| q25one − B1 | 100k | +0.63 ± 0.61 | +2.04 |
| **q25one − q25** (one pass − 1.5 passes) | 100k | **−0.10 ± 0.59** | −0.33 |
| q25one vs mortal_v4 | 100k | −2.05 ± 0.44 | |

- **Verdict (≥ +0.5 and z ≥ 2 against v11): not passed.**
- **Repetition was not the handicap.** One pass and 1.5 passes over the same selected games come out the same within
  ±0.6 pt, and the one-pass arm used 35% fewer steps. The expectation stated in v9 (q25one − q25 > 0) is not borne
  out. The two factors the arm cannot split (fewer repeats, fewer steps) would have to cancel exactly for a real
  repetition cost to hide here; the simpler reading is that 1.5 passes do not hurt.
- **The selection effect, read across both arms.** q25 +0.55 and q25one +0.45 against v11 on the same 100k: two
  trainings of the same selection agree on about +0.5 pt. Each falls short of z ≥ 2 on its own. The two estimates
  share v11's noise, so taken together they do not make a pass either.
- **Practical point.** The same strength costs 65% of the training steps: a cheaper recipe for any later base on this
  selection.

Mechanism readouts (s14000 8k vs v4):

| | q25one | q25 | symaug |
| --- | ---: | ---: | ---: |
| call rate (v4 ≈ 30.6%) | 32.83% | 31.28% | 30.24% |
| deal-in rate vs v4 | +0.40 pp | +0.38 pp | +0.52 pp |
| win rate vs v4 | +0.10 pp | +0.03 pp | +0.02 pp |
| riichi rate vs v4 | −0.49 pp | +0.50 pp | +1.14 pp |
| net points per kyoku vs v4 (attack / defence) | −27 (+1 / −28) | −26 (+2 / −27) | +2 (+35 / −33) |
| 4th place vs v4 | +1.25 pp | +1.32 pp | +1.12 pp |
| busted | 7.03% (v4 6.31%) | 6.80% (v4 6.44%) | 6.78% (v4 6.66%) |

Both q25 arms keep the low deal-in excess; q25one calls more and riichis less than q25.

## v10 (2026-10-19, before the run): select twice as hard (q12one), user decision

After q25one the user chose to tighten the selection ("按A来做吧").

- **Why.** Both q25 arms land about +0.5 pt above v11 at 100k, each short of z ≥ 2, and one pass is as good as 1.5. If
  the gain comes from who is playing, a stronger selection should give more. If halving the data again costs more
  than the stronger selection adds, the arm will show it.
- **Data.** The best 12.5% by the same M3 game score: 314,053 games, the first half of q25's order (rerun of
  `quality_player_select.py` with a 12.5% output; the q25 list reproduces byte for byte). On the held-out sample,
  excess regret against v11's set: v4 −0.0384 ± 0.0019 (q25 −0.0286), C' −0.0111 ± 0.0008 (q25 −0.0081), about 1.35
  times q25's. The mix leans recent: 2024–2025 games are 42% of it (q25: 33%).
- **Recipe.** q25one item for item (`configs/local_train_q12one.toml`), one pass. 658.3 samples per game on 3,617
  scanned files of the list (SE 0.43%) put the pass at about 403,800 steps. Cosine 1e-4 → 1e-5 over 386,000 steps,
  stop at 396,000, about 2% (4.5 SE) short of the pass end. Same loader, which skips a seat's game with an illegal
  logged action.
- **Gates.** As q25one: no 12k futility stop, 100k always. Pass: q12one − v11 ≥ +0.5 with z ≥ 2. Reported with
  CIs: q12one − q25one (twice as hard a selection, one pass each) and q12one − q25.
- **Expectation, stated now.** If the gain scales with the regret gap, q12one − v11 ≈ +0.6–0.7 and q12one − q25one ≈
  +0.15–0.2. That difference is far inside the ±0.6 of a 100k pairing, so a pass against v11 is the readable
  outcome. A clearly negative q12one − q25one would mean data quantity binds at 314k games.
- **Where.** 5060 Ti: about 12 hours of training at 9 steps/s, then about 7.5 hours of evaluation.
