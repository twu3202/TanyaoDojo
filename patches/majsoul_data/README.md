# New-data arm: Tenhou houou + Mahjong Soul throne / jade (2026-10-06)

## Why

The user had another session collect recent game records and asked whether more, newer data makes a better offline
base. The earlier data-doubling attempt (v18) added older Tenhou years (2009–2018) and did not help. This time the
data is recent.

The snapshot (handoff `training_handoff_20261006`, source `nikketryhard/tenhou-houou-mjai` @ 6450bb58) is
**almost entirely Mahjong Soul**, not newer Tenhou:

| source | window 2024-10-06 … 2026-10-06 | note |
|---|---|---|
| Mahjong Soul jade (玉の間) | 1,254,065 | 93% of the new data; 2026 coverage ends early February |
| Mahjong Soul throne (王座の間) | 89,192 | no 2025 shard (the file named throne-2025 holds Tenhou 2024 ids, excluded) |
| Tenhou houou, raw XML | 5,105 | 0.4% of our 1,289,661; needs XML→MJAI; **not used in this arm** |
| Mahjong Soul official raw | 14 | 13 duplicate the archive, 1 unverified; **not used** |

So the question this arm answers is "does adding a related, recent, but different-platform source help", with a
lower-skill majority (jade is generally weaker than Tenhou houou; throne is close to it).

**Prior.** Low. v18's data doubling ≈ 0, and the step21 learning curve said label count is not the bottleneck.

## Data audit (`jax_rl/mjai_bot/mjs_convert.py`)

- **Throne** is standard mjai and passes through unchanged.
- **Jade** comes from a different converter with systematic defects, measured on a 2000-game sample:

| defect | how common | repair |
|---|---|---|
| dealer's 14 tiles all in `tehais`, no first draw | every kyoku | 14th tile → first `tsumo` (which tile was drawn is unknowable; the dealer's first discard is always flagged tedashi) |
| call target = actor | 100% of chi / pon / daiminkan | target = actor of the claimed (last) discard |
| added kan labelled `ankan`, concealed kan labelled `kakan` without `consumed` | 758/758 and 826/826 | labels swapped back; kakan `consumed` from the actor's pon |
| `reach_accepted` before a ron on the riichi tile | ~0.3 per game | dropped |
| no kan dora ever revealed | 0/2020 kans | **kyoku with a kan dropped** (indicator unknowable) |
| no `deltas` / `ura_markers` | every game | **all-last kyoku dropped** (final ranking unknowable); placeholder deltas only to let libriichi load |

Dropped kyoku are listed in `start_game.drop_kyoku`, and the training loader skips their samples (`dataloader.patch`).
On 20k jade games this drops 17.5% of kyoku. Everything the repairs write is checked by the libriichi replay below:
an illegal call, kan or discard fails the game.

**Mode.** Both rooms are taken as hanchan:
- all throne games that end in the East round end with a bust (76/76);
- jade's ending positions have the same distribution as throne's;
- tonpuu games would pile up at E4.

## Checks

- **Structural, every game** (in the converter): event types, haipai, at most 4 copies of each tile kind with
  draws and indicators, ending present. Throne fails 5 of 20,000 (all-last without hora/ryukyoku); jade fails 0 of
  20,000.
- **libriichi replay** (`mjs_validate.py`) on a stratified 3000-game sample: load_log, every logged action inside
  its mask, kyoku indices against the GRP features. One full pass costs about as much as a training epoch's
  loading (0.78 s/game), so the full set is replayed by the training loader itself. Unreadable logs are skipped
  and logged, and the counts are reported with the result.

## Source-quality probe and the pre-registered jade rule (`mjs_quality_probe.py`)

For each source, 300 games: how often the human's action equals mortal_v4's greedy action.
- Non-forced decisions only.
- The same kind of kyoku for every source: kyoku with a kan and the all-last kyoku are excluded everywhere.
- Game-clustered CIs.

Let gap = agreement(Tenhou houou 2025) − agreement(source), in percentage points, on point estimates. Fixed before
the probe runs:

| jade gap | jade in the arm |
|---|---|
| ≤ 2 pp | all games |
| 2–4 pp | a fixed 50% (by UUID hash) |
| > 4 pp | excluded |

Throne is included if its gap is ≤ 2 pp. If jade is excluded, the arm would add only ~7% data, which a 100k
evaluation cannot resolve, so the arm is **not run** and the result is reported as is.

## Arm (single variable against v11)

The v11 recipe item for item: 192×40, pts [6, 4, 2, 0], CQL 5, batch 512, LR cosine 1e-4 → 1e-5, one epoch, no
symmetry augmentation. Only the data changes: v11's 1,289,661 Tenhou logs plus the admitted Mahjong Soul games,
uniformly shuffled together.

**Steps.** One epoch over the larger set. `max_steps` scales v11's 1.21M by the ratio of training samples (Tenhou +
admitted Mahjong Soul, after drops) to Tenhou alone, and training stops 10k steps after it (v11: 1.21M → 1.22M).
The arm therefore also trains longer. This is part of its definition, not a hidden second variable:
- if it fails, data and training length are ruled out together;
- if it passes, a Tenhou-only run with the same steps is the next control, to separate the two.

**Gates (same as B1).**
- 12k duplicate vs v4 paired with v11: a point estimate < 0 stops.
- Otherwise 100k: arm − v11 ≥ +0.5 with z ≥ 2 passes.
- Mechanism readouts with every eval.

Logs and derived data stay on our machines and are never committed or published.

## Results (2026-10-06): the arm is not run

**Conversion.** Every game of the snapshot window was converted:
- throne: 89,181 of 89,192 (11 fail: the all-last kyoku has no hora / ryukyoku);
- jade: 1,254,064 of 1,254,065 (1 fail: a ron without a matching discard).

Jade repairs, all deterministic:
- every haipai and every call target;
- 460k mislabelled added kans and 650k mislabelled concealed kans swapped back;
- 352k `reach_accepted` before a ron removed.

Dropped kyoku: 17.4% of jade's (2.29M of 13.1M).

**libriichi replay.** The stratified 3000-game sample (1500 throne + 1500 jade) all pass:
- 1,953,253 decisions, 1,773,612 left after the drops;
- decisions per game: throne 646.7, jade 535.8 after drops, Tenhou houou 661.6 (600-game sample).

**Probe** (300 games per source; non-forced decisions in kyoku without a kan, all-last excluded):

| source | decisions | agreement with v4 | discard | riichi | call |
|---|---|---|---|---|---|
| Tenhou houou 2025 | 154,141 | 82.06 ± 0.26 | 78.26 | 82.76 | 94.17 |
| Mahjong Soul throne | 151,132 | 79.71 ± 0.34 | 75.48 | 79.78 | 93.32 |
| Mahjong Soul jade | 146,131 | 76.50 ± 0.35 | 71.43 | 77.13 | 92.57 |

**Rule outcome.** jade gap 5.56 pp > 4, so jade is excluded. Throne gap 2.35 pp > 2, so throne is excluded too.
Nothing is left to add, so the arm is not run, as pre-registered.

Agreement with v4 partly measures similarity to v4's style. The platforms also differ in uma and incentives, so
part of the gap may be style rather than skill. The rule was fixed before the probe, and the gaps are in the
expected order (houou > throne > jade).

The converted, checked logs are kept on the 5060 Ti (WSL `data/majsoul_20261006/`, per-game metadata under
`meta/`), so a later decision can use them without redoing the conversion.

## 2026-10-07: run anyway (user decision)

After seeing the probe, the user chose to run the arm anyway, to get a direct answer. The jade / throne rule above
is therefore overridden by an explicit decision; the strength gates are unchanged.

**Data.** v11's 1,289,661 Tenhou logs, 89,181 throne logs and 1,254,064 repaired jade logs (2,632,906 files, one
shuffled file index). Jade's dropped kyoku are skipped by the loader.
- Loader check (`Mortal_mjs/mortal/dataloader.py`): Tenhou and throne logs give exactly the samples of the
  original path; jade logs give that set minus the dropped kyoku.

**Steps.** Training samples are estimated from decisions per game: Tenhou 661.6, throne 646.7, jade 535.8 after
drops. The total is 1.855 times Tenhou alone, so:
- `max_steps` = 1.21M × 1.855 = 2,245,200, and the run stops at 2,255,200;
- this is the same fraction of an epoch that v11 trained;
- about 3 days on the 5060 Ti.

**Start.** It starts automatically when the symaug evaluation chain finishes (`runs/mjs/mjs_after_symaug.sh`;
copies of both scripts are in this directory). It started on 2026-10-07 at 14:30, at about 9.5 steps/s.
`mjs_chain.sh` then runs:
- 12k paired with v11: a point estimate < 0 stops;
- otherwise 100k: mjs − v11 ≥ +0.5 with z ≥ 2 passes.

## 2026-10-08: Tenhou-only length control (v11long, trx)

With trx free, the length control pre-registered above (run only if the arm passes) is started now, in parallel, so
that both answers arrive together instead of three days apart.

- **Config** `configs/local_train_v11long.toml`: the mjs config item for item without the Mahjong Soul globs.
  - v11's 1,289,661 Tenhou logs (file index copied from the v11 run: same set, same order);
  - the same schedule, `max_steps` 2,245,200, stopping at 2,255,200.
  - Against v11 (identical config apart from the step count and `num_workers`), it trains 1.855 times as many
    steps over the same data.
- **Code** `Mortal_mjs/mortal` on trx, every file md5-identical to the WSL tree (Tenhou logs load exactly as on the
  original path).
- **Reading.**
  - mjs − v11long is the effect of the Mahjong Soul data at equal steps;
  - v11long − v11 is the effect of training length alone.
- **Gates** (`v11long_chain.sh` on the trx evaluation station, same as B1): 12k paired with v11 < 0 stops;
  otherwise 100k, and v11long − v11 ≥ +0.5 with z ≥ 2 says length alone helps.

## Result (2026-10-10): futility stop at 12k

mjs reached 2,255,200 steps at 15:07 (`mjs_final`, infer md5 4defd70c), with no crash or loader error in three
days. Duplicate vs v4 at 12k (s10000 4k + s14000 8k), paired with v11 at the same seeds (pt/hanchan, 95% CI over seed
groups):

| | difference | z | arm | twin |
|---|---|---|---|---|
| mjs − v11 | **−1.38 ± 1.77** | −1.53 | −5.17 | −3.79 |
| mjs − B1 | −1.85 ± 1.83 | −1.98 | −5.17 | −3.32 |

The point estimate is below 0, so the arm stops here as pre-registered and is not extended to 100k.

**Mechanism readouts** (s14000 8k, against v4 in the same games; symaug, which ties v11 at 100k, alongside):

| | mjs | symaug |
|---|---|---|
| avg pt on s14000 | −5.18 | −1.12 |
| call rate (v4 ≈ 30.6%) | **34.25%** | 30.24% |
| deal-in rate vs v4 | **+1.98 pp** | +0.52 pp |
| win rate vs v4 | +0.29 pp | +0.02 pp |
| riichi rate vs v4 | +0.12 pp | +1.14 pp |
| net points per kyoku vs v4 (attack / defence) | −104 (+18 / −122) | +2 (+35 / −33) |
| 4th place vs v4 | +4.40 pp | +1.12 pp |
| busted | 7.88% (v4 6.36%) | 6.78% (v4 6.66%) |

The loss is almost all defence: mjs calls about 4 points more often and deals in 2 points more often than v4, and
ends 4th more often. That matches what the source-quality probe saw: jade players agree least with v4, and the
base copies their more open, less defensive style. Training-time test_play said the same throughout. Over the last
fifth of each schedule, mjs averaged −5.3 and symaug −2.8 on the same fixed seeds.

**Reading.** Adding the 1.34M Mahjong Soul games (93% jade) makes the offline base worse, not better. The
pre-registered inclusion rule (exclude both sources) is confirmed by the direct test.

The Tenhou-only length control (v11long) was paused on trx at 550,800 steps on 2026-10-09, when trx was needed.
Since mjs did not pass, it no longer serves to split data from length. On its own, it would only test whether
training 1.855 times longer on Tenhou helps.

**2026-10-10: v11long finished on the 5060 Ti.** The user chose to finish it on the 5060 Ti, now free.
- The 550,800-step state was copied from trx (md5 78b3b82c) and resumed at 16:10. The Mortal_mjs tree is
  md5-identical on both machines, and the file index is the symaug one, verified identical to v11's. As with B1,
  the data epoch is reshuffled on resume.
- `v11long_chain_wsl.sh` (the mjs chain with the tag changed) runs the same gates on the 5060 Ti.

## Result (2026-10-13): training 1.855 times longer on Tenhou makes the base worse

v11long reached 2,255,200 steps at 10:11 (one silent exit at 2,220,800, resumed from the last save by
`symaug_run.sh`); final weights md5 b2af6e59.

| comparison | seeds | Δ pt / hanchan | z |
| --- | --- | ---: | ---: |
| v11long − v11 | 12k (s10000 + s14000) | +0.28 ± 1.82 | +0.30 |
| v11long − v11 | **100k** (s10000–34999) | **−1.26 ± 0.62** | **−3.98** |
| v11long − v11 | the 88k outside the gate | −1.47 ± 0.66 | −4.36 |
| v11long − B1 | 100k | −1.08 ± 0.63 | −3.38 |
| mjs − v11long (data effect) | 12k | −1.67 ± 1.78 | −1.83 |

The 12k gate passed (≥ 0), so the chain extended to 100k as pre-registered. At 100k the length effect is clearly
negative, the opposite of the +0.5 / z ≥ 2 the control was set to detect. The 12k seeds happened to favour v11long
by about 1.6 pt relative to the rest, which is within the 12k noise (±1.8).

Mechanism readouts (s14000 8k vs v4), next to mjs and symaug (≈ v11 recipe, 1.21M steps):

| | v11long | mjs | symaug |
| --- | ---: | ---: | ---: |
| call rate (v4 ≈ 30.6%) | 32.65% | 34.25% | 30.24% |
| deal-in rate vs v4 | +1.01 pp | +1.98 pp | +0.52 pp |
| win rate vs v4 | +0.32 pp | +0.29 pp | +0.02 pp |
| riichi rate vs v4 | +0.59 pp | +0.12 pp | +1.14 pp |
| net points per kyoku vs v4 (attack / defence) | −41 (+21 / −62) | −104 (+18 / −122) | +2 (+35 / −33) |
| 4th place vs v4 | +1.83 pp | +4.40 pp | +1.12 pp |
| busted | 7.03% (v4 6.29%) | 7.88% (v4 6.36%) | 6.78% (v4 6.66%) |

**Reading.**
- More passes over the same 1.29M Tenhou games hurt by about 1.3 pt. The base drifts the same way mjs did (more
  calls, more deal-ins, losses on defence) by about half as much.
- So part of mjs's deficit against v11 was its schedule length, not the Mahjong Soul data. The data effect proper
  (mjs − v11long on the 12k seeds) is −1.7 ± 1.8: most likely negative, not established.
- For any later arm, training on past the data costs. A sample count (data-quality README v9) gives about 659
  training samples per game, so one pass over v11's 1.29M games is about 1.66M steps: v11 stops at about 0.74 of a
  pass, and v11long reaches about 1.36. That bears on q25, which makes about 1.5 passes over 628k games in the
  same 1.22M steps.
