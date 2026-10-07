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
