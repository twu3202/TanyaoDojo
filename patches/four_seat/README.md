# Four-seat self-play, variant S2: does training on self-play from every seat beat 1v3 against a frozen start?

Pre-registered 2026-11-10, before the run. 5060 Ti. The user chose variant S2 on 11-09 and, after the check below, the
one-rotation form. The design draft and the background research are in [DESIGN.md](DESIGN.md).

## Why

- Every online stage so far trains one seat (自对弈的单座位训练，one-seat training): the trainee plays 1v3 against
  copies of a fixed pool, and only its own seat's samples train.
- The control stage of the v11 line (A80k → A160k) uses the pool base, v4, base, base: three workers play against the
  frozen start (A80k), one against v4.
- Four-seat self-play (四座位自对弈，four-seat self-play) is what DouZero and the IJCAI mahjong teams use: the current
  policy in every seat, every seat's samples trained.
- Prior: about 10–15% that it beats the control by the screening rule (DESIGN.md explains why it is low).

## The check before the run (11-10, 100 games from A80k)

`fs_check.py` played one worker round of four-seat self-play with the patched `player.py`:
- **The arena accepts it.** The challenger and a second engine on the same modules, both named `trainee`; the
  trainer's reader returns four trajectories per file (672 samples per file, against 157 for a 1v3 file).
- **The champion engine follows every parameter update:** it wraps the trainee's own module objects.
- **The four rotations of a seed are mostly the same game.** The arena plays each seed four times. With one
  near-deterministic policy (T 0.1, ε 0.005) in every seat, 55 of 75 rotation pairs were identical games once the
  engines' metadata is stripped, and 90% of kyokus were identical.
  - So "four times the samples per game" was wrong: feeding all rotations would add ~1.4 times the unique data and
    quietly raise sample reuse to about 5.
- **Decision:** keep only rotation a of each seed. Each kept game is one deal with four distinct seats trained.
  - Unique samples per game played equal the control's (672 per 4 games, against 4 × 157).
  - So this run tests the opponents and the all-seat training at equal compute, not data efficiency.
  - A one-game-per-seed arena (a libriichi change, buildable on trx) would realize the efficiency. It is built only if
    this run wins.

## Design

- **Arm FS** (`runs/xfer_v11fs`, port 5597) is the control's stage with workers 1, 3 and 4 switched to four-seat
  self-play (pool `self,v4,self,self`).
  - Same start (the A80k archive), anchor (λ 0.2 to A80k), reward, temperature, LR, 80,000 batches (80k → 160k).
  - Worker 2 plays 1v3 against v4 as in the control. About a quarter of the samples come from it, as in the control.
- **Two config lines change.** The chain stops if anything else differs from v11s2's config.
  - The trainer's loader workers (`num_workers`) 8 → 4, throughput only. With 8, the second start (11-10 01:21) took
    WSL from 7.5 GB to 1.2 GB free at the trainer's start, and the memory watch stopped it: each loader worker holds
    a whole four-seat file, 4.3 times a 1v3 file. The watch now stops the arm under 1 GB (4 GB of swap behind it).
  - `capacity` 4,000 → 1,750 files. A four-seat round keeps 200 files of 672 samples, a 1v3 round 800 files of 157.
    With 1,750 the buffer should hold about 1.13 M samples per drain, as the control's 7,200 one-seat files did, so
    reuse should stay near 1.8.
  - `file_batch_size` stays 1. It was first set to 4 (about three games per batch, as the control has), but at the
    trainer's start (11-10 01:02) its loader ran WSL down to 247 MB free and the memory watch stopped the arm before
    any training step was saved. With 1, a batch of 512 holds most of one game's four seats; with 8 loader workers
    taking turns, an optimizer step (8 batches) still spans about 6 games, 24 seat trajectories, against the
    control's ~26 games of one seat.
  - The trainer starts once 1,750 files are buffered (MINBUF), about the control's 600,000 samples.
- **Code:** `player.py` (FOUR_SEAT=1: the second engine, rotation a only) and `run_selfplay_m1.sh` (pool member
  `self`). Both are off by default; the 1v3 path is unchanged. `xfer_arm.sh` takes MINBUF from the environment
  (default 4,000, unchanged).

## Evaluation and verdict

- 12k of FS160k first, then SPRT against the **pooled control** (the mean of A160k and C160k per group): 0 vs +1,
  α 0.025, β 0.2, a look every 1,000 groups from seed 10000, capped at the 16,000 groups both controls have.
  - The pooled control replaces A160k alone, used by the earlier screens: the control was trained twice (+0.28 and
    +0.38 over A80k), and the mean halves its training draw.
- **Promote:** four-seat self-play goes on, with the one-game arena built first.
- **Drop:** the four-seat direction closes for the v11 line.
- **Undecided at the cap:** the fixed 16k is reported.
- **Also reported:** FS160k vs v4, − A160k, − C160k, − A80k, − tpt 408k; mechanism readouts at 12k and on all
  groups played (riichi, deal-in, call rate, attack and defence per kyoku, 4th place); sample reuse per drain.

## Departures (R8)

| | The control, or convention | Here | Why, and the cost |
| --- | --- | --- | --- |
| 1 | Three workers 1v3 against the frozen start | Four-seat self-play of the current policy | The variable under test |
| 2 | Convention: one game per deal | The arena plays four rotations; three are discarded | Rotations are mostly duplicates. Discarding them wastes three quarters of the self-play game compute, the same compute per sample as the control. The fix is a libriichi change, built only on a win |
| 3 | capacity 4,000 files | 1,750 | Same samples per drain; measured afterwards (reuse per drain is reported) |
| 4 | A batch holds samples of ~3 games, one seat each | Most of one game, all four seats; an optimizer step spans ~6 games | file_batch_size 4 would match the control but needs ~4 times the loader memory, which WSL does not have (the 01:02 start). The seats of one game are correlated, so each step carries somewhat less independent information |
| 5 | Opponents use the rule-based agari guard (the control's base and v4 seats) | No seat uses it in self-play | Every seat must be the behavior policy; the trainee never had the guard |
| 6 | Earlier screens: SPRT against A160k | Against the pooled control | A better estimate of the control; A160k alone is also reported |
| 7 | 8 trainer loader workers | 4 | Memory (see above). Throughput only: if the loader limits the trainer, the stage takes longer, but the samples per drain are set by the capacity |

## Cost

About 9 hours of training (the control took 8.7), then up to 3 hours of evaluation. The chain marks the card BUSY,
serves bookings at the arm start and at each evaluation call, and lends the card for a day after it ends.

## Files

- `fs_check.py`: the check above. `player.patch` (against Mortal's `player.py`) and `run_selfplay_m1.sh`: the
  deployed changes (the script's original is in `orig/`).
- `fs_chain.sh` (`runs/xfer_tools/fs_chain.sh`): log `runs/xfer_tools/fs_chain.log`, markers `FS_CHAIN_DONE` /
  `FS_CHAIN_STOPPED`.

## Glossary

| 中文 | English | Meaning here |
| --- | --- | --- |
| 四座位自对弈 | four-seat self-play | the current policy in all four seats, all seats' samples trained |
| 单座位训练 | one-seat training (1v3) | the trainee in one seat against three copies of a pool model |
| 轮换 | rotation (split a–d) | the arena's four games per seed, the challenger in each seat |
| 样本复用 | sample reuse | samples trained ÷ fresh samples generated |
| 序贯检验 | sequential test (SPRT) | stop the evaluation as soon as the evidence decides |
| 合并对照 | pooled control | the mean of the two control runs per seed group |

## Result (11-10 11:18): dropped at 3,000 groups; four-seat self-play closes for the v11 line

Training ran 01:39-10:16 (80,000 batches, ~2.6 batches/s). It saw 22.7 M fresh trainee samples, reuse 1.81 (the
control's 1.83). WSL's free memory never fell below 1.29 GB.

| Readout | Groups | FS160k minus it | z |
| --- | --- | --- | --- |
| SPRT against the pooled control (seeds 10000-12999) | 3,000 | -0.05 ± 1.13, LLR -1.68 < -1.58 → **H0, drop** | |
| pooled control (all groups played) | 5,000 | -0.47 ± 0.88 | -1.04 |
| A160k | 5,000 | -0.58 ± 1.00 | -1.14 |
| C160k | 5,000 | -0.36 ± 1.03 | -0.68 |
| A80k (its start) | 5,000 | -0.21 ± 0.96 | -0.43 |
| tpt 408k | 5,000 | -1.21 ± 1.34 | -1.77 |
| against v4 | 5,000 | -1.93 ± 1.03 | -3.68 |

- The 12k alone (seeds 10000-10999, 14000-15999) read -0.51 ± 1.13 against the pooled control. The SPRT segments
  were 10000-12999.
- **Mechanism readouts (all 20,000 games, against v4 at the same tables):** call rate 30.54% (v4 30.56), win -0.42 pp,
  deal-in +0.14 pp, riichi -0.18 pp, net -26 per kyoku (attack -16 / defence -10), bust 6.88% (v4 6.73), 4th place
  +1.25 pp.
- The control at 64,000 games: A160k deal-in +0.20, riichi +0.09, net -27 (-14 / -13), 4th +1.20; C160k +0.33, -0.19,
  -35 (-14 / -21), +1.15.
- So FS160k plays like the control: its defence reads a little better and its attack a little worse, all within the
  spread of the two control runs.
- **Reading.** At equal compute and an equal amount of unique data, replacing the frozen-start opponents with the
  current policy in every seat changes neither the strength nor the profile. Training all four seats of one deal did
  not help, and the opponents' identity did not either. The one-game arena is not built.
