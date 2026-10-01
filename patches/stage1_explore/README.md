# Branch B Stage 1: targeted exploration at the decision boundary (2026-09-30)

Stage 0 (`jax_rl/mjai_bot/stage0_stale.py`) found that on push/fold and riichi decisions, the Q of the action actually
taken is calibrated, while the untaken side is underestimated by about 49 pt (left over from offline CQL and frozen since
rl1), and online RL never gives it a target. Here the workers explore the other side at the decision boundary, and
training drops the samples contaminated by that exploration.

- `explore_engine.py`: goes into `Mortal/mortal/`. A worker-side engine that requests the v5 observation from libriichi
  (v4 + 10 rows of defense features), while the network consumes only the first 1012 rows; when the Q gap between the
  greedy action and its push/fold or riichi/dama counterpart is < `margin_pt`, it switches to the counterpart with
  probability `prob`. `is_greedy` is now recorded as "action == argmax Q".
- `player_dataloader.patch`: patches `Mortal/mortal/player.py` (with `[explore] enable`, TrainPlayer uses the engine above
  and logs the explorable/explored counts every round) and `dataloader.py` (with `drop_pre_explore`, it drops the samples
  in a game that come before that game's last non-greedy action; samples are aligned one by one with the log events, and
  games that fail to align are kept whole and counted). With the config absent, both patches are completely inert.
  Apply with: `cd ~/Mortal && patch -p1 < player_dataloader.patch`.
- `test_frontier.py`: compares the engine's vectorized classification, decision by decision, against a pure-Python
  reference that replays PlayerState (trx: 40,114 decisions, 0 mismatches; kan-choice flag rows 494/494).
- `test_explore_run.py`: a small batch of real games plus loader alignment (trx: 200 exploration games + 800 old games,
  0 misalignments; about 1.07 explorations per game; 8.1% of samples dropped).
- `stage1_setup.sh`: sets up `runs/stage1_explore` on trx (starting from tpt 408k, md5 6fb1d3e1…), the config, start/stop
  scripts, archiving every 8000 steps, and the crash guard. Apart from `[explore]`, everything is identical to the
  historical tpt 408k→448k continuation (7 workers, opponent pool base×4 / v4×2 / C392k×1, anchor cprime_final λ=0.1,
  ε 0.005, temperature 0.1).

Start: `bash scripts/run_selfplay_stage1.sh --resume --workers 7 --pool base,base,v4,<C392k>,base,base,v4`;
stop: `bash scripts/stop_selfplay_stage1.sh` (by PID process group, including the archiver and the guard).
