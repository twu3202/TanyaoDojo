# sichuan

The Sichuan (Xuezhan Daodi, "bloody to the end") line: pure RL from random initialization, sharing the JAX stack and
the duplicate-evaluation method with the riichi lines. Rules are frozen in [RULES.md](RULES.md). Results are in the
[README](../README.md) and the [project log](../HISTORY.md#sichuan-line-xuezhan-bloody-to-the-end-progress).

| File | Contents |
| --- | --- |
| `RULES.md` | The rules, frozen at v0: the single source of truth for both implementations |
| `reference_impl.py` | Reference implementation (correctness first): the differential-test oracle and the executable arbiter of rule questions |
| `env_jax.py` | The JAX environment, implementing Mahjax's `Env` protocol so the existing trainers run unchanged |
| `test_sichuan.py`, `test_diff_jax.py`, `test_disc_obs.py` | Unit cases; the decision-by-decision differential test against the reference (the gate before any training); the discard-observation differential test |
| `obs.py`, `net.py` | Observation (27 × 22 planes + 26 scalars) and the actor-critic network (61 actions) |
| `shaping.py`, `check_shaping.py`, `reward_density.py`, `check_objective.py` | Shanten-potential shaping and its numerical checks: reward density, telescoping, and training objective = evaluation objective |
| `ppo_sichuan.py` | Self-play PPO trainer |
| `bots.py` | Rule-bot ladder L0–L3: the out-of-lineage ruler |
| `arena.py` | Duplicate arena (the challenger takes all four seats on the same wall) |
| `eval_net.py`, `eval_h2h.py` | Network vs the rule ladder; direct head-to-head between snapshots |
| `diag_coldstart.py`, `discard_headroom.py` | Cold-start signal diagnosis; discard-efficiency headroom of the learned policy |
