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


## Response-priority correction (2026-11-15)

Both engines now defer a pung or exposed-kong claim until all remaining seats have responded. Any ron cancels
the claim; multiple winners still settle, and a declined win allows the pending claim to resolve exactly once.
No hand, river, kong payment or replacement draw is changed while the claim is pending. This implements the
existing RULES.md priority; it does not introduce a new rules variant.

`test_response_priority.py` includes independent rule cases for all seat rotations, later/multiple winners,
passing on a win, an earlier win, empty-wall kong resolution and a fresh discard clearing the pending claim.
The JAX cases execute the compiled engine. `test_diff_jax.py` additionally checks phase, river, finished seats,
wall consumption and conservation, and exits nonzero on a mismatch.

A 100,000-game audit of the pre-fix reference engine (seeds 0–99,999, random legal actions with win priority)
found 289 games with a suppressed legal ron opportunity (0.289%). This is not a trained-policy occurrence rate
or an estimate of score bias. Reproduce with `audit_response_priority.py --legacy <pre-fix-reference.py>`.
Historical trained-model scores retain their old-engine meaning; re-evaluate frozen weights on the corrected
engine before making a new strength claim. No retraining or remote deployment was performed for this fix.

The observation/action shapes and network weights are unchanged. Serialized environment states need migration
or a fresh environment because the JAX State now has two pending-claim fields. Deferred responses also change
trajectory lengths and reward timing, so old and new environment rollouts must not be silently mixed.


### Authorized follow-up training

After the local fix review, the user authorized training on the 5060 Ti on 2026-11-15. The corrected engine and
bounded two-seed baseline chain were deployed into an isolated run directory; the legacy deployment and weights
remain unchanged. See [the frozen design](../patches/sichuan_rulefix/README.md). Both training seeds and five fixed
evaluations finished on 2026-11-16. Both seeds lost head-to-head to the frozen P2 b2272 checkpoint.

## Draw-turn correction (2026-11-18)

The Python reference, JAX environment, and web adapter now keep an explicit drawn tile for the active seat.
A pung leaves the draw indicator empty and permits only discards. A regular or replacement draw sets the
indicator and enables the existing self-draw and kong rules. The network keeps the same input dimensions.
Observation plane 4 is zero when the active player did not draw, so the previous player's draw cannot enter
the next caller's observation after a pung. Restoring a pre-fix parameter checkpoint uses a fresh environment.

The follow-up job re-evaluates the three frozen checkpoints on the same fixed walls. The batched evaluator
runs the Python reference for every game and checks the compiled shadow after each decision. Its CUDA gate
compares complete action histories and terminal scores against the serial evaluator on separate walls.
