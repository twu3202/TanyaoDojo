# Environment setup

This repository contains **only this project's own code**. The evaluation bridge depends on `libriichi` from upstream
[Mortal](https://github.com/Equim-chan/Mortal) (AGPL-3.0); clone and build it yourself under its license terms. It is
not distributed with this repository.

## 1. Dependency versions (pinned; do not upgrade)

```
python 3.10 · jax[cuda12]==0.6.2 · flax==0.10.7 · optax==0.2.8
distrax==0.1.5 · chex==0.1.90 · numpy==2.2.6 · pydantic==2.13.4 · omegaconf==2.3.1
```

The environment install script is [`jax_rl/cloud_setup.sh`](jax_rl/cloud_setup.sh) (brings up the whole stack in one go on a cloud or fresh machine).

## 2. Mahjong environment (Mahjax)

```bash
git clone https://github.com/nissymori/mahjax ~/mahjax
export PYTHONPATH=~/mahjax
```

### 2.1 Mandatory wall-RNG patch (versions before v0.1.3)

**Before training, confirm that this patch is in place; otherwise the training data will be silently corrupted.**
In mahjax before `v0.1.3` (2026-09-04, upstream commit `fade6de` / #73), `red_mahjong/env.py::_init` calls
`_make_state` without passing `rng_key`, so that field keeps the dataclass default `PRNGKey(0)`; yet the walls from the
2nd hand onward are drawn from `split(round_state.rng_key)`, and `Env.step` does `del key` on its very first line after
receiving the key. The result is that **hands 2..9 of every hanchan use the same fixed set of 8 decks**, identical across
seeds, parallel environments, episodes and runs. `round_mode="single"` is unaffected;
**under `round_mode="half"`, 88.9% of training steps land on constant walls** (measured).

It does not crash and raises no error; tile conservation, zero-sum scoring, mask legality and end-of-game settlement all
pass as usual. It just quietly turns the training data into repeats of 8 decks. And because evaluation runs through
`libriichi.arena` and never touches mahjax, the symptom is "internal metrics look good, external strength does not
move". About 2.98B steps of RL in this project, from 2026-08-23 to 09-07 (≈131 GPU-h), fell into this trap.

If you are on an unpatched version, two changes to `_init` are enough (you do **not** need to upgrade to v0.1.3 — it
brings API breakage such as `step` requiring a key and a rewritten observation schema, with no benefit for this project):

```python
def _init(rng: PRNGKey, game_config=None) -> State:
    dealer_key, wall_key, round_key = jax.random.split(rng, 3)      # was: rng, subkey = split(rng)
    current_player = jnp.int8(jax.random.randint(dealer_key, (), 0, 4))   # was: rng
    ...
    deck = Tile.from_tile_id_to_tile(
        jax.random.permutation(wall_key, jnp.arange(136))).astype(jnp.int8)   # was: rng
    ...
    state = _make_state(..., rng_key=round_key)                     # this argument was missing
```

(The second change also fixes "the dealer and the wall share the same `rng`", which made the dealer a deterministic
function of the wall.)

**Verify after patching** — don't just look at the diff:

```bash
python - <<'EOF'
import jax, numpy as np, mahjax
env = mahjax.make("red_mahjong", round_mode="half", observe_type="dict")
a = np.asarray(jax.jit(env.init)(jax.random.PRNGKey(1)).round_state.rng_key)
b = np.asarray(jax.jit(env.init)(jax.random.PRNGKey(999)).round_state.rng_key)
assert not np.array_equal(a, b), "rng_key is still unseeded; the patch did not take effect"
print("OK", a, b)
EOF
```

## 3. Evaluation bridge dependency (libriichi, AGPL-3.0)

```bash
git clone https://github.com/Equim-chan/Mortal ~/Mortal   # next to the repo root also works
cd ~/Mortal/libriichi && cargo build --release            # produces libriichi.so
```

The evaluation scripts need `Mortal/mortal` on `PYTHONPATH`, plus a set of opponent weights in `baseline/` (the weights
are not distributed with this repository; see upstream for how to request them).

### 3.1 Anchored value fine-tuning patch (value line C', optional)

This project's current strongest model (12k duplicate `-0.75 ± 1.31` vs mortal_v4) relies on an **anchor loss** added
to the upstream trainer: during online training, the squared error between the Q-values of **all legal actions** and
those of a frozen base is penalized, suppressing the slow failure mode "the Q-values of untaken actions drift without
supervision → the action ranking erodes". The patch is inert for upstream — with no `[anchor]` section in the config, or
with `weight = 0`, behavior is identical to upstream.

```bash
cd ~/Mortal && patch -p1 < /path/to/TanyaoDojo/patches/anchor_train.patch
```

λ (i.e. `[anchor] weight`) is the main knob of this scheme, not an incidental decimal: at λ=0.5 the anchor pins the
policy in place, and about 40 hours of training give a 12k pairing of only `+0.19` (z=0.33); relaxed to 0.2, it reaches
`+1.56` (z=2.37) in 9 hours. Config: [`configs/online_selfplay.toml`](configs/online_selfplay.toml);
interpretation: [RESULTS.md](RESULTS.md).

## 4. Training data

**Not distributed**: the BC datasets are built from Tenhou Phoenix-room game logs, which are bound by Tenhou's terms;
this project does not redistribute the raw logs or any datasets derived from them. Bring your own logs and build the
datasets yourself with [`jax_rl/data_bridge/make_bc_dataset.py`](jax_rl/data_bridge/make_bc_dataset.py):

```bash
PYTHONPATH=~/mahjax python make_bc_dataset.py "<log glob>" <num games> <output dir> v2
```

## 5. Smoke self-check

```bash
python jax_rl/mjai_bot/test_tracker_diff.py "<log glob>" 100 v2   # the observation diff should match exactly
```
