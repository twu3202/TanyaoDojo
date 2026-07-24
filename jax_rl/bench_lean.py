"""
R2 选型微基准:上游 ACNet(dict obs + 双 transformer 抽取器)vs LeanACNet。

测 fwd+bwd(策略 CE + 价值 MSE 的代表性损失)的每样本耗时,两点法扣除编译。
另做 obs_lean 正确性冒烟:真实 env state 上取观测,查形状/NaN/取值域。
用法:PYTHONPATH=~/mahjax:~/mahjax/examples python bench_lean.py [mb] [iters]
"""
from __future__ import annotations
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, "/home/r/mahjax/examples")
from networks.red_network import ACNet
from net_lean import LeanACNet
from obs_lean import observe_lean, NUM_PLANES, NUM_SCALARS

MB = int(sys.argv[1]) if len(sys.argv) > 1 else 512
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 20


def fake_dict_obs(rng, mb):
    r = np.random.default_rng(rng)
    return {
        "hand": jnp.asarray(r.integers(-1, 37, (mb, 14)), jnp.int32),
        "last_draw": jnp.asarray(r.integers(-1, 37, (mb,)), jnp.int32),
        "action_history": jnp.asarray(r.integers(-1, 4, (mb, 3, 200)), jnp.int32),
        "shanten_count": jnp.asarray(r.integers(0, 7, (mb,)), jnp.int32),
        "furiten": jnp.zeros((mb,), jnp.int32),
        "scores": jnp.asarray(r.integers(0, 500, (mb, 4)), jnp.int32),
        "round": jnp.zeros((mb,), jnp.int32),
        "honba": jnp.zeros((mb,), jnp.int32),
        "kyotaku": jnp.zeros((mb,), jnp.int32),
        "prevalent_wind": jnp.zeros((mb,), jnp.int32),
        "seat_wind": jnp.zeros((mb,), jnp.int32),
        "dora_indicators": jnp.asarray(r.integers(-1, 37, (mb, 5)), jnp.int32),
    }


def fake_lean_obs(rng, mb):
    r = np.random.default_rng(rng)
    return {
        "planes": jnp.asarray(r.random((mb, 34, NUM_PLANES)), jnp.float32),
        "scalars": jnp.asarray(r.random((mb, NUM_SCALARS)), jnp.float32),
    }


def bench(name, net, obs):
    params = net.init(jax.random.PRNGKey(0), obs)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    labels = jnp.zeros((MB,), jnp.int32)
    targets = jnp.zeros((MB,), jnp.float32)

    def loss_fn(p):
        logits, value = net.apply(p, obs)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return ce + ((value - targets) ** 2).mean()

    step = jax.jit(jax.grad(loss_fn))
    g = step(params)
    jax.block_until_ready(g)  # 编译+首跑
    t0 = time.perf_counter()
    for _ in range(ITERS):
        g = step(params)
    jax.block_until_ready(g)
    dt = (time.perf_counter() - t0) / ITERS
    per_sample_us = dt / MB * 1e6
    print(f"{name:10s} params={n_params/1e6:6.2f}M  fwd+bwd={dt*1e3:8.2f} ms/iter  "
          f"{per_sample_us:7.2f} us/sample")
    return dt


def smoke_obs_lean():
    from mahjax.red_mahjong.env import RedMahjong

    env = RedMahjong(round_mode="single")
    state = env.init(jax.random.PRNGKey(7))
    obs = jax.jit(observe_lean)(state)
    p, s = np.asarray(obs["planes"]), np.asarray(obs["scalars"])
    assert p.shape == (34, NUM_PLANES) and s.shape == (NUM_SCALARS,), (p.shape, s.shape)
    assert np.isfinite(p).all() and np.isfinite(s).all()
    assert (p >= 0).all() and (p <= 1.0 + 1e-6).all(), (p.min(), p.max())
    n_hand = p[:, 0:4].sum()  # >=1..>=4 四平面之和 = 手牌总张数(庄家刚摸 = 14)
    assert n_hand in (13.0, 14.0), n_hand
    print(f"obs_lean smoke OK: planes{p.shape} scalars{s.shape} "
          f"hand_tiles={n_hand:.0f} visible_max={p[:, 18].max()*4:.0f}")


if __name__ == "__main__":
    print(f"device={jax.devices()[0].platform}  mb={MB}  iters={ITERS}")
    smoke_obs_lean()
    d_up = bench("upstream", ACNet(), fake_dict_obs(0, MB))
    d_ln = bench("lean", LeanACNet(), fake_lean_obs(1, MB))
    print(f"speedup(update侧) = {d_up/d_ln:.1f}x")
