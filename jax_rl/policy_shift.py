"""
复核一条我们自己的、正在被用来做决策的判断:"锚把策略钉住了,所以该放开信任域"。

依据一直是 mag_kl = 0.023 这个数——但 0.023 nats 到底意味着多大的行为差异,
从没换算过。本脚本在同一批状态上直接量两个策略的**行为**距离:

  · top-1 一致率(评测走 argmax,这才是真正被打分的量)
  · 双向 KL、总变差 TV
  · 分动作类型拆解(弃张 / 鸣牌 / 立直)

判读:
  一致率 ≥ 99%  → 策略几乎没动,"锚过紧"成立,放开信任域是对的
  一致率 ≤ 95%  → 策略其实动了不少,瓶颈不在锚,放开也未必有用

用法:
  PYTHONPATH=~/mahjax ~/jax_env/bin/python policy_shift.py rl.pkl base.pkl \
      [--num-envs 128] [--num-steps 64]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import mahjax
from mahjax.wrappers.auto_reset_wrapper import auto_reset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from net_lean import LeanACNet          # noqa: E402
from obs_v2 import observe_v2           # noqa: E402

NEG = -1e9
TSUMOGIRI, RIICHI = 71, 72
CALL_LO, CALL_HI = 74, 83


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rl")
    ap.add_argument("base")
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--num-steps", type=int, default=64)
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obs", default="v2", choices=("v2", "lean"))
    args = ap.parse_args()

    global observe_v2
    if args.obs == "lean":
        from obs_lean import observe_lean as observe_v2   # noqa: F811

    env = mahjax.make("red_mahjong", round_mode="half", observe_type="dict")
    step_fn = auto_reset(env.step, env.init)
    net = LeanACNet(channels=args.channels, blocks=args.blocks)
    with open(args.rl, "rb") as f:
        p_rl = pickle.load(f)
    with open(args.base, "rb") as f:
        p_bs = pickle.load(f)

    def one_step(carry, _):
        state, rng = carry
        rng, k_env = jax.random.split(rng)
        obs = jax.vmap(observe_v2)(state)
        mask = state.legal_action_mask.astype(jnp.bool_)
        l_rl = jnp.where(mask, net.apply(p_rl, obs)[0], NEG)
        l_bs = jnp.where(mask, net.apply(p_bs, obs)[0], NEG)
        a_rl = jnp.argmax(l_rl, -1)
        a_bs = jnp.argmax(l_bs, -1)
        p1 = jax.nn.softmax(l_rl, -1)
        p2 = jax.nn.softmax(l_bs, -1)
        lg = jnp.log(jnp.maximum(p1, 1e-12)) - jnp.log(jnp.maximum(p2, 1e-12))
        kl_ab = jnp.sum(p1 * lg, -1)
        kl_ba = jnp.sum(p2 * -lg, -1)
        tv = 0.5 * jnp.sum(jnp.abs(p1 - p2), -1)
        # 轨迹由 RL 策略生成(它才是被部署的那个)
        nxt = jax.vmap(step_fn)(state, a_rl, jax.random.split(k_env, args.num_envs))
        return (nxt, rng), (a_rl, a_bs, kl_ab, kl_ba, tv, mask.sum(-1))

    rng = jax.random.PRNGKey(args.seed)
    rng, k = jax.random.split(rng)
    st = jax.vmap(env.init)(jax.random.split(k, args.num_envs))
    _, out = lax.scan(one_step, (st, rng), None, length=args.num_steps)
    a_rl, a_bs, kl_ab, kl_ba, tv, n_legal = [np.asarray(x).ravel() for x in out]

    m = n_legal > 1
    agree = (a_rl == a_bs)[m]
    print(f"决策点(有得选) n={m.sum():,}  平均合法动作 {n_legal[m].mean():.2f}")
    print(f"  top-1 一致率 = {agree.mean()*100:.2f}%   (不一致 {100-agree.mean()*100:.2f}%)")
    print(f"  KL(RL‖base) = {kl_ab[m].mean():.4f}   KL(base‖RL) = {kl_ba[m].mean():.4f}")
    print(f"  总变差 TV    = {tv[m].mean():.4f}")

    def bucket(name, sel):
        s = sel & m
        if s.sum() < 50:
            return
        print(f"    {name:<8}: n={int(s.sum()):>7,}  一致率 {(a_rl==a_bs)[s].mean()*100:5.2f}%  "
              f"TV {tv[s].mean():.4f}")
    bucket("弃张", (a_rl <= 36) | (a_rl == TSUMOGIRI))
    bucket("立直", a_rl == RIICHI)
    bucket("鸣牌", (a_rl >= CALL_LO) & (a_rl <= CALL_HI))
    bucket("其它", (a_rl > 36) & (a_rl != TSUMOGIRI) & (a_rl != RIICHI) &
           ~((a_rl >= CALL_LO) & (a_rl <= CALL_HI)))

    a = agree.mean()
    print()
    if a >= 0.99:
        print("[判读] 一致率 ≥99%:策略几乎没动,'锚过紧'成立 → 放开信任域是对的")
    elif a <= 0.95:
        print("[判读] 一致率 ≤95%:策略其实动了不少,瓶颈不在锚 → 放开未必有用")
    else:
        print("[判读] 一致率在 95~99%:策略动了但有限,放开信任域值得一试但不是必然解")


if __name__ == "__main__":
    main()
