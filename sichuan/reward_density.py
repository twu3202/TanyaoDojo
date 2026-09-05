"""
P2 设计前置测量:川麻的奖励到底稀不稀疏?

立直线上"信号密度 0.1%"是 GRP 势能塑形存在的唯一理由。川麻结算方式完全不同
(点杠即时给钱、逐家胡牌即时结算、终局查大叫),奖励**可能本来就是密的**。
密的话就不该照搬塑形——塑形不是免费的,它要一个训练好的势函数,且只在稀疏时值钱。

量三件事:
  · 非零奖励步占比(信号密度)
  · 每盘的非零奖励事件数
  · 奖励绝对值分布(终局结算是否压倒性地大于过程奖励)

用法: PYTHONPATH=~/mahjax ~/jax_env/bin/python reward_density.py [n_deals]
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_jax as E


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    env = E.make()
    step = jax.jit(env.step)
    init = jax.jit(env.init)

    def pick(mask, key):
        return jax.random.categorical(key, jnp.where(mask, 0.0, -jnp.inf))

    steps_tot = 0
    nz_steps = 0
    per_deal_events = []
    mags = []
    term_mags = []
    for d in range(n):
        key = jax.random.PRNGKey(10_000 + d)
        st = init(key)
        ev = 0
        k = 0
        while not bool(st.terminated) and k < 4000:
            key, k1, k2 = jax.random.split(key, 3)
            a = pick(st.legal_action_mask, k2)
            st = step(st, a, k1)
            r = np.asarray(st.rewards, np.float64)
            steps_tot += 1
            k += 1
            if np.abs(r).max() > 0:
                nz_steps += 1
                ev += 1
                mags.append(np.abs(r).max())
                if bool(st.terminated):
                    term_mags.append(np.abs(r).max())
        per_deal_events.append(ev)

    mags = np.array(mags)
    pe = np.array(per_deal_events)
    print(f"n={n} 盘  总步数={steps_tot:,}  平均每盘 {steps_tot/n:.1f} 步")
    print(f"信号密度(非零奖励步占比) = {100*nz_steps/steps_tot:.2f}%"
          f"   —— 立直线塑形前是 0.10%,塑形后 1.10%")
    print(f"每盘非零奖励事件 = {pe.mean():.2f} ± {pe.std():.2f}(中位 {np.median(pe):.0f})")
    print(f"奖励绝对值:中位 {np.median(mags):.0f}  均值 {mags.mean():.1f}  最大 {mags.max():.0f}")
    if len(term_mags):
        tm = np.array(term_mags)
        print(f"  其中终局那一步:n={len(tm)}  中位 {np.median(tm):.0f}  "
              f"占全部奖励量级的 {100*tm.sum()/mags.sum():.1f}%")
    print()
    if nz_steps / steps_tot > 0.05:
        print("[判读] 奖励已经很密(>5%),**不需要照搬 GRP 势能塑形**;")
        print("       塑形的成本(训练势函数 + 多一层近似)在这里换不到东西。")
    else:
        print("[判读] 奖励偏稀疏,塑形仍有价值,需要为川麻单独训练势函数。")


if __name__ == "__main__":
    main()
