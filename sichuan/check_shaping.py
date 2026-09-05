"""
验塑形的策略不变性(数值,不是读代码)。

势能塑形只有在**精确 telescoping** 时才不改变最优策略。断言:

  Σ_t r'_i(t)  ==  Σ_t r_i(t)/SCALE  −  φ_i(s_0)

即整条轨迹上塑形项加总只剩一个开局常数。任何一处 φ 取错时机(比如在 auto_reset
重置之后才取 φ(s'))都会让这个等式崩掉——立直线上就踩过。

用法: PYTHONPATH=~/mahjax ~/jax_env/bin/python check_shaping.py [n_deals] [w]
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sichuan import env_jax as E
from sichuan.shaping import auto_reset_shaped, potential, REWARD_SCALE


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    w = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    env = E.make()
    raw_step = jax.jit(env.step)
    shaped_step = jax.jit(auto_reset_shaped(env.step, env.init, w))
    init = jax.jit(env.init)
    phi = jax.jit(lambda s: potential(s, w))

    def pick(mask, key):
        return jax.random.categorical(key, jnp.where(mask, 0.0, -jnp.inf))

    bad = 0
    max_err = 0.0
    dens_raw = dens_shaped = steps = 0
    for d in range(n):
        key = jax.random.PRNGKey(50_000 + d)
        st = init(key)
        st_s = st
        phi0 = np.asarray(phi(st), np.float64)
        acc_raw = np.zeros(E.NUM_PLAYERS, np.float64)
        acc_shp = np.zeros(E.NUM_PLAYERS, np.float64)
        k = 0
        while not bool(st.terminated) and k < 4000:
            key, k1, k2 = jax.random.split(key, 3)
            a = pick(st.legal_action_mask, k2)
            st = raw_step(st, a, k1)
            st_s = shaped_step(st_s, a, k1)
            r = np.asarray(st.rewards, np.float64)
            rs = np.asarray(st_s.rewards, np.float64)
            acc_raw += r
            acc_shp += rs
            steps += 1
            dens_raw += int(np.abs(r).max() > 0)
            dens_shaped += int(np.abs(rs).max() > 1e-9)
            k += 1
        expect = acc_raw / REWARD_SCALE - phi0
        err = np.abs(acc_shp - expect).max()
        max_err = max(max_err, err)
        if err > 1e-4:
            bad += 1
            if bad <= 3:
                print(f"  [失败] deal {d}: Σ塑形={acc_shp}  期望={expect}")

    print(f"n={n} 盘  w={w}")
    print(f"  telescoping 精确性: {'通过' if bad == 0 else f'失败 {bad} 盘'}"
          f"  (最大偏差 {max_err:.3g})")
    print(f"  信号密度: 塑形前 {100*dens_raw/steps:.2f}%  →  塑形后 "
          f"{100*dens_shaped/steps:.2f}%  (×{dens_shaped/max(dens_raw,1):.1f})")
    if bad == 0:
        print("\n[结论] 塑形项整条轨迹加总 = 开局常数,策略不变性成立。")


if __name__ == "__main__":
    main()
