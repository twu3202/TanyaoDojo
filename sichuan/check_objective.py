"""
开 P2 之前先验一条 P2 完全依赖的前提:川麻线"结构性免疫目标错配"。

立直线的教训是:mahjax 只把 order_points 写进 state.score,**从不写进 rewards**,
四次 RL 因此优化了一个不是评测目标的函数,烧掉约 1e10 步。那次错误是"读代码觉得
没问题"造成的,所以这次直接测——不看注释,只看数。

三项断言:
  A. 逐步 rewards 累加 == 终局 score(奖励确实承载了结算,含查大叫/呼叫转移)
  B. 终局 score 零和(分数守恒,没有凭空产生的钱)
  C. 过 auto_reset 包装后终局那一步的 reward 不丢(立直线正是栽在这一环)

用法: PYTHONPATH=~/mahjax python check_objective.py [n_deals]
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_jax as E
from mahjax.wrappers.auto_reset_wrapper import auto_reset


def pick(mask, key):
    """合法动作里均匀随机 —— 定死取首个合法会绕开胡/杠/流局等结算路径。"""
    logits = jnp.where(mask, 0.0, -jnp.inf)
    return jax.random.categorical(key, logits)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    env = E.make()
    step_raw = jax.jit(env.step)
    step_auto = jax.jit(auto_reset(env.step, env.init))
    init = jax.jit(env.init)

    bad_a = bad_b = bad_c = 0
    max_err = 0.0
    for d in range(n):
        key = jax.random.PRNGKey(d)
        st = init(key)
        acc = np.zeros(E.NUM_PLAYERS, np.float64)
        st_auto = st
        acc_auto = np.zeros(E.NUM_PLAYERS, np.float64)
        steps = 0
        while not bool(st.terminated) and steps < 4000:
            key, k1, k2 = jax.random.split(key, 3)
            a = pick(st.legal_action_mask, k2)
            st = step_raw(st, a, k1)
            acc += np.asarray(st.rewards, np.float64)
            # 同一条轨迹在 auto_reset 包装下重放,验证终局 reward 不被吞
            st_auto = step_auto(st_auto, a, k1)
            acc_auto += np.asarray(st_auto.rewards, np.float64)
            steps += 1

        final = np.asarray(st.score, np.float64)
        err = np.abs(acc - final).max()
        max_err = max(max_err, err)
        if err > 1e-6:
            bad_a += 1
            if bad_a <= 3:
                print(f"  [A 失败] deal {d}: Σrewards={acc} 终局 score={final}")
        if abs(final.sum()) > 1e-6:
            bad_b += 1
            if bad_b <= 3:
                print(f"  [B 失败] deal {d}: 终局 score 和 = {final.sum()}")
        if np.abs(acc_auto - acc).max() > 1e-6:
            bad_c += 1
            if bad_c <= 3:
                print(f"  [C 失败] deal {d}: auto_reset 下 Σrewards={acc_auto} vs 裸 {acc}")

    print(f"n={n} 盘")
    print(f"  A 逐步 rewards 累加 == 终局 score : {'通过' if bad_a==0 else f'失败 {bad_a}'}"
          f"  (最大偏差 {max_err:.3g})")
    print(f"  B 终局 score 零和                 : {'通过' if bad_b==0 else f'失败 {bad_b}'}")
    print(f"  C auto_reset 不吞终局 reward      : {'通过' if bad_c==0 else f'失败 {bad_c}'}")
    if bad_a == bad_b == bad_c == 0:
        print("\n[结论] 训练奖励 == 评测目标(该盘分数转移),川麻线不存在立直线那层错配。")


if __name__ == "__main__":
    main()
