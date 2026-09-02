"""
复核诊断报告主张⑥:"熵系数方向可能反了"(报告自陈不确定)。

前提(先查清再谈):**评测走 argmax**(`mjai_bot/jax_engine.py:280`),
所以 ent_coef 不会在评测时直接花掉顺位点——它只能通过"探索质量"间接起作用。

本仪表用自家中心化 Q-critic 量温度对比:
  V_τ(s) = Σ_a π_τ(a|s) Q(s,a),  π_τ ∝ softmax(logits/τ) (非法动作置 -inf)
  Δ(τ)   = V_τ − V_{τ=1}         (归一化尺度,×135 → 顺位点)
τ→0 即评测所用的贪心策略,τ=1 即训练采样策略。Δ 的符号与量级回答
"策略的随机性在它自己的价值模型下值不值钱"。

⚠️ 第一版用 max_a Q 当基线,得出 0.836 pt/决策的"熵税",是**错的**:
9 个合法动作上各带估计噪声时 E[max] 天然高出均值约 1.5σ,量到的是 critic
的高估偏差而非熵的代价(佐证:该值在 τ∈[0.5,2] 上纹丝不动)。温度对比两
边都是期望、无 max 算子,才是无偏的问法。

用法:
  PYTHONPATH=~/mahjax python entropy_tax.py actor.pkl critic.pkl \
      [--num-envs 64] [--num-steps 512] [--channels 256] [--blocks 10]
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
from net_lean import LeanACNet, LeanQCriticNet          # noqa: E402
from obs_oracle import observe_oracle_v2                # noqa: E402

NEG = -1e9
PT_SCALE = 135.0
LEAN_PLANES, LEAN_SCALARS = 36, 32
TEMPS = (0.01, 0.5, 0.75, 1.0, 1.5, 2.0)   # 0.01 ≈ 贪心(评测所用)
BASE_IX = TEMPS.index(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actor")
    ap.add_argument("critic")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--num-steps", type=int, default=512)
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--critic-channels", type=int, default=128)
    ap.add_argument("--critic-blocks", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = mahjax.make("red_mahjong", round_mode="half", observe_type="dict")
    step_fn = auto_reset(env.step, env.init)

    with open(args.actor, "rb") as f:
        a_params = pickle.load(f)
    with open(args.critic, "rb") as f:
        c_params = pickle.load(f)
    net = LeanACNet(channels=args.channels, blocks=args.blocks)
    critic = LeanQCriticNet(channels=args.critic_channels, blocks=args.critic_blocks)

    def one_step(carry, _):
        state, rng = carry
        rng, k_act, k_env = jax.random.split(rng, 3)
        obs = jax.vmap(observe_oracle_v2)(state)
        obs_a = {"planes": obs["planes"][..., :LEAN_PLANES],
                 "scalars": obs["scalars"][..., :LEAN_SCALARS]}
        mask = state.legal_action_mask.astype(jnp.bool_)
        logits, _ = net.apply(a_params, obs_a)
        logits = jnp.where(mask, logits, NEG)
        q_all = critic.apply(c_params, obs)

        def v_at(t):
            p = jax.nn.softmax(jnp.where(mask, logits / t, NEG), axis=-1)
            return jnp.sum(p * q_all, axis=-1)

        v_t = jnp.stack([v_at(t) for t in TEMPS])            # (T, B)
        p1 = jax.nn.softmax(logits, axis=-1)
        ent = -jnp.sum(p1 * jnp.log(jnp.maximum(p1, 1e-12)), axis=-1)
        n_legal = mask.sum(-1)
        choice = n_legal > 1                                  # 单一合法动作处熵恒 0

        act = jax.random.categorical(k_act, logits)
        nxt = jax.vmap(step_fn)(state, act, jax.random.split(k_env, args.num_envs))
        done = jnp.asarray(state.terminated | state.truncated, dtype=jnp.bool_)
        return (nxt, rng), (v_t, ent, choice, n_legal, done)

    rng = jax.random.PRNGKey(args.seed)
    rng, k = jax.random.split(rng)
    st = jax.vmap(env.init)(jax.random.split(k, args.num_envs))
    _, out = lax.scan(one_step, (st, rng), None, length=args.num_steps)
    v_t, ent, choice, n_legal, done = jax.tree.map(np.asarray, out)

    m = choice
    n = int(m.sum())
    n_ep = max(1, int(done.sum()))
    print(f"决策点(有得选)={n:,}  半庄终局={n_ep:,}  每半庄≈{m.sum()/n_ep:.0f} 个")
    print(f"平均合法动作数={n_legal[m].mean():.2f}  平均熵={ent[m].mean():.4f} nats "
          f"(等效动作数 {np.exp(ent[m].mean()):.2f})")
    print()
    print("温度 τ    Δ = V_τ − V_{τ=1}(顺位点/决策)")
    base = v_t[:, BASE_IX][m]
    for i, t in enumerate(TEMPS):
        d = (v_t[:, i][m] - base) * PT_SCALE
        se = d.std(ddof=1) / np.sqrt(len(d))
        tag = "  ← 评测所用(贪心)" if t < 0.1 else ("  ← 训练采样" if t == 1.0 else "")
        print(f"  {t:<6.2f}  {d.mean():+.5f}  ± {1.96*se:.5f}{tag}")
    g = (v_t[:, 0][m] - base) * PT_SCALE
    print()
    print(f"[判读] 贪心相对采样策略每决策 {g.mean():+.5f} pt(95% CI ±{1.96*g.std(ddof=1)/np.sqrt(len(g)):.5f})")
    print("      评测本就走贪心 → 这份差值我们已经在收,ent_coef 不在评测端花钱;")
    print("      其影响只能经由探索质量,而探索质量无法用本仪表判向。")


if __name__ == "__main__":
    main()
