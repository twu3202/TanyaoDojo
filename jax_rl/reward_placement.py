"""
顺位对齐的奖励包装(2026-08-23 目标错配修正)。

背景(实测证据,见 MAHJAX_MIGRATION 坑志):
  mahjax 的 state.rewards **永远是素点转移**(百点单位),任何 round_mode 都一样;
  order_points 只影响终局的 state.round_state.score,**从不进 rewards**。而
  round_mode="single" 让 episode 在一盘结束即终止。于是此前全部 RL 训练的目标是
  "在孤立一盘内最大化素点期望",而竞技场评的是"整个半庄的顺位点 [90,45,0,-135]"。
  两者是不同的函数——这是四次 RL 失败(-19.5/-14.8/-9.7/-8.4)最可能的共同根因:
  单盘素点最大化器没有"第四名"的概念,应当无限激进,而实测失分正是四位率偏高。

本模块提供 auto_reset 的替代:episode 内奖励恒为 0,**终局发竞技场同款顺位点**。
  reward_t = 0                              (非终局)
  reward_T = PT[rank(final_score)] / SCALE  (终局,PT 默认 [90,45,0,-135])
配 round_mode="half"(= 竞技场的半庄)与 gae_lambda=1.0(纯 MC,终局奖励下无偏)。

注意 auto_reset 在终局把 state 换成新局、仅保留 (terminated, truncated, rewards),
终局分数会丢失,故顺位必须在重置前算好并写入 rewards——本模块即做这件事。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# 竞技场同款顺位点(神圣协议),归一到 O(1)
ARENA_PT = jnp.array([90.0, 45.0, 0.0, -135.0], dtype=jnp.float32)
PT_SCALE = 135.0


def placement_reward(score: jnp.ndarray, pt: jnp.ndarray = ARENA_PT,
                     scale: float = PT_SCALE) -> jnp.ndarray:
    """(4,) 终局分数 → (4,) 顺位点/scale。并列按座次序(与竞技场一致)。"""
    order = jnp.argsort(-score)                 # 名次 r 上坐着谁
    rank_pt = jnp.zeros(4, jnp.float32).at[order].set(pt)
    return rank_pt / scale


def auto_reset_placement(step_fn, init_fn, pt: jnp.ndarray = ARENA_PT,
                         scale: float = PT_SCALE):
    """替代 mahjax.wrappers.auto_reset:丢弃素点流,只在终局发顺位点。"""

    def wrapped(state, action, key):
        key1, key2 = jax.random.split(key)
        state = jax.lax.cond(
            state.terminated | state.truncated,
            lambda: state.replace(terminated=jnp.bool_(False), truncated=jnp.bool_(False),
                                  rewards=jnp.zeros_like(state.rewards)),
            lambda: state,
        )
        state = step_fn(state, action, key1)
        done = state.terminated | state.truncated
        # 终局分数已含 order_points(单调于名次,不影响排序);重置前算好顺位奖励
        rew = jnp.where(
            done,
            placement_reward(state.round_state.score.astype(jnp.float32), pt, scale),
            jnp.zeros_like(state.rewards),
        )
        state = state.replace(rewards=rew)
        init_state = init_fn(key2)
        return jax.lax.cond(
            done,
            lambda: init_state.replace(terminated=state.terminated,
                                       truncated=state.truncated, rewards=state.rewards),
            lambda: state,
        )

    return wrapped


# ---------------------------------------------------------------- GRP 势函数塑形
# 终局奖励虽然对了,但每个 rollout 只有约 0.75 个半庄终局(实测 eps_rate=2.29e-5),
# 10 亿步跑完策略几乎没动(mag_kl 0.011)。潜能塑形把终局信号摊到每一盘的边界:
#   r'_t = Φ(s_{t+1}) − Φ(s_t)   (非终局)
#   r'_T = pt − Φ(s_T)           (终局)
# 求和телескоп为 pt − Φ(s_0),与原目标只差一个与策略无关的常数 → **策略不变**
# (Ng et al. 1999 的势能塑形定理),但信号密度 ×8(一副半庄 8 局)。
# Φ 由 GRP(局面→终局顺位点期望)给出,752 万人类对局样本训练,解释方差 0.313。

import flax.linen as nn


class GRPNet(nn.Module):
    """局面(15 维)→ 四座终局顺位点期望(已按 PT_SCALE 归一)。与 grp_train.py 同构。"""

    @nn.compact
    def __call__(self, x):
        for _ in range(3):
            x = nn.relu(nn.Dense(128)(x))
        return nn.Dense(4)(x)


def grp_features(rs) -> jnp.ndarray:
    """从 round_state 取 GRP 特征。分数单位:env 为百点,训练时用的是点/50000。"""
    sc = rs.score.astype(jnp.float32) / 500.0            # 250(=25000点) → 0.5
    return jnp.concatenate([
        jnp.array([rs.round / 8.0,
                   jnp.clip(rs.honba, 0, 8) / 8.0,
                   jnp.clip(rs.kyotaku, 0, 8) / 8.0], jnp.float32),
        sc,
        sc - sc.mean(),
        jax.nn.one_hot(rs.dealer, 4, dtype=jnp.float32),
    ])


def auto_reset_shaped(step_fn, init_fn, grp_params, pt=ARENA_PT, scale=PT_SCALE):
    """顺位终局奖励 + GRP 潜能塑形(策略不变,信号密度 ×8)。"""
    net = GRPNet()

    def phi(state):
        return net.apply(grp_params, grp_features(state.round_state)[None])[0]  # (4,)

    def wrapped(state, action, key):
        key1, key2 = jax.random.split(key)
        state = jax.lax.cond(
            state.terminated | state.truncated,
            lambda: state.replace(terminated=jnp.bool_(False), truncated=jnp.bool_(False),
                                  rewards=jnp.zeros_like(state.rewards)),
            lambda: state,
        )
        phi_before = phi(state)
        state = step_fn(state, action, key1)
        done = state.terminated | state.truncated
        phi_after = phi(state)
        term_r = placement_reward(state.round_state.score.astype(jnp.float32), pt, scale)
        rew = jnp.where(done, term_r - phi_before, phi_after - phi_before)
        state = state.replace(rewards=rew)
        init_state = init_fn(key2)
        return jax.lax.cond(
            done,
            lambda: init_state.replace(terminated=state.terminated,
                                       truncated=state.truncated, rewards=state.rewards),
            lambda: state,
        )

    return wrapped
