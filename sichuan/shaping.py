"""
川麻势能塑形(P2)。

为什么需要:`reward_density.py` 实测随机策略下非零奖励步只占 **0.36%**,
中位数是每盘 **0** 个非零奖励事件——随机打没人听牌,查大叫也无钱可结,
冷启动迈不出第一步。目标虽然没错配(见 `check_objective.py`),信号仍然稀疏。

为什么不照搬立直线的 GRP:GRP 要一个"局面 → 期望终局顺位点"的回归模型,
是从 752 万人类对局训出来的。川麻线是从零开始、没有人类数据,也不该引入。
这里用**手工势函数**:

    φ_i(s) = -w · shanten_i(s)      (已胡离场者 φ=0,没有further进展可言)

按 Ng et al. 1999,**任何** φ 的势能差都不改变最优策略,所以手工 φ 不引入偏置;
它只把"向听前进了一步"这件事立刻兑现成奖励,而不必等到胡牌那一刻。

    r'_i = r_i + γ·φ_i(s') − φ_i(s)        γ=1
    终局: φ(terminal) ≡ 0,整条轨迹加总 = −φ_i(s_0),每盘一个常数 → 策略不变

用法(训练器里):
    STEP_FN = auto_reset_shaped(env.step, env.init, w=0.5)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from sichuan.env_jax import NUM_PLAYERS, State, _shanten_fn

REWARD_SCALE = 8.0     # 奖励归一化尺度(实测单次结算中位 3、最大 12)


def shanten_all(state: State) -> jnp.ndarray:
    """(4,) 四家向听。已胡离场者记 0(不再有进展空间)。"""
    def one(i):
        return _shanten_fn(state.hand[i].astype(jnp.int32),
                           state.n_melds[i].astype(jnp.int32),
                           state.void[i].astype(jnp.int32)).astype(jnp.float32)
    sh = jax.vmap(one)(jnp.arange(NUM_PLAYERS))
    return jnp.where(state.finished, 0.0, sh)


def potential(state: State, w: float) -> jnp.ndarray:
    """φ(s) = -w · shanten,(4,)。终局态由包装器置 0,不在这里判。"""
    return -w * shanten_all(state)


def auto_reset_shaped(step_fn, init_fn, w: float = 0.5):
    """带势能塑形的 auto_reset。与 mahjax 原版行为一致,只是改写 rewards。

    ⚠️ 顺序很重要:必须在 step **之前**取 φ(s)、在 step 之后但**重置之前**取 φ(s'),
    否则 φ(s') 取到的是新一盘的开局手牌 —— 立直线上正是这一环出过事。
    """
    def wrapped(state: State, action, key):
        key1, key2 = jax.random.split(key)
        state = jax.lax.cond(
            state.terminated | state.truncated,
            lambda: state.replace(terminated=jnp.bool_(False),
                                  truncated=jnp.bool_(False),
                                  rewards=jnp.zeros_like(state.rewards)),
            lambda: state,
        )
        phi_before = potential(state, w)
        nxt = step_fn(state, action, key1)
        done = nxt.terminated | nxt.truncated
        phi_after = jnp.where(done, 0.0, potential(nxt, w))   # 终局势能恒 0
        raw = nxt.rewards / REWARD_SCALE
        shaped = raw + (phi_after - phi_before)
        nxt = nxt.replace(rewards=shaped)

        init_state = init_fn(key2)
        return jax.lax.cond(
            done,
            lambda: init_state.replace(terminated=nxt.terminated,
                                       truncated=nxt.truncated,
                                       rewards=nxt.rewards),
            lambda: nxt,
        )
    return wrapped
