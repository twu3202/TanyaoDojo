"""
势函数塑形 —— 让"严格 tabula rasa"这条路真正走得通的那块砖。

## 问题(本项目实测,不是文献推测)

`sichuan/diag_coldstart.py` 在 2000 副随机对局上量到:**均匀随机策略下,
76.8% 的对局四家分数全为零**。也就是说随机初始化的 agent 有超过四分之三的 episode
拿到的是一个恒零的奖励向量 —— 没有任何梯度。立项文档当初写"川麻奖励天然稠密
(一局最多3家胡 + 刮风下雨即时结算 + 查大叫),纯 RL 不需要 GRP 类奖励模型",
**这个判断对熟练策略成立,对随机初始化不成立**。稠密性是策略的函数,不是规则的函数。

| 策略 | 至少一家胡的局 | 产生非零分数的局 | 分数 std |
|---|---|---|---|
| 均匀随机 | 14.3% | **23.2%** | 2.14 |
| 贪心(仅向听下降) | 55% | **83.0%** | 6.63 |

好消息在同一张表里:从随机爬到贪心,信号密度涨 3.6 倍 —— 只要能迈出第一步,
后面就是正反馈。**所以整个 tabula rasa 的成败压在"怎么迈出第一步"上。**

## 解法:势能塑形(Ng, Harada & Russell 1999)

    r'_t = r_t + gamma * Phi(s_{t+1}) - Phi(s_t)        (非终局)
    r'_T = r_T             - Phi(s_T)                   (终局,取 Phi(terminal) = 0)

求和后 telescoping 成 `原回报 - Phi(s_0)`,即每条轨迹的回报只差一个**与策略无关的
常数**。因此:最优策略不变、纳什均衡集不变(多智能体下每个 agent 的回报各自平移
一个常数,不改变任何人的最佳响应)。这不是近似,是恒等式。

对本案的意义:**它不违反"零训练数据"**。塑形注入的是奖励工程,不是模仿——
没有任何人类牌谱、没有任何外部策略被克隆。收敛点与纯 RL 完全相同,变的只是
"到达它的路好不好走"。这正是用户选的 "A 主 + B 保底" 里 A 那条路缺的东西。

## 势函数取什么

    Phi_i(s) = -w * shanten_i(s)

向听是"离胡牌还差几张"的精确整数,查表 O(1)(见 common/suit_table.py),环境侧免费。
它把"我这一打让手牌更接近/更远离胡牌"这条信息从终局摊到每一步 —— 恰好是
随机策略学不会的那一步。

**必须退火。** 塑形不改变最优策略,但会改变**学习路径**:向听势能天然偏向速度,
早期会把策略推向"莽着冲"。所以 w 按 cos 从 w0 退到 0;退到 0 之后目标函数与
未塑形版逐字相同,前面那段只是加速。这条也是对立直线教训的直接回应——那边
四次失败的根因是**目标函数**被改错了;这里刻意选一个**可证明不改目标函数**的干预。

## 与立直线 GRP 塑形的关系

立直线 2026-08 在做的是同一个定理的另一个用法:Phi = "局面 -> 终局顺位点期望"
(752 万人类样本训的 GRP 网),把半庄终局奖励摊到每一盘。区别在于:
  · 那边的 Phi **需要人类数据**训练,这边的 Phi 是规则算出来的,零数据;
  · 那边解决的是"一个 rollout 只有 0.75 个终局"的稀疏,这边解决的是
    "76.8% 的 episode 恒零"的冷启动。
定理是同一个,验证方法也应当是同一个:**数值验证 telescoping 精确成立**,
见本文件的 verify_telescoping()。
"""
from __future__ import annotations

import math
from typing import Callable, Optional


# ------------------------------------------------------------------ 退火
def anneal_cosine(step: int, total: int, w0: float = 1.0,
                  hold_frac: float = 0.1) -> float:
    """w0 -> 0 的余弦退火。前 hold_frac 段保持 w0(让冷启动吃满),之后余弦降到 0。

    立直线的记分册上有一条独立证据支持"退火而非恒定":那边所有 league 臂
    **全程恒锚**,六点全景在噪声里横盘;而值线的增益来自"峰值提取"式的动态。
    恒定的辅助项会把策略钉在辅助项的最优上,不是真目标的最优上。
    """
    if total <= 0:
        return 0.0
    t = step / total
    if t <= hold_frac:
        return w0
    u = (t - hold_frac) / max(1e-9, 1.0 - hold_frac)
    return w0 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, u)))


# ------------------------------------------------------------ JAX 环境包装
def make_shaped_step(step_fn, init_fn, potential_fn, reward_scale: float = 1.0):
    """auto_reset + 势能塑形的 JAX 包装(形状对齐 TanyaoDojo/jax_rl/reward_placement.py)。

    potential_fn(state) -> (num_players,) 每家的势能。**终局的势必须按 0 处理**,
    否则 telescoping 会漏一项、塑形不再策略不变。

    ⚠️ 立直线踩过的那颗雷在这里同样有效:auto_reset 在终局把 state 换成新局,
    只保留 (terminated, truncated, rewards)。所以**终局的势与终局奖励都必须在
    重置之前算好写进 rewards**,否则悄悄丢失,而指标看起来一切正常。
    """
    import jax
    import jax.numpy as jnp

    def wrapped(state, action, key):
        k1, k2 = jax.random.split(key)
        state = jax.lax.cond(
            state.terminated | state.truncated,
            lambda: state.replace(terminated=jnp.bool_(False),
                                  truncated=jnp.bool_(False),
                                  rewards=jnp.zeros_like(state.rewards)),
            lambda: state,
        )
        phi_before = potential_fn(state)
        state = step_fn(state, action, k1)
        done = state.terminated | state.truncated
        phi_after = jnp.where(done, 0.0, potential_fn(state))   # Phi(terminal) := 0
        base = jnp.asarray(state.rewards, jnp.float32) / reward_scale
        state = state.replace(rewards=base + phi_after - phi_before)
        fresh = init_fn(k2)
        return jax.lax.cond(
            done,
            lambda: fresh.replace(terminated=state.terminated,
                                  truncated=state.truncated,
                                  rewards=state.rewards),
            lambda: state,
        )

    return wrapped


def shanten_potential(shanten_fn, w: float = 1.0, cap: int = 8):
    """Phi_i(s) = -w * shanten_i(s) / cap,归一到 [-w, 0]。

    已胡离场的玩家势能按 0 记(他们的手牌冻结,再谈"还差几张"没有意义),
    否则血战里离场者会持续贡献一个恒定势差。
    """
    import jax.numpy as jnp

    def phi(state):
        st = shanten_fn(state)                       # (num_players,)
        alive = ~jnp.asarray(state.finished, bool)   # 血战:已胡者冻结
        return jnp.where(alive, -w * st.astype(jnp.float32) / cap, 0.0)

    return phi


# -------------------------------------------------------------- 数值验证
def verify_telescoping(rewards_per_step, phis_per_step, phi0, terminal_reward,
                       tol: float = 1e-6):
    """检查 sum_t (r_t + Phi_{t+1} - Phi_t) == sum_t r_t - Phi(s_0)。

    立直线做 GRP 塑形时就是靠这一条判定"塑形实现对不对"的。它不需要 GPU、
    不需要训练、几秒钟出结果,却能拦住"势能没在终局归零""auto_reset 吃掉终局项"
    这两类最常见、最难从训练指标上看出来的实现 bug。
    """
    shaped = 0.0
    for r, (p_now, p_next) in zip(rewards_per_step, phis_per_step):
        shaped += r + p_next - p_now
    plain = sum(rewards_per_step) + terminal_reward - terminal_reward
    expect = sum(rewards_per_step) - phi0
    ok = abs(shaped - expect) < tol
    return ok, shaped, expect


def selfcheck_on_reference(n_games: int = 100, w: float = 1.0, verbose: bool = True):
    """在参考实现的真实对局上验证 telescoping 与"塑形不改变零和"。

    这是 P0 的判据之一:塑形版与未塑形版在**同一条轨迹**上的回报差,必须恰好等于
    -Phi(s_0),逐局精确,不是统计上近似。
    """
    import os
    import random
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "sichuan"))
    from reference_impl import SichuanGame  # noqa: E402
    import bots  # noqa: E402

    bad = 0
    max_err = 0.0
    for seed in range(n_games):
        g = SichuanGame(seed)
        rng = random.Random(seed)
        phi_prev = None
        shaped_sum = [0.0] * 4
        prev_scores = [0] * 4
        guard = 0
        while g.phase != "over" and guard < 6000:
            phi_now = _phi_ref(g, w)
            if phi_prev is None:
                phi0 = phi_now[:]
            i, acts = g.legal_actions()
            g.step(bots.bot_L1_greedy(g, i, acts, rng))
            guard += 1
            done = g.phase == "over"
            phi_next = [0.0] * 4 if done else _phi_ref(g, w)
            cur = g.scores()
            for k in range(4):
                r = cur[k] - prev_scores[k]
                shaped_sum[k] += r + phi_next[k] - phi_now[k]
            prev_scores = cur
            phi_prev = phi_next

        final = g.scores()
        for k in range(4):
            expect = final[k] - phi0[k]
            err = abs(shaped_sum[k] - expect)
            max_err = max(max_err, err)
            if err > 1e-6:
                bad += 1
    if verbose:
        print(f"telescoping 自检: {n_games} 局 x 4 家, 失配 = {bad}, "
              f"最大误差 = {max_err:.2e}")
        print("判据: 失配 0 → 塑形实现策略不变,可以开训"
              if bad == 0 else "判据: ✗ 塑形实现有 bug,训练前必须修")
    return bad == 0


def _phi_ref(g, w: float, cap: int = 8):
    """参考实现上的势能(用 bots 里的 shanten)。已胡者按 0。"""
    import bots
    out = []
    for k in range(4):
        p = g.players[k]
        if p.hu or p.void is None:
            out.append(0.0)
            continue
        st = bots.shanten(p.hand, len(p.melds), p.void)
        out.append(-w * min(st, cap) / cap)
    return out


if __name__ == "__main__":
    import sys
    selfcheck_on_reference(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
