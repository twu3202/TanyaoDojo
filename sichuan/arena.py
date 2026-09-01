"""
川麻复式竞技场 —— 强度的唯一裁决口径。

协议(照搬立直线那套"神圣不可变"的形状,只换了记分单位):
  · 同一副牌山(seed 决定洗牌)打 4 遍,**挑战者轮流坐 4 个座位**,其余三座是对手;
  · 挑战者成绩 = 4 次的平均分 → 座次偏差被完全消除,牌运被共同随机数抵消;
  · 报 avg_pt = 每盘平均分 ± 95% CI。CI 全域 > 0 = 真的强于对手。

与立直线的两处刻意差异:
  1) **记分单位是"钱"不是顺位点**。川麻现实中就是逐盘结账(底分 x 2^番 + 刮风下雨
     + 查大叫),所以 episode = 一盘、奖励 = 该盘分数转移 —— 训练目标与评测目标
     **是同一个函数**。立直线四连负的根因(单盘素点 vs 半庄顺位点)在这里被结构性消除,
     不是靠小心,是靠这个变种本来就这么算钱。
  2) 没有 champion。没有公开的川麻 AI 可打,所以对手是自造的规则梯子(bots.py L0–L3)。

内置两个免费自检(立直线调研里的 T0-2 / T0-8,零 GPU):
  · A(pi,pi) == 0:四座同策略的复式期望必然恒等于零(代数事实)。测出来不是
    0.000 ± 噪声,就是竞技场或环境有 bug —— 在任何强度结论之前先过这一关。
  · rho:同一副牌山内 4 次轮换的组内相关。它决定复式到底买到了多少方差缩减,
    也决定"要测多少盘才能分辨 X 分的差距"。

用法:
    python3 arena.py selfcheck 200          # A(pi,pi)==0
    python3 arena.py ladder 300             # L1vsL0 / L2vsL1 / L3vsL2 全梯
    python3 arena.py duel L2 L1 500
"""
from __future__ import annotations

import math
import random
import statistics
import sys

from reference_impl import SichuanGame
import bots


def play_deal(seed: int, seat_policies, rng_seed: int) -> list:
    """打一盘,返回四家分数。seed 决定牌山(复式的关键:同 seed = 同牌山)。"""
    g = SichuanGame(seed)
    rng = random.Random(rng_seed)
    guard = 0
    while g.phase != "over" and guard < 6000:
        i, acts = g.legal_actions()
        g.step(seat_policies[i](g, i, acts, rng))
        guard += 1
    assert g.phase == "over", "对局未终止(疑似死循环)"
    return g.scores()


def duplicate(n_deals: int, challenger, opponent, verbose_every: int = 0):
    """复式 1v3:挑战者轮坐 4 座打同一副牌山。返回 (每盘平均分列表, 每副牌山4次的原始值)。"""
    per_deal = []          # 每副牌山:挑战者 4 次轮换的平均
    raw = []               # 每次轮换的原始分(算 rho 用)
    for d in range(n_deals):
        vals = []
        for seat in range(4):
            pol = [opponent] * 4
            pol[seat] = challenger
            sc = play_deal(d, pol, rng_seed=d * 4 + seat)
            vals.append(sc[seat])
        per_deal.append(sum(vals) / 4.0)
        raw.append(vals)
        if verbose_every and (d + 1) % verbose_every == 0:
            m = statistics.mean(per_deal)
            se = statistics.pstdev(per_deal) / math.sqrt(len(per_deal))
            print(f"  [{d+1}/{n_deals}] avg={m:+.3f} +/- {1.96*se:.3f}", flush=True)
    return per_deal, raw


def report(per_deal, raw, label):
    n = len(per_deal)
    m = statistics.mean(per_deal)
    sd = statistics.pstdev(per_deal)
    se = sd / math.sqrt(n)
    ci = 1.96 * se

    flat = [v for vals in raw for v in vals]
    var_tot = statistics.pvariance(flat) if len(flat) > 1 else 0.0
    var_within = statistics.mean(
        [statistics.pvariance(vals) for vals in raw]) if n else 0.0
    rho = (var_tot - var_within) / var_tot if var_tot > 0 else float("nan")

    print(f"\n=== {label} ===")
    print(f"  n 副牌山      : {n}  (= {4*n} 局对战)")
    if m - ci > 0:
        verdict = "CI 全域 > 0 → 显著更强"
    elif m + ci < 0:
        verdict = "CI 全域 < 0 → 显著更弱"
    else:
        verdict = "CI 跨 0 → 未分出胜负"
    print(f"  avg_pt/盘     : {m:+.4f}  ± {ci:.4f} (95% CI)   {verdict}")
    print(f"  每盘分数 std  : {sd:.3f}(复式后) / {math.sqrt(var_tot):.3f}(单次)")
    print(f"  组内相关 rho  : {rho:.3f}   "
          f"(复式买到的方差缩减 = {math.sqrt(var_tot/max(sd*sd,1e-9)):.2f}x)")
    if abs(m) > 1e-12:
        need = (1.96 * sd / (0.5 * abs(m))) ** 2 if m else float("inf")
        print(f"  分辨 0.1 分需 : {int((1.96*sd/0.1)**2):,} 副牌山")
    return m, ci, rho


def selfcheck(n: int, policy=bots.bot_L1_greedy):
    """A(pi,pi) == 0 自检:四座同策略,复式期望必须恒为 0。"""
    print("A(pi,pi) == 0 自检:四座同策略的复式期望是代数恒等式,测不出 0 就是有 bug")
    per_deal, raw = duplicate(n, policy, policy)
    m, ci, _ = report(per_deal, raw, "自检 L1 vs L1(期望恒等于 0)")
    ok = abs(m) < max(ci, 1e-9)
    print(f"\n判据: {'✓ 通过' if ok else '✗ 失败 —— 竞技场或环境有 bug,先修这个'}"
          f"  |{m:+.5f}| {'<' if ok else '>='} CI {ci:.5f}")
    return ok


def ladder(n: int):
    """整条梯子:每一级必须显著强于下一级,否则这级不该存在。"""
    rungs = [("L1", bots.bot_L1_greedy, "L0", bots.bot_L0_uniform),
             ("L2", bots.bot_L2_defensive, "L1", bots.bot_L1_greedy),
             ("L2", bots.bot_L2_defensive, "L0", bots.bot_L0_uniform)]
    out = []
    for hi_n, hi, lo_n, lo in rungs:
        per_deal, raw = duplicate(n, hi, lo)
        m, ci, rho = report(per_deal, raw, f"{hi_n} vs 3x{lo_n}")
        out.append((f"{hi_n}>{lo_n}", m, ci))
    print("\n===== 梯子汇总 =====")
    for name, m, ci in out:
        verdict = "台阶成立" if m - ci > 0 else "⚠ 台阶不成立(这级白做)"
        print(f"  {name:10s} {m:+8.3f} ± {ci:.3f}   {verdict}")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selfcheck"
    n = int(sys.argv[2]) if len(sys.argv) > 2 and cmd != "duel" else 200
    if cmd == "selfcheck":
        sys.exit(0 if selfcheck(n) else 1)
    elif cmd == "ladder":
        ladder(n)
    elif cmd == "duel":
        a, b = sys.argv[2], sys.argv[3]
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 300
        pd, raw = duplicate(n, bots.LADDER[a], bots.LADDER[b], verbose_every=max(1, n // 5))
        report(pd, raw, f"{a} vs 3x{b}")
    else:
        print(__doc__)
