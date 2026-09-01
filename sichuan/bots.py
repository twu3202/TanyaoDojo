"""
川麻规则 bot 阶梯 L0–L3 —— 这条线的"标尺",也是 league 里唯一不会协同演化的对手。

**为什么这是全案最重要的一块基建。** 立直线四次 RL 失败后的复盘给出两条独立结论:
  (a) 自家族 league(池子全是自己的快照)在 10–30 亿步任何尺度都不产外部增益,
      病名 competitive overfitting —— 剥削自家弱点 ≠ 逼近真正的强手;
  (b) 药方是"对手池混入协同适应谱系外的成分 + 高频外部探针"。
立直线当时唯一的谱系外成分是 Mortal v4(别人训的)。川麻没有 Mortal —— 但川麻可以
**自己造一条谱系外的梯子**:手写的规则 bot 永远不会跟着 agent 一起漂移,它们是钉死的
坐标原点。于是"没有公开 baseline"这个看似的劣势,反而换来了一个立直线求之不得的东西:
一个成本为零、永不污染、可以无限次复测的外部标尺。

阶梯设计与**实测标定**(复式 400 副牌山 = 1600 局,见 arena.py):

  L0 uniform    均匀随机(能胡则胡)—— 坐标零点
  L1 greedy     纯向听下降,零防守 —— **+3.866 ± 0.298 vs 3xL0,台阶成立**
  L2 defensive  L1 + 同向听内安全牌优先 —— **-0.124 ± 0.210 vs 3xL1,统计打平**
  L2b folding   立直式弃和防守(**对照臂,不在梯子里**)—— **-0.229 ± 0.215,显著更弱**
  L3 rollout    每候选打牌做 K 次蒙特卡洛续局 —— 手写上限,但 26.8s/局,**需 JAX 批量版**

**两条实测结论,都与立直麻将的直觉相反,值得记在这里:**

1. **弃和在川麻是负期望。** L2b 用的是立直麻将最可靠的防守动作(手牌差 + 对手副露多
   → 宁慢求安),实测显著更弱。原因是川麻有**查大叫**:流局时未听牌者要向每个听牌者
   赔钱。所以"不听牌"本身就是要付费的,弃和不是止损,是直接认罚。
2. **纯安全牌优先只能打平,加不出台阶。** L2 已经用上了川麻独有的硬信号——每家开局
   公开定缺,而胡牌手里不得含缺门牌,所以**打出某家缺门花色的牌对那家 100% 安全**
   (立直麻将的现物/筋只是概率性安全)。即便如此仍只是统计打平。

**所以梯子定版为 L0 → L1 → L3 → RL 自快照,跳过 L2 那一级。** 手写台阶在 L1 之上
就造不动了,这本身是对 RL 有利的证据:川麻在"向听下降"之上的策略并不平凡。

接口与 reference_impl 对齐:bot(game, player, actions, rng) -> action。
CPU 版用于标尺标定与评测;L0–L2 是"对候选动作打分取 argmax"的形状,可无损搬成
JAX 算子当 league 冻结对手(L3 太贵,只做评测对手)。
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

from reference_impl import NUM_TILES, SichuanGame, suit_of, is_hu

# 向听:优先用查表(快),退化到递归实现
try:
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from common.suit_table import load_table, shanten_np
    _TAB = load_table(build_if_missing=False)

    def shanten(hand, num_melds, void):
        return shanten_np(_TAB, hand, num_melds, void if void is not None else -1)
except Exception:                                          # 表还没构建时的退路
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def _suit_prof(c9):
        best = [[-1, -1] for _ in range(5)]
        c = list(c9)

        def rec(i, sets, pair, part):
            if sets <= 4 and part > best[sets][pair]:
                best[sets][pair] = part
            if i >= 9:
                return
            rec(i + 1, sets, pair, part)
            if c[i] >= 3:
                c[i] -= 3; rec(i, sets + 1, pair, part); c[i] += 3
            if c[i] >= 2:
                if not pair:
                    c[i] -= 2; rec(i, sets, 1, part); c[i] += 2
                c[i] -= 2; rec(i, sets, pair, part + 1); c[i] += 2
            if i <= 6 and c[i + 1] and c[i + 2]:
                c[i] -= 1; c[i + 1] -= 1; c[i + 2] -= 1
                rec(i, sets + 1, pair, part)
                c[i] += 1; c[i + 1] += 1; c[i + 2] += 1
            if i <= 7 and c[i + 1]:
                c[i] -= 1; c[i + 1] -= 1
                rec(i, sets, pair, part + 1)
                c[i] += 1; c[i + 1] += 1
            if i <= 6 and c[i + 2]:
                c[i] -= 1; c[i + 2] -= 1
                rec(i, sets, pair, part + 1)
                c[i] += 1; c[i + 2] += 1

        rec(0, 0, 0, 0)
        return tuple(tuple(r) for r in best)

    @lru_cache(maxsize=None)
    def _shanten_key(hand, num_melds, void):
        tabs = [_suit_prof((0,) * 9 if s == void else hand[s * 9:(s + 1) * 9])
                for s in range(3)]

        def merge(a, b):
            out = [[-1, -1] for _ in range(5)]
            for s1 in range(5):
                for p1 in range(2):
                    if a[s1][p1] < 0:
                        continue
                    for s2 in range(5 - s1):
                        for p2 in range(2 - p1):
                            if b[s2][p2] < 0:
                                continue
                            v = a[s1][p1] + b[s2][p2]
                            if v > out[s1 + s2][p1 + p2]:
                                out[s1 + s2][p1 + p2] = v
            return out

        t = merge(merge(tabs[0], tabs[1]), tabs[2])
        best = 99
        for sets in range(5):
            for pair in range(2):
                if t[sets][pair] < 0:
                    continue
                ts = num_melds + sets
                if ts > 4:
                    continue
                best = min(best, 8 - 2 * ts - min(t[sets][pair], 4 - ts) - pair)
        if num_melds == 0:
            live = [hand[i] if suit_of(i) != void else 0 for i in range(NUM_TILES)]
            pairs = sum(1 for x in live if x >= 2)
            kinds = sum(1 for x in live if x >= 1)
            best = min(best, 6 - pairs + max(0, 7 - kinds))
        return best

    def shanten(hand, num_melds, void):
        return _shanten_key(tuple(hand), num_melds, void if void is not None else -1)


# --------------------------------------------------------------------- 公共件
def _hu_action(acts):
    for a in acts:
        if a[0] in ("zimo", "ron"):
            return a
    return None


def _void_choice(p):
    """定缺:选手里张数最少的门。平手取序号小者(与参考实现的缺省一致,便于对拍)。"""
    cnt = [sum(p.hand[t] for t in range(NUM_TILES) if suit_of(t) == s) for s in range(3)]
    return min(range(3), key=lambda s: (cnt[s], s))


def _visible_counts(g, me):
    """全场可见的每型张数(自家手牌 + 所有牌河 + 所有副露)。用于估"还剩几张"。"""
    seen = [0] * NUM_TILES
    for t in range(NUM_TILES):
        seen[t] += g.players[me].hand[t]
    for i in range(4):
        for t in g.discards[i]:
            seen[t] += 1
        for kind, t in g.players[i].melds:
            seen[t] += 4 if kind.startswith("gang") else 3
    return seen


def _discard_scores(g, me, cand: List[int], defense_w: float = 0.0,
                    use_width: bool = False):
    """对每个候选打牌打分(越大越好)。分 = -向听 + 进张宽度 - 防守罚。

    ⚠️ 成本注记(实测教训):`use_width` 每张候选要再跑 27 次 shanten,把一次打牌决策
    从 ~14 次 shanten 抬到 ~378 次,整条评测慢 27 倍 —— 120 副牌山的自检就从分钟级
    变成跑不完。默认关掉;它只在"向听持平的两张之间选"时才有意义,而那种情形本就
    该交给 RL 去学,不该由手写 bot 精雕。**梯子的价值在于钉死、便宜、可无限复测,
    不在于每一级都尽可能强。**
    """
    p = g.players[me]
    seen = _visible_counts(g, me) if defense_w else None
    out = []
    for t in cand:
        p.hand[t] -= 1
        st = shanten(p.hand, len(p.melds), p.void)
        width = 0
        if use_width:
            for u in range(NUM_TILES):
                if suit_of(u) == p.void or p.hand[u] >= 4:
                    continue
                p.hand[u] += 1
                if shanten(p.hand, len(p.melds), p.void) < st:
                    width += 1
                p.hand[u] -= 1
        p.hand[t] += 1
        score = -4.0 * st + (0.10 * width if use_width else 0.0)
        if defense_w:
            score -= defense_w * _danger(g, me, t, seen)
        out.append((score, t))
    return out


def _danger(g, me, tile, seen) -> float:
    """打出 tile 的危险度。川麻专属:对手的缺门牌 100% 安全,这是最硬的一条。

    其余用两个廉价代理:
      - 该牌全场已现张数越多,构成他家面子的可能越低;
      - 副露多 / 已听味浓的对手权重更高(用副露数当威胁度)。
    """
    d = 0.0
    for j in range(4):
        if j == me or g.players[j].hu:
            continue
        q = g.players[j]
        if q.void is not None and suit_of(tile) == q.void:
            continue                                   # 对该家 100% 安全
        threat = 1.0 + 0.6 * len(q.melds)              # 副露越多越危险
        live = max(0, 4 - seen[tile])                  # 还没现的张数
        d += threat * (0.25 + 0.25 * live)
    return d


# --------------------------------------------------------------------- L0
def bot_L0_uniform(g, i, acts, rng: random.Random):
    """均匀随机,但能胡就胡(否则终局事件太少,阶梯零点会退化成"从不结算")。"""
    hu = _hu_action(acts)
    return hu if hu else rng.choice(acts)


# --------------------------------------------------------------------- L1
def bot_L1_greedy(g, i, acts, rng: random.Random):
    """纯速度:向听下降优先,杠一律接(刮风下雨是白拿的钱),碰只在不亏向听时接。
    零防守、零前瞻、不看分数、不看局势。"""
    hu = _hu_action(acts)
    if hu:
        return hu
    p = g.players[i]

    if any(a[0] == "void" for a in acts):
        return ("void", _void_choice(p))
    for k, arg in acts:
        if k in ("ankan", "bugang"):
            return (k, arg)
    for k, _ in acts:
        if k == "zhigang":
            return (k, None)

    cand = [a[1] for a in acts if a[0] == "discard"]
    if ("peng", None) in acts:
        _, dt = g.pending_discard
        cur = shanten(p.hand, len(p.melds), p.void)
        h = p.hand[:]
        h[dt] -= 2
        if shanten(h, len(p.melds) + 1, p.void) <= cur:
            return ("peng", None)
    if cand:
        return ("discard", max(_discard_scores(g, i, cand), key=lambda x: x[0])[1])
    return ("pass", None)


# --------------------------------------------------------------------- L2
def bot_L2_defensive(g, i, acts, rng: random.Random):
    """L1 + **同向听内的安全牌优先**。除打牌的并列拆解外,与 L1 逐字相同。

    ⚠️ 这一级被重写过一次,教训值得写在代码里。初版把防守做成"权衡":手牌差时
    调高危险罚、宁可慢也要安全(立直麻将的标准弃和直觉)。实测 **L2 vs 3xL1
    = -1.71 ± 0.45,显著更弱**。原因是川麻有 **查大叫**:流局时未听牌的人要向每个
    听牌的人赔钱。于是"弃和"在川麻里不是止损,是**直接认罚**——立直麻将里最可靠的
    防守动作,在这里是负期望。

    正确的形状是**严格改良**而非权衡:先按 L1 选出向听最优的那批牌,只在**它们之间**
    挑最安全的一张。这样速度一分不让,安全是白捡的 —— 构造上不可能弱于 L1。
    川麻还额外送了一个立直麻将没有的硬信号:**对手的定缺花色牌对那家 100% 安全**
    (胡牌手里不得含缺门牌),所以"白捡"的量并不小。
    """
    hu = _hu_action(acts)
    if hu:
        return hu
    p = g.players[i]

    if any(a[0] == "void" for a in acts):
        return ("void", _void_choice(p))
    for k, arg in acts:
        if k in ("ankan", "bugang"):
            return (k, arg)
    for k, _ in acts:
        if k == "zhigang":
            return (k, None)

    cand = [a[1] for a in acts if a[0] == "discard"]
    if ("peng", None) in acts:                     # 与 L1 完全一致
        _, dt = g.pending_discard
        cur = shanten(p.hand, len(p.melds), p.void)
        h = p.hand[:]
        h[dt] -= 2
        if shanten(h, len(p.melds) + 1, p.void) <= cur:
            return ("peng", None)
    if not cand:
        return ("pass", None)

    scored = _discard_scores(g, i, cand)           # 纯速度分,无防守项
    top = max(s for s, _ in scored)
    tied = [t for s, t in scored if s >= top - 1e-9]
    if len(tied) == 1:
        return ("discard", tied[0])
    seen = _visible_counts(g, i)
    return ("discard", min(tied, key=lambda t: (_danger(g, i, t, seen), t)))


# --------------------------------------------------------------------- L3
def bot_L3_rollout(g, i, acts, rng: random.Random, n_rollout: int = 24,
                   base=bot_L1_greedy):
    """蒙特卡洛续局:对每个候选打牌,把未知牌洗匀发给对手,用 base 策略打到终局,
    取平均分最高者。这是手写策略的天花板,也是 RL 必须超过的最后一级台阶。

    注意这**不是决策时搜索的推荐用法**——立直线的调研结论是"搜索作为特征赢、
    作为决策规则输"。L3 在本案里只当**评测对手**(它慢,但慢无所谓),
    用来回答"RL 学到的东西超过朴素前瞻了吗"。
    """
    hu = _hu_action(acts)
    if hu:
        return hu
    p = g.players[i]
    if any(a[0] == "void" for a in acts):
        return ("void", _void_choice(p))
    cand = [a for a in acts if a[0] == "discard"]
    if len(cand) <= 1:
        return base(g, i, acts, rng)

    # 只对 L1 打分靠前的几张做续局,省算力
    scored = sorted(_discard_scores(g, i, [a[1] for a in cand]),
                    key=lambda x: -x[0])[:4]
    best, best_v = None, -1e18
    for _, t in scored:
        tot = 0.0
        for r in range(n_rollout):
            tot += _rollout_once(g, i, t, rng, base)
        v = tot / n_rollout
        if v > best_v:
            best, best_v = t, v
    return ("discard", best)


def _rollout_once(g, me, tile, rng, base) -> float:
    """从当前局面复制一份、我方打 tile、其余未知牌重洗后打到终局,返回我方分数增量。"""
    sim = _clone_with_resample(g, me, rng)
    if sim is None:
        return 0.0
    s0 = sim.players[me].score_delta
    try:
        sim.step(("discard", tile))
        guard = 0
        while sim.phase != "over" and guard < 3000:
            j, a = sim.legal_actions()
            sim.step(base(sim, j, a, rng))
            guard += 1
    except Exception:
        return 0.0
    return sim.players[me].score_delta - s0


def _clone_with_resample(g: SichuanGame, me: int, rng) -> Optional[SichuanGame]:
    """深拷贝当前局面,把三家手牌 + 牌墙当作未知牌池重新洗匀分配(信息集采样)。"""
    import copy
    sim = copy.deepcopy(g)
    pool = list(sim.wall)
    for j in range(4):
        if j == me or sim.players[j].hu:
            continue
        for t in range(NUM_TILES):
            pool.extend([t] * sim.players[j].hand[t])
            sim.players[j].hand[t] = 0
    rng.shuffle(pool)
    for j in range(4):
        if j == me or sim.players[j].hu:
            continue
        n = 3 * (4 - len(sim.players[j].melds)) + 1
        if len(pool) < n:
            return None
        for _ in range(n):
            sim.players[j].hand[pool.pop()] += 1
    sim.wall = pool
    return sim


LADDER = {
    "L0": bot_L0_uniform,
    "L1": bot_L1_greedy,
    "L2": bot_L2_defensive,
    "L3": bot_L3_rollout,
}


def bot_L2b_folding(g, i, acts, rng: random.Random, defense_w: float = 1.0,
                    fold_shanten: int = 3):
    """**对照臂,不属于梯子**:立直麻将式的"权衡型"防守——手牌差且对手副露多时弃和。

    留在仓库里是因为它的实测结果是本案的一条领域证据:在川麻里弃和是负期望,
    因为**查大叫**会让流局时未听牌者向每个听牌者赔钱。
    """
    hu = _hu_action(acts)
    if hu:
        return hu
    p = g.players[i]
    if any(a[0] == "void" for a in acts):
        return ("void", _void_choice(p))

    cur = shanten(p.hand, len(p.melds), p.void)
    threat = max((len(g.players[j].melds) for j in range(4)
                  if j != i and not g.players[j].hu), default=0)
    folding = cur >= fold_shanten and threat >= 2
    w = defense_w * (2.5 if folding else 1.0)

    for k, arg in acts:
        if k in ("ankan", "bugang"):
            return (k, arg)
    for k, _ in acts:
        if k == "zhigang":
            return (k, None)
    if ("peng", None) in acts and not folding:
        _, dt = g.pending_discard
        h = p.hand[:]
        h[dt] -= 2
        if shanten(h, len(p.melds) + 1, p.void) <= cur:
            return ("peng", None)
    cand = [a[1] for a in acts if a[0] == "discard"]
    if cand:
        return ("discard", max(_discard_scores(g, i, cand, defense_w=w),
                               key=lambda x: x[0])[1])
    return ("pass", None)


LADDER["L2b"] = bot_L2b_folding
