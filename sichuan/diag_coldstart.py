"""
川麻冷启动诊断(零 GPU)。回答一个问题:随机初始化的策略,在川麻里能不能拿到
足够的学习信号?这是 tabula rasa 可行性的第一道判据。

测三件事:
  1) 终局类型分布 / 胡次数 —— "赢过牌"这件事在随机策略下多罕见
  2) 奖励信号密度 —— 有多少比例的对局产生非零分数(刮风下雨/查大叫也算信号)
  3) 有没有可学的梯度 —— 贪心(向听下降)对 3 个随机,分差有多大

用法: python3 -u diag_coldstart.py [n_random] [n_greedy]
"""
import sys, os, random, statistics
from collections import Counter
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_impl import SichuanGame, NUM_TILES, suit_of  # noqa: E402

NEG = -1


# ---------------------------------------------------------------- 向听(川麻版)
# 改用 common/suit_table 的位掩码查表(见那里的注记:初版递归漏了 `c[i]>=1` 检查,
# 会从负计数里造顺子 → 低估向听。本文件早期版本带同一个 bug,已换成查表)。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.suit_table import load_table, shanten_np  # noqa: E402

_TAB = load_table()


def shanten(counts, num_melds, void_suit):
    return shanten_np(_TAB, counts, num_melds,
                      void_suit if void_suit is not None else -1)


# ---------------------------------------------------------------- 策略
def policy_random(g, i, acts, rng):
    hu = [a for a in acts if a[0] in ("zimo", "ron")]
    return rng.choice(hu) if hu else rng.choice(acts)


def _best_discard(p, discards):
    best, bt = 99, discards[0][1]
    for _, t in discards:
        p.hand[t] -= 1
        s = shanten(p.hand, len(p.melds), p.void)
        p.hand[t] += 1
        if s < best:
            best, bt = s, t
    return bt


def policy_greedy(g, i, acts, rng):
    """贪心:能胡就胡;定缺选手中最少的花色;打牌选让向听最小的那张;
    杠一律接(刮风下雨白拿钱);碰只在不升高向听时接。零防守、零前瞻。"""
    hu = [a for a in acts if a[0] in ("zimo", "ron")]
    if hu:
        return hu[0]
    p = g.players[i]

    voids = [a for a in acts if a[0] == "void"]
    if voids:
        cnt = [sum(p.hand[t] for t in range(NUM_TILES) if suit_of(t) == s) for s in range(3)]
        return ("void", min(range(3), key=lambda s: (cnt[s], s)))

    for k, arg in acts:                      # 杠白拿钱,先杠
        if k in ("ankan", "bugang"):
            return (k, arg)
    for k, _ in acts:
        if k == "zhigang":
            return (k, None)

    discards = [a for a in acts if a[0] == "discard"]
    if ("peng", None) in acts:
        dp, dt = g.pending_discard
        cur = shanten(p.hand, len(p.melds), p.void)
        h = p.hand[:]
        h[dt] -= 2
        if shanten(h, len(p.melds) + 1, p.void) <= cur:
            return ("peng", None)
    if discards:
        return ("discard", _best_discard(p, discards))
    return ("pass", None)


# ---------------------------------------------------------------- 跑分
def play(seed, seat_policies):
    g = SichuanGame(seed)
    rng = random.Random(seed ^ 0xABCDEF)
    n_dec = 0
    while g.phase != "over":
        i, acts = g.legal_actions()
        g.step(seat_policies[i](g, i, acts, rng))
        n_dec += 1
    return g, n_dec


def run(n, seat_policies, label):
    terms, fans = Counter(), Counter()
    hus, decs, scores, absmax = [], [], [], []
    nonzero = 0
    for s in range(n):
        g, nd = play(s, seat_policies)
        hus.append(len(g.hu_order))
        decs.append(nd)
        sc = g.scores()
        scores.extend(sc)
        absmax.append(max(abs(x) for x in sc))
        if any(x != 0 for x in sc):
            nonzero += 1
        terms["3胡终局" if len(g.hu_order) >= 3 else "流局/墙空"] += 1
        for _, fan, _ in g.hu_order:
            fans[fan] += 1
    print(f"\n=== {label}  (n={n}) ===")
    print(f"  终局类型            : {dict(terms)}")
    print(f"  平均胡次/局         : {statistics.mean(hus):.3f}   分布={dict(sorted(Counter(hus).items()))}")
    print(f"  产生非零分数的局占比: {nonzero/n:.1%}   <-- 奖励信号是否存在")
    print(f"  |终局分数| 均值/最大: {statistics.mean(absmax):.2f} / {max(absmax)}")
    print(f"  单座分数 std        : {statistics.pstdev(scores):.2f}")
    print(f"  决策数/局           : {statistics.mean(decs):.1f}")
    print(f"  胡牌番数分布        : {dict(sorted(fans.items()))}")


if __name__ == "__main__":
    nr = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    ng = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    R, G = policy_random, policy_greedy

    run(nr, [R] * 4, "四座全随机 (tabula rasa 第 0 步的真实处境)")
    run(ng, [G] * 4, "四座全贪心 (向听下降,无防守)")

    seat0, others = [], []
    for s in range(ng):
        g, _ = play(s, [G, R, R, R])
        sc = g.scores()
        seat0.append(sc[0])
        others.extend(sc[1:])
    sem = statistics.pstdev(seat0) / ng ** 0.5
    print(f"\n=== 贪心 vs 3 随机  (n={ng}) ===")
    print(f"  贪心座平均分 : {statistics.mean(seat0):+.3f}  "
          f"(std {statistics.pstdev(seat0):.2f}, SEM {sem:.3f}, z={statistics.mean(seat0)/max(sem,1e-9):.1f})")
    print(f"  随机座平均分 : {statistics.mean(others):+.3f}")
    print(f"  --> 这个差值 = tabula rasa 早期可追的梯度大小")
