"""
川麻胡牌/向听判定的 O(1) 查表核心 —— P1 的拱心石。

**为什么需要它**:递归的 agari/shanten 判定在 CPU 上没问题(参考实现就是这么写的),
但 GPU 向量化环境里每个 env 每步都要判"能不能胡/听没听",递归是 jit 的天敌。
川麻正好有一个别的麻将变种没有的便利:**只有 3 门数牌、无字牌**,于是单门花色的手牌
就是 9 个 0..4 的计数,可以编码成 5 进制整数索引一张预计算表。

    索引空间 5^9 = 1,953,125;每条存 best[sets][pair] = 该拆法下的最大搭子数(-1=不可达)
    → (5^9, 10) int8 = 19.5 MB,**常驻显存,查一次 = 一次 gather**

三门各查一次,再做两次 (max,+) 卷积合并,即得 (胡牌?, 向听)。全程定长、无递归、
jit/vmap 友好。这把川麻环境里最脏的一块变成了纯访存。

对比:立直麻将做不了这么干净的表——34 种牌型、字牌不成顺、还有赤宝/役种,
上游 mahjax 为此写了相当复杂的判定。川麻的规则简化在这里直接兑换成工程简化。

构建实测 **1 秒**(位掩码 DP,见下文注记),结果缓存为 .npy。仓库里同时保留了一个
"笨但显然对"的递归版 `suit_profile` 当**差分测试的另一端**——它慢 4 万倍,但正是它和 DP
互相对拍时,抓出了递归版自己的一个 bug(见 `suit_profile` 里的注记)。

用法:
    python3 -m common.suit_table build          # 构表并落盘(1 秒)
    python3 -m common.suit_table verify 30000   # 与参考实现 is_hu 对拍

另见本文件末尾的 `is_ting_np` —— 川麻的"听牌"**必须带缺门约束**,直接复用
国标/日麻 shanten 库会在那里**静默出错**。
"""
from __future__ import annotations

import os
import sys
import time
from functools import lru_cache

import numpy as np

NUM_TILES = 27          # 万/筒/条 各 9
NEG = -1
POW5 = np.array([5 ** i for i in range(9)], dtype=np.int64)
TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "suit_table.npy")

# 合并用的索引映射:把 (sets,pair) 展平成 10 槽,(i,j) -> 目标槽 或 -1(越界)
_SLOTS = [(s, p) for s in range(5) for p in range(2)]
MERGE_IDX = np.full((10, 10), -1, dtype=np.int8)
for _i, (_s1, _p1) in enumerate(_SLOTS):
    for _j, (_s2, _p2) in enumerate(_SLOTS):
        _s, _p = _s1 + _s2, _p1 + _p2
        if _s <= 4 and _p <= 1:                    # 全局最多一个对子
            MERGE_IDX[_i, _j] = _s * 2 + _p


# --------------------------------------------------------------------- 构表
@lru_cache(maxsize=None)
def suit_profile(c9: tuple) -> tuple:
    """单门 9 计数 → best[sets][pair] = 最大搭子数。笨递归,只在构表时跑。

    拆法枚举:刻子(3)、对子(2,全局至多一个当雀头)、搭子对(2)、顺子(i,i+1,i+2)、
    两面/坎张搭子(i,i+1) 与 (i,i+2)。每到一个节点就记账——记下的都是某个
    合法子拆解,故取 max 安全。"""
    best = [[NEG, NEG] for _ in range(5)]
    c = list(c9)

    def rec(i, sets, pair, part):
        if sets <= 4 and part > best[sets][pair]:
            best[sets][pair] = part
        if i >= 9:
            return
        rec(i + 1, sets, pair, part)                                   # 放弃 rank i
        if c[i] >= 3:
            c[i] -= 3; rec(i, sets + 1, pair, part); c[i] += 3         # 刻子
        if c[i] >= 2:
            if not pair:
                c[i] -= 2; rec(i, sets, 1, part); c[i] += 2            # 雀头
            c[i] -= 2; rec(i, sets, pair, part + 1); c[i] += 2         # 对子搭
        # ⚠️ 这三个分支的 `c[i] >= 1` 是必须的 —— 初版漏了,被位掩码 DP 的对拍抓出来:
        # 走到 rank i 时 c[i] 可能已被 i-1 / i-2 起的顺子吃空,漏检就会从 **负计数**
        # 里造出顺子/搭子,**高估面子数 → 低估向听**。反例 c=[3,1,2,0,2,2,0,1,1]:
        # 漏检版报 sets=2 可达,实际最多 1。教训:差分测试的价值不在"确认对",
        # 在于两个实现里**先出错的那个不一定是新写的那个**。
        if i <= 6 and c[i] and c[i + 1] and c[i + 2]:
            c[i] -= 1; c[i + 1] -= 1; c[i + 2] -= 1                    # 顺子
            rec(i, sets + 1, pair, part)
            c[i] += 1; c[i + 1] += 1; c[i + 2] += 1
        if i <= 7 and c[i] and c[i + 1]:
            c[i] -= 1; c[i + 1] -= 1                                   # 两面/边张
            rec(i, sets, pair, part + 1)
            c[i] += 1; c[i + 1] += 1
        if i <= 6 and c[i] and c[i + 2]:
            c[i] -= 1; c[i + 2] -= 1                                   # 坎张
            rec(i, sets, pair, part + 1)
            c[i] += 1; c[i + 2] += 1

    rec(0, 0, 0, 0)
    return tuple(tuple(r) for r in best)


# ---------------------------------------------- 位掩码 DP(实际用的构表器)
# ⚠️ 实测教训:上面那个 suit_profile 递归在稠密手牌上要 **1.7 秒**(指数展开,
#    同一个 (i, 剩余计数) 状态被不同顺序反复到达),40.5 万条要跑 ~11 小时——
#    不可行。它保留下来只当**差分测试的另一端**(笨但显然对)。
#
# 真正的构表器换成一个位掩码 DP,基于两点观察:
#   1) 可达状态 (sets, pair, part) 的取值空间只有 5x2x5 = 50 个,可以整体塞进一个
#      **uint64 位掩码**;于是"取并集"就是 `|`,"sets+1 / 记雀头 / part+1"就是**移位**
#      (位序设计成 bit = (sets*2 + pair)*5 + part,三种操作恰好是 +10 / +5 / +1)。
#   2) 每种拆法都把 5 进制索引**严格变小**(总是消耗最低非零位的牌),所以
#      x 从小到大扫一遍即可,无需递归。
# 于是整表是一次 O(5^9 x 6) 的线性扫描,秒级完成,且与递归版逐条可对拍。
_BITS = [(s, p, q) for s in range(5) for p in range(2) for q in range(5)]
_BIT_OF = {t: i for i, t in enumerate(_BITS)}
_M_SETS_LT4 = sum(1 << i for i, (s, p, q) in enumerate(_BITS) if s <= 3)
_M_PAIR0 = sum(1 << i for i, (s, p, q) in enumerate(_BITS) if p == 0)
_M_PART_LT4 = sum(1 << i for i, (s, p, q) in enumerate(_BITS) if q <= 3)
_ALL = (1 << 50) - 1


def build_table(verbose: bool = True) -> np.ndarray:
    """(5^9, 10) int8。best[sets][pair] = 最大搭子数(-1 = 该 (sets,pair) 不可达)。"""
    t0 = time.time()
    N = 5 ** 9
    f = np.zeros(N, dtype=np.uint64)
    f[0] = np.uint64(1 << _BIT_OF[(0, 0, 0)])
    pw = [5 ** i for i in range(9)]

    M_S, M_P, M_Q = np.uint64(_M_SETS_LT4), np.uint64(_M_PAIR0), np.uint64(_M_PART_LT4)
    TEN, FIVE, ONE = np.uint64(10), np.uint64(5), np.uint64(1)
    ALL = np.uint64(_ALL)

    digits = [0] * 9
    for x in range(1, N):
        # 增量维护 5 进制位(比每次 divmod 快)
        k = 0
        while digits[k] == 4:
            digits[k] = 0
            k += 1
        digits[k] += 1

        i = 0                                   # 最低非零位
        while digits[i] == 0:
            i += 1
        ci = digits[i]
        p = pw[i]

        m = f[x - p]                                             # 该张当废牌
        if ci >= 3:
            m |= (f[x - 3 * p] & M_S) << TEN                     # 刻子
        if ci >= 2:
            g = f[x - 2 * p]
            m |= (g & M_P) << FIVE                               # 雀头
            m |= (g & M_Q) << ONE                                # 对子搭
        if i <= 6 and digits[i + 1] and digits[i + 2]:
            m |= (f[x - p - pw[i + 1] - pw[i + 2]] & M_S) << TEN  # 顺子
        if i <= 7 and digits[i + 1]:
            m |= (f[x - p - pw[i + 1]] & M_Q) << ONE              # 两面/边张
        if i <= 6 and digits[i + 2]:
            m |= (f[x - p - pw[i + 2]] & M_Q) << ONE              # 坎张
        f[x] = m & ALL

        if verbose and x % 500000 == 0:
            print(f"  ... {x:,}/{N:,}  ({time.time()-t0:.0f}s)", flush=True)

    # 位掩码 → best[sets][pair] = max part
    if verbose:
        print(f"  DP 完成 {time.time()-t0:.0f}s,解码中...", flush=True)
    tab = np.full((N, 10), NEG, dtype=np.int8)
    for bit, (s, p, q) in enumerate(_BITS):
        has = (f >> np.uint64(bit)) & np.uint64(1)
        slot = s * 2 + p
        np.maximum(tab[:, slot], np.where(has.astype(bool), q, NEG).astype(np.int8),
                   out=tab[:, slot])
    if verbose:
        print(f"构表完成: {time.time()-t0:.0f}s, {tab.nbytes/1e6:.1f} MB", flush=True)
    return tab


def _crosscheck_recursive(tab, n: int = 300, seed: int = 0):
    """位掩码 DP vs 笨递归,逐条对拍(递归慢,只抽查)。"""
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(n):
        tot = int(rng.integers(0, 15))
        c = [0] * 9
        for _ in range(tot):
            j = int(rng.integers(0, 9))
            if c[j] < 4:
                c[j] += 1
        ref = suit_profile(tuple(c))
        got = tab[int(np.dot(c, POW5))]
        for s in range(5):
            for p in range(2):
                # 搭子数在查询时恒被 min(part, 4-sets) 夹住,故 >=4 一律等价;
                # 位掩码只留 0..4 是**无损**饱和,对拍时把参考值同样夹到 4。
                r = ref[s][p]
                r = min(r, 4) if r >= 0 else NEG
                if int(got[s * 2 + p]) != r:
                    bad += 1
                    if bad <= 3:
                        print(f"  DP/递归不符 c={c} sets={s} pair={p} "
                              f"DP={got[s*2+p]} 递归(夹4)={r}")
    print(f"DP vs 递归对拍 {n} 例: 不符 = {bad}")
    return bad


def load_table(build_if_missing: bool = True) -> np.ndarray:
    if os.path.exists(TABLE_PATH):
        return np.load(TABLE_PATH)
    if not build_if_missing:
        raise FileNotFoundError(TABLE_PATH)
    tab = build_table()
    np.save(TABLE_PATH, tab)
    return tab


# ------------------------------------------------------- numpy 查询(参考路径)
def _merge_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(10,) x (10,) 的 (max,+) 卷积。"""
    s = a[:, None].astype(np.int16) + b[None, :].astype(np.int16)
    out = np.full(10, NEG, dtype=np.int16)
    ok = MERGE_IDX >= 0
    both = (a[:, None] >= 0) & (b[None, :] >= 0) & ok
    np.maximum.at(out, MERGE_IDX[both], s[both])
    return out


def query_np(tab, hand27, num_melds: int, void_suit: int):
    """→ (shanten, is_agari_standard)。hand27 为 27 长计数;void_suit 门整门置零。"""
    prof = []
    for s in range(3):
        if s == void_suit:
            prof.append(tab[0])
        else:
            seg = np.asarray(hand27[s * 9:(s + 1) * 9], dtype=np.int64)
            prof.append(tab[int(seg @ POW5)])
    m = _merge_np(_merge_np(prof[0], prof[1]), prof[2])

    best, agari = 99, False
    for k, (sets, pair) in enumerate(_SLOTS):
        part = int(m[k])
        if part < 0:
            continue
        ts = num_melds + sets
        if ts > 4:
            continue
        if ts == 4 and pair == 1:
            agari = True
        best = min(best, 8 - 2 * ts - min(part, 4 - ts) - pair)
    return best, agari


def shanten_np(tab, hand27, num_melds: int, void_suit: int) -> int:
    """含七对分支的完整向听(门清才有七对)。"""
    st, _ = query_np(tab, hand27, num_melds, void_suit)
    if num_melds == 0:
        live = [hand27[t] if t // 9 != void_suit else 0 for t in range(NUM_TILES)]
        pairs = sum(1 for x in live if x >= 2)
        kinds = sum(1 for x in live if x >= 1)
        st = min(st, 6 - pairs + max(0, 7 - kinds))
    return st


# ------------------------------------------------------------ JAX 查询(热路径)
def make_jax_ops(tab: np.ndarray):
    """返回 (agari_fn, shanten_fn),表常驻设备。两者都是 jit/vmap 友好的定长算子。"""
    import jax
    import jax.numpy as jnp

    TAB = jnp.asarray(tab, dtype=jnp.int8)                      # (5^9, 10)
    P5 = jnp.asarray(POW5, dtype=jnp.int32)
    MIDX = jnp.asarray(MERGE_IDX, dtype=jnp.int32)
    SETS = jnp.asarray([s for s, _ in _SLOTS], dtype=jnp.int32)
    PAIR = jnp.asarray([p for _, p in _SLOTS], dtype=jnp.int32)

    def _merge(a, b):
        s = a[:, None].astype(jnp.int32) + b[None, :].astype(jnp.int32)
        ok = (MIDX >= 0) & (a[:, None] >= 0) & (b[None, :] >= 0)
        s = jnp.where(ok, s, -100)
        idx = jnp.where(MIDX >= 0, MIDX, 0)
        return jnp.full((10,), -100, jnp.int32).at[idx.reshape(-1)].max(s.reshape(-1))

    def _profiles(hand27, void_suit):
        segs = hand27.reshape(3, 9).astype(jnp.int32)
        idx = (segs * P5[None, :]).sum(axis=1)                   # (3,)
        idx = jnp.where(jnp.arange(3) == void_suit, 0, idx)      # 缺门整门置空
        return TAB[idx]                                          # (3,10)

    def _core(hand27, num_melds, void_suit):
        pr = _profiles(hand27, void_suit)
        m = _merge(_merge(pr[0], pr[1]), pr[2])                  # (10,)
        ts = num_melds + SETS
        ok = (m >= 0) & (ts <= 4)
        part = jnp.minimum(m, jnp.maximum(4 - ts, 0))
        st = jnp.where(ok, 8 - 2 * ts - part - PAIR, 99)
        agari = jnp.any(ok & (ts == 4) & (PAIR == 1))
        return jnp.min(st), agari

    def shanten(hand27, num_melds, void_suit):
        st, _ = _core(hand27, num_melds, void_suit)
        live = jnp.where((jnp.arange(NUM_TILES) // 9) == void_suit, 0, hand27)
        pairs = jnp.sum(live >= 2)
        kinds = jnp.sum(live >= 1)
        qidui = 6 - pairs + jnp.maximum(0, 7 - kinds)
        return jnp.where(num_melds == 0, jnp.minimum(st, qidui), st)

    def agari(hand27, num_melds, void_suit):
        """标准型 或 七对(门清)。缺门牌存在时直接判否(规则约束)。"""
        _, ag = _core(hand27, num_melds, void_suit)
        void_mask = (jnp.arange(NUM_TILES) // 9) == void_suit
        has_void = jnp.any(jnp.where(void_mask, hand27, 0) > 0)
        total = jnp.sum(hand27)
        need = 3 * (4 - num_melds) + 2
        qidui = (num_melds == 0) & (total == 14) & jnp.all(
            (hand27 == 0) | (hand27 == 2) | (hand27 == 4))
        return (ag | qidui) & (total == need) & (~has_void)

    return jax.jit(agari), jax.jit(shanten)


# ------------------------------------------------------------------- CLI
def _verify(n: int):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sichuan.reference_impl import is_hu  # noqa: E402

    tab = load_table()
    rng = np.random.default_rng(0)
    bad_hu = 0
    n_true = 0
    t0 = time.time()
    for k in range(n):
        num_melds = int(rng.integers(0, 5))
        void = int(rng.integers(0, 3))
        need = 3 * (4 - num_melds) + 2
        live = [t for t in range(NUM_TILES) if t // 9 != void]
        hand = [0] * NUM_TILES
        if k % 3 == 0:                                   # 均匀:几乎全是诈胡
            for t in rng.choice(live, size=need, replace=True):
                if hand[t] < 4:
                    hand[t] += 1
        else:                                            # 聚集:制造真胡牌
            seeds = list(rng.choice(live, size=max(2, need // 3), replace=False))
            for _ in range(need * 3):
                if sum(hand) >= need:
                    break
                t = int(rng.choice(seeds)) + int(rng.integers(-2, 3))
                if not (0 <= t < NUM_TILES) or t // 9 == void or hand[t] >= 4:
                    continue
                hand[t] += 1
            while sum(hand) < need:                      # 补满
                t = int(rng.choice(live))
                if hand[t] < 4:
                    hand[t] += 1

        st, ag = query_np(tab, hand, num_melds, void)
        qidui = (num_melds == 0 and sum(hand) == 14
                 and all(c in (0, 2, 4) for c in hand))
        mine = (ag or qidui) and sum(hand) == need
        ref = is_hu(hand, num_melds, void)
        if mine != ref:
            bad_hu += 1
            if bad_hu <= 5:
                print(f"  MISMATCH melds={num_melds} void={void} 表={mine} 参考={ref}\n"
                      f"    hand={hand}")
        n_true += int(ref)
    dt = time.time() - t0
    print(f"对拍 {n:,} 例(其中真胡牌 {n_true:,}): 胡牌不符 = {bad_hu}   {dt:.1f}s")
    print("判据: 0 不符 → 查表可直接搬进 JAX 环境" if bad_hu == 0 else "判据: ✗ 需修")
    return bad_hu


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "build":
        tab = build_table()
        np.save(TABLE_PATH, tab)
        print(f"落盘 {TABLE_PATH} ({os.path.getsize(TABLE_PATH)/1e6:.1f} MB)")
    elif cmd == "verify":
        sys.exit(1 if _verify(int(sys.argv[2]) if len(sys.argv) > 2 else 20000) else 0)
    else:
        print(__doc__)


# ------------------------------------------------------------ 听牌(带缺门约束)
def is_ting_np(tab, hand27, num_melds: int, void_suit: int) -> bool:
    """川麻的"听牌"**不是纯牌型概念** —— 必须带缺门约束。

    ⚠️ 这条是规则调研里最容易造成**静默错误**的一处。参考引擎 lonng/nanoserver 的
    `IsTing` / `TingTiles` / `MaxMultiple` **全部不看 Que**,后果有两个,都不会报异常:
      ① 手里还留着缺门牌的人被判为"听牌",在**查大叫**里照样收钱;
      ② `MaxMultiple` 可能挑中一张**缺门牌**当作"最大番的那张",于是整笔赔叫金额
         建立在一张永远不可能胡的牌上。
    对照 cdxzmj 的 `IsTingCard → AnalyseChiHuCard → if(IsHuaZhu(...)) return WIK_NULL`
    ——它的听牌判定是带花色约束的。

    **任何直接复用国标/日麻 shanten 库的做法都会在这里出错。** 本函数即为此存在。
    """
    if any(hand27[t] for t in range(NUM_TILES) if t // 9 == void_suit):
        return False                       # 手里还有缺门牌 → 不可能听牌
    for t in range(NUM_TILES):
        if t // 9 == void_suit or hand27[t] >= 4:
            continue
        c = list(hand27)
        c[t] += 1
        _, ag = query_np(tab, c, num_melds, void_suit)
        qidui = (num_melds == 0 and sum(c) == 14
                 and all(x in (0, 2, 4) for x in c))
        if ag or qidui:
            return True
    return False


def waiting_tiles_np(tab, hand27, num_melds: int, void_suit: int):
    """听张列表(同样带缺门约束)。查大叫算"最大可能番"时枚举的就是这个集合。"""
    if any(hand27[t] for t in range(NUM_TILES) if t // 9 == void_suit):
        return []
    out = []
    for t in range(NUM_TILES):
        if t // 9 == void_suit or hand27[t] >= 4:
            continue
        c = list(hand27)
        c[t] += 1
        _, ag = query_np(tab, c, num_melds, void_suit)
        qidui = (num_melds == 0 and sum(c) == 14
                 and all(x in (0, 2, 4) for x in c))
        if ag or qidui:
            out.append(t)
    return out
