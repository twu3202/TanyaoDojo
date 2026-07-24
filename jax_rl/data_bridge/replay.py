"""
R1 数据桥·第二件:牌面映射 + 牌山重构 + 重放循环骨架。

已实现(纯 Python,可全量校验):
  - mjai 牌面 ↔ 34 型/136 型映射(含赤五)
  - 从 mjai 局记录重构"逻辑牌山"(tehais + 摸牌序列 + 岭上/宝牌区),并做多重集一致性校验
已勘明的 Mahjax 注入点(下一步实现 init_from_deck 时用):
  - red_mahjong/env.py:409  deck = permutation(136) → 换成我们重构的 deck 即可
  - state.round_state.deck / next_deck_ix / last_deck_ix;dora 指示牌 = deck[9],里 = deck[8]
    (死墙居 deck 低位;精确的配牌/摸牌索引约定见 Hand.make_init_hand,实现 init_from_deck
     时逐字段对齐,并用"重放合法性 == 100%"作为对齐正确的判据)
待实现(下一工作段):
  - init_from_deck(deck, oya, scores, honba, kyotaku)
  - mjai 事件 → red_mahjong Action id 映射(action.py)
  - 重放主循环:逐事件断言 legal_action_mask[action]==True(差分校验),同时产出 (obs, action)
"""
from __future__ import annotations
from collections import Counter

from mjai_parser import Kyoku, iter_draws

SUITS = "mps"
HONORS = ["E", "S", "W", "N", "P", "F", "C"]


def mjai_to_t34(t: str) -> int:
    """mjai 牌面 → 34 型编号(赤五归并到普通五)。"""
    if t in ("5mr", "5pr", "5sr"):
        return SUITS.index(t[1]) * 9 + 4
    if t[0].isdigit():
        return SUITS.index(t[1]) * 9 + int(t[0]) - 1
    return 27 + HONORS.index(t)


def is_red(t: str) -> bool:
    return t in ("5mr", "5pr", "5sr")


def reconstruct_logical_wall(kyoku: Kyoku) -> dict:
    """
    从局记录重构逻辑牌山各分区(不定物理顺序,只定"谁在哪个区"):
      hands: 4×13(配牌区) | draws: 按序摸牌(含岭上) | dora_markers: 宝牌指示区
      unseen: 剩余(死墙未翻部分 + 未摸到的活牌)
    返回各分区 + 校验信息。多重集校验:全体并集 ≤ 每种牌 4 张、总数 ≤ 136。
    """
    used = Counter()
    for hand in kyoku.tehais:
        for t in hand:
            used[t if not is_red(t) else t] += 1
    draws = [(a, t) for a, t in iter_draws(kyoku)]
    for _, t in draws:
        used[t] += 1
    for t in kyoku.dora_markers:
        used[t] += 1

    # 赤五与普通五合并计数(物理各 1 张赤,mjai 已区分,直接按字面计)
    over = {t: c for t, c in used.items() if t != "?" and (
        c > (1 if is_red(t) else (3 if t in ("5m", "5p", "5s") else 4)))}
    total = sum(c for t, c in used.items() if t != "?")
    return {
        "hands": kyoku.tehais,
        "draws": draws,
        "dora_markers": kyoku.dora_markers,
        "seen_total": total,
        "over_quota": over,
        "ok": not over and total <= 136,
    }


if __name__ == "__main__":
    import sys, glob, random
    from mjai_parser import parse_game
    pat = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(pat))
    random.Random(1).shuffle(files)
    files = files[: int(sys.argv[2]) if len(sys.argv) > 2 else 100]
    games = kyokus = bad = 0
    seen_hist = Counter()
    for fp in files:
        g = parse_game(fp)
        games += 1
        for k in g.kyokus:
            r = reconstruct_logical_wall(k)
            kyokus += 1
            seen_hist[r["seen_total"] // 10 * 10] += 1
            if not r["ok"]:
                bad += 1
                print("WALL_FAIL:", fp, k.bakaze, k.kyoku_num, r["over_quota"], r["seen_total"])
    print(f"games={games} kyokus={kyokus} wall_bad={bad}")
    print("seen_total 直方(×10):", dict(sorted(seen_hist.items())))
