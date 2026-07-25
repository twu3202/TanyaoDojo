"""
R1 数据桥·第三件:mjai 牌谱 → Mahjax red_mahjong 逐步重放。

三个部件:
  build_deck(kyoku)     从 mjai 局记录重构 Mahjax 牌山(136 槽,0-36 面值,赤=34/35/36)
  init_from_deck(...)   镜像 env._init 但注入给定牌山/庄家/分数(jit 一次)
  replay_kyoku(...)     逐事件驱动 verify_step,断言 legal_action_mask[action]==True

对齐判据:重放合法率 100%(每步动作都在 env 的合法掩码内)+ 终局类型一致。

牌山布局(读自 env.py/_init,_draw,_kan,_draw_after_kan):
  deck[84+13p : 84+13p+13] = 玩家 p 配牌(绝对座位;槽内顺序无关,只计数)
  deck[83], deck[82], ...  = 自摸序(庄家第一张 = deck[83],在 _init 内摸出)
  deck[10+k]               = 第 k+1 次杠的岭上牌 (k=0..3)
  deck[9], deck[9-2k]      = 宝牌指示(初始 / 第 k 次新指示)
  deck[8], deck[8-2k]      = 里宝指示
动作空间关键事实:本回合动作(0-70 打/暗加杠,71 摸切,72 立直,73 自摸,85 九种)
与响应动作(74 荣,75/76 碰,77 明杠,78-83 吃,84 过)id 集合不相交 →
驱动规则"日志下一事件属于当前被服务玩家且合法则执行,否则 PASS"无歧义。
"""
from __future__ import annotations
import sys
from collections import Counter

import numpy as np

from mjai_parser import Kyoku, parse_game
from replay import is_red

BAKAZE = {"E": 0, "S": 1, "W": 2, "N": 3}
SUITS = "mps"
HONORS = ["E", "S", "W", "N", "P", "F", "C"]
RED_BY_SUIT = {"m": 34, "p": 35, "s": 36}


def mjai_to_tile(t: str) -> int:
    """mjai 牌面 → Mahjax tile(0-36,赤五=34/35/36)。"""
    if t in ("5mr", "5pr", "5sr"):
        return RED_BY_SUIT[t[1]]
    if t[0].isdigit():
        return SUITS.index(t[1]) * 9 + int(t[0]) - 1
    return 27 + HONORS.index(t)


def full_multiset() -> Counter:
    c = Counter()
    for tt in range(34):
        c[tt] = 4
    for black, red in ((4, 34), (13, 35), (22, 36)):
        c[black] -= 1
        c[red] = 1
    return c


class ReplayReject(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


def build_deck(kyoku: Kyoku) -> np.ndarray:
    """从局记录填充 136 槽牌山;未见牌用剩余多重集补齐。失败抛 ReplayReject。"""
    deck = np.full(136, -1, dtype=np.int8)
    remain = full_multiset()

    def place(ix: int, t: str, what: str):
        v = mjai_to_tile(t)
        if deck[ix] != -1:
            if deck[ix] != v:
                raise ReplayReject("slot_conflict", f"{what} deck[{ix}] {deck[ix]}!={v}")
            return
        if remain[v] <= 0:
            raise ReplayReject("multiset_overflow", f"{what} tile {t}")
        remain[v] -= 1
        deck[ix] = v

    for p, hand in enumerate(kyoku.tehais):
        for j, t in enumerate(hand):
            if t == "?":
                raise ReplayReject("hidden_tehai", f"player {p}")
            place(84 + 13 * p + j, t, "tehai")

    live_ix = 83
    kan_count = 0
    dora_k = 0
    pending_rinshan = -1
    place(9, kyoku.dora_markers[0], "dora0")
    for ev in kyoku.events:
        t = ev["type"]
        if t == "tsumo":
            if ev["pai"] == "?":
                raise ReplayReject("hidden_draw", "")
            if pending_rinshan == ev["actor"]:
                if kan_count >= 4:
                    raise ReplayReject("too_many_kans", "")
                place(10 + kan_count, ev["pai"], "rinshan")
                kan_count += 1
                pending_rinshan = -1
            else:
                if live_ix < 14:
                    raise ReplayReject("live_wall_underflow", "")
                place(live_ix, ev["pai"], "draw")
                live_ix -= 1
        elif t in ("daiminkan", "ankan", "kakan"):
            pending_rinshan = ev["actor"]
        elif t == "dora":
            dora_k += 1
            if dora_k > 4:
                raise ReplayReject("too_many_doras", "")
            place(9 - 2 * dora_k, ev["dora_marker"], "kan_dora")
        elif t == "hora":
            uras = ev.get("ura_markers") or ev.get("uradora_markers") or []
            for i, u in enumerate(uras):
                place(8 - 2 * i, u, "ura")

    holes = [ix for ix in range(136) if deck[ix] == -1]
    filler = [v for v, c in sorted(remain.items()) for _ in range(c)]
    if len(holes) != len(filler):
        raise ReplayReject("multiset_mismatch", f"holes={len(holes)} filler={len(filler)}")
    for ix, v in zip(holes, filler):
        deck[ix] = v
    return deck


# ---------------------------------------------------------------- mahjax 侧
import jax
import jax.numpy as jnp
from mahjax.red_mahjong import env as M
from mahjax.red_mahjong.action import Action
from mahjax.red_mahjong.tile import Tile


def _init_from_deck_core(deck, dealer, score, honba, kyotaku, round_idx):
    """镜像 env._init(env.py:402-458),把随机牌山/随机庄家换成给定值。"""
    init_hand_with_red = M.Hand.make_init_hand(deck)
    init_hand = jax.vmap(M.Hand.to_34)(init_hand_with_red)
    state = M._make_state(
        current_player=dealer,
        dealer=dealer,
        init_wind=M._calc_wind(dealer),
        seat_wind=M._calc_wind(dealer),
        last_player=jnp.int8(-1),
        deck=deck,
        dora_indicators=jnp.array([deck[9], -1, -1, -1, -1], dtype=jnp.int8),
        ura_dora_indicators=jnp.array([deck[8], -1, -1, -1, -1], dtype=jnp.int8),
        hand=init_hand,
        hand_with_red=init_hand_with_red,
        score=score,
        honba=honba,
        kyotaku=kyotaku,
        round=round_idx,
    )
    can_ron = M.v_can_win(state.players.hand, M.TILE_RANGE)
    c_p = state.current_player
    new_tile = state.round_state.deck[state.round_state.next_deck_ix]
    new_tile_type = Tile.to_tile_type(new_tile)
    next_deck_ix = state.round_state.next_deck_ix - 1
    eval_state = M._replace_state(state, last_draw=new_tile)
    _, yakuman_num, _ = M.Yaku.judge_yakuman(
        state.players.hand_with_red[c_p], jnp.bool_(False), c_p, eval_state
    )
    hand = state.players.hand.at[c_p].set(M.Hand.add(state.players.hand[c_p], new_tile))
    hand_with_red = state.players.hand_with_red.at[c_p].set(
        M.Hand.add(state.players.hand_with_red[c_p], new_tile)
    )
    legal_c_p = M._make_legal_action_mask_after_draw(state, hand_with_red, c_p, new_tile, None)
    legal_4p = M.ZERO_MASK_2D.at[c_p, :].set(legal_c_p)
    state = M._replace_state(
        state,
        has_yaku=state.players.has_yaku.at[c_p, 0].set(can_ron[c_p, new_tile_type]),
        fan=state.players.fan.at[c_p, 0].set(jnp.int32(yakuman_num)),
        fu=state.players.fu.at[c_p, 0].set(jnp.int32(0)),
        can_win=can_ron,
        legal_action_mask=legal_4p,
        next_deck_ix=next_deck_ix,
        hand=hand,
        hand_with_red=hand_with_red,
        last_draw=new_tile,
        target=jnp.int8(-1),
    )
    return M._replace_state(
        state, legal_action_mask=state.players.legal_action_mask[state.current_player]
    )


_init_jit = jax.jit(_init_from_deck_core)
_vstep_jit = jax.jit(lambda s, a: M.verify_step(s, a))


def init_from_deck(deck: np.ndarray, kyoku: Kyoku):
    scores = kyoku.scores or [25000] * 4
    return _init_jit(
        jnp.asarray(deck, dtype=jnp.int8),
        jnp.int8(kyoku.oya),
        jnp.array([s // 100 for s in scores], dtype=jnp.int32),
        jnp.int8(kyoku.honba),
        jnp.int8(kyoku.kyotaku),
        jnp.int8(BAKAZE[kyoku.bakaze] * 4 + (kyoku.kyoku_num - 1)),
    )


# ---------------------------------------------------------------- 动作映射
DECISION_TYPES = {"dahai", "reach", "pon", "chi", "daiminkan", "ankan", "kakan", "hora"}


def event_to_action(ev: dict) -> int:
    t = ev["type"]
    if t == "dahai":
        return int(Action.TSUMOGIRI) if ev.get("tsumogiri") else mjai_to_tile(ev["pai"])
    if t == "reach":
        return int(Action.RIICHI)
    if t == "hora":
        return int(Action.TSUMO) if ev["actor"] == ev["target"] else int(Action.RON)
    if t == "pon":
        red = any(is_red(x) for x in ev["consumed"])
        return int(Action.PON_RED) if red else int(Action.PON)
    if t == "daiminkan":
        return int(Action.OPEN_KAN)
    if t == "ankan":
        return 37 + int(Tile.to_tile_type(mjai_to_tile(ev["consumed"][0])))
    if t == "kakan":
        return 37 + int(Tile.to_tile_type(mjai_to_tile(ev["pai"])))
    if t == "chi":
        called = mjai_to_tile(ev["pai"])
        called_t = int(Tile.to_tile_type(called))
        cons = sorted(int(Tile.to_tile_type(mjai_to_tile(x))) for x in ev["consumed"])
        red = any(is_red(x) for x in ev["consumed"])
        if called_t < cons[0]:
            base = Action.CHI_L
        elif called_t > cons[1]:
            base = Action.CHI_R
        else:
            base = Action.CHI_M
        return int(base) + (1 if red else 0)
    raise ReplayReject("bad_decision_event", t)


MAX_STEPS = 400


def replay_kyoku(kyoku: Kyoku, on_decision=None) -> dict:
    """重放一局。返回 {steps, end_ok};不合法/无法推进抛 ReplayReject。

    on_decision(state, action, from_log):每个将要执行的动作前回调——
    from_log=True 是牌谱显式动作,False 是隐式 PASS(有响应权但未叫,亦是真人决策)。
    """
    deck = build_deck(kyoku)
    state = init_from_deck(deck, kyoku)
    decisions = [ev for ev in kyoku.events if ev["type"] in DECISION_TYPES]
    # 九种九牌:tenhou-mjai 无显式动作事件,以"无任何决策事件且流局"近似检测
    ptr, steps = 0, 0
    while not bool(state.round_state.terminated_round):
        if steps >= MAX_STEPS:
            raise ReplayReject("step_overflow", f"ptr={ptr}/{len(decisions)}")
        if bool(state.terminated):
            break
        mask = np.asarray(state.legal_action_mask)
        cp = int(state.current_player)
        from_log = False
        if ptr < len(decisions) and decisions[ptr]["actor"] == cp:
            a = event_to_action(decisions[ptr])
            if mask[a]:
                ptr += 1
                from_log = True
            elif mask[Action.PASS]:
                a = int(Action.PASS)
            else:
                raise ReplayReject(
                    "illegal_action",
                    f"step={steps} cp={cp} ev={decisions[ptr]} legal={np.flatnonzero(mask).tolist()}",
                )
        elif mask[Action.PASS]:
            a = int(Action.PASS)
        elif ptr >= len(decisions) and kyoku.end_type == "ryukyoku" and mask[Action.KYUUSHU]:
            # 九种九牌:mjai 无显式动作事件;verify_step 的 KYUUSHU 分支会直接
            # 推进到下一局(带新随机牌山),所以断言合法后就地终局,不再步进检查。
            if on_decision is not None:
                on_decision(state, int(Action.KYUUSHU), True)
            _, illegal = _vstep_jit(state, jnp.int32(int(Action.KYUUSHU)))
            if bool(illegal):
                raise ReplayReject("verify_illegal", f"step={steps} a=KYUUSHU")
            return {"steps": steps + 1, "end_ok": True}
        else:
            nxt = decisions[ptr] if ptr < len(decisions) else None
            raise ReplayReject(
                "serve_mismatch",
                f"step={steps} cp={cp} next_ev={nxt} legal={np.flatnonzero(mask).tolist()}",
            )
        if on_decision is not None:
            on_decision(state, a, from_log)
        state, illegal = _vstep_jit(state, jnp.int32(a))
        if bool(illegal):
            raise ReplayReject("verify_illegal", f"step={steps} a={a}")
        steps += 1
    if ptr != len(decisions):
        raise ReplayReject("events_left", f"{ptr}/{len(decisions)}")
    ended_hora = bool(np.asarray(state.players.has_won).any())
    end_ok = ended_hora == (kyoku.end_type == "hora")
    if not end_ok:
        raise ReplayReject("end_type_mismatch", f"env_hora={ended_hora} log={kyoku.end_type}")
    return {"steps": steps, "end_ok": end_ok}


if __name__ == "__main__":
    import glob, random, time

    pat = sys.argv[1] if len(sys.argv) > 1 else "."
    n_games = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    files = sorted(glob.glob(pat))
    random.Random(1).shuffle(files)
    files = files[:n_games]
    n_k = ok = 0
    rejects = Counter()
    first_fail = {}
    t0 = time.time()
    for fp in files:
        g = parse_game(fp)
        for i, k in enumerate(g.kyokus):
            n_k += 1
            try:
                replay_kyoku(k)
                ok += 1
            except ReplayReject as e:
                rejects[e.reason] += 1
                first_fail.setdefault(e.reason, f"{fp}#k{i} {e}")
    dt = time.time() - t0
    print(f"games={len(files)} kyokus={n_k} ok={ok} ({ok/max(n_k,1):.1%})  {dt:.0f}s")
    for r, c in rejects.most_common():
        print(f"  REJECT {r}: {c}\n    e.g. {first_fail[r][:300]}")
