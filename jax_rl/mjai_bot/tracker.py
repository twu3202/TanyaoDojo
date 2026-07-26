"""
R4 评测桥·第一件:mjai 事件流 → obs_lean 等价观测(无状态重建,纯 numpy)。

用途:libriichi 'mjai-log' 引擎每个决策回调给 events_json(自 start_kyoku 起),
本模块从事件前缀重建与 jax_rl/obs_lean.observe_lean 完全一致的 (planes, scalars)。
正确性判据 = test_tracker_diff.py:对 houou 牌谱逐决策点与 replay_env 的 env 真值全等。

镜像语义要点(对照 obs_lean.py 与 env.py 的实测行为):
  - 牌河 append-only(被鸣的牌留河,同 Mahjax river 灰牌);
  - 副露按"解码语义"计数:碰3/杠4/吃各1;加杠把碰升级成4(melds 编码如此);
  - 可见计数减非暗杠副露的鸣走牌(obs_lean._called_correction34);
  - last_draw = 自家最近摸牌,且其后无任何 dahai/鸣牌事件(抢杠窗口天然为 -1,
    对应 obs_lean 的 kan_declared 守卫);
  - furiten_by_discard = 自家听牌(shape waits)∩ 自家河;waits 缓存按 env 时点更新:
    自家打牌后(13张)与自家杠后(13张,岭上前);
  - furiten_by_pass = 见逃置位,自家摸牌时若未立直则清除;置位由外部信号
    note_ron_passed() 驱动(评测时来自 libriichi cans.can_ron_agari + 我们选择 PASS;
    差分测试来自 env legal_mask[RON]——两侧语义与 env 完全一致,不做事件近似);
  - live_left = 70 - 活墙摸牌数 - 王牌位移(暗杠即时 +1;明/加杠在岭上摸时 +1)。
"""
from __future__ import annotations

import numpy as np

SUITS = "mps"
HONORS = ["E", "S", "W", "N", "P", "F", "C"]
RED_BY_SUIT = {"m": 34, "p": 35, "s": 36}
BAKAZE = {"E": 0, "S": 1, "W": 2, "N": 3}
BLACK5 = {34: 4, 35: 13, 36: 22}
NUM_PLANES = 20
NUM_SCALARS = 26

MELD_EVS = ("pon", "chi", "daiminkan", "ankan", "kakan")


def t2i(t: str) -> int:
    """mjai 牌面 → 0-36(赤=34/35/36)。"""
    if t in ("5mr", "5pr", "5sr"):
        return RED_BY_SUIT[t[1]]
    if t[0].isdigit():
        return SUITS.index(t[1]) * 9 + int(t[0]) - 1
    return 27 + HONORS.index(t)


def to34(v: int) -> int:
    return BLACK5.get(v, v)


class KyokuTracker:
    """从某一席位视角追踪一局。shanten_fn(hand34)->int 与 waits_fn(hand34)->(34,)bool
    由外部注入(默认用 mahjax 的 Shanten/Hand.can_ron,见 make_jax_helpers)。"""

    def __init__(self, player_id: int, shanten_fn, waits_fn):
        self.me = player_id
        self.shanten_fn = shanten_fn
        self.waits_fn = waits_fn

    # ---------------------------------------------------------------- feed
    def start(self, ev: dict):
        me = self.me
        self.oya = ev["oya"]
        self.round = BAKAZE[ev["bakaze"]] * 4 + (ev["kyoku"] - 1)
        self.honba = ev["honba"]
        self.kyotaku = ev.get("kyotaku", 0)
        self.scores = [s // 100 for s in ev.get("scores", [25000] * 4)]
        self.hand37 = np.zeros(37, np.int8)
        for t in ev["tehais"][me]:
            self.hand37[t2i(t)] += 1
        self.dora_ind = [t2i(ev["dora_marker"])]
        self.rivers = [[] for _ in range(4)]          # 0-36 append-only
        self.rivers_str = [[] for _ in range(4)]      # 原始 mjai 牌面串(回发用)
        self.melds = [[] for _ in range(4)]           # [kind, tile_types, target34, red_suits, strs]
        self.riichi = [False] * 4
        self.last_draw = -1
        self.last_draw_str = None
        self.forbidden34 = np.zeros(34, bool)         # 食替禁打(own claim 后一手内)
        self.live_draws = 0
        self.wall_shift = 0                            # last_deck_ix 相对 14 的位移
        self.pending_rinshan = None                    # (actor, needs_shift)
        self.waits = np.zeros(34, bool)                # 自家 shape waits 缓存
        self.furiten_pass = False
        self._recompute_waits()

    def note_ron_passed(self):
        """外部信号:本席位刚对一张可荣的牌选择了 PASS(见逃)。"""
        self.furiten_pass = True

    def _recompute_waits(self):
        if int(self.hand37.sum()) % 3 == 1:            # 13 张形才有听
            self.waits = np.asarray(self.waits_fn(self._hand34()), bool)

    def _hand34(self) -> np.ndarray:
        h = self.hand37[:34].astype(np.int8).copy()
        for r, b in BLACK5.items():
            h[b] += self.hand37[r]
        return h

    def feed(self, ev: dict):
        t = ev["type"]
        me = self.me
        if t == "start_kyoku":
            self.start(ev)
            return
        if t == "tsumo":
            actor = ev["actor"]
            if self.pending_rinshan is not None and self.pending_rinshan[0] == actor:
                if self.pending_rinshan[1]:
                    self.wall_shift += 1               # 明/加杠:岭上摸时移墙
                self.pending_rinshan = None
            else:
                self.live_draws += 1
            if actor == me:
                v = t2i(ev["pai"])
                self.hand37[v] += 1
                self.last_draw = v
                self.last_draw_str = ev["pai"]
                if not self.riichi[me]:
                    self.furiten_pass = False          # env: furiten_by_pass &= riichi
        elif t == "dahai":
            actor = ev["actor"]
            v = t2i(ev["pai"])
            self.rivers[actor].append(v)
            self.rivers_str[actor].append(ev["pai"])
            if actor == me:
                self.hand37[v] -= 1
                self.forbidden34[:] = False            # 食替禁打只约束鸣后首打
                self._recompute_waits()
            self.last_draw = -1
        elif t in MELD_EVS:
            self._feed_meld(ev)
        elif t == "reach_accepted":
            a = ev["actor"]
            self.riichi[a] = True
            self.scores[a] -= 10
            self.kyotaku += 1
        elif t == "dora":
            self.dora_ind.append(t2i(ev["dora_marker"]))

    def _feed_meld(self, ev: dict):
        t, actor, me = ev["type"], ev["actor"], self.me
        cons = [t2i(x) for x in ev.get("consumed", [])]
        pai = t2i(ev["pai"]) if "pai" in ev else -1
        red_suits = set(v - 34 for v in cons + ([pai] if pai >= 34 else []) if v >= 34)
        if actor == me:
            if t == "kakan":
                self.hand37[pai] -= 1  # mjai 的 kakan.consumed 是已亮的碰牌,不在手里!
            else:
                for v in cons:
                    self.hand37[v] -= 1
        strs = list(ev.get("consumed", [])) + ([ev["pai"]] if "pai" in ev else [])
        if t == "pon":
            tt = to34(pai)
            self.melds[actor].append(["pon", [tt] * 3, tt, red_suits, strs])
            self.last_draw = -1
            if actor == me:
                self.forbidden34[:] = False
                self.forbidden34[tt] = True            # 碰后禁打同型(食替)
        elif t == "chi":
            tts = sorted(to34(v) for v in cons + [pai])
            self.melds[actor].append(["chi", tts, to34(pai), red_suits, strs])
            self.last_draw = -1
            if actor == me:
                self.forbidden34[:] = False
                ct = to34(pai)
                self.forbidden34[ct] = True            # 吃后禁打同型
                if ct == tts[0] and ct % 9 <= 5:
                    self.forbidden34[ct + 3] = True    # 低位吃禁打高位食替(如 [4]56 禁 7)
                elif ct == tts[2] and ct % 9 >= 3:
                    self.forbidden34[ct - 3] = True    # 高位吃禁打低位食替(如 45[6] 禁 3)
        elif t == "daiminkan":
            tt = to34(pai)
            if tt in (4, 13, 22):
                red_suits.add(tt // 9)                 # 五的杠必含赤
            self.melds[actor].append(["kan_open", [tt] * 4, tt, red_suits, strs])
            self.pending_rinshan = (actor, True)
            self.last_draw = -1
        elif t == "ankan":
            tt = to34(cons[0])
            if tt in (4, 13, 22):
                red_suits.add(tt // 9)
            self.melds[actor].append(["kan_closed", [tt] * 4, tt, red_suits, strs])
            self.wall_shift += 1                       # 暗杠即时移墙+翻宝
            self.pending_rinshan = (actor, False)
        elif t == "kakan":
            tt = to34(pai)
            if tt in (4, 13, 22):
                red_suits.add(tt // 9)
            for m in self.melds[actor]:                # 升级碰 → 杠
                if m[0] == "pon" and m[2] == tt:
                    m[0] = "kan_added"
                    m[1] = [tt] * 4
                    m[3] = m[3] | red_suits
                    m[4] = m[4] + ([ev["pai"]] if "pai" in ev else [])
                    break
            self.pending_rinshan = (actor, True)
        if actor == me and t in ("pon", "chi", "daiminkan", "ankan"):
            self._recompute_waits()
        if t in ("daiminkan", "ankan", "kakan") and actor == me:
            self._recompute_waits()

    # ---------------------------------------------------------------- obs
    def build_obs(self) -> dict:
        me = self.me
        rel = [(me + i) % 4 for i in range(4)]
        hand34 = self._hand34().astype(np.float32)
        planes = np.zeros((20, 34), np.float32)
        for k in range(4):
            planes[k] = (hand34 >= k + 1)
        for r, b in BLACK5.items():
            planes[4][b] = float(self.hand37[r])
        if self.last_draw >= 0:
            planes[5][to34(self.last_draw)] = 1.0
        for i, p in enumerate(rel):
            for v in self.rivers[p]:
                planes[6 + i][to34(v)] += 1.0
            if self.rivers[p]:
                planes[10 + i][to34(self.rivers[p][-1])] = 1.0
        planes[6:10] /= 4.0
        called = np.zeros(34, np.float32)
        meld_all = np.zeros(34, np.float32)
        for i, p in enumerate(rel):
            for kind, tts, tgt, _, _s in self.melds[p]:
                for tt in tts:
                    planes[14 + i][tt] += 1.0
                    meld_all[tt] += 1.0
                if kind != "kan_closed":
                    called[tgt] += 1.0
        planes[14:18] /= 4.0
        river_all = np.zeros(34, np.float32)
        for p in range(4):
            for v in self.rivers[p]:
                river_all[to34(v)] += 1.0
        dora_cnt = np.zeros(34, np.float32)
        for v in self.dora_ind:
            dora_cnt[to34(v)] += 1.0
        planes[18] = (hand34 + river_all + meld_all - called + dora_cnt) / 4.0
        planes[19] = dora_cnt / 4.0

        shanten = float(self.shanten_fn(self._hand34()))
        furiten_discard = bool(
            (self.waits & np.isin(np.arange(34), [to34(v) for v in self.rivers[me]])).any()
        )
        live_left = 70.0 - self.live_draws - self.wall_shift
        red_seen = 0.0
        for s, r in enumerate((34, 35, 36)):
            in_river = any(v == r for p in range(4) for v in self.rivers[p])
            in_meld = any(s in m[3] for p in range(4) for m in self.melds[p])
            in_dora = r in self.dora_ind
            red_seen += float(in_river or in_meld or in_dora)
        sc = np.zeros(26, np.float32)
        sc[0:4] = [float(self.riichi[p]) for p in rel]
        sc[4:8] = [self.scores[p] / 500.0 for p in rel]
        sc[8] = shanten / 6.0
        sc[9] = float(furiten_discard or self.furiten_pass)
        sc[10] = self.round / 12.0
        sc[11] = self.honba / 10.0
        sc[12] = self.kyotaku / 10.0
        sc[13 + self.round // 4] = 1.0
        sc[17 + (me - self.oya) % 4] = 1.0
        sc[21] = (len(self.dora_ind) - 1) / 4.0
        sc[22] = red_seen / 3.0
        sc[23] = len(self.melds[me]) / 4.0
        sc[24] = float(live_left <= 0.0)
        sc[25] = live_left / 70.0
        return {"planes": planes.T, "scalars": sc}


def make_jax_helpers():
    """默认 shanten/waits 实现(mahjax,jit 缓存)。"""
    import jax
    import jax.numpy as jnp
    from mahjax.red_mahjong.shanten import Shanten
    from mahjax.red_mahjong.hand import Hand

    shan = jax.jit(Shanten.number)
    waits = jax.jit(
        lambda h: jax.vmap(Hand.can_ron, in_axes=(None, 0))(
            h.astype(jnp.int8), jnp.arange(34, dtype=jnp.int32)
        )
    )
    return (lambda h34: int(shan(np.asarray(h34, np.int8)))), (
        lambda h34: np.asarray(waits(np.asarray(h34, np.int8)))
    )
