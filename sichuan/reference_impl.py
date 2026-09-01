"""
川麻(血战到底)参考实现 v0 —— 对应 RULES.md v0,正确性优先、速度无关紧要。
角色:JAX 环境的差分测试 oracle + 规则争议的可执行裁决。

牌编码:0..26,suit = t // 9 (0=万,1=筒,2=条),rank = t % 9 + 1。
动作:(kind, arg) 元组:
  ("void", suit)        定缺
  ("discard", tile)     打牌
  ("ankan", tile)       暗杠
  ("bugang", tile)      补杠(碰后摸到第4张)
  ("zimo", None)        自摸胡
  ("ron", None)         荣胡(响应)
  ("peng", None)        碰(响应)
  ("zhigang", None)     直杠(响应)
  ("pass", None)        过(响应)
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional

NUM_TILES = 27
FAN_CAP = 4
BASE = 1


def suit_of(t): return t // 9


def fmt_tile(t):
    return f"{t % 9 + 1}{'万筒条'[t // 9]}"


# ---------------- 胡牌型判定 ----------------

def _can_form_sets(counts, n_sets):
    """counts(27) 能否拆成 n_sets 个面子(刻子/顺子)。"""
    if n_sets == 0:
        return all(c == 0 for c in counts)
    for i in range(NUM_TILES):
        if counts[i] > 0:
            first = i
            break
    else:
        return False
    c = counts[:]
    # 刻子
    if c[first] >= 3:
        c[first] -= 3
        if _can_form_sets(c, n_sets - 1):
            return True
        c[first] += 3
    # 顺子(不跨花色)
    if first % 9 <= 6 and c[first + 1] > 0 and c[first + 2] > 0:
        c[first] -= 1; c[first + 1] -= 1; c[first + 2] -= 1
        if _can_form_sets(c, n_sets - 1):
            return True
    return False


def is_hu(hand_counts, num_melds, void_suit):
    """hand_counts 含胡张(即 3k+2 张)。需满足缺门。"""
    total = sum(hand_counts)
    if any(hand_counts[t] for t in range(NUM_TILES) if suit_of(t) == void_suit):
        return False
    if total != 3 * (4 - num_melds) + 2:
        return False
    # 七对(门清)
    if num_melds == 0 and total == 14 and all(c in (0, 2, 4) for c in hand_counts):
        return True
    # 标准型
    for p in range(NUM_TILES):
        if hand_counts[p] >= 2:
            c = hand_counts[:]
            c[p] -= 2
            if _can_form_sets(c, 4 - num_melds):
                return True
    return False


def _decomp_all_kotsu(hand_counts, num_meld_kotsu, num_melds):
    """能否按"全刻子"拆(对对胡):所有副露须为碰/杠(顺子副露不存在于川麻,恒真)。"""
    for p in range(NUM_TILES):
        if hand_counts[p] >= 2:
            c = hand_counts[:]
            c[p] -= 2
            need = 4 - num_melds
            ok = True
            for t in range(NUM_TILES):
                if c[t] % 3 != 0:
                    ok = False
                    break
            if ok and sum(c) == 3 * need:
                return True
    return False


def calc_fan(hand_counts, melds, void_suit, ctx):
    """
    ctx: dict(zimo, gang_shang_hua, gang_shang_pao, qiang_gang, hai_di)
    melds: list of (kind, tile) kind in {peng, gang_ming, gang_an, gang_bu}
    返回(fan, 名目列表)。hand_counts 含胡张。
    """
    fans = 0
    names = []
    num_melds = len(melds)
    total_counts = hand_counts[:]
    for kind, tile in melds:
        total_counts[tile] += 4 if kind.startswith("gang") else 3

    qidui = num_melds == 0 and sum(hand_counts) == 14 and all(c in (0, 2, 4) for c in hand_counts)
    long_qidui = qidui and any(c == 4 for c in hand_counts)
    if long_qidui:
        fans += 3; names.append("龙七对")
    elif qidui:
        fans += 2; names.append("七对")
    elif _decomp_all_kotsu(hand_counts, None, num_melds):
        fans += 1; names.append("对对胡")

    suits = {suit_of(t) for t in range(NUM_TILES) if total_counts[t] > 0}
    if len(suits) == 1:
        fans += 2; names.append("清一色")

    # 金钩钓:副露 4 组,手中仅单钓成对
    if num_melds == 4 and sum(hand_counts) == 2:
        fans += 1; names.append("金钩钓")

    # 根:每组 4 张同牌(杠或手中4张);龙七对的那组不另计
    gen = 0
    for t in range(NUM_TILES):
        if total_counts[t] == 4:
            gen += 1
    if long_qidui:
        gen -= sum(1 for c in hand_counts if c == 4)
    if gen > 0:
        fans += gen; names.append(f"根x{gen}")

    for key, name in (("zimo", "自摸"), ("gang_shang_hua", "杠上花"),
                      ("gang_shang_pao", "杠上炮"), ("qiang_gang", "抢杠"), ("hai_di", "海底")):
        if ctx.get(key):
            fans += 1; names.append(name)

    return min(fans, FAN_CAP), names


# ---------------- 对局状态机 ----------------

@dataclass
class Player:
    hand: list = field(default_factory=lambda: [0] * NUM_TILES)
    melds: list = field(default_factory=list)     # (kind, tile)
    void: Optional[int] = None
    hu: bool = False
    hu_fan: int = 0
    score_delta: int = 0


class SichuanGame:
    """
    phase: "void" -> "action"(当前玩家摸后决策) -> "response"(打牌后逐家响应) -> ... -> "over"
    响应实现:打牌后依优先级征询(先胡后碰杠),内部逐家询问;多家胡全部成立。
    """

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.wall = [t for t in range(NUM_TILES) for _ in range(4)]
        self.rng.shuffle(self.wall)
        self.players = [Player() for _ in range(4)]
        self.discards = [[] for _ in range(4)]
        self.gang_ledger = []      # (payer, receiver, amount) 刮风下雨记录,退税用
        self.cur = 0               # 当前行动者
        self.phase = "void"
        self.void_declared = 0
        self.pending_discard = None      # (player, tile)
        self.response_queue = []         # 待询问的响应者
        self.last_draw_was_gang = False  # 杠上花/杠上炮标记
        self.hu_order = []
        self.turn_guard = 0
        for p in self.players:
            for _ in range(13):
                self._draw_to(p)
        # 庄家(0)的第 14 张在定缺后摸(简化:定缺完成后进入 0 的 action 并摸牌)

    # ----- 基础 -----
    def _draw_to(self, p: Player):
        t = self.wall.pop()
        p.hand[t] += 1
        return t

    def alive(self, i):
        return not self.players[i].hu

    def alive_players(self):
        return [i for i in range(4) if self.alive(i)]

    def scores(self):
        return [p.score_delta for p in self.players]

    def tile_conservation(self):
        n = len(self.wall)
        for i, p in enumerate(self.players):
            n += sum(p.hand)
            for kind, _ in p.melds:
                n += 4 if kind.startswith("gang") else 3
            n += len(self.discards[i])
        return n

    # ----- 合法动作 -----
    def legal_actions(self):
        """返回 (player, [(kind, arg), ...])。phase=void 时依次征询。"""
        if self.phase == "over":
            return None, []
        if self.phase == "void":
            i = self.void_declared
            return i, [("void", s) for s in range(3)]
        if self.phase == "action":
            p = self.players[self.cur]
            acts = []
            post_draw = sum(p.hand) % 3 == 2   # 摸牌后 3k+2;碰后 3k+1(只能打牌)
            void_tiles = [t for t in range(NUM_TILES) if suit_of(t) == p.void and p.hand[t] > 0]
            if void_tiles:
                acts = [("discard", t) for t in void_tiles]      # 缺门必打
            else:
                acts = [("discard", t) for t in range(NUM_TILES) if p.hand[t] > 0]
                if post_draw:
                    for t in range(NUM_TILES):
                        if p.hand[t] == 4 and suit_of(t) != p.void:
                            acts.append(("ankan", t))
                        if p.hand[t] == 1 and ("peng", t) in p.melds:
                            acts.append(("bugang", t))
                    if is_hu(p.hand, len(p.melds), p.void):
                        acts.append(("zimo", None))
            return self.cur, acts
        if self.phase == "response":
            i = self.response_queue[0]
            p = self.players[i]
            dp, dt = self.pending_discard
            acts = [("pass", None)]
            if suit_of(dt) != p.void:
                c = p.hand[:]
                c[dt] += 1
                if is_hu(c, len(p.melds), p.void):
                    acts.append(("ron", None))
                if p.hand[dt] >= 2:
                    acts.append(("peng", None))
                if p.hand[dt] == 3:
                    acts.append(("zhigang", None))
            return i, acts
        raise RuntimeError(self.phase)

    # ----- 结算工具 -----
    def _pay(self, payer, receiver, amount, gang=False):
        self.players[payer].score_delta -= amount
        self.players[receiver].score_delta += amount
        if gang:
            self.gang_ledger.append((payer, receiver, amount))

    def _gang_money(self, ganger, kind, provider=None):
        targets = [i for i in self.alive_players() if i != ganger]
        if kind == "gang_ming":
            self._pay(provider, ganger, 2 * BASE, gang=True)
        elif kind == "gang_bu":
            for i in targets:
                self._pay(i, ganger, 1 * BASE, gang=True)
        elif kind == "gang_an":
            for i in targets:
                self._pay(i, ganger, 2 * BASE, gang=True)

    def _settle_hu(self, winner, hand_counts, ctx, discarder=None):
        p = self.players[winner]
        fan, names = calc_fan(hand_counts, p.melds, p.void, ctx)
        amount = 2 ** fan * BASE
        if ctx.get("zimo"):
            for i in self.alive_players():
                if i != winner:
                    self._pay(i, winner, amount)
        else:
            self._pay(discarder, winner, amount)
        p.hu = True
        p.hu_fan = fan
        self.hu_order.append((winner, fan, tuple(names)))

    # ----- 流局:查叫与退税 -----
    def _waits_max_fan(self, i):
        p = self.players[i]
        best = None
        for t in range(NUM_TILES):
            if p.hand[t] >= 4:
                continue
            c = p.hand[:]
            c[t] += 1
            if is_hu(c, len(p.melds), p.void):
                fan, _ = calc_fan(c, p.melds, p.void, {})
                best = fan if best is None else max(best, fan)
        return best  # None = 未听

    def _liuju_settle(self):
        waits = {}
        for i in range(4):
            if self.players[i].hu:
                waits[i] = "hu"
            else:
                waits[i] = self._waits_max_fan(i)
        # 查大叫
        ting = [i for i in range(4) if waits[i] not in (None, "hu")]
        noting = [i for i in range(4) if waits[i] is None]
        for loser in noting:
            for winner in ting:
                self._pay(loser, winner, 2 ** waits[winner] * BASE)
        # 退税:未听者退杠钱
        for payer, receiver, amount in self.gang_ledger:
            if waits.get(receiver) is None:
                self.players[receiver].score_delta -= amount
                self.players[payer].score_delta += amount
        self.phase = "over"

    # ----- 推进 -----
    def _next_alive(self, i):
        for k in range(1, 5):
            j = (i + k) % 4
            if self.alive(j):
                return j
        return None

    def _begin_turn(self, i, from_gang=False):
        """i 摸牌进入 action;墙空则流局。"""
        if len(self.hu_order) >= 3:
            self.phase = "over"
            return
        if not self.wall:
            self._liuju_settle()
            return
        self.cur = i
        self.last_draw_was_gang = from_gang
        self.hai_di = len(self.wall) == 1
        self._draw_to(self.players[i])
        self.phase = "action"

    def step(self, action):
        self.turn_guard += 1
        assert self.turn_guard < 4000, "疑似死循环"
        kind, arg = action
        if self.phase == "void":
            self.players[self.void_declared].void = arg
            self.void_declared += 1
            if self.void_declared == 4:
                self._begin_turn(0)
            return
        if self.phase == "action":
            p = self.players[self.cur]
            if kind == "discard":
                assert p.hand[arg] > 0
                p.hand[arg] -= 1
                self.discards[self.cur].append(arg)
                self.discard_was_gang = self.last_draw_was_gang
                self.pending_discard = (self.cur, arg)
                self.response_queue = [j for j in (self._resp_order()) ]
                self.ron_winners = []
                self.phase = "response" if self.response_queue else None
                if not self.response_queue:
                    self._begin_turn(self._next_alive(self.cur))
                return
            if kind == "ankan":
                p.hand[arg] -= 4
                p.melds.append(("gang_an", arg))
                self._gang_money(self.cur, "gang_an")
                self._begin_turn_same(from_gang=True)
                return
            if kind == "bugang":
                # 抢杠窗口:征询其他在场者是否荣该张
                robbed = False
                for j in self.alive_players():
                    if j == self.cur:
                        continue
                    q = self.players[j]
                    if suit_of(arg) == q.void:
                        continue
                    c = q.hand[:]
                    c[arg] += 1
                    if is_hu(c, len(q.melds), q.void):
                        c2 = q.hand[:]
                        c2[arg] += 1
                        self._settle_hu(j, c2, {"qiang_gang": True}, discarder=self.cur)
                        robbed = True
                if robbed:
                    p.hand[arg] -= 1   # 该张被抢(离开杠者;计入胡家结算,牌面归属仅记账)
                    self.discards[self.cur].append(arg)
                    nxt = self._next_alive(self.cur)
                    if nxt is None or len(self.hu_order) >= 3:
                        self.phase = "over"
                    else:
                        self._begin_turn(nxt)
                    return
                p.hand[arg] -= 1
                p.melds.remove(("peng", arg))
                p.melds.append(("gang_bu", arg))
                self._gang_money(self.cur, "gang_bu")
                self._begin_turn_same(from_gang=True)
                return
            if kind == "zimo":
                ctx = {"zimo": True,
                       "gang_shang_hua": self.last_draw_was_gang,
                       "hai_di": getattr(self, "hai_di", False)}
                self._settle_hu(self.cur, self.players[self.cur].hand, ctx)
                nxt = self._next_alive(self.cur)
                if nxt is None or len(self.hu_order) >= 3:
                    self.phase = "over"
                else:
                    self._begin_turn(nxt)
                return
            raise ValueError(f"action phase 不接受 {kind}")
        if self.phase == "response":
            i = self.response_queue[0]
            dp, dt = self.pending_discard
            p = self.players[i]
            if kind == "ron":
                c = p.hand[:]
                c[dt] += 1
                ctx = {"gang_shang_pao": getattr(self, "discard_was_gang", False)}
                self._settle_hu(i, c, ctx, discarder=dp)
                self.ron_winners.append(i)
                self.response_queue.pop(0)
            elif kind == "pass":
                self.response_queue.pop(0)
            elif kind in ("peng", "zhigang"):
                # 有人已荣则碰/杠不成立(牌被胡家拿走)
                if self.ron_winners:
                    self.response_queue.pop(0)
                else:
                    self.response_queue = []
                    if kind == "peng":
                        p.hand[dt] -= 2
                        p.melds.append(("peng", dt))
                        self.discards[dp].pop()
                        self.cur = i
                        self.phase = "action_after_claim"
                        self.phase = "action"
                        self.last_draw_was_gang = False
                        return
                    else:
                        p.hand[dt] -= 3
                        p.melds.append(("gang_ming", dt))
                        self.discards[dp].pop()
                        self._gang_money(i, "gang_ming", provider=dp)
                        self._begin_turn(i, from_gang=True)
                        return
            else:
                raise ValueError(f"response phase 不接受 {kind}")
            if not self.response_queue:
                if self.ron_winners:
                    self.discards[dp].pop()
                    self.players[self.ron_winners[0]].hand[dt] += 1   # 牌归首个胡家(守恒)
                    if len(self.hu_order) >= 3:
                        self.phase = "over"
                        return
                    nxt = self._next_alive(dp)
                    if nxt is None:
                        self.phase = "over"
                    else:
                        self._begin_turn(nxt)
                else:
                    self._begin_turn(self._next_alive(dp))
            return
        raise RuntimeError(f"phase={self.phase}")

    def _resp_order(self):
        dp, _ = self.pending_discard
        order = []
        j = dp
        for _ in range(3):
            j = (j + 1) % 4
            if self.alive(j) and j != dp:
                order.append(j)
        return order

    def _begin_turn_same(self, from_gang):
        """杠后本家补摸。"""
        if not self.wall:
            self._liuju_settle()
            return
        self.last_draw_was_gang = from_gang
        self.hai_di = len(self.wall) == 1
        self._draw_to(self.players[self.cur])
        self.phase = "action"


# ---------------- 随机对局与不变量 ----------------

def random_playout(seed=0, policy=None):
    """policy(game, player, actions) -> action;缺省均匀随机(胡优先,提高终局覆盖)。"""
    g = SichuanGame(seed)
    rng = random.Random(seed ^ 0xABCDEF)
    while g.phase != "over":
        i, acts = g.legal_actions()
        assert acts, f"无合法动作 phase={g.phase}"
        if policy:
            a = policy(g, i, acts)
        else:
            hu_acts = [a for a in acts if a[0] in ("zimo", "ron")]
            a = rng.choice(hu_acts) if hu_acts else rng.choice(acts)
        g.step(a)
        assert g.tile_conservation() == 108, "牌数守恒破坏"
        assert sum(g.scores()) == 0, "零和破坏"
    return g
