"""
川麻(血战到底)JAX 环境 —— P1 主体。

设计原则(方案 §5.1/§5.2):**不 fork mahjax,而是实现它的 Env 协议**
(init / step / num_players / legal_action_mask / current_player / terminated /
truncated / rewards),因此现有训练器(ppo_qcritic 等)一行不改即可复用。

正确性基准:sichuan/reference_impl.py 的状态机逐条镜像(硬闸门 = 百万局逐决策点
零失配)。全部字段定长、无动态形状、无递归,jit/vmap 友好。

动作空间(61):
   0..26  打牌 t
  27..53  杠 t(暗杠/补杠由手牌状态消歧:手中 4 张=暗杠;手中 1 张且已碰=补杠)
  54      碰      55  直杠(点杠)   56  胡(自摸/荣由 phase 消歧)
  57      过      58..60  定缺(万/筒/条)

phase: 0=定缺 1=行动(摸后决策) 2=响应(打牌后逐家询问) 3=终局

关键实现注记:
  · 血战"离场"用定长 bool 掩码 finished 表示,轮转 = 定长 4 循环跳过离场者;
  · 响应队列定长 3(-1 填充),按 reference 的 _resp_order 顺序;
  · 一炮多响:队列内每家独立结算,ron_flag 记录;碰/杠在已有荣时不成立;
  · 刮风下雨记账用 (4,4) 矩阵 gang_ledger[payer, receiver],流局退税直接按行列求和;
  · 抢杠:补杠时 vmap 检查其余在场者能否荣该张,成立则补杠不成立。
"""
from __future__ import annotations

import functools
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.suit_table import load_table, make_jax_ops, NUM_TILES

NUM_PLAYERS = 4
NUM_ACTIONS = 61
WALL_SIZE = 108
MAX_MELDS = 4
FAN_CAP = 4
BASE = 1

A_DISCARD = 0          # 0..26
A_GANG = 27            # 27..53(自家杠:暗杠/补杠)
A_PENG = 54
A_ZHIGANG = 55
A_HU = 56
A_PASS = 57
A_VOID = 58            # 58..60

PH_VOID, PH_ACTION, PH_RESP, PH_OVER = 0, 1, 2, 3
MK_NONE, MK_PENG, MK_GANG_MING, MK_GANG_AN, MK_GANG_BU = 0, 1, 2, 3, 4

TRUE, FALSE = jnp.bool_(True), jnp.bool_(False)
SUIT = jnp.arange(NUM_TILES) // 9


@struct.dataclass
class State:
    # --- 牌面 ---
    hand: jnp.ndarray            # (4,27) int8
    melds_kind: jnp.ndarray      # (4,4) int8
    melds_tile: jnp.ndarray      # (4,4) int8
    n_melds: jnp.ndarray         # (4,) int8
    river: jnp.ndarray           # (4,27) int8(被鸣走的牌会减回去,与 reference 的 pop 一致)
    wall: jnp.ndarray            # (108,) int8
    wall_ix: jnp.ndarray         # int32 已摸张数(从尾部摸,对齐 reference 的 pop)
    # --- 玩家 ---
    void: jnp.ndarray            # (4,) int8,-1 未定
    finished: jnp.ndarray        # (4,) bool 已胡离场
    score: jnp.ndarray           # (4,) int32
    hu_fan: jnp.ndarray          # (4,) int8
    gang_ledger: jnp.ndarray     # (4,4) int32 [payer, receiver]
    # --- 流程 ---
    phase: jnp.ndarray           # int32
    cur: jnp.ndarray             # int32
    void_declared: jnp.ndarray   # int32
    pending_tile: jnp.ndarray    # int32
    pending_from: jnp.ndarray    # int32
    resp_queue: jnp.ndarray      # (3,) int32,-1 填充
    resp_ix: jnp.ndarray         # int32
    ron_flag: jnp.ndarray        # (4,) bool 本次打牌的荣和者
    first_ron: jnp.ndarray       # int32 队列顺序里**第一个**荣和者(-1 无);牌归他
    last_draw_was_gang: jnp.ndarray  # bool
    discard_was_gang: jnp.ndarray    # bool
    hai_di: jnp.ndarray          # bool
    n_hu: jnp.ndarray            # int32
    # --- 协议字段(对齐 mahjax)---
    legal_action_mask: jnp.ndarray   # (61,) bool
    terminated: jnp.ndarray      # bool
    truncated: jnp.ndarray       # bool
    rewards: jnp.ndarray         # (4,) float32
    step_count: jnp.ndarray      # int32
    current_player: jnp.ndarray  # int32


_TAB = load_table()
_agari_fn, _shanten_fn = make_jax_ops(_TAB)


def _is_hu(hand27, n_melds, void):
    """含胡张的 3k+2 手牌是否成胡(带缺门约束)。"""
    return _agari_fn(hand27.astype(jnp.int32), n_melds.astype(jnp.int32), void.astype(jnp.int32))


# ------------------------------------------------------------------ 番数
def _all_kotsu(hand27):
    """能否全刻拆(对对胡):去掉一对后其余每型计数均为 3 的倍数。"""
    def try_pair(p):
        c = hand27.at[p].add(-2)
        return (hand27[p] >= 2) & jnp.all((c >= 0) & (c % 3 == 0))
    return jnp.any(jax.vmap(try_pair)(jnp.arange(NUM_TILES)))


def _calc_fan(hand27, melds_kind, melds_tile, n_melds, void,
              zimo, gang_hua, gang_pao, qiang_gang, hai_di):
    """与 reference_impl.calc_fan 逐条对应。hand27 含胡张。"""
    hand27 = hand27.astype(jnp.int32)
    slot = jnp.arange(MAX_MELDS)
    active = slot < n_melds
    add = jnp.where(melds_kind >= MK_GANG_MING, 4, 3) * active
    total = hand27
    total = total.at[jnp.clip(melds_tile, 0, NUM_TILES - 1)].add(jnp.where(active, add, 0))

    n_hand = jnp.sum(hand27)
    qidui = (n_melds == 0) & (n_hand == 14) & jnp.all((hand27 == 0) | (hand27 == 2) | (hand27 == 4))
    long_qidui = qidui & jnp.any(hand27 == 4)
    duidui = _all_kotsu(hand27)

    fan = jnp.where(long_qidui, 3, jnp.where(qidui, 2, jnp.where(duidui, 1, 0)))
    # 清一色:全部同花色
    present = total > 0
    suits_used = jnp.array([jnp.any(present & (SUIT == s)) for s in range(3)])
    fan = fan + jnp.where(jnp.sum(suits_used) == 1, 2, 0)
    # 金钩钓
    fan = fan + jnp.where((n_melds == 4) & (n_hand == 2), 1, 0)
    # 根(龙七对里那组 4 张不另计)
    gen = jnp.sum(total == 4) - jnp.where(long_qidui, jnp.sum(hand27 == 4), 0)
    fan = fan + jnp.maximum(gen, 0)
    fan = fan + zimo.astype(jnp.int32) + gang_hua.astype(jnp.int32) \
              + gang_pao.astype(jnp.int32) + qiang_gang.astype(jnp.int32) + hai_di.astype(jnp.int32)
    return jnp.minimum(fan, FAN_CAP)


# ------------------------------------------------------------------ 结算原语
def _pay(st: State, payer, receiver, amount, is_gang=False):
    """payer → receiver 支付 amount。is_gang 时同时记入退税账本。"""
    score = st.score.at[payer].add(-amount).at[receiver].add(amount)
    ledger = jnp.where(is_gang, st.gang_ledger.at[payer, receiver].add(amount), st.gang_ledger)
    return st.replace(score=score, gang_ledger=ledger)


def _pay_all_alive(st: State, receiver, amount, is_gang=False):
    """所有"未胡且在场"且非 receiver 者各付 amount(向量化,离场者付 0)。"""
    tgt = (~st.finished) & (jnp.arange(NUM_PLAYERS) != receiver)
    delta = jnp.where(tgt, -amount, 0)
    score = st.score + delta
    score = score.at[receiver].add(amount * jnp.sum(tgt))
    ledger = jnp.where(is_gang,
                       st.gang_ledger.at[:, receiver].add(jnp.where(tgt, amount, 0)),
                       st.gang_ledger)
    return st.replace(score=score, gang_ledger=ledger)


def _settle_hu(st: State, winner, hand_with_win, zimo, discarder,
               gang_hua=FALSE, gang_pao=FALSE, qiang_gang=FALSE, hai_di=FALSE):
    fan = _calc_fan(hand_with_win, st.melds_kind[winner], st.melds_tile[winner],
                    st.n_melds[winner].astype(jnp.int32), st.void[winner].astype(jnp.int32),
                    zimo, gang_hua, gang_pao, qiang_gang, hai_di)
    amount = (2 ** fan) * BASE
    st = jax.lax.cond(zimo,
                      lambda s: _pay_all_alive(s, winner, amount),
                      lambda s: _pay(s, discarder, winner, amount), st)
    return st.replace(finished=st.finished.at[winner].set(True),
                      hu_fan=st.hu_fan.at[winner].set(fan.astype(jnp.int8)),
                      n_hu=st.n_hu + 1)


# ------------------------------------------------------------------ 流局
def _waits_max_fan(st: State, i):
    """i 的听牌最大番;未听返回 -1。枚举 27 张补入后是否成胡。"""
    hand = st.hand[i].astype(jnp.int32)
    nm = st.n_melds[i].astype(jnp.int32)
    vd = st.void[i].astype(jnp.int32)

    def one(t):
        c = hand.at[t].add(1)
        ok = (hand[t] < 4) & _is_hu(c, nm, vd)
        fan = _calc_fan(c, st.melds_kind[i], st.melds_tile[i], nm, vd,
                        FALSE, FALSE, FALSE, FALSE, FALSE)
        return jnp.where(ok, fan, -1)

    return jnp.max(jax.vmap(one)(jnp.arange(NUM_TILES)))


def _liuju_settle(st: State) -> State:
    """查大叫 + 退税。已胡者视同听牌、不参与查叫。"""
    waits = jax.vmap(lambda i: _waits_max_fan(st, i))(jnp.arange(NUM_PLAYERS))
    is_ting = (waits >= 0) & (~st.finished)
    no_ting = (waits < 0) & (~st.finished)
    # 查大叫:每个未听者向每个听牌者付 2^其最大番
    pay = (2 ** jnp.clip(waits, 0, FAN_CAP)) * BASE
    mat = no_ting[:, None] & is_ting[None, :]          # [payer, receiver]
    amt = jnp.where(mat, pay[None, :], 0)
    score = st.score - amt.sum(axis=1) + amt.sum(axis=0)
    # 退税:未听者(不含已胡)退还本局收到的全部杠钱
    refund = jnp.where(no_ting[None, :], st.gang_ledger, 0)   # [payer, receiver]
    score = score - refund.sum(axis=0) + refund.sum(axis=1)
    return st.replace(score=score, phase=jnp.int32(PH_OVER), terminated=TRUE)


# ------------------------------------------------------------------ 轮转
def _next_alive(st: State, i):
    """i 之后的下一个在场者;全部离场返回 -1。"""
    def body(k, acc):
        j = (i + k) % NUM_PLAYERS
        return jnp.where((acc < 0) & (~st.finished[j]), j, acc)
    return jax.lax.fori_loop(1, NUM_PLAYERS + 1, body, jnp.int32(-1))


def _draw(st: State, i, from_gang):
    """i 摸一张进入 action。调用前须确保牌墙非空。

    ⚠️ JAX 对越界索引**静默截断**到末元素(不报错),所以这里显式钳位并由调用方
    保证 wall_ix < WALL_SIZE;差分测试另有断言兜底。"""
    ix = jnp.clip(st.wall_ix, 0, WALL_SIZE - 1)
    t = st.wall[WALL_SIZE - 1 - ix].astype(jnp.int32)
    return st.replace(hand=st.hand.at[i, t].add(1),
                      wall_ix=st.wall_ix + 1,
                      cur=jnp.int32(i),
                      last_draw_was_gang=from_gang,
                      hai_di=(WALL_SIZE - st.wall_ix - 1) == 0,
                      phase=jnp.int32(PH_ACTION))


def _begin_turn(st: State, i, from_gang=FALSE):
    """i 摸牌开始回合;3 家已胡→终局;墙空→流局结算。"""
    wall_empty = st.wall_ix >= WALL_SIZE
    return jax.lax.cond(
        (st.n_hu >= 3) | (i < 0),
        lambda s: s.replace(phase=jnp.int32(PH_OVER), terminated=TRUE),
        lambda s: jax.lax.cond(wall_empty, _liuju_settle,
                               lambda ss: _draw(ss, i, from_gang), s),
        st)


# ------------------------------------------------------------------ 合法动作
def _legal_mask(st: State) -> jnp.ndarray:
    """与 reference_impl.legal_actions 逐条对应。"""
    m = jnp.zeros(NUM_ACTIONS, dtype=bool)

    # --- 定缺 ---
    m_void = m.at[A_VOID:A_VOID + 3].set(True)

    # --- 行动阶段 ---
    i = st.cur
    hand = st.hand[i].astype(jnp.int32)
    nm = st.n_melds[i].astype(jnp.int32)
    vd = st.void[i].astype(jnp.int32)
    n_hand = jnp.sum(hand)
    post_draw = (n_hand % 3) == 2                      # 摸后 3k+2;鸣牌后 3k+1 只能打
    is_void_tile = SUIT == vd
    has_void = jnp.any(jnp.where(is_void_tile, hand, 0) > 0)
    # 缺门必打:手里还有缺门牌时,打牌掩码仅开放缺门张
    discard_ok = jnp.where(has_void, (hand > 0) & is_void_tile, hand > 0)
    m_act = m.at[A_DISCARD:A_DISCARD + NUM_TILES].set(discard_ok)
    # 暗杠 / 补杠(仅摸后、且不在"缺门必打"状态)
    has_peng = jax.vmap(lambda t: jnp.any(
        (st.melds_kind[i] == MK_PENG) & (st.melds_tile[i] == t)))(jnp.arange(NUM_TILES))
    ankan = (hand == 4) & (~is_void_tile)
    bugang = (hand == 1) & has_peng
    gang_ok = (ankan | bugang) & post_draw & (~has_void)
    m_act = m_act.at[A_GANG:A_GANG + NUM_TILES].set(gang_ok)
    zimo_ok = post_draw & (~has_void) & _is_hu(hand, nm, vd)
    m_act = m_act.at[A_HU].set(zimo_ok)

    # --- 响应阶段 ---
    j = st.resp_queue[jnp.clip(st.resp_ix, 0, 2)]
    j = jnp.clip(j, 0, NUM_PLAYERS - 1)
    dt = st.pending_tile
    jh = st.hand[j].astype(jnp.int32)
    jnm = st.n_melds[j].astype(jnp.int32)
    jvd = st.void[j].astype(jnp.int32)
    not_void = (dt // 9) != jvd
    ron_ok = not_void & _is_hu(jh.at[dt].add(1), jnm, jvd)
    peng_ok = not_void & (jh[dt] >= 2)
    zg_ok = not_void & (jh[dt] == 3)
    m_resp = m.at[A_PASS].set(True).at[A_HU].set(ron_ok) \
              .at[A_PENG].set(peng_ok).at[A_ZHIGANG].set(zg_ok)

    return jax.lax.switch(jnp.clip(st.phase, 0, 3),
                          [lambda: m_void, lambda: m_act, lambda: m_resp, lambda: m])


def _resp_order(st: State, dp):
    """打牌者 dp 之后的在场者顺序,定长 3,-1 填充(对齐 reference._resp_order)。"""
    def one(k):
        j = (dp + 1 + k) % NUM_PLAYERS
        return jnp.where((~st.finished[j]) & (j != dp), j, -1)
    q = jax.vmap(one)(jnp.arange(3))
    # 压紧:把 -1 挪到尾部,保持相对顺序
    valid = q >= 0
    order = jnp.argsort(jnp.where(valid, jnp.arange(3), 3 + jnp.arange(3)))
    return q[order]


# ------------------------------------------------------------------ 各动作
def _do_discard(st: State, t):
    i = st.cur
    st = st.replace(hand=st.hand.at[i, t].add(-1),
                    river=st.river.at[i, t].add(1),
                    pending_tile=jnp.int32(t), pending_from=i,
                    discard_was_gang=st.last_draw_was_gang,
                    ron_flag=jnp.zeros(NUM_PLAYERS, dtype=bool),
                    first_ron=jnp.int32(-1))
    q = _resp_order(st, i)
    st = st.replace(resp_queue=q, resp_ix=jnp.int32(0))
    return jax.lax.cond(q[0] >= 0,
                        lambda s: s.replace(phase=jnp.int32(PH_RESP)),
                        lambda s: _begin_turn(s, _next_alive(s, i)), st)


def _do_ankan(st: State, t):
    i = st.cur
    k = st.n_melds[i]
    st = st.replace(hand=st.hand.at[i, t].add(-4),
                    melds_kind=st.melds_kind.at[i, k].set(MK_GANG_AN),
                    melds_tile=st.melds_tile.at[i, k].set(t),
                    n_melds=st.n_melds.at[i].add(1))
    st = _pay_all_alive(st, i, 2 * BASE, is_gang=True)
    # 杠后本家补摸(墙空则流局)
    return jax.lax.cond(st.wall_ix >= WALL_SIZE, _liuju_settle,
                        lambda s: _draw(s, i, TRUE), st)


def _do_bugang(st: State, t):
    """补杠:先开抢杠窗口,任一在场者能荣该张则补杠不成立。"""
    i = st.cur

    def can_rob(j):
        jh = st.hand[j].astype(jnp.int32)
        return ((j != i) & (~st.finished[j]) & ((t // 9) != st.void[j])
                & _is_hu(jh.at[t].add(1), st.n_melds[j].astype(jnp.int32),
                         st.void[j].astype(jnp.int32)))

    robbers = jax.vmap(can_rob)(jnp.arange(NUM_PLAYERS))
    robbed = jnp.any(robbers)

    def do_rob(s: State):
        def settle_one(k, ss):
            return jax.lax.cond(
                robbers[k],
                lambda x: _settle_hu(x, k, x.hand[k].astype(jnp.int32).at[t].add(1),
                                     FALSE, i, qiang_gang=TRUE),
                lambda x: x, ss)
        s = jax.lax.fori_loop(0, NUM_PLAYERS, settle_one, s)
        # 该张离开杠者,记入其牌河(与 reference 一致)
        s = s.replace(hand=s.hand.at[i, t].add(-1), river=s.river.at[i, t].add(1))
        return _begin_turn(s, _next_alive(s, i))

    def do_gang(s: State):
        slot = jnp.argmax((s.melds_kind[i] == MK_PENG) & (s.melds_tile[i] == t))
        s = s.replace(hand=s.hand.at[i, t].add(-1),
                      melds_kind=s.melds_kind.at[i, slot].set(MK_GANG_BU))
        s = _pay_all_alive(s, i, 1 * BASE, is_gang=True)
        return jax.lax.cond(s.wall_ix >= WALL_SIZE, _liuju_settle,
                            lambda x: _draw(x, i, TRUE), s)

    return jax.lax.cond(robbed, do_rob, do_gang, st)


def _do_zimo(st: State):
    i = st.cur
    st = _settle_hu(st, i, st.hand[i].astype(jnp.int32), TRUE, i,
                    gang_hua=st.last_draw_was_gang, hai_di=st.hai_di)
    return _begin_turn(st, _next_alive(st, i))


# ------------------------------------------------------------------ 响应阶段
def _resp_advance(st: State) -> State:
    """队列推进一位;队列空时按 reference 的收尾逻辑结算。"""
    ix = st.resp_ix + 1
    st = st.replace(resp_ix=ix)
    more = (ix < 3) & (st.resp_queue[jnp.clip(ix, 0, 2)] >= 0)

    def finish(s: State):
        dp = s.pending_from
        dt = s.pending_tile
        any_ron = jnp.any(s.ron_flag)

        def with_ron(x: State):
            # 牌离开打牌者牌河、归**队列顺序**里首个胡家(守恒),与 reference 一致。
            # ⚠️ 不能用 argmax(ron_flag)——那是玩家编号最小者;队列从打牌者下家起算,
            # 一炮多响时两者不同(实测 seed 402578:队列 [3,0,1],3 与 0 同荣,牌应归 3)。
            first = jnp.clip(x.first_ron, 0, NUM_PLAYERS - 1)
            x = x.replace(river=x.river.at[dp, dt].add(-1),
                          hand=x.hand.at[first, dt].add(1))
            return jax.lax.cond(x.n_hu >= 3,
                                lambda y: y.replace(phase=jnp.int32(PH_OVER), terminated=TRUE),
                                lambda y: _begin_turn(y, _next_alive(y, dp)), x)

        return jax.lax.cond(any_ron, with_ron,
                            lambda x: _begin_turn(x, _next_alive(x, dp)), s)

    return jax.lax.cond(more, lambda s: s, finish, st)


def _do_ron(st: State) -> State:
    j = st.resp_queue[jnp.clip(st.resp_ix, 0, 2)]
    dt = st.pending_tile
    st = _settle_hu(st, j, st.hand[j].astype(jnp.int32).at[dt].add(1),
                    FALSE, st.pending_from, gang_pao=st.discard_was_gang)
    st = st.replace(ron_flag=st.ron_flag.at[j].set(True),
                    first_ron=jnp.where(st.first_ron < 0, j, st.first_ron))
    return _resp_advance(st)


def _do_peng(st: State) -> State:
    """已有荣和时碰不成立(牌被胡家拿走),仅跳过本家。"""
    j = st.resp_queue[jnp.clip(st.resp_ix, 0, 2)]
    dt, dp = st.pending_tile, st.pending_from

    def claim(s: State):
        k = s.n_melds[j]
        s = s.replace(hand=s.hand.at[j, dt].add(-2),
                      melds_kind=s.melds_kind.at[j, k].set(MK_PENG),
                      melds_tile=s.melds_tile.at[j, k].set(dt),
                      n_melds=s.n_melds.at[j].add(1),
                      river=s.river.at[dp, dt].add(-1),
                      cur=j, last_draw_was_gang=FALSE,
                      phase=jnp.int32(PH_ACTION),
                      resp_ix=jnp.int32(3))
        return s

    return jax.lax.cond(jnp.any(st.ron_flag), _resp_advance, claim, st)


def _do_zhigang(st: State) -> State:
    j = st.resp_queue[jnp.clip(st.resp_ix, 0, 2)]
    dt, dp = st.pending_tile, st.pending_from

    def claim(s: State):
        k = s.n_melds[j]
        s = s.replace(hand=s.hand.at[j, dt].add(-3),
                      melds_kind=s.melds_kind.at[j, k].set(MK_GANG_MING),
                      melds_tile=s.melds_tile.at[j, k].set(dt),
                      n_melds=s.n_melds.at[j].add(1),
                      river=s.river.at[dp, dt].add(-1),
                      resp_ix=jnp.int32(3))
        s = _pay(s, dp, j, 2 * BASE, is_gang=True)
        return jax.lax.cond(s.wall_ix >= WALL_SIZE, _liuju_settle,
                            lambda x: _draw(x, j, TRUE), s)

    return jax.lax.cond(jnp.any(st.ron_flag), _resp_advance, claim, st)


# ------------------------------------------------------------------ Env
def _init_core(key) -> State:
    wall = jax.random.permutation(key, jnp.repeat(jnp.arange(NUM_TILES, dtype=jnp.int8), 4))
    return _init_from_wall(wall)


def _init_from_wall(wall) -> State:
    """注入指定牌墙(差分测试用:与 reference_impl 打同一副牌)。摸牌一律从尾部取,
    与 reference 的 wall.pop() 对齐。"""
    hand = jnp.zeros((NUM_PLAYERS, NUM_TILES), jnp.int8)
    # 各发 13 张(从尾部摸,对齐 reference 的 wall.pop())
    def deal(carry, k):
        h, ix = carry
        t = wall[WALL_SIZE - 1 - ix].astype(jnp.int32)
        p = (k // 13).astype(jnp.int32)
        return (h.at[p, t].add(1), ix + 1), None
    (hand, wall_ix), _ = jax.lax.scan(deal, (hand, jnp.int32(0)), jnp.arange(52))

    st = State(
        hand=hand,
        melds_kind=jnp.zeros((NUM_PLAYERS, MAX_MELDS), jnp.int8),
        melds_tile=jnp.full((NUM_PLAYERS, MAX_MELDS), -1, jnp.int8),
        n_melds=jnp.zeros(NUM_PLAYERS, jnp.int8),
        river=jnp.zeros((NUM_PLAYERS, NUM_TILES), jnp.int8),
        wall=wall, wall_ix=wall_ix,
        void=jnp.full(NUM_PLAYERS, -1, jnp.int8),
        finished=jnp.zeros(NUM_PLAYERS, dtype=bool),
        score=jnp.zeros(NUM_PLAYERS, jnp.int32),
        hu_fan=jnp.zeros(NUM_PLAYERS, jnp.int8),
        gang_ledger=jnp.zeros((NUM_PLAYERS, NUM_PLAYERS), jnp.int32),
        phase=jnp.int32(PH_VOID), cur=jnp.int32(0), void_declared=jnp.int32(0),
        pending_tile=jnp.int32(-1), pending_from=jnp.int32(-1),
        resp_queue=jnp.full(3, -1, jnp.int32), resp_ix=jnp.int32(0),
        ron_flag=jnp.zeros(NUM_PLAYERS, dtype=bool), first_ron=jnp.int32(-1),
        last_draw_was_gang=FALSE, discard_was_gang=FALSE, hai_di=FALSE,
        n_hu=jnp.int32(0),
        legal_action_mask=jnp.zeros(NUM_ACTIONS, dtype=bool),
        terminated=FALSE, truncated=FALSE,
        rewards=jnp.zeros(NUM_PLAYERS, jnp.float32),
        step_count=jnp.int32(0), current_player=jnp.int32(0),
    )
    return _refresh(st)


def _refresh(st: State) -> State:
    """更新 current_player 与 legal_action_mask(每次 step 末尾调用)。"""
    cp = jax.lax.switch(jnp.clip(st.phase, 0, 3),
                        [lambda: st.void_declared,
                         lambda: st.cur,
                         lambda: jnp.clip(st.resp_queue[jnp.clip(st.resp_ix, 0, 2)], 0, 3),
                         lambda: st.cur])
    return st.replace(current_player=cp, legal_action_mask=_legal_mask(st))


def _step_core(st: State, action) -> State:
    prev_score = st.score
    a = jnp.int32(action)

    def f_void(s: State):
        i = s.void_declared
        s = s.replace(void=s.void.at[i].set((a - A_VOID).astype(jnp.int8)),
                      void_declared=i + 1)
        return jax.lax.cond(s.void_declared >= NUM_PLAYERS,
                            lambda x: _begin_turn(x, jnp.int32(0)), lambda x: x, s)

    def f_action(s: State):
        t_dis = a - A_DISCARD
        t_gang = a - A_GANG
        is_dis = a < A_GANG
        is_gang = (a >= A_GANG) & (a < A_PENG)
        return jax.lax.cond(
            is_dis, lambda x: _do_discard(x, t_dis),
            lambda x: jax.lax.cond(
                is_gang,
                lambda y: jax.lax.cond(y.hand[y.cur, t_gang] == 4,
                                       lambda z: _do_ankan(z, t_gang),
                                       lambda z: _do_bugang(z, t_gang), y),
                _do_zimo, x), s)

    def f_resp(s: State):
        return jax.lax.switch(
            jnp.where(a == A_HU, 0, jnp.where(a == A_PENG, 1,
                      jnp.where(a == A_ZHIGANG, 2, 3))),
            [_do_ron, _do_peng, _do_zhigang, _resp_advance], s)

    st = jax.lax.switch(jnp.clip(st.phase, 0, 3),
                        [f_void, f_action, f_resp, lambda s: s], st)
    st = st.replace(rewards=(st.score - prev_score).astype(jnp.float32),
                    step_count=st.step_count + 1)
    return _refresh(st)


class SichuanEnv:
    """实现 mahjax 的 Env 协议,现有训练器可直接使用。"""

    num_players = NUM_PLAYERS

    def init(self, key):
        return _init_core(key)

    def step(self, state, action, key=None):
        return _step_core(state, action)


def make(*_args, **_kwargs):
    return SichuanEnv()
