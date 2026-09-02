"""
川麻观测(P2)。planes (27, 22) + scalars (26),比立直线的 (34,20)+(26) 更小。

要点(方案 §5.3):
  · **定缺是公开信息**,四家各占一个平面——这是川麻独有的强信号(打某家缺门花色的牌
    对那家 100% 安全,而立直的现物/筋只是概率性安全);
  · 向听查表免费(O(1) gather),直接进标量;
  · **finished 四家标志进标量**——血战里"还剩几家在场"直接改变期望;
  · 相对座位序(自/下家/对家/上家),避免 agent 只学会 0 号位的打法。

平面表(27 列 = 万筒条各 9):
   0-3  自家手牌计数 one-hot(≥1..≥4)
   4    刚摸的牌 one-hot
   5-8  四家牌河计数/4(相对序)
   9-12 四家副露暴露计数/4(碰 3、杠 4)
  13-16 四家定缺花色(整门置 1)—— 公开信息
  17-20 四家"已胡离场"标志(整平面)
  21    全场可见计数/4(自手+四河+四副露)
标量表:
   0-3  四家分数/16(相对序)      4-7  四家副露数/4
   8    自家向听/8               9    自家是否听牌
  10    牌墙剩余/56              11   已胡人数/3
  12-14 自家定缺 one-hot         15   自家手牌张数/14
  16-18 三家对手是否已胡          19   自家是否已胡
  20    是否刚杠过(杠上花窗口)    21   海底
  22-25 四家手牌张数/14(相对序)
"""
from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.suit_table import load_table, make_jax_ops, NUM_TILES
from sichuan.env_jax import State, NUM_PLAYERS, WALL_SIZE, MK_GANG_MING, MAX_MELDS

NUM_PLANES = 22
NUM_SCALARS = 26
_TAB = load_table()
_agari_fn, _shanten_fn = make_jax_ops(_TAB)
_SUIT = jnp.arange(NUM_TILES) // 9


def _meld_counts(melds_kind, melds_tile, n_melds):
    """(4,) 副露 → (27,) 暴露张数(碰 3 / 杠 4)。"""
    slot = jnp.arange(MAX_MELDS)
    active = slot < n_melds
    n = jnp.where(melds_kind >= MK_GANG_MING, 4.0, 3.0) * active
    idx = jnp.clip(melds_tile, 0, NUM_TILES - 1)
    return jnp.zeros(NUM_TILES, jnp.float32).at[idx].add(jnp.where(active, n, 0.0))


def observe(state: State) -> dict:
    cp = state.current_player
    rel = (jnp.arange(NUM_PLAYERS) + cp) % NUM_PLAYERS      # 自/下/对/上

    hand = state.hand[cp].astype(jnp.float32)
    planes = [ (hand >= k).astype(jnp.float32) for k in (1, 2, 3, 4) ]

    # 刚摸的牌:摸后手牌为 3k+2,用"最后一次摸的牌"近似不可得,故用牌墙指针回溯
    last = state.wall[jnp.clip(WALL_SIZE - state.wall_ix, 0, WALL_SIZE - 1)].astype(jnp.int32)
    drawn = jax.nn.one_hot(last, NUM_TILES, dtype=jnp.float32) * ((jnp.sum(hand) % 3) == 2)
    planes.append(drawn)

    river = state.river[rel].astype(jnp.float32)
    planes += [river[i] / 4.0 for i in range(NUM_PLAYERS)]

    melds = jax.vmap(_meld_counts)(state.melds_kind[rel], state.melds_tile[rel],
                                   state.n_melds[rel].astype(jnp.int32))
    planes += [melds[i] / 4.0 for i in range(NUM_PLAYERS)]

    # 定缺(公开):整门置 1;未定缺时全 0
    void = state.void[rel].astype(jnp.int32)
    planes += [((_SUIT == void[i]) & (void[i] >= 0)).astype(jnp.float32)
               for i in range(NUM_PLAYERS)]

    fin = state.finished[rel]
    planes += [jnp.full(NUM_TILES, fin[i].astype(jnp.float32)) for i in range(NUM_PLAYERS)]

    visible = hand + river.sum(0) + melds.sum(0)
    planes.append(visible / 4.0)

    P = jnp.stack(planes, axis=0).T                          # (27, 22)

    nm = state.n_melds[cp].astype(jnp.int32)
    vd = state.void[cp].astype(jnp.int32)
    shan = _shanten_fn(state.hand[cp].astype(jnp.int32), nm, vd).astype(jnp.float32)
    n_hand = jnp.sum(hand)
    S = jnp.concatenate([
        state.score[rel].astype(jnp.float32) / 16.0,
        state.n_melds[rel].astype(jnp.float32) / 4.0,
        jnp.stack([
            shan / 8.0,
            (shan <= 0).astype(jnp.float32),
            (WALL_SIZE - state.wall_ix).astype(jnp.float32) / 56.0,
            state.n_hu.astype(jnp.float32) / 3.0,
        ]),
        jax.nn.one_hot(jnp.clip(vd, 0, 2), 3, dtype=jnp.float32) * (vd >= 0),
        jnp.stack([n_hand / 14.0]),
        fin[1:].astype(jnp.float32),
        jnp.stack([
            fin[0].astype(jnp.float32),
            state.last_draw_was_gang.astype(jnp.float32),
            state.hai_di.astype(jnp.float32),
        ]),
        jnp.sum(state.hand[rel], axis=1).astype(jnp.float32) / 14.0,
    ])                                                        # (26,)
    return {"planes": P, "scalars": S}
