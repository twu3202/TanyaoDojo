"""
R2 轻量观测:Mortal 式结构化特征,直接从 red_mahjong State 读取。

设计动机(实测定量依据见 MAHJAX_MIGRATION.md):
  上游 dict obs 把牌河/副露/立直等信息压在 (3,200) 动作流水账里,网络被迫用
  200-token 注意力重建牌局状态(update 阶段占吞吐 ~92%)。而 State 里这些
  信息本就是结构化数组——直接拼 (34, C) 特征平面 + 标量向量,注意力整段免除。

输出(全 jit/vmap 友好,固定形状):
  planes : (34, 20) float32 —— 34 牌型列 × 20 通道
  scalars: (26,)   float32 —— 全局标量
通道表:
   0-3  自家手牌计数 one-hot(≥1..≥4,34 型)
   4    自家赤五持有(标在 4/13/22 列)
   5    刚摸的牌 one-hot
   6-9  四家牌河计数/4(相对座位序:自/右/对/左)
  10-13 四家最后一张舍牌 one-hot
  14-17 四家副露暴露计数/4
  18    全可见计数/4(河+副露+指示牌+自手)
  19    宝牌指示牌计数/4
标量表:
   0-3  四家立直(相对序)  4-7 分数/500(相对序,中心 250)
   8    向听/6   9 振听   10 局序/12   11 本场/10   12 供托/10
  13-16 场风 one-hot   17-20 自风 one-hot
  21    王牌已翻杠宝数/4   22 赤五已见数/3
  23    自家副露数/4   24 海底旗   25 剩余活牌墙/70
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from mahjax.red_mahjong.env import State
from mahjax.red_mahjong.tile import Tile, River
from mahjax.red_mahjong.meld import Meld

NUM_PLANES = 20
NUM_SCALARS = 26

# ⚠️ 数据源勘误(2026-07-25,差分核查发现):env 从不写 state.players.discards 与
# meld_tiles(只在 _replace_state 路由表里挂名)。真实牌河在 players.river(uint16 编码,
# 经 River.decode_tile 解),真实副露在 players.melds(uint16 编码,经 Meld.* 解)。
# 初版误用死数组导致牌河/副露平面恒零——现已全部改从活数组解码。


def _count_by_type(tiles: jnp.ndarray) -> jnp.ndarray:
    """(N,) tile id(0-36,-1=空)→ (34,) 计数(赤并入普通五)。"""
    valid = tiles >= 0
    t34 = Tile.to_tile_type(jnp.where(valid, tiles, 0))
    oh = jax.nn.one_hot(t34, 34, dtype=jnp.float32) * valid[:, None]
    return oh.sum(axis=0)


def _one_hot_tile(tile: jnp.ndarray) -> jnp.ndarray:
    """标量 tile(0-36,负=无)→ (34,) one-hot。"""
    ok = tile >= 0
    t34 = Tile.to_tile_type(jnp.where(ok, tile, 0))
    return jax.nn.one_hot(t34, 34, dtype=jnp.float32) * ok


def _meld_counts34(melds_row: jnp.ndarray) -> jnp.ndarray:
    """(4,) uint16 编码副露 → (34,) 暴露牌计数(碰3/杠4/吃三连各1)。"""

    def one(m):
        act = Meld.action(m)
        tt = Meld.target(m)
        valid = tt >= 0
        is_chi = Meld.is_chi(m)
        is_pon = Meld.is_pon(m)
        is_kan = Meld.is_kan(m)
        n = is_pon.astype(jnp.float32) * 3.0 + is_kan.astype(jnp.float32) * 4.0
        cnt = jnp.zeros(34, jnp.float32)
        cnt = cnt.at[jnp.clip(tt, 0, 33)].add(n * valid)
        chi_base = tt - Meld._chi_index(act)
        for k in range(3):
            cnt = cnt.at[jnp.clip(chi_base + k, 0, 33)].add(
                is_chi.astype(jnp.float32) * valid
            )
        return cnt

    return jax.vmap(one)(melds_row).sum(axis=0)


def _called_correction34(melds_all: jnp.ndarray) -> jnp.ndarray:
    """(4,4) 全桌副露 → (34,) 鸣走牌校正:被鸣的那张同时留在牌河(灰)与副露中,
    可见计数会双计 → 每个非暗杠副露在 target 型上减 1。"""
    m = melds_all.reshape(-1)
    tt = jax.vmap(Meld.target)(m)
    is_closed = jax.vmap(Meld.is_closed_kan)(m)
    valid = (tt >= 0) & (~is_closed)
    oh = jax.nn.one_hot(jnp.clip(tt, 0, 33), 34, dtype=jnp.float32) * valid[:, None]
    return oh.sum(axis=0)


def observe_lean(state: State) -> dict:
    c_p = state.current_player
    rel = (jnp.arange(4) + c_p) % 4  # 相对座位:自/右/对/左

    hand34 = state.players.hand[c_p].astype(jnp.float32)  # (34,)
    hand_oh = jnp.stack([(hand34 >= k).astype(jnp.float32) for k in (1, 2, 3, 4)])  # (4,34)
    red_flags34 = jnp.zeros(34, dtype=jnp.float32)
    for red, black in ((34, 4), (35, 13), (36, 22)):
        red_flags34 = red_flags34.at[black].set(
            state.players.hand_with_red[c_p, red].astype(jnp.float32)
        )
    # 抢杠窗口守卫:kan_declared 时 last_draw 是杠家的暗牌(env 不重置),对被服务的
    # 响应者是信息泄漏且 mjai 侧不可复现 → 置 -1。
    drawn_visible = jnp.where(
        state.round_state.kan_declared, jnp.int32(-1), state.round_state.last_draw.astype(jnp.int32)
    )
    drawn = _one_hot_tile(drawn_visible)

    river_tiles = River.decode_tile(state.players.river[rel])  # (4,24) 0-36/-1,含被鸣走的(灰)
    river_cnt = jax.vmap(_count_by_type)(river_tiles)  # (4,34)
    dc = state.players.discard_counts[rel].astype(jnp.int32)  # (4,)
    last_ix = jnp.maximum(dc - 1, 0)
    last_tiles = River.decode_tile(state.players.river[rel, last_ix])
    last_tiles = jnp.where(dc > 0, last_tiles, -1)
    last_oh = jax.vmap(_one_hot_tile)(last_tiles)  # (4,34)

    meld_cnt = jax.vmap(_meld_counts34)(state.players.melds[rel])  # (4,34)

    dora_ind_cnt = _count_by_type(state.round_state.dora_indicators.astype(jnp.int32))
    visible = (
        hand34
        + river_cnt.sum(axis=0)
        + meld_cnt.sum(axis=0)
        - _called_correction34(state.players.melds)
        + dora_ind_cnt
    )

    planes = jnp.concatenate(
        [
            hand_oh,
            red_flags34[None],
            drawn[None],
            river_cnt / 4.0,
            last_oh,
            meld_cnt / 4.0,
            (visible / 4.0)[None],
            (dora_ind_cnt / 4.0)[None],
        ],
        axis=0,
    ).T  # (34, 20)

    riichi = state.players.riichi[rel].astype(jnp.float32)
    scores = state.round_state.score[rel].astype(jnp.float32) / 500.0
    shanten = state.round_state.shanten_current_player.astype(jnp.float32) / 6.0
    furiten = (
        state.players.furiten_by_discard[c_p] | state.players.furiten_by_pass[c_p]
    ).astype(jnp.float32)
    rnd = state.round_state.round.astype(jnp.float32)
    prevalent = jax.nn.one_hot(state.round_state.round // 4, 4, dtype=jnp.float32)
    seat = jax.nn.one_hot(state.round_state.seat_wind[c_p], 4, dtype=jnp.float32)
    m_flat = state.players.melds.reshape(-1)
    m_red = jax.vmap(Meld.contains_red)(m_flat)
    m_suit = jax.vmap(Meld.target)(m_flat) // 9  # 空副露 target=-1 → -1,不会命中 0-2
    red_flags = []
    for s, r in enumerate((34, 35, 36)):
        red_flags.append(
            (
                jnp.any(river_tiles == r)
                | jnp.any(state.round_state.dora_indicators == r)
                | jnp.any(m_red & (m_suit == s))
            ).astype(jnp.float32)
        )
    reds_seen = jnp.stack(red_flags).sum()
    live_left = (
        state.round_state.next_deck_ix.astype(jnp.float32)
        - state.round_state.last_deck_ix.astype(jnp.float32)
        + 1.0
    )
    scalars = jnp.concatenate(
        [
            riichi,
            scores,
            jnp.stack(
                [
                    shanten,
                    furiten,
                    rnd / 12.0,
                    state.round_state.honba.astype(jnp.float32) / 10.0,
                    state.round_state.kyotaku.astype(jnp.float32) / 10.0,
                ]
            ),
            prevalent,
            seat,
            jnp.stack(
                [
                    state.round_state.n_kan_doras.astype(jnp.float32) / 4.0,
                    reds_seen / 3.0,
                    state.players.meld_counts[c_p].astype(jnp.float32) / 4.0,
                    state.round_state.is_haitei.astype(jnp.float32),
                    live_left / 70.0,
                ]
            ),
        ]
    )  # (26,)
    return {"planes": planes, "scalars": scalars}
