"""
Oracle 观测(仅供 critic 训练期使用,eval/actor 绝不接触):
在 obs_lean 的合法 20 平面 / 26 标量之上,追加隐藏信息通道。

动机(MAHJAX_MIGRATION.md 负结果三连):自家族 league 在 10-30 亿步内对外部
对手无增益。Suphx 式非对称 AC——critic 看全信息给出低方差值估计,actor 仍只看
合法信息——是 RL 重设计头牌。本模块只负责"critic 看什么"。

输出(jit/vmap 友好,固定形状):
  planes : (34, 37) = 合法 20 + oracle 17
  scalars: (29,)    = 合法 26 + oracle 3
oracle 平面表(索引接在合法 20 之后):
  20-31 三家对手手牌计数 one-hot(≥1..≥4;相对座位序:右/对/左)
  32-34 三家对手赤五持有(标在 4/13/22 列;右/对/左)
  35    未摸牌墙计数/4(活牌墙 [last_deck_ix, next_deck_ix] + 未用岭上)
  36    里宝指示牌计数/4
oracle 标量表(索引接在合法 26 之后):
  26-28 三家对手向听/6(右/对/左)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from mahjax.red_mahjong.env import State
from mahjax.red_mahjong.shanten import Shanten

from obs_lean import observe_lean, _count_by_type

NUM_ORACLE_PLANES = 37       # lean 底座
NUM_ORACLE_SCALARS = 29
NUM_ORACLE_V2_PLANES = 53    # v2 底座
NUM_ORACLE_V2_SCALARS = 35


def oracle_extras(state: State):
    """(34,17) 特权平面 + (3,) 特权标量。与合法底座(lean / v2)解耦,两者共用。"""
    c_p = state.current_player
    opp = (jnp.arange(1, 4) + c_p) % 4  # 相对座位:右/对/左

    opp_hand = state.players.hand[opp].astype(jnp.float32)  # (3,34)
    opp_oh = jnp.stack(
        [(opp_hand >= k).astype(jnp.float32) for k in (1, 2, 3, 4)], axis=1
    ).reshape(12, 34)  # (3,4,34)->(12,34) 每家连续 4 通道

    opp_red = jnp.zeros((3, 34), dtype=jnp.float32)
    for red, black in ((34, 4), (35, 13), (36, 22)):
        opp_red = opp_red.at[:, black].set(
            state.players.hand_with_red[opp, red].astype(jnp.float32)
        )

    # 未摸牌:活牌墙(自摸序从 next_deck_ix 递减至 last_deck_ix)+ 未用岭上
    idx = jnp.arange(136)
    live = (idx >= state.round_state.last_deck_ix) & (idx <= state.round_state.next_deck_ix)
    rinshan = (idx >= 10 + state.players.n_kan.sum()) & (idx <= 13)
    deck = state.round_state.deck.astype(jnp.int32)
    wall_cnt = _count_by_type(jnp.where(live | rinshan, deck, -1))

    ura_cnt = _count_by_type(state.round_state.ura_dora_indicators.astype(jnp.int32))

    extras = jnp.concatenate(
        [
            opp_oh.T,  # (34,12)
            opp_red.T,  # (34,3)
            (wall_cnt / 4.0)[:, None],
            (ura_cnt / 4.0)[:, None],
        ],
        axis=1,
    )  # (34, 17)
    opp_shanten = jax.vmap(Shanten.number)(state.players.hand[opp]).astype(jnp.float32)
    return extras, opp_shanten / 6.0


def observe_oracle(state: State) -> dict:
    """lean 底座(20+26)+ 特权 → (34,37) / (29,)。只喂 critic,永不进 actor/评测。"""
    legal = observe_lean(state)
    ex_p, ex_s = oracle_extras(state)
    return {"planes": jnp.concatenate([legal["planes"], ex_p], axis=1),
            "scalars": jnp.concatenate([legal["scalars"], ex_s])}


def observe_oracle_v2(state: State) -> dict:
    """v2 底座(36+32)+ 特权 → (34,53) / (35,)。配 obs v2 系基座(如 g186)。"""
    from obs_v2 import observe_v2
    legal = observe_v2(state)
    ex_p, ex_s = oracle_extras(state)
    return {"planes": jnp.concatenate([legal["planes"], ex_p], axis=1),
            "scalars": jnp.concatenate([legal["scalars"], ex_s])}
