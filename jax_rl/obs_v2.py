"""
obs v2:在 obs_lean(34×20 + 26)之上追加 16 平面 + 6 标量——BC 末级火箭。

设计原则:每个新特征必须在 env(State 数组)与评测桥(mjai 事件流 tracker)
两侧都能精确构造(见 obs_lean 的勘误传统);时序/摸切/立直位置/现物是
lean 完全缺失、而强对手明显在用的信息(危险度估计的原料)。

输出:planes (34, 36) = lean 20 + v2 16;scalars (32,) = lean 26 + v2 6
v2 平面表(接在 lean 20 之后;相对座位序 自/右/对/左):
  20-23 四家牌河时序:每型最后一次舍出的位置 (idx+1)/24,未舍=0
  24-27 四家手切(非摸切)计数/4(摸切数=河计数−手切,网络可自行差分)
  28-31 四家立直宣言牌 one-hot(未立直全零)
  32-34 三家对手河现物 binary(该家河中出现过的牌型;立直家的绝对安牌子集)
  35    表宝牌实体计数/4(指示牌经 wrap 映射,含杠宝)
v2 标量表(接在 lean 26 之后):
  26-28 三家对手立直宣言时点(宣言牌河内 idx+1)/18,未立直=0
  29    自家巡目 discard_count/18
  30    全场总舍牌/70
  31    自家手切率(手切/舍牌,无舍牌=0)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from mahjax.red_mahjong.env import State
from mahjax.red_mahjong.tile import Tile, River

from obs_lean import observe_lean, _count_by_type

NUM_V2_PLANES = 36
NUM_V2_SCALARS = 32

# 指示牌 → 实宝牌 的 34 型映射(数牌 9→1 wrap;东南西北循环;白发中循环)
_DORA_NEXT = []
for _t in range(34):
    if _t < 27:
        _DORA_NEXT.append(_t - 8 if _t % 9 == 8 else _t + 1)
    elif _t < 31:
        _DORA_NEXT.append(27 + (_t - 27 + 1) % 4)
    else:
        _DORA_NEXT.append(31 + (_t - 31 + 1) % 3)
DORA_NEXT = jnp.array(_DORA_NEXT, dtype=jnp.int32)


def observe_v2(state: State) -> dict:
    legal = observe_lean(state)
    c_p = state.current_player
    rel = (jnp.arange(4) + c_p) % 4

    dec = jax.vmap(River.decode_river)(state.players.river[rel])  # (4,6,24)
    tile, riichi_f, _gray, tsumog, _src, _mt = (dec[:, k] for k in range(6))
    valid = tile >= 0
    t34 = Tile.to_tile_type(jnp.where(valid, tile, 0))
    oh = jax.nn.one_hot(t34, 34, dtype=jnp.float32) * valid[..., None]  # (4,24,34)
    pos = (jnp.arange(24, dtype=jnp.float32) + 1.0)[None, :, None]

    time_last = (oh * pos).max(axis=1) / 24.0                       # (4,34)
    tedashi = (oh * ((tsumog == 0) & valid)[..., None]).sum(1) / 4.0  # (4,34)
    riichi_tile = jnp.clip((oh * (riichi_f == 1)[..., None]).sum(1), 0.0, 1.0)  # (4,34)
    opp_seen = jnp.clip(oh.sum(1), 0.0, 1.0)[1:]                    # (3,34) 河现物

    ind_cnt = _count_by_type(state.round_state.dora_indicators.astype(jnp.int32))
    dora_real = jnp.zeros(34, jnp.float32).at[DORA_NEXT].add(ind_cnt)

    planes = jnp.concatenate(
        [
            legal["planes"],
            time_last.T,
            tedashi.T,
            riichi_tile.T,
            opp_seen.T,
            (dora_real / 4.0)[:, None],
        ],
        axis=1,
    )  # (34, 36)

    p24 = (jnp.arange(24, dtype=jnp.float32) + 1.0)[None, :]        # (1,24)
    riichi_pos = jnp.where(riichi_f == 1, p24, 0.0).max(axis=1)     # (4,)
    dc = state.players.discard_counts[rel].astype(jnp.float32)
    n_ted = tedashi.sum(axis=1) * 4.0
    scalars = jnp.concatenate(
        [
            legal["scalars"],
            riichi_pos[1:] / 18.0,
            jnp.stack(
                [
                    dc[0] / 18.0,
                    dc.sum() / 70.0,
                    jnp.where(dc[0] > 0, n_ted[0] / jnp.maximum(dc[0], 1.0), 0.0),
                ]
            ),
        ]
    )  # (32,)
    return {"planes": planes, "scalars": scalars}
