"""川麻弃张效率头寸:策略实选的弃张,离"向听/受入最优"还差多少。

**动机**。P3 观测(打出每张后的向听)把早期学习加快了(5033 万处 +0.240,4/4 seed,
p=0.023),但 1.51 亿处的平台没动。平台由什么决定?第一个该排除的候选是
"效率已经到顶,再喂效率信息也没用"。立直线上同一个问题是用 `jax_rl/ukeire_headroom.py`
回答的——那边实测向听最优 96.4%、受入亏欠 3.85%,**没有头寸**,于是"接单人求解器
特征"那条主张被直接排除。本脚本是川麻侧的同一把尺子。

**做法**。策略自博弈跑出打牌决策点;对每个合法弃张 t 精确算
    s'(t)  = shanten(手牌 − t)
    uk'(t) = Σ_u (4 − 己见 u) · 1[shanten(手牌 − t + u) = s'(t) − 1]
再比"策略实选"与"最优"的差。向听是 O(1) 查表,27×27 次查一把 vmap 完。

**口径**:
  · 己见 = 自家手牌 + 四家牌河 + 四家副露(比立直线那把尺子更全,那边只扣自家手牌);
  · **只统计有选择的节点**:合法弃张 ≥2 张。川麻"缺门必打"会把可选张压到一张,
    那种节点没有效率可言,计入会凭空抬高最优率;
  · 受入亏欠只在"实选已是向听最优"的节点上算,否则两种错误会混在一起;
  · 贪心取动作(argmax),与 eval_h2h / eval_net 一致。

用法: python -m sichuan.discard_headroom <a.pkl> [b.pkl ...] [--num-envs 256] [--num-steps 200]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.suit_table import load_table, make_jax_ops, NUM_TILES
from sichuan import env_jax as E
from sichuan.net import SichuanACNet
from sichuan.obs import observe, NUM_PLANES, NUM_PLANES_DISC, _meld_counts

_, _shanten = make_jax_ops(load_table())
EYE = jnp.eye(NUM_TILES, dtype=jnp.int32)
NEG = -1e9


def infer_arch(params):
    d = params["params"] if "params" in params else params
    ch = int(d["Conv_0"]["kernel"].shape[-1])
    nb = len([k for k in d if k.startswith("ResBlock1D_")])
    n_emb = int(d["Dense_0"]["kernel"].shape[-1])
    n_planes = int(d["Conv_0"]["kernel"].shape[-2]) - n_emb
    return ch, nb, n_planes == NUM_PLANES_DISC


def _metrics_one(hand, nm, vd, seen, legal_t, chosen):
    """单个决策点。返回 (有选择?, 向听最优?, 受入亏欠, 算亏欠?)。"""
    after = jnp.maximum(hand[None, :] - EYE, 0)                  # (27,27)
    st_after = jax.vmap(lambda h: _shanten(h, nm, vd))(after)    # (27,)
    live = jnp.maximum(4 - seen, 0)

    def uk_of(row, s0):
        plus = row[None, :] + EYE
        st_plus = jax.vmap(lambda h: _shanten(h, nm, vd))(plus)
        return jnp.sum(jnp.where(st_plus < s0, live, 0))
    uk = jax.vmap(uk_of)(after, st_after)                        # (27,)

    BIG = jnp.int32(99)
    st_legal = jnp.where(legal_t, st_after, BIG)
    best_st = jnp.min(st_legal)
    has_choice = jnp.sum(legal_t) >= 2
    is_opt = st_after[chosen] == best_st

    # 受入亏欠:只在"向听最优"的那批候选里比
    opt_set = legal_t & (st_after == best_st)
    best_uk = jnp.max(jnp.where(opt_set, uk, -1))
    short = jnp.where(best_uk > 0,
                      (best_uk - uk[chosen]) / jnp.maximum(best_uk, 1), 0.0)
    count_short = has_choice & is_opt & (best_uk > 0) & (jnp.sum(opt_set) >= 2)
    return has_choice, has_choice & is_opt, short, count_short


def run(params, ch, nb, disc, n_env, n_step, seed=0):
    """params=None → 均匀随机合法动作。这是**阳性对照**:随机策略必须明显更差,
    否则说明这把尺子没在量东西(见 [[eval-scale-saturation]] 的第 2 条纪律)。"""
    if params is not None:
        net = SichuanACNet(channels=ch, blocks=nb)
        obs_fn = lambda s: observe(s, disc)

    @jax.jit
    def step(st, key):
        mask = jax.vmap(E._legal_mask)(st)
        if params is None:
            a = jax.random.categorical(key, jnp.where(mask, 0.0, NEG), axis=-1)
        else:
            logits, _ = net.apply(params, jax.vmap(obs_fn)(st))
            a = jnp.argmax(jnp.where(mask, logits, NEG), axis=-1)

        i = st.cur
        hand = jnp.take_along_axis(st.hand, i[:, None, None], 1)[:, 0].astype(jnp.int32)
        nm = jnp.take_along_axis(st.n_melds, i[:, None], 1)[:, 0].astype(jnp.int32)
        vd = jnp.take_along_axis(st.void, i[:, None], 1)[:, 0].astype(jnp.int32)
        melds = jax.vmap(jax.vmap(_meld_counts))(
            st.melds_kind, st.melds_tile, st.n_melds.astype(jnp.int32)).sum(1)
        seen = (hand + st.river.sum(1).astype(jnp.int32)
                + melds.astype(jnp.int32))       # 己见 = 自家手牌 + 四家河 + 四家副露
        legal_t = mask[:, E.A_DISCARD:E.A_DISCARD + NUM_TILES]
        is_dis = a < NUM_TILES
        chosen = jnp.clip(a, 0, NUM_TILES - 1)
        hc, opt, short, cs = jax.vmap(_metrics_one)(hand, nm, vd, seen, legal_t, chosen)
        hc = hc & is_dis
        opt = opt & is_dis
        cs = cs & is_dis
        nxt = jax.vmap(E._step_core)(st, a)
        return nxt, (hc.sum(), opt.sum(), jnp.where(cs, short, 0.0).sum(), cs.sum())

    walls = jnp.stack([jax.random.permutation(jax.random.PRNGKey(seed * 100000 + s),
                                              jnp.arange(E.WALL_SIZE) // 4)
                       for s in range(n_env)])
    st = jax.vmap(E._init_from_wall)(walls)
    key = jax.random.PRNGKey(seed)
    tot = np.zeros(4, np.float64)
    for _ in range(n_step):
        key, sub = jax.random.split(key)
        st, out = step(st, sub)
        tot += np.array([float(x) for x in out])
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params", nargs="+")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--num-steps", type=int, default=200)
    args = ap.parse_args()

    print(f"  {'快照':<24}{'决策点':>10}{'向听最优':>10}{'受入亏欠':>10}")
    print("  " + "-" * 56)
    for p in args.params:
        if p == "random":                      # 阳性对照
            n_hc, n_opt, s_short, n_short = run(None, 0, 0, False,
                                                args.num_envs, args.num_steps)
            print(f"  {'random(阳性对照)':<24}{int(n_hc):>10,}{n_opt / n_hc:>10.2%}"
                  f"{(s_short / max(n_short, 1)):>10.2%}")
            continue
        with open(p, "rb") as f:
            prm = pickle.load(f)
        ch, nb, disc = infer_arch(prm)
        n_hc, n_opt, s_short, n_short = run(prm, ch, nb, disc,
                                            args.num_envs, args.num_steps)
        name = Path(p).name
        if n_hc == 0:
            print(f"  {name:<24}{'0':>10}   (没有可选的打牌节点)")
            continue
        print(f"  {name:<24}{int(n_hc):>10,}{n_opt / n_hc:>10.2%}"
              f"{(s_short / max(n_short, 1)):>10.2%}"
              f"   [{ch}x{nb} obs={'P3' if disc else 'P2'}]")


if __name__ == "__main__":
    main()
