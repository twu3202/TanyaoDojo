"""
川麻 P2 外评:把训练好的网络放进复式竞技场,对规则梯子测强度。

桥接方式(和立直线的 mjai_bot/jax_engine.py 同一思路,但更省事):
规则机器人 L0-L3 都是写给 `reference_impl.SichuanGame` 的,所以对局仍在参考实现里跑;
同时**并排维护一份 JAX 影子状态**,每一步用同一个动作推进两边,网络只看影子状态。
P1 的百万局差分已经证明两边逐决策点完全同步,所以影子可信;这样也保证**训练与评测
看到的是同一个 `observe()`**,不会重演立直线上 BC 与在线特征错位那种事。

判据(方案 P2):net vs 3×L0 必须**显著超过 L1 vs 3×L0 的 +3.867**。

用法:
  PYTHONPATH=~/mahjax:<repo> python sichuan/eval_net.py <params.pkl> [n_deals] [--opp L0|L1]
"""
from __future__ import annotations

import math
import pickle
import random
import statistics
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sichuan import env_jax as E
from sichuan.net import SichuanACNet
from sichuan.obs import observe
from reference_impl import SichuanGame
import bots

NEG = -1e9
L1_VS_L0 = 3.867      # 判据基准(arena.ladder 实测)

_init = jax.jit(E._init_from_wall)
_step = jax.jit(E._step_core)
_obs = jax.jit(observe)


def ref_to_id(a):
    k, arg = a
    return {"discard": lambda: E.A_DISCARD + arg, "ankan": lambda: E.A_GANG + arg,
            "bugang": lambda: E.A_GANG + arg, "zimo": lambda: E.A_HU,
            "ron": lambda: E.A_HU, "peng": lambda: E.A_PENG,
            "zhigang": lambda: E.A_ZHIGANG, "pass": lambda: E.A_PASS,
            "void": lambda: E.A_VOID + arg}[k]()


def infer_arch(params):
    """从权重推断 (channels, blocks)。

    写死规格会在换网络大小时炸(实测:容量实验的 256x10 权重喂给写死的 128x6,
    flax 抛 ScopeParamShapeError)。规格本来就完整编码在权重里,没有理由再传一次。
    """
    d = params["params"] if "params" in params else params
    ch = int(d["Conv_0"]["kernel"].shape[-1])
    nb = len([k for k in d if k.startswith("ResBlock1D_")])
    return ch, nb


def make_net_fn(params, channels, blocks):
    net = SichuanACNet(channels=channels, blocks=blocks)

    @jax.jit
    def pick(st):
        o = _obs(st)
        logits, _ = net.apply(params, o)
        logits = jnp.where(st.legal_action_mask, logits[0], NEG)
        return jnp.argmax(logits)          # 与立直线评测一致:贪心
    return pick


STATS = {"net_decisions": 0, "fallback": 0, "mask_mismatch": 0}


def play_deal(seed: int, net_seat: int, pick, opponent, rng_seed: int):
    """一盘。参考实现跑对局,JAX 影子同步推进供网络观测。

    ⚠️ 影子一旦与参考失步,网络就是在看错的局面下棋,而分数照样算得出来——
    这种 bug 不会崩,只会安静地把评测变成噪声。所以每个网络决策点都对一次
    合法集,并统计回退次数;两个计数必须为 0 才认这次读数。"""
    g = SichuanGame(seed)
    full = [t for t in range(E.NUM_TILES) for _ in range(4)]
    random.Random(seed).shuffle(full)
    st = _init(jnp.asarray(full, jnp.int8))
    rng = random.Random(rng_seed)
    guard = 0
    while g.phase != "over" and guard < 6000:
        i, acts = g.legal_actions()
        if i == net_seat:
            STATS["net_decisions"] += 1
            ref_ids = sorted({ref_to_id(a) for a in acts})
            jax_ids = sorted(np.flatnonzero(np.asarray(st.legal_action_mask)).tolist())
            if ref_ids != jax_ids or int(st.current_player) != i:
                STATS["mask_mismatch"] += 1
            aid = int(pick(st))
            cand = [a for a in acts if ref_to_id(a) == aid]
            if not cand:
                STATS["fallback"] += 1
            a = cand[0] if cand else acts[0]
        else:
            a = opponent(g, i, acts, rng)
        g.step(a)
        st = _step(st, jnp.int32(ref_to_id(a)))
        guard += 1
    assert g.phase == "over", "对局未终止"
    return g.scores()


def play_deal_rule(seed, challenger, opponent, net_seat, rng_seed):
    """纯规则挑战者走同一条评测路径,保证与网络用的是同一批牌山和同一套 rng 约定。"""
    g = SichuanGame(seed)
    rng = random.Random(rng_seed)
    guard = 0
    while g.phase != "over" and guard < 6000:
        i, acts = g.legal_actions()
        g.step((challenger if i == net_seat else opponent)(g, i, acts, rng))
        guard += 1
    assert g.phase == "over", "对局未终止"
    return g.scores()


def main():
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    opp_name = "L0"
    if "--opp" in sys.argv:
        opp_name = sys.argv[sys.argv.index("--opp") + 1]
    opponent = {"L0": bots.bot_L0_uniform, "L1": bots.bot_L1_greedy,
                "L2": bots.bot_L2_defensive}[opp_name]
    if path == "L1":                    # 让 L1 走同一套评测代码、同一批牌山
        pick = None
        challenger = bots.bot_L1_greedy
    else:
        with open(path, "rb") as f:
            params = pickle.load(f)
        pick = make_net_fn(params, *infer_arch(params))
        challenger = None

    per_deal = []
    for d in range(n):
        vals = []
        for seat in range(4):
            if pick is not None:
                sc = play_deal(d, seat, pick, opponent, rng_seed=d * 4 + seat)
            else:
                sc = play_deal_rule(d, challenger, opponent, seat, rng_seed=d * 4 + seat)
            vals.append(sc[seat])
        per_deal.append(sum(vals) / 4.0)
        if (d + 1) % 100 == 0:
            m = statistics.mean(per_deal)
            se = statistics.pstdev(per_deal) / math.sqrt(len(per_deal))
            print(f"  [{d+1}/{n}] {m:+.3f} ± {1.96*se:.3f}", flush=True)

    m = statistics.mean(per_deal)
    se = statistics.pstdev(per_deal) / math.sqrt(len(per_deal))
    ci = 1.96 * se
    tag = "L1" if pick is None else Path(path).name
    print(f"\n{tag} vs 3x{opp_name}:  {m:+.3f} ± {ci:.3f}   (n={n} 副牌)")
    if pick is not None:
        print(f"影子同步自检: 网络决策 {STATS['net_decisions']:,} 次,"
              f"合法集不符 {STATS['mask_mismatch']},动作回退 {STATS['fallback']}"
              f"  → {'✓ 可信' if STATS['mask_mismatch']==0 and STATS['fallback']==0 else '✗ 本次读数作废'}")
    # ⚠️ 不能落 /tmp:WSL 被回收时 /tmp 会清空,配对数据丢了就得整轮重跑
    out = Path.home() / "Projects/better_mortal/runs/sichuan_eval"
    out.mkdir(parents=True, exist_ok=True)
    fp = out / f"{tag}_{opp_name}_{n}.npy"
    np.save(fp, np.array(per_deal))
    print(f"逐副牌读数已存 → {fp}(供配对比较)")


if __name__ == "__main__":
    main()
