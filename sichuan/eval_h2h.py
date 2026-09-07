"""测试 3:川麻快照之间的**直接对打**(复式 1v3,挑战者轮坐四座)。

动机:我们量过 b800 和 b2272 各自 vs 3xL0 的配对差(-0.528, z=-2.13),
但**从没量过两者直接对打**。在非传递博弈里这两件事可以给出相反结论:
"A 和 B 打同一个弱对手的分差" 与 "A 打 B 的分差" 不是一回事。
若出现 b2272 打不过 b800 但两者都打得过 L1,那 -0.528 就不是"学不动"
而是**非传递漂移**,对策完全不同(保留最好的 checkpoint,而不是加熵)。

沿用 eval_net.py 的桥接:参考实现跑对局,JAX 影子状态同步供网络观测,
每个网络决策点核对合法集,失配/回退必须为 0 才认读数。
A(pi,pi)=0:同一份权重自己打自己期望为 0,可作自检。

用法: python eval_h2h.py <A.pkl|L1> <B.pkl|L1> [n_deals]
"""
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
_init = jax.jit(E._init_from_wall)
_step = jax.jit(E._step_core)
_obs = jax.jit(observe)
STATS = {"net_decisions": 0, "fallback": 0, "mask_mismatch": 0}


def ref_to_id(a):
    k, arg = a
    return {"discard": lambda: E.A_DISCARD + arg, "ankan": lambda: E.A_GANG + arg,
            "bugang": lambda: E.A_GANG + arg, "zimo": lambda: E.A_HU,
            "ron": lambda: E.A_HU, "peng": lambda: E.A_PENG,
            "zhigang": lambda: E.A_ZHIGANG, "pass": lambda: E.A_PASS,
            "void": lambda: E.A_VOID + arg}[k]()


def make_pick(path):
    """返回 (kind, fn)。kind='net' 时 fn(st)->action_id;'bot' 时 fn(g,i,acts,rng)->action。"""
    if path == "L1":
        return "bot", bots.bot_L1_greedy
    if path == "L0":
        return "bot", bots.bot_L0_uniform
    with open(path, "rb") as f:
        params = pickle.load(f)
    net = SichuanACNet(channels=128, blocks=6)

    @jax.jit
    def pick(st):
        logits, _ = net.apply(params, _obs(st))
        return jnp.argmax(jnp.where(st.legal_action_mask, logits[0], NEG))
    return "net", pick


def play_deal(seed, a_seat, A, B, rng_seed):
    """A 坐 a_seat,B 坐其余三席。返回四家分数。"""
    (ka, fa), (kb, fb) = A, B
    g = SichuanGame(seed)
    full = [t for t in range(E.NUM_TILES) for _ in range(4)]
    random.Random(seed).shuffle(full)
    st = _init(jnp.asarray(full, jnp.int8))
    rng = random.Random(rng_seed)
    guard = 0
    while g.phase != "over" and guard < 6000:
        i, acts = g.legal_actions()
        kind, fn = (ka, fa) if i == a_seat else (kb, fb)
        if kind == "net":
            STATS["net_decisions"] += 1
            ref_ids = sorted({ref_to_id(x) for x in acts})
            jax_ids = sorted(np.flatnonzero(np.asarray(st.legal_action_mask)).tolist())
            if ref_ids != jax_ids or int(st.current_player) != i:
                STATS["mask_mismatch"] += 1
            aid = int(fn(st))
            cand = [x for x in acts if ref_to_id(x) == aid]
            if not cand:
                STATS["fallback"] += 1
            a = cand[0] if cand else acts[0]
        else:
            a = fn(g, i, acts, rng)
        g.step(a)
        st = _step(st, jnp.int32(ref_to_id(a)))
        guard += 1
    assert g.phase == "over", "对局未终止"
    return g.scores()


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    A, B = make_pick(a_path), make_pick(b_path)
    per_deal = []
    for d in range(n):
        vals = [play_deal(d, s, A, B, rng_seed=d * 4 + s)[s] for s in range(4)]
        per_deal.append(sum(vals) / 4.0)
        if (d + 1) % 50 == 0:
            m = statistics.mean(per_deal)
            se = statistics.pstdev(per_deal) / math.sqrt(len(per_deal))
            print(f"  [{d+1}/{n}] {m:+.3f} ± {1.96*se:.3f}", flush=True)
    m = statistics.mean(per_deal)
    se = statistics.pstdev(per_deal) / math.sqrt(len(per_deal))
    an, bn = Path(a_path).name, Path(b_path).name
    print(f"\n{an} vs 3x{bn}:  {m:+.3f} ± {1.96*se:.3f}   (n={n} 副牌)")
    print(f"影子自检: 网络决策 {STATS['net_decisions']:,},合法集不符 "
          f"{STATS['mask_mismatch']},回退 {STATS['fallback']}  → "
          f"{'✓ 可信' if STATS['mask_mismatch'] == 0 and STATS['fallback'] == 0 else '✗ 作废'}")
    out = Path.home() / "Projects/better_mortal/runs/sichuan_eval"
    out.mkdir(parents=True, exist_ok=True)
    fp = out / f"h2h_{an}_vs_{bn}_{n}.npy"
    np.save(fp, np.array(per_deal))
    print(f"逐副牌读数 → {fp}")


if __name__ == "__main__":
    main()
