"""
复核诊断报告主张⑤:"接入 sp(单人求解器)特征值 +1.0~1.5pt"。

无法不训练就直接证伪一个反事实增益,但可以量**头寸(headroom)**:
sp 求解器的主项是"打哪张牌后的向听/受入最好"。若我们的策略在这条维度上
已经贴着最优,那 1.0-1.5pt 就不可能从效率这条路来。

做法(GPU,几分钟):策略自博弈跑出打牌决策点,对每个合法弃张精确算
  s'      = shanten(手牌 − 该张)
  ukeire' = Σ_u (4 − 己见 u) · 1[shanten(手牌 − 该张 + u) = s' − 1]
再比较"策略实选弃张"与"受入最优弃张"的差。

口径说明:己见只扣自家手牌(不扣牌河/副露),同一决策点内各候选共用同一
权重,故候选间比较不受影响;红 5(动作 34/35/36)与摸切(71)都归一到牌型。

用法:
  PYTHONPATH=~/mahjax ~/jax_env/bin/python ukeire_headroom.py a.pkl [b.pkl ...] \
      [--num-envs 128] [--num-steps 96]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import mahjax
from mahjax.wrappers.auto_reset_wrapper import auto_reset
from mahjax.red_mahjong.shanten import Shanten

sys.path.insert(0, str(Path(__file__).resolve().parent))
from net_lean import LeanACNet          # noqa: E402
from obs_v2 import observe_v2           # noqa: E402

NEG = -1e9
RED_TO_TYPE = {34: 4, 35: 13, 36: 22}
TSUMOGIRI = 71


def build_candidate_types():
    """动作 → 牌型(0-33);非弃张动作置 -1。摸切(71)在运行时按 last_draw 填。"""
    t = np.full(87, -1, np.int32)
    for a in range(34):
        t[a] = a
    for a, ty in RED_TO_TYPE.items():
        t[a] = ty
    return jnp.asarray(t)


CAND_TYPE = build_candidate_types()


def ukeire_of(hand13):
    """hand13: (34,) 打出后 13 张手牌计数 → (shanten, ukeire 枚数)。"""
    s0 = Shanten.number(hand13)
    eye = jnp.eye(34, dtype=hand13.dtype)
    plus = hand13[None, :] + eye                      # (34,34)
    s1 = jax.vmap(Shanten.number)(plus)
    room = jnp.maximum(4 - hand13, 0)
    ok = (s1 == s0 - 1) & (room > 0)
    return s0, jnp.sum(jnp.where(ok, room, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actors", nargs="+")
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--num-steps", type=int, default=96)
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = mahjax.make("red_mahjong", round_mode="half", observe_type="dict")
    step_fn = auto_reset(env.step, env.init)
    net = LeanACNet(channels=args.channels, blocks=args.blocks)

    def run(params):
        def one_step(carry, _):
            state, rng = carry
            rng, k_env = jax.random.split(rng)
            obs = jax.vmap(observe_v2)(state)
            mask = state.legal_action_mask.astype(jnp.bool_)
            logits, _ = net.apply(params, obs)
            logits = jnp.where(mask, logits, NEG)
            act = jnp.argmax(logits, axis=-1)          # 与评测一致:贪心
            cur = jnp.asarray(state.current_player, jnp.int32)
            hand = state.players.hand[jnp.arange(state.players.hand.shape[0]), cur]
            ld = jnp.asarray(state.round_state.last_draw, jnp.int32)
            nxt = jax.vmap(step_fn)(state, act, jax.random.split(k_env, args.num_envs))
            return (nxt, rng), (hand, mask, act, ld)

        rng = jax.random.PRNGKey(args.seed)
        rng, k = jax.random.split(rng)
        st = jax.vmap(env.init)(jax.random.split(k, args.num_envs))
        _, out = lax.scan(one_step, (st, rng), None, length=args.num_steps)
        return jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), out)

    # 候选弃张的牌型(含摸切),非弃张候选置 -1
    def cand_types(mask, ld):
        t = jnp.where(mask[:87], CAND_TYPE, -1)
        ld_t = jnp.where(ld >= 34, jnp.asarray([4, 13, 22, -1])[jnp.clip(ld - 34, 0, 3)], ld)
        return t.at[TSUMOGIRI].set(jnp.where(mask[TSUMOGIRI] & (ld >= 0), ld_t, -1))

    @jax.jit
    def analyse(hand, mask, act, ld):
        ct = cand_types(mask, ld)                       # (87,) 牌型或 -1
        ok = (ct >= 0) & (hand[jnp.clip(ct, 0, 33)] > 0)
        h13 = hand[None, :] - jax.nn.one_hot(jnp.clip(ct, 0, 33), 34, dtype=hand.dtype)
        s, u = jax.vmap(ukeire_of)(h13)
        # 只在"有多个不同弃张牌型可选"的决策点比较
        s = jnp.where(ok, s, 99)
        u = jnp.where(ok, u, -1)
        best = jnp.max(jnp.where(s == jnp.min(s), u, -1))
        chosen_ok = ok[act]
        return (jnp.sum(ok) , chosen_ok, s[act], u[act], jnp.min(s), best)

    for path in args.actors:
        with open(path, "rb") as f:
            params = pickle.load(f)
        hand, mask, act, ld = jax.device_get(run(params))
        res = [analyse(jnp.asarray(hand[i]), jnp.asarray(mask[i]),
                       jnp.asarray(act[i]), jnp.asarray(ld[i]))
               for i in range(len(act))]
        n_cand, ok, s_c, u_c, s_b, u_b = [np.asarray(x) for x in zip(*res)]
        m = ok & (n_cand > 1)
        n = int(m.sum())
        same_s = (s_c[m] == s_b[m])
        # 受入只在同向听下可比(更差向听的手牌受入天然更大),故一律条件在向听最优上
        d = (u_b[m] - u_c[m])[same_s]
        ub = u_b[m][same_s]
        sb = s_b[m][same_s]
        print(f"\n=== {Path(path).name}  弃张决策点 n={n:,} (候选>1)")
        print(f"  向听最优率            : {same_s.mean()*100:.2f}%  "
              f"(漏失 {100-same_s.mean()*100:.2f}%)")
        print(f"  向听最优内 受入最优率 : {(d == 0).mean()*100:.2f}%")
        print(f"  向听最优内 平均亏欠   : {d.mean():.3f} 枚 = 最优受入的 "
              f"{100*d.mean()/ub.mean():.2f}%")
        for lo, hi, lbl in ((0, 0, "听牌"), (1, 1, "一向听"), (2, 2, "二向听"), (3, 9, "三向听+")):
            k = (sb >= lo) & (sb <= hi)
            if k.sum() < 30:
                continue
            print(f"    {lbl:<6}: n={int(k.sum()):>6,}  最优率 {(d[k]==0).mean()*100:5.1f}%  "
                  f"亏欠 {d[k].mean():5.2f}/{ub[k].mean():5.1f} 枚 "
                  f"({100*d[k].mean()/ub[k].mean():.2f}%)")


if __name__ == "__main__":
    main()
