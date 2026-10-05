"""D3 (10-03): would bootstrapped (TD) Q targets cut the noise of Mortal's online-RL targets, without bias?

Mortal's online loss regresses Q(s_t, a_t) on a pure Monte Carlo target: the kyoku reward, i.e. the GRP expected
placement pt at the next kyoku start (or the final placement pt) minus that at this kyoku's start (gamma = 1). The
per-sample spread of that target is large next to the 0.5-1 pt differences between actions that matter. If the Q of
the action actually taken is calibrated (Stage 0: +0.18 ± 0.19 pt), the Q the same model reported n decisions later
in the same kyoku is an unbiased stand-in for the rest of the kyoku, with much less of the future's randomness:

    y_n = Q(s_{t+n}, a_{t+n})   (or the kyoku reward if the kyoku ends first),   y_lambda = TD(lambda) mix of the y_n

Reads 1v3 arena logs whose challenger wrote `meta.q_values` (Mortal engine; values in training units,
sacred pt x 2.390/90). For every greedy decision of the challenger, compares the target noise Var(y − Q_t) of MC vs
n-step / lambda targets, and the bias mean(y − G) overall, by Q decile and by decisions-to-kyoku-end (seed-group CIs).

Reading: worth building into the trainer's data loader if the noise drops >= 3x for some n or lambda while every
bias cell stays within ±0.3 pt (the size of the effects we chase) — otherwise M2 is dropped.

    python td_target_variance.py --log-dir DIR --grp grp.pth --games 2000 [--ref-name mortal-v4]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("RIICHI_EVAL_SRC", os.path.expanduser("~/riichi-eval/src")))
from riichi_eval.logs import NAME_RE, read_game  # noqa: E402
from riichi_eval.protocol import SACRED, placement_pt  # noqa: E402
from riichi_eval.rollout.value import GrpValuer  # noqa: E402

PT_SCALE = 2.390 / 90
NS = (1, 2, 4, 8)
LAMS = (0.5, 0.8, 0.9, 0.95)


def kyoku_rewards(g, seat, grp):
    """Sacred-pt kyoku rewards for `seat`, as Mortal's RewardCalculator computes them (pts in sacred units)."""
    feats = [grp.feature(r.index, r.honba, r.kyotaku, r.scores) for r in g.rounds]
    exp = grp.expected_pt([feats[:i + 1] for i in range(len(feats))], [seat] * len(feats))
    exp.append(placement_pt(g.final_scores(), seat, SACRED.pts))
    return np.diff(np.array(exp))


def decisions(rd, seat):
    """Q of the challenger's greedy decisions in this kyoku, in order (sacred pt); None marks a non-greedy one."""
    out = []
    for ev in rd.events:
        m = ev.get("meta")
        if ev.get("actor") != seat or not m or "q_values" not in m:
            continue
        out.append(max(m["q_values"]) / PT_SCALE if m.get("is_greedy", True) else None)
    return out


def targets(q, G):
    """Per decision: Q_t, MC target, n-step targets, lambda targets, decisions left (greedy decisions only)."""
    T = len(q)
    rows = []
    for t in range(T):
        if q[t] is None:
            continue
        # bootstrap only through greedy decisions; a non-greedy one in between ends the chain at the MC target
        nxt = []
        for u in range(t + 1, T):
            if q[u] is None:
                break
            nxt.append(q[u])
        yn = [nxt[n - 1] if n <= len(nxt) else G for n in NS]
        yl = []
        for lam in LAMS:
            # y_lambda = sum_{n>=1} (1-lam) lam^(n-1) y_n, with y_n = G once the chain ends
            acc, w = 0.0, 1.0
            for v in nxt:
                acc += (1 - lam) * w * v
                w *= lam
            yl.append(acc + w * G)
        rows.append((q[t], G, *yn, *yl, len(nxt)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--grp", required=True)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--ref-name", default="mortal-v4")
    args = ap.parse_args()

    grp = GrpValuer(args.grp)
    paths = sorted(p for p in glob.glob(os.path.join(args.log_dir, "*.json.gz")) if NAME_RE.search(os.path.basename(p)))
    paths = paths[: args.games]
    R, S = [], []
    for p in paths:
        g = read_game(p)
        seat = next(i for i, n in enumerate(g.names) if n != args.ref_name)
        rew = kyoku_rewards(g, seat, grp)
        for rd, G in zip(g.rounds, rew):
            for r in targets(decisions(rd, seat), float(G)):
                R.append(r)
                S.append(g.seed[0])
    R = np.array(R)
    S = np.array(S)
    q, G = R[:, 0], R[:, 1]
    left = R[:, -1]
    names = ["MC"] + [f"n={n}" for n in NS] + [f"lam={l}" for l in LAMS]
    Y = [G] + [R[:, 2 + i] for i in range(len(NS))] + [R[:, 2 + len(NS) + i] for i in range(len(LAMS))]
    us = np.unique(S)
    idx = [np.flatnonzero(S == s) for s in us]

    def ci_mean(x):
        y = np.array([x[i].sum() for i in idx]) / len(x) * len(us)
        return x.mean(), 1.96 * y.std(ddof=1) / np.sqrt(len(us))

    print(f"{len(paths):,} games, {len(us):,} seed groups, {len(q):,} greedy decisions of the challenger")
    b, bci = ci_mean(G - q)
    print(f"calibration of Q_t against the MC target: mean(G − Q) {b:+.3f} ± {bci:.3f} pt")
    print(f"\n{'target':<10}{'SD(y−Q)':>9}{'var ratio':>11}{'bias mean(y−G)':>18}")
    v0 = np.var(G - q)
    for n, y in zip(names, Y):
        bb, bc = ci_mean(y - G)
        print(f"{n:<10}{np.sqrt(np.var(y - q)):>9.2f}{np.var(y - q) / v0:>11.3f}{bb:>+11.3f} ± {bc:.3f}")

    print("\nbias mean(y − G) by Q decile (pt), seed-group 95% CI:")
    dec = np.quantile(q, np.linspace(0, 1, 11))
    cols = [names.index(c) for c in ("n=1", "n=8", "lam=0.8")]
    cn = [names[c] for c in cols]
    print(f"{'Q range':<20}" + "".join(f"{c:>18}" for c in cn))
    for a, bnd in zip(dec[:-1], dec[1:]):
        m = (q >= a) & (q <= bnd)
        cells = []
        for c in cols:
            x = np.where(m, Y[c] - G, 0.0)
            y = np.array([x[i].sum() for i in idx]) / m.sum() * len(us)
            cells.append(f"{x.sum() / m.sum():+8.3f} ± {1.96 * y.std(ddof=1) / np.sqrt(len(us)):.3f}")
        print(f"[{a:+7.2f},{bnd:+7.2f}] " + "".join(f"{c:>18}" for c in cells))

    print("\nby decisions left in the kyoku (greedy chain), var ratio and bias for lam=0.8:")
    lc = names.index("lam=0.8")
    for lo, hi in ((0, 0), (1, 2), (3, 5), (6, 10), (11, 99)):
        m = (left >= lo) & (left <= hi)
        if m.sum() < 100:
            continue
        vr = np.var(Y[lc][m] - q[m]) / np.var(G[m] - q[m])
        print(f"  {lo:>2}-{hi:<2} n={m.sum():>8,}  SD MC {np.std(G[m] - q[m]):6.2f}  var ratio {vr:.3f}  "
              f"bias {np.mean(Y[lc][m] - G[m]):+.3f}")


if __name__ == "__main__":
    main()
