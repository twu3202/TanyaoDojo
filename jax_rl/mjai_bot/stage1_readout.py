"""Branch B Stage 1 mechanism readout on Stage 0's labelled states (reads a stage0_stale.py npz run with the
Stage 1 snapshots and the historical 408k->448k snapshot as --extra models, all in sacred-proportional pts).

For the states where tpt 408k took `ours` and the rollout priced the alternative, per kind and per 408k margin bin:
  - untaken-side residual (T_alt - Q_m(alt)) and taken-side residual, pt: exploration should lift the untaken side
  - model margin Q_m(ours) - Q_m(alt), pt, and the share of states where the model now prefers alt
  - calibration slope of d_grp on the model's predicted difference
Targets T come from 408k's own continuation (the labels), so they are exact for 408k and approximate for later
snapshots; the comparison that matters is Stage 1 vs the historical continuation at the same step count.

    python stage1_readout.py stage0_s1.npz
"""
from __future__ import annotations

import sys

import numpy as np

PT = 2.390 / 90
KINDS = ("push2fold", "fold2push", "riichi2dama", "dama2riichi")
BINS = ((0, 5), (5, 10), (10, 20), (20, 1e9))


def cl_mean(y, g):
    _, gi = np.unique(g, return_inverse=True)
    s, n = np.bincount(gi, weights=y), np.bincount(gi).astype(float)
    m = s.sum() / n.sum()
    G = len(s)
    return m, 1.96 * np.sqrt(G / (G - 1) * ((s - m * n) ** 2).sum()) / n.sum()


def cl_slope(x, y, g):
    xc, yc = x - x.mean(), y - y.mean()
    sxx = (xc * xc).sum()
    b = (xc * yc).sum() / sxx
    _, gi = np.unique(g, return_inverse=True)
    sc = np.bincount(gi, weights=xc * (yc - b * xc))
    G = len(sc)
    return b, 1.96 * np.sqrt(G / (G - 1) * (sc ** 2).sum()) / sxx


def main():
    z = np.load(sys.argv[1], allow_pickle=False)
    c = {str(k): z["data"][:, i] for i, k in enumerate(z["cols"])}
    extras = [str(x) for x in z["extras"]]
    g, d, t_o = c["seed"], c["d_grp"], c["t_ours"]
    t_a = t_o + d
    m408 = (c["q408_o"] - c["q408_a"]) / PT
    models = {"408k": (c["q408_o"], c["q408_a"]), "448k (m448)": (c["q448_o"], c["q448_a"])}
    for n in extras:
        models[n] = (c[f"{n}_q_o"], c[f"{n}_q_a"])
    games = len(np.unique(c["seed"] * 4 + c["rot"]))
    print(f"{len(d):,} pairs, {games:,} games; bins by the 408k margin (pt); residual + = Q too low\n")
    for k, kind in enumerate(KINDS):
        mk = c["kind"] == k
        print(f"== {kind}")
        for lo, hi in BINS:
            b = mk & (m408 >= lo) & (m408 < hi)
            if b.sum() < 300:
                continue
            dm, dc = cl_mean(d[b], g[b])
            print(f"  408k margin [{lo:>2},{hi if hi < 1e9 else 'inf':>3}) pt: {b.sum() / games:5.2f}/game, "
                  f"rollout d(alt-ours) {dm:+.2f}±{dc:.2f}")
            for name, (qo, qa) in models.items():
                ru, ruc = cl_mean(t_a[b] - qa[b] / PT, g[b])
                rt, rtc = cl_mean(t_o[b] - qo[b] / PT, g[b])
                mm = ((qo[b] - qa[b]) / PT).mean()
                flip = np.mean(qa[b] > qo[b])
                print(f"    {name:<16} untaken res {ru:+7.2f}±{ruc:4.2f}  taken res {rt:+6.2f}±{rtc:4.2f}  "
                      f"margin {mm:+7.2f}  prefers alt {flip:6.1%}")
        for name, (qo, qa) in models.items():
            s, sc = cl_slope((qa[mk] - qo[mk]) / PT, d[mk], g[mk])
            print(f"  calibration slope {name:<16} {s:+.4f} ± {sc:.4f}")
        print()


if __name__ == "__main__":
    main()
