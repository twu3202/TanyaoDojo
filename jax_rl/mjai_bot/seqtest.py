"""Sequential test on duplicate seed groups (Wald SPRT, truncated): stop an evaluation as soon as the games played
decide between "no better than δ0" and "at least δ1 better", instead of always playing a fixed 12k / 100k.

Unit: one seed group (four rotations of one wall), its mean placement points; for a comparison, the per-group
difference of two models on the same seeds. With per-group SD σ (about 37 pt absolute vs v4, about 48 pt for a
paired difference of two of our models), after n groups

    LLR_n = (δ1 − δ0) / σ² · Σ (d_i − (δ0 + δ1) / 2)

stop for H1 when LLR_n ≥ log((1 − β) / α), for H0 when LLR_n ≤ log(β / (1 − α)). σ² is the running sample variance
(plug-in). Looks happen every `look` groups only (an evaluation segment), and at `nmax` groups an undecided test
ends with the fixed-sample result. Groups are taken in ascending seed order, fixed before any game is played.

The decision is the output. The mean at a stop is biased towards the boundary it crossed, so a stopped test reports
a decision, not an effect size; records stay on the fixed 100k.

    python seqtest.py A_GLOB [B_GLOB] [--delta0 0] [--delta1 0.5] [--alpha 0.05] [--beta 0.2] [--look 1000]
                      [--nmax 25000] [--lo 10000]
A alone tests A vs mortal_v4; with B it tests A − B on shared seeds. Dumps are group_ci.py --dump npz files (seed, mean).
Exit code: 0 = H1 (promote), 1 = H0 (drop), 3 = continue (more groups needed), 4 = cap reached undecided,
2 = bad input.
"""
from __future__ import annotations

import argparse
import glob
import math
import sys

import numpy as np


def bounds(alpha, beta):
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)


def sprt(d, delta0=0.0, delta1=0.5, alpha=0.05, beta=0.2, look=1000, nmax=25000):
    """Runs the test over the per-group values d (already in the fixed order). Returns (decision, n, llr, mean, se):
    decision 'H1' / 'H0' at a look that crosses a bound, 'fixed' when nmax is reached undecided, 'continue' when d
    runs out before either."""
    lo_b, hi_b = bounds(alpha, beta)
    d = np.asarray(d, dtype=float)[:nmax]
    cs, cs2 = np.cumsum(d), np.cumsum(d * d)
    n = 0
    for n in range(look, len(d) + 1, look):
        mean = cs[n - 1] / n
        var = (cs2[n - 1] - n * mean * mean) / (n - 1)
        llr = (delta1 - delta0) / var * (cs[n - 1] - n * (delta0 + delta1) / 2)
        if llr >= hi_b or llr <= lo_b:
            return ("H1" if llr >= hi_b else "H0"), n, llr, mean, math.sqrt(var / n)
    if n == 0:
        return "continue", 0, 0.0, float("nan"), float("nan")
    mean = cs[n - 1] / n
    var = (cs2[n - 1] - n * mean * mean) / (n - 1)
    llr = (delta1 - delta0) / var * (cs[n - 1] - n * (delta0 + delta1) / 2)
    return ("fixed" if n >= nmax else "continue"), n, llr, mean, math.sqrt(var / n)


def load(pattern):
    m, bad = {}, False
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        for s, v in zip(z["seed"], z["mean"]):
            s, v = int(s), float(v)
            if s in m and m[s] != v:
                bad = True
            m.setdefault(s, v)
    return m, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--delta0", type=float, default=0.0)
    ap.add_argument("--delta1", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--look", type=int, default=1000)
    ap.add_argument("--nmax", type=int, default=25000)
    ap.add_argument("--lo", type=int, default=10000, help="first seed of the fixed ascending order")
    args = ap.parse_args()
    a, bad = load(args.a)
    b, bad_b = load(args.b) if args.b else ({}, False)
    if bad or bad_b:
        print("CONFLICT: a seed dumped twice with different values")
        sys.exit(2)
    order = range(args.lo, args.lo + args.nmax)
    # only a gap-free prefix of the fixed order counts: a missing group in the middle would let the order depend on
    # which games happened to finish
    d = []
    for s in order:
        if s not in a or (args.b and s not in b):
            break
        d.append(a[s] - (b[s] if args.b else 0.0))
    dec, n, llr, mean, se = sprt(d, args.delta0, args.delta1, args.alpha, args.beta, args.look, args.nmax)
    lo_b, hi_b = bounds(args.alpha, args.beta)
    what = f"{args.a}" + (f" - {args.b}" if args.b else " vs mortal_v4")
    print(f"{what}: H0 delta <= {args.delta0:+g} vs H1 delta >= {args.delta1:+g}, alpha {args.alpha}, beta {args.beta}, "
          f"looks every {args.look} groups, cap {args.nmax}")
    print(f"  {len(d):,} groups available (gap-free from seed {args.lo}); at n = {n:,}: mean {mean:+.3f} "
          f"(+- {1.96 * se:.3f}), LLR {llr:+.2f} in ({lo_b:+.2f}, {hi_b:+.2f})")
    msg = {"H1": "STOP: H1 (at least delta1 better)", "H0": "STOP: H0 (no better than delta0)",
           "fixed": "CAP reached undecided: report the fixed-sample CI", "continue": "CONTINUE: play the next segment"}
    print(f"  {msg[dec]}")
    sys.exit({"H1": 0, "H0": 1, "fixed": 4, "continue": 3}[dec])


if __name__ == "__main__":
    main()
