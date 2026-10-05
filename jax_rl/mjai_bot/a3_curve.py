"""A3 learning curve (10-02): does the rollout-label fine-tune get better with more labels?

Reads the `<out>.testpred.npz` files written by `a3_finetune.py --test-pairs` for several `--train-frac` runs.
Each holds the model's predicted margin (sacred pt) on the same low-noise test pairs at every eval step.
For each run the step is chosen on half A of the test seed groups (even seeds) by the correlation of the
prediction with the wall-averaged target, and reported on half B (odd seeds), so the choice of step does not
inflate the reported number. Paired comparisons between runs use a seed-group bootstrap on half B.

Pre-registered reading (written before the runs):
  data-limited  <=> corr(100%) − corr(25%) > 0 with the 95% bootstrap CI above 0, and 50% between them.
                    Then a 10x label set is worth considering.
  not           <=> no fraction beats the base model's correlation, or no ordering by data size.
                    Then scaling the labels is not justified and A3 is closed.

    python a3_curve.py NAME=path.testpred.npz [NAME=...]
"""
from __future__ import annotations

import sys

import numpy as np


def corr(p, t):
    return float(np.corrcoef(p, t)[0, 1])


def main():
    runs = {}
    for a in sys.argv[1:]:
        name, path = a.split("=", 1)
        runs[name] = np.load(path, allow_pickle=True)
    first = next(iter(runs.values()))
    tgt, seed = first["tgt"], first["seed"]
    for r in runs.values():
        assert np.array_equal(r["tgt"], tgt), "runs were scored on different test pairs"
    A, B = seed % 2 == 0, seed % 2 == 1
    ub = np.unique(seed[B])
    nB = 4 * len(ub)
    print(f"test pairs {len(tgt):,} ({A.sum():,} in half A for step choice, {B.sum():,} in half B for the report)")
    base = first["pred"][0]
    print(f"base model (step 0): corr B {corr(base[B], tgt[B]):+.4f}")
    chosen = {}
    print(f"\n{'run':<8}{'step*':>7}{'corr A':>9}{'corr B':>9}{'slope B':>9}{'flips/g':>9}{'gain/g B':>10}  curve of corr A over steps")
    for name, r in runs.items():
        cA = [corr(p[A], tgt[A]) for p in r["pred"]]
        k = int(np.argmax(cA))
        p = r["pred"][k]
        pb, tb = p[B], tgt[B]
        flip = pb > 0
        slope = float(np.cov(pb, tb)[0, 1] / pb.var(ddof=1))
        chosen[name] = p
        curve = " ".join(f"{s}:{c:.3f}" for s, c in zip(r["steps"], cA))
        print(f"{name:<8}{int(r['steps'][k]):>7}{cA[k]:>9.4f}{corr(pb, tb):>9.4f}{slope:>9.3f}"
              f"{flip.sum() / nB:>9.2f}{tb[flip].sum() / nB:>+10.3f}  {curve}")

    rng = np.random.default_rng(0)
    pos = {s: np.flatnonzero(B & (seed == s)) for s in ub}
    names = list(chosen)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]] + [(n, "base") for n in names]
    chosen["base"] = base
    boots = {pr: [] for pr in pairs}
    for _ in range(2000):
        idx = np.concatenate([pos[s] for s in rng.choice(ub, len(ub))])
        c = {n: corr(chosen[n][idx], tgt[idx]) for n in chosen}
        for a, b in pairs:
            boots[(a, b)].append(c[a] - c[b])
    print("\npaired corr differences on half B (seed-group bootstrap, 95% CI):")
    for (a, b), v in boots.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        pt = corr(chosen[a][B], tgt[B]) - corr(chosen[b][B], tgt[B])
        print(f"  {a:>6} − {b:<6} {pt:+.4f}   [{lo:+.4f}, {hi:+.4f}]")
    print("\nflip gain on half B with the low-noise target, per game (seed-group 95% CI):")
    for n in names:
        p = chosen[n]
        y = np.array([tgt[pos[s]][p[pos[s]] > 0].sum() / 4 for s in ub])
        print(f"  {n:<8}{y.mean():+.3f} ± {1.96 * y.std(ddof=1) / np.sqrt(len(y)):.3f}  "
              f"({(p[B] > 0).sum() / nB:.2f} flips/game)")


if __name__ == "__main__":
    main()
