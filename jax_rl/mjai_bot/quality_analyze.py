"""Data-quality study analysis (2026-10-11, patches/data_quality/README.md).

    python quality_analyze.py METRICS.tsv OUT_SCORES.tsv [--sample QS.tsv ...]

1. Player strength (M2): for every player name, the average placement over all of their other games in the pool
   (leave-one-out), shrunk toward 2.5 with an empirical-Bayes prior. A game's M2 is the mean over its four players
   (lower = stronger); the weakest player's value is kept too, as an exploratory variant.
2. Writes one score row per game.
3. With --sample (quality_sample.py output): e1 / bal / all agreement with v4 by quartile of each metric, worst
   minus best quartile with game-clustered 95% CIs, plus the selection check of rule 4.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

V11_YEARS = range(2019, 2027)
N_SELECT = 1_289_661


def player_strength(m: pd.DataFrame) -> pd.DataFrame:
    long = pd.concat([pd.DataFrame({"g": m.index, "name": m[f"n{i}"], "rank": m[f"r{i}"]}) for i in range(4)])
    long = long[~long["name"].str.startswith("EXAMPLE")]
    agg = long.groupby("name")["rank"].agg(["sum", "count"])
    # empirical-Bayes prior strength k = within-player rank variance / between-player variance of true means
    big = agg[agg["count"] >= 200]
    means = big["sum"] / big["count"]
    within = 1.25                      # variance of a uniformly random placement 1..4
    tau2 = max(means.var() - (within / big["count"]).mean(), 1e-4)
    k = within / tau2
    long = long.join(agg, on="name")
    long["loo"] = (long["sum"] - long["rank"] + k * 2.5) / (long["count"] - 1 + k)
    g = long.groupby("g")["loo"].agg(["mean", "max", "count"])
    out = pd.DataFrame(index=m.index)
    out["m2_mean"] = g["mean"]
    out["m2_weakest"] = g["max"]
    out["m2_known"] = g["count"]
    print(f"players {len(agg):,}; with >= 200 games {len(big):,}; between-player SD of true mean rank "
          f"{np.sqrt(tau2):.3f}; prior k = {k:.0f} games")
    print(f"games per player: median {agg['count'].median():.0f}, mean {agg['count'].mean():.1f}, "
          f"share of seats held by players with >= 200 games {big['count'].sum() / agg['count'].sum():.1%}")
    return out


def clustered(df: pd.DataFrame, key: str):
    n, h = df[f"n_{key}"].to_numpy(float), df[f"h_{key}"].to_numpy(float)
    keep = n > 0
    n, h = n[keep], h[keep]
    p = h.sum() / n.sum()
    var = ((h - p * n) ** 2).sum() / (len(n) * (len(n) - 1)) / n.mean() ** 2
    return p, var, len(n)


def diff_line(a: pd.DataFrame, b: pd.DataFrame, key: str):
    pa, va, na = clustered(a, key)
    pb, vb, nb = clustered(b, key)
    d, se = pa - pb, np.sqrt(va + vb)
    return f"{100 * pa:6.2f} vs {100 * pb:6.2f}  diff {100 * d:+5.2f} ± {196 * se:4.2f} pp (z {d / se:+.1f}; {na}/{nb} games)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics")
    ap.add_argument("out")
    ap.add_argument("--sample", nargs="*", default=[])
    a = ap.parse_args()
    m = pd.read_csv(a.metrics, sep="\t", dtype={f"n{i}": str for i in range(4)}, keep_default_na=False,
                    na_values=[])
    assert (m[["s0", "s1", "s2", "s3"]].sum(1) == 100000).mean() > 0.999, "final scores do not sum to 100,000"
    ps = player_strength(m)
    m = m.join(ps)
    sc = m[["path", "year", "final_spread", "gap12", "gap34", "mean_spread", "max_spread", "busted",
            "m2_mean", "m2_weakest", "m2_known"]]
    sc.to_csv(a.out, sep="\t", index=False)

    print("\n== descriptives (all 2009-2026)")
    q = m["final_spread"].quantile([.25, .5, .75]).tolist()
    print(f"final spread quartiles {q}; busted {m['busted'].mean():.1%}; "
          f"one-low (3rd-4th >= 20k) {(m['gap34'] >= 20000).mean():.1%}; one-high (1st-2nd >= 20k) {(m['gap12'] >= 20000).mean():.1%}")
    print("Spearman correlations between metrics:")
    print(m[["final_spread", "gap12", "gap34", "mean_spread", "m2_mean", "m2_weakest"]].corr("spearman").round(3).to_string())
    print("by era: mean M2 (lower = stronger), mean final spread")
    m["era"] = np.where(m["year"] >= 2019, "2019-2026", "2009-2018")
    print(m.groupby("era")[["m2_mean", "final_spread", "busted"]].mean().round(3).to_string())

    if not a.sample:
        return
    s = pd.concat([pd.read_csv(f, sep="\t") for f in a.sample])
    s["path"] = s["path"].str.replace(r"^.*/houou/", "", regex=True)
    j = s.merge(m, on="path", how="inner")
    print(f"\n== yardstick: {len(j):,} sampled games with results")
    for key in ("e1", "bal", "all"):
        p, v, n = clustered(j, key)
        print(f"  overall {key}: {100 * p:.2f} ± {196 * np.sqrt(v):.2f} pp ({n} games)")
    # metric -> sign so that "best" quartile = higher expected quality under the hypothesis
    tests = [("final_spread", "M1 final spread (close best)", 1), ("gap34", "M1 one-low gap (small best)", 1),
             ("gap12", "M1 one-high gap (small best)", 1), ("mean_spread", "M1 mean in-game spread", 1),
             ("m2_mean", "M2 player strength, mean of 4", 1), ("m2_weakest", "M2 weakest player (exploratory)", 1)]
    print("\n== rule 1: best quartile minus worst quartile (pre-registered: e1 diff >= 1.0 pp, CI excludes 0)")
    for col, label, sign in tests:
        qs = j[col].rank(pct=True, method="first")
        best = j[qs <= 0.25] if sign > 0 else j[qs > 0.75]
        worst = j[qs > 0.75] if sign > 0 else j[qs <= 0.25]
        print(f"  {label}")
        for key in ("e1", "bal", "all"):
            print(f"    {key:<4} {diff_line(best, worst, key)}")
    print("\n== era: 2019-2026 minus 2009-2018")
    for key in ("e1", "all"):
        print(f"    {key:<4} {diff_line(j[j['year'] >= 2019], j[j['year'] < 2019], key)}")
    # rule 4: the 1,289,661 best games by M2 vs v11's set (2019-2026), on the sample
    cut = m["m2_mean"].nsmallest(N_SELECT).max()
    sel = j[j["m2_mean"] <= cut]
    v11 = j[j["year"].isin(V11_YEARS)]
    print(f"\n== rule 4 (if M2 selects): top {N_SELECT:,} by M2 (cut {cut:.4f}) vs v11 set")
    for key in ("e1", "all"):
        print(f"    {key:<4} {diff_line(sel, v11, key)}")
    yrs = m.loc[m["m2_mean"] <= cut, "year"].value_counts().sort_index()
    print("    selected games by year: " + ", ".join(f"{y}:{c:,}" for y, c in yrs.items()))


if __name__ == "__main__":
    main()
