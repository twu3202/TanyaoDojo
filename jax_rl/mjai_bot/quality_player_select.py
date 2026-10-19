"""Data-quality study v6: score players by their own regret and select games (patches/data_quality/README.md, v6).

    python quality_player_select.py --players PL.npz [PL2.npz ...] --heldout RG_A.npz RG_B.npz \
        --metrics game_metrics_houou.tsv --out-dir DIR

1. Stratum means of v4 / C' regret come from the held-out 12,000-game sample (houou); excess = regret - mean.
2. Player score = mean excess v4 regret over the player's seats in the player-balanced sample, shrunk toward 0 with
   an empirical-Bayes prior k = seat variance / between-player variance. Game score = mean of its four players'
   scores (unscored players count as 0, the population mean). C' is not used to score.
3. Held-out checks (rules 1 and 4 of v6), both judges, thresholds from the calibration J (jade - houou on the same
   held-out sample): rule 1 worst minus best quartile of game score >= J/3; rule 4 the best 1,289,661 pool games
   minus v11's set <= -J/5. The best 25% is reported only.
4. Writes the selected file lists (relative paths) to DIR for the arm, whatever the verdict.
v10 (2026-10-19): also the best 12.5% (select_m3_q12.txt, the first half of the q25 order), reported the same way.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from quality_regret_analyze import COARSE, FINE, diff, load, per_game

N_SELECT = 1_289_661
JUDGES = ("v4", "cp")


def strata_means(ref: pd.DataFrame, col: str):
    h = ref[ref["source"] == "houou"]
    f = h.groupby(FINE)[col].agg(["mean", "count"]).rename(columns={"mean": "mf", "count": "cf"})
    c = h.groupby(COARSE)[col].mean().rename("mc")
    return f, c, h[col].mean()


def apply_excess(target: pd.DataFrame, col: str, tables):
    f, c, overall = tables
    x = target[FINE].join(f, on=FINE).join(c, on=COARSE)
    base = np.where(x["cf"].fillna(0).to_numpy() >= 30, x["mf"], x["mc"])
    base = np.where(np.isnan(base), overall, base)
    target[f"{col}_ex"] = target[col].to_numpy() - base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", nargs="+", required=True)
    ap.add_argument("--heldout", nargs="+", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    ho = load(a.heldout)
    for j in JUDGES:
        ho[f"{j}_reg"] = ho[f"{j}_regret"]
    tables = {j: strata_means(ho, f"{j}_reg") for j in JUDGES}
    for j in JUDGES:
        apply_excess(ho, f"{j}_reg", tables[j])

    comp = (ho["kan"] == 0) & (ho["last"] == 0)
    J = {}
    for j in JUDGES:
        gh = per_game(ho, f"{j}_reg_ex", comp & (ho.source == "houou"))
        gj = per_game(ho, f"{j}_reg_ex", comp & (ho.source == "jade"))
        J[j] = diff(gj, gh)[0]
    print("calibration J (jade - houou): " + ", ".join(f"{j} {J[j]:+.5f}" for j in JUDGES))

    m = pd.read_csv(a.metrics, sep="\t", usecols=["path", "year", "n0", "n1", "n2", "n3"], dtype={f"n{i}": str for i in range(4)},
                    keep_default_na=False, na_values=[])
    held = set(ho.loc[ho.source == "houou", "rel"])

    pl = load(a.players)
    pl["v4_reg"] = pl["v4_regret"]
    apply_excess(pl, "v4_reg", tables["v4"])
    seats = pl.groupby(["game", "seat"]).agg(v=("v4_reg_ex", "mean"), rel=("rel", "first")).reset_index()
    assert not seats["rel"].isin(held).any(), "player sample overlaps the held-out games"
    names = m.set_index("path")[["n0", "n1", "n2", "n3"]]
    seats["name"] = [names.at[r, f"n{s}"] for r, s in zip(seats["rel"], seats["seat"])]
    seats = seats[~seats["name"].str.startswith("EXAMPLE")]
    g = seats.groupby("name")["v"].agg(["mean", "count", "var"])
    sw2 = g.loc[g["count"] >= 5, "var"].mean()
    tau2 = max(g.loc[g["count"] >= 5, "mean"].var() - (sw2 / g.loc[g["count"] >= 5, "count"]).mean(), 1e-12)
    k = sw2 / tau2
    g["score"] = g["count"] / (g["count"] + k) * g["mean"]
    rel25 = tau2 / (tau2 + sw2 / 25)
    print(f"player sample: {pl['game'].nunique():,} games, {len(seats):,} seats, {len(g):,} players; seat SD {np.sqrt(sw2):.4f}, "
          f"between-player SD {np.sqrt(tau2):.4f}, prior k {k:.1f} seats, reliability at 25 seats {rel25:.2f}")

    sc = g["score"]
    gs = np.zeros(len(m))
    for i in range(4):
        gs += m[f"n{i}"].map(sc).fillna(0.0).to_numpy()
    m["gscore"] = gs / 4
    order = np.argsort(m["gscore"].to_numpy(), kind="stable")
    sel = set(m["path"].to_numpy()[order[:N_SELECT]])
    q25 = set(m["path"].to_numpy()[order[: len(m) // 4]])
    q12 = set(m["path"].to_numpy()[order[: len(m) // 8]])
    pd.Series(sorted(sel)).to_csv(os.path.join(a.out_dir, "select_m3_1289661.txt"), index=False, header=False)
    pd.Series(sorted(q25)).to_csv(os.path.join(a.out_dir, "select_m3_q25.txt"), index=False, header=False)
    pd.Series(sorted(q12)).to_csv(os.path.join(a.out_dir, "select_m3_q12.txt"), index=False, header=False)
    y12 = m.loc[m["path"].isin(q12), "year"].value_counts().sort_index()
    print(f"best 12.5%: {len(q12):,} games; by year: " + ", ".join(f"{y}:{c:,}" for y, c in y12.items()))
    ys = m.loc[m["path"].isin(sel), "year"].value_counts().sort_index()
    print("selected 1,289,661 by year: " + ", ".join(f"{y}:{c:,}" for y, c in ys.items()))
    v11 = set(m.loc[m["year"] >= 2019, "path"])
    print(f"overlap with v11's set: {len(sel & v11):,} of {len(v11):,}")

    hg = ho[ho.source == "houou"][["game", "rel"]].drop_duplicates().merge(m[["path", "year", "gscore"]], left_on="rel", right_on="path")
    q = hg["gscore"].rank(pct=True, method="first")
    best, worst = set(hg.loc[q <= .25, "game"]), set(hg.loc[q > .75, "game"])
    print(f"\n== v6 rule 1 on {len(hg):,} held-out houou games: worst minus best quartile of game score (need >= J/3)")
    ok1 = True
    for j in JUDGES:
        dd, se, _, _ = diff(*(lambda x: (x[x.index.isin(worst)], x[x.index.isin(best)]))(per_game(ho, f"{j}_reg_ex")))
        ok = dd >= J[j] / 3 and dd - 1.96 * se > 0
        ok1 &= ok
        print(f"  {j}: {dd:+.5f} ± {1.96 * se:.5f} (z {dd / se:+.1f}); need >= {J[j] / 3:.5f} -> {'pass' if ok else 'fail'}")
    print(f"\n== v6 rule 4: selected minus v11's set on the held-out sample (need <= -J/5)")
    ok4 = True
    for label, s in (("best 1,289,661 (rule 3)", sel), ("best 25% (reported only)", q25), ("best 12.5% (v10, reported)", q12)):
        gsel = set(hg.loc[hg["path"].isin(s), "game"])
        gv11 = set(hg.loc[hg["year"] >= 2019, "game"])
        for j in JUDGES:
            x = per_game(ho, f"{j}_reg_ex")
            dd, se, na, nb = diff(x[x.index.isin(gsel)], x[x.index.isin(gv11)])
            ok = dd <= -J[j] / 5
            if label.startswith("best 1,289,661"):
                ok4 &= ok
            print(f"  {label:<26} {j}: {dd:+.5f} ± {1.96 * se:.5f} ({na}/{nb} games); need <= {-J[j] / 5:.5f} -> {'pass' if ok else 'fail'}")
    print(f"\nverdict: rule 1 {'PASS' if ok1 else 'fail'}, rule 4 {'PASS' if ok4 else 'fail'} -> "
          f"{'train the M3 arm' if ok1 and ok4 else 'do not train (pre-registered)'}")


if __name__ == "__main__":
    main()
