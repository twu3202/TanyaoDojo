"""Analysis for the data-quality study (patches/data_quality/README.md, sections v2 to v5).

    python quality_regret_analyze.py --npz A.npz B.npz --ctx A_ctx.npz B_ctx.npz --leverage leverage_houou.tsv \
        --metrics game_metrics_houou.tsv --scores game_scores_houou.tsv

Decisions come from quality_regret.py (one npz per worker). Steps:
  1. variants per decision: reg_j (v2 regret), tol_j (regret beyond the band delta_j), blunder (all three judges -
     v4, C', human-school hu - say avoid), sev (blunder x tol_v4); delta_j = 95th pct of judge j's cost of the other
     judges' choices where they disagree (houou);
  1b. v4: human-vs-AI disagreement types (human action class, AI action class, phase group) on AI-consensus
     decisions; realised cost per type from a regression of seat-kyoku adjusted point change on type counts, fitted
     on even-numbered games; "typed" = tol_v4 on confirmed-avoid types (z <= -2) only;
  2. excess = value - mean value of the decision's situation stratum (houou; sparse strata fall back to coarser);
  3. choose the primary variant: most negative z between a seat-kyoku's summed excess and its stratum-adjusted point
     change in that kyoku, on odd-numbered games (the types were confirmed on even ones); secondary: East-round
     value vs South-round point change per seat;
  4. calibration J = jade - houou mean excess (kyoku without a kan, not all-last) for the primary variant;
  5. rule 1 (worst minus best quartile >= J/3, CI excludes 0; per-judge variants must pass under both judges),
     exploratory M3 / judge agreement / the user's close-vs-lopsided view, rule 4 (selected set vs v11's <= -J/5).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

JUDGES = ("v4", "cp", "hu")
N_SELECT = 1_289_661
EDGES = [1, 4000, 8000, 16000]          # gap buckets: 0, <4k, <8k, <16k, >=16k
FINE = ["phase", "last", "rank", "ab", "bb", "kind"]
COARSE = ["phase", "last", "rank", "kind"]
FAMILIES = {"reg": ["v4_reg", "cp_reg"], "tol": ["v4_tol", "cp_tol"], "blunder": ["blunder"], "sev": ["sev"],
            "typed": ["typed"], "wtol": ["wtol"], "wsev": ["wsev"]}
CATS = ["open", "mid", "defence", "first_call", "call", "riichi", "agari", "other"]
STRAT = ["phase", "last", "rank", "ab", "bb"]
CLS = np.array(["discard"] * 37 + ["riichi", "chi", "chi", "chi", "pon", "kan", "agari", "draw", "pass"])
PHASE_GROUP = np.array(["E"] * 4 + ["S12"] * 2 + ["S34"] * 2 + ["W"])


def load(npzs):
    parts, off = [], 0
    for f in npzs:
        z = np.load(f, allow_pickle=False)
        d = pd.DataFrame({k: z[k] for k in z.files if k not in ("paths", "ok")})
        d["path"] = z["paths"][d["game"].to_numpy()]
        d["game"] += off
        off += len(z["paths"])
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["source"] = np.select([d["path"].str.contains("/houou/"), d["path"].str.contains("/throne/"),
                             d["path"].str.contains("/jade/")], ["houou", "throne", "jade"], "other")
    d["rel"] = d["path"].str.replace(r"^.*/houou/", "", regex=True)
    d["late"] = (d["phase"] >= 6) | (d["last"] == 1)
    d["e1"] = (d["phase"] == 0) & (d["kyoku"] == 0)
    d["ab"], d["bb"] = np.digitize(d["above"], EDGES), np.digitize(d["below"], EDGES)
    return d


def add_context(d, ctx_files, lev_file):
    c = pd.concat([pd.DataFrame({k: np.load(f)[k] for k in ("game", "seat", "kind_check", "nlegal", "at_turn",
                                                                "shanten", "opp_riichi", "melds")}).assign(part=i)
                   for i, f in enumerate(ctx_files)], ignore_index=True)
    assert len(c) == len(d) and (c["kind_check"].to_numpy() == d["act"].to_numpy()).all() \
        and (c["nlegal"].to_numpy() == d["nlegal"].to_numpy()).all() and (c["seat"].to_numpy() == d["seat"].to_numpy()).all(), \
        "context rows do not line up with the scoring rows"
    for k in ("at_turn", "shanten", "opp_riichi", "melds"):
        d[k] = c[k].to_numpy()
    kind = d["kind"].to_numpy()
    d["cat"] = np.select(
        [(kind == 0) & (d["opp_riichi"] >= 1), (kind == 0) & (d["at_turn"] <= 6) & (d["opp_riichi"] == 0) & (d["shanten"] >= 2),
         kind == 0, (kind == 2) & (d["melds"] == 0), kind == 2, kind == 1, kind == 3],
        ["defence", "open", "mid", "first_call", "call", "riichi", "agari"], "other")
    lev = pd.read_csv(lev_file, sep="\t")
    coarse = lev.dropna(subset=["leverage"]).groupby(["phase", "last", "rank"]).apply(
        lambda g: (g["leverage"] * g["n"]).sum() / g["n"].sum(), include_groups=False).rename("lc")
    x = d[STRAT].join(lev.set_index(STRAT)["leverage"], on=STRAT).join(coarse, on=["phase", "last", "rank"])
    lv = np.where(np.isnan(x["leverage"]), x["lc"], x["leverage"])
    d["lev"] = lv
    h = d["source"] == "houou"
    d["w_place"] = lv / np.nanmean(lv[h.to_numpy()])
    d["w_place"] = d["w_place"].fillna(1.0)
    print("  decision categories (houou): " + ", ".join(f"{k} {v:.1%}" for k, v in d.loc[h, "cat"].value_counts(normalize=True).items()))
    print(f"  placement weight w_place: houou quartiles {np.nanpercentile(d.loc[h, 'w_place'], [25, 50, 75]).round(2).tolist()}")
    return d


def add_category_weights(d):
    """v5: what does one unit of AI-judged error (tol_v4) cost in pt, per category? Fitted on even games."""
    sk = seat_kyoku(d, [], cats=True)
    even = (sk["game"] % 2 == 0).to_numpy()
    # v5.1: decision counts per category as controls, so a riichi error is compared with riichi decisions without one
    X = np.column_stack([np.ones(len(sk))] + [sk[f"tol_{c}"].to_numpy(float) for c in CATS]
                        + [sk[f"n_{c}"].to_numpy(float) for c in CATS if c != "other"])
    beta, se = ols_hc1(X[even], sk.loc[even, "adj_pt"].to_numpy(float))
    h = d[d["source"] == "houou"]
    n = h["cat"].value_counts()
    raw = {c: max(0.0, -beta[i + 1]) for i, c in enumerate(CATS)}
    norm = sum(raw[c] * n.get(c, 0) for c in CATS) / max(sum(n.get(c, 0) for c in CATS), 1)
    w = {c: (raw[c] / norm if norm > 0 else 1.0) for c in CATS}
    print("\n== v5 category weights: pt-relevant loss per unit of tol_v4 (even games), weight = max(0, -coef) / mean")
    print(f"  {'category':<11}{'decisions':>10}{'mean tol_v4':>13}{'coef':>10}{'se':>9}{'z':>7}{'weight':>8}")
    for i, c in enumerate(CATS):
        print(f"  {c:<11}{n.get(c, 0):>10,}{h.loc[h['cat'] == c, 'v4_tol'].mean():>13.4f}{beta[i + 1]:>10.3f}"
              f"{se[i + 1]:>9.3f}{beta[i + 1] / se[i + 1]:>7.1f}{w[c]:>8.2f}")
    d["w_cat"] = d["cat"].map(w).fillna(1.0)
    d["wtol"] = d["w_place"] * d["w_cat"] * d["v4_tol"]
    d["wsev"] = d["w_place"] * d["w_cat"] * d["sev"]
    return d


def add_variants(d):
    h = d["source"] == "houou"
    delta = {}
    for j in JUDGES:
        others = [k for k in JUDGES if k != j]
        x = d[[f"{j}_x_{k}" for k in others]].max(axis=1)
        dis = h & (x > 0)
        delta[j] = float(np.percentile(x[dis], 95))
        d[f"{j}_reg"] = d[f"{j}_regret"]
        d[f"{j}_tol"] = np.maximum(0.0, d[f"{j}_regret"] - delta[j])
        print(f"  {j}: disagrees with another judge on {dis.sum() / h.sum():.1%} of houou decisions; band {delta[j]:.3f}; "
              f"human = its greedy on {d.loc[h, f'{j}_hit'].mean():.1%}")
    d["blunder"] = np.logical_and.reduce([d[f"{j}_regret"] > delta[j] for j in JUDGES]).astype(float)
    d["sev"] = d["blunder"] * d["v4_tol"]
    for c in ("v4_tol", "cp_tol", "hu_tol", "blunder"):
        print(f"  share of houou decisions with {c} > 0: {(d.loc[h, c] > 0).mean():.2%}")
    return d


def add_types(d):
    """v4 layer 2: type the human-vs-AI disagreements and price them by realised kyoku outcomes."""
    cons = d["v4_best"] == d["cp_best"]
    pg = PHASE_GROUP[np.minimum(d["phase"].to_numpy(), 8)]
    d["ai_cls"] = CLS[d["v4_best"].to_numpy().astype(int)]
    d["hu_cls"] = CLS[d["act"].to_numpy().astype(int)]
    d["dev"] = cons & (d["act"] != d["v4_best"])
    d["type"] = np.where(d["dev"], d["hu_cls"] + "|ai:" + d["ai_cls"] + "|" + pg, "")
    h = d[(d["source"] == "houou")]
    opp = h[cons[h.index]].assign(pg=pg[h.index][cons[h.index].to_numpy()]).groupby(["ai_cls", "pg"]).size()
    counts = h.loc[h["dev"], "type"].value_counts()
    keep = list(counts[counts >= 300].index)
    sk = seat_kyoku(d, ["dev"], extra=keep)
    even = sk["game"] % 2 == 0
    # v5.1: control for the opportunity (how many AI-consensus decisions of each AI class the seat faced), so a type's
    # coefficient compares deviating with following the AI in the same kind of spot, not tenpai hands with all others
    opp_cols = [c for c in sk.columns if c.startswith("opp_")]
    X = np.column_stack([np.ones(len(sk))] + [sk[t].to_numpy(float) for t in keep] + [sk[c].to_numpy(float) for c in opp_cols])
    beta, se = ols_hc1(X[even.to_numpy()], sk.loc[even, "adj"].to_numpy(float))
    rows = []
    for i, t in enumerate(keep):
        hc, ac, g = t.split("|")
        ai = ac[3:]
        rate = counts[t] / opp.get((ai, g), np.nan)
        sub = h[h["type"] == t]
        rows.append((t, counts[t], rate, sub["v4_regret"].mean(), sub["cp_regret"].mean(), sub["hu_regret"].mean(),
                     beta[i + 1], se[i + 1]))
    tab = pd.DataFrame(rows, columns=["type", "n", "dev_rate", "reg_v4", "reg_cp", "reg_hu", "cost", "se"])
    tab["z"] = tab["cost"] / tab["se"]
    tab = tab.sort_values("z")
    pd.set_option("display.width", 200)
    print("\n== v4 layer 2: human-vs-AI disagreement types (AI = v4 and C' agree), realised cost per deviation "
          "(points, fitted on even games)")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    confirmed = set(tab.loc[tab["z"] <= -2, "type"])
    print(f"  confirmed avoid (z <= -2): {len(confirmed)} types, {tab.loc[tab['z'] <= -2, 'n'].sum():,} deviations; "
          f"treated as style: {len(tab) - len(confirmed)} types, {tab.loc[tab['z'] > -2, 'n'].sum():,} deviations")
    d["typed"] = np.where(d["type"].isin(confirmed), d["v4_tol"], 0.0)
    return d


def ols_hc1(X, y):
    # numpy 2 on macOS Accelerate raises spurious matmul warnings; results are checked against lstsq
    with np.errstate(all="ignore"):
        X, y = np.asarray(X, np.float64), np.asarray(y, np.float64)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        XtX_inv = np.linalg.pinv(X.T @ X)
        r = y - X @ beta
        meat = (X * (r ** 2)[:, None]).T @ X
        cov = XtX_inv @ meat @ XtX_inv * len(y) / (len(y) - X.shape[1])
    return beta, np.sqrt(np.diag(cov))


def seat_kyoku(d, sumcols, extra=(), cats=False):
    """houou seat-kyokus with their stratum-adjusted point change (adj, points) and pt-relevant loss (adj_pt =
    adj x the stratum's placement leverage); sums of sumcols, counts of each extra type, tol_v4 sums per category."""
    h = d[d["source"] == "houou"]
    agg = {"score": ("score", "first"), "phase": ("phase", "first"), "last": ("last", "first"),
           "rank": ("rank", "first"), "ab": ("ab", "first"), "bb": ("bb", "first"), "rel": ("rel", "first"),
           "lev": ("lev", "first")}
    agg.update({c: (c, "sum") for c in sumcols})
    sk = h.groupby(["game", "seat", "kyoku"]).agg(**agg).reset_index()
    if cats:
        tc = h.groupby(["game", "seat", "kyoku", "cat"])["v4_tol"].agg(["sum", "size"]).unstack(fill_value=0.0)
        tc.columns = [f"{'tol' if a == 'sum' else 'n'}_{c}" for a, c in tc.columns]
        sk = sk.join(tc, on=["game", "seat", "kyoku"])
        for c in CATS:
            for pre in ("tol", "n"):
                sk[f"{pre}_{c}"] = sk.get(f"{pre}_{c}", 0.0)
                sk[f"{pre}_{c}"] = sk[f"{pre}_{c}"].fillna(0.0)
    if extra:
        cons = h[h["v4_best"] == h["cp_best"]]
        oc = cons.assign(opp=cons["ai_cls"] + "|" + PHASE_GROUP[np.minimum(cons["phase"].to_numpy(), 8)]) \
            .groupby(["game", "seat", "kyoku", "opp"]).size().unstack(fill_value=0)
        oc.columns = [f"opp_{c}" for c in oc.columns]
        sk = sk.join(oc, on=["game", "seat", "kyoku"])
        sk[list(oc.columns)] = sk[list(oc.columns)].fillna(0)
    if extra:
        tc = h[h["type"].isin(extra)].groupby(["game", "seat", "kyoku", "type"]).size().unstack(fill_value=0)
        sk = sk.join(tc, on=["game", "seat", "kyoku"])
        sk[list(extra)] = sk[list(extra)].fillna(0)
    sk = sk.sort_values(["game", "seat", "kyoku"])
    nxt = sk.groupby(["game", "seat"])[["kyoku", "score"]].shift(-1)
    fin = np.array([FINAL.at[r, f"s{s}"] if r in FINAL.index else np.nan for r, s in zip(sk["rel"], sk["seat"])])
    end = np.where(sk["last"] == 1, fin, np.where(nxt["kyoku"] == sk["kyoku"] + 1, nxt["score"], np.nan))
    sk["delta"] = end - sk["score"]
    sk = sk.dropna(subset=["delta"])
    sk["adj"] = sk["delta"] - sk.groupby(["phase", "last", "rank", "ab", "bb"])["delta"].transform("mean")
    sk["adj_pt"] = sk["adj"] / 1000.0 * sk["lev"].fillna(np.nanmean(sk["lev"]))
    return sk


def add_excess(d, cols):
    h = d[d["source"] == "houou"]
    for col in cols:
        f = h.groupby(FINE)[col].agg(["mean", "count"]).rename(columns={"mean": "mf", "count": "cf"})
        c = h.groupby(COARSE)[col].mean().rename("mc")
        x = d[FINE].join(f, on=FINE).join(c, on=COARSE)
        base = np.where(x["cf"].fillna(0).to_numpy() >= 30, x["mf"], x["mc"])
        base = np.where(np.isnan(base), h[col].mean(), base)
        d[f"{col}_ex"] = d[col].to_numpy() - base
    return d


def per_game(d, col, mask=None):
    x = d if mask is None else d[mask]
    return x.groupby("game")[col].agg(["sum", "count"])


def ratio(g):
    s, n = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    p = s.sum() / n.sum()
    var = ((s - p * n) ** 2).sum() / (len(n) * (len(n) - 1)) / n.mean() ** 2
    return p, var, len(n)


def diff(ga, gb):
    pa, va, na = ratio(ga)
    pb, vb, nb = ratio(gb)
    return pa - pb, np.sqrt(va + vb), na, nb


def kyoku_validation(d, m, cols):
    h = d[d["source"] == "houou"]
    final = FINAL
    sk = seat_kyoku(d, [f"{c}_ex" for c in cols])
    sk = sk[sk["game"] % 2 == 1]
    out = {}
    print(f"\n== primary choice (odd games): seat-kyoku summed excess vs pt-relevant loss "
          f"(adjusted point change x leverage; raw points in brackets) ({len(sk):,} seat-kyokus)")
    for c in cols:
        ok = sk[f"{c}_ex"].std() > 0
        r = np.corrcoef(sk[f"{c}_ex"], sk["adj_pt"])[0, 1] if ok else 0.0
        r2 = np.corrcoef(sk[f"{c}_ex"], sk["adj"])[0, 1] if ok else 0.0
        z = r * np.sqrt(len(sk) - 3)
        out[c] = z
        print(f"  {c:<8} r {r:+.4f}  z {z:+.1f}   [points: r {r2:+.4f}  z {r2 * np.sqrt(len(sk) - 3):+.1f}]")
    # secondary: East-round value -> South-round change, per seat
    seat = h.groupby(["game", "seat"]).agg(rel=("rel", "first"))
    east = h[h["phase"] <= 3].groupby(["game", "seat"])[[f"{c}_ex" for c in cols]].mean()
    s1 = h[h["phase"] >= 4].sort_values("kyoku").groupby(["game", "seat"]).agg(s1=("score", "first"), r1=("rank", "first"))
    ps = seat.join(east, how="inner").join(s1, how="inner").reset_index()
    ps["final"] = [final.at[r, f"s{s}"] if r in final.index else np.nan for r, s in zip(ps["rel"], ps["seat"])]
    ps["chg"] = ps["final"] - ps["s1"]
    ps["adj"] = ps["chg"] - ps.groupby("r1")["chg"].transform("mean")
    ps = ps.dropna(subset=["adj"])
    print(f"  secondary, East-round mean excess vs South-round adjusted change ({len(ps):,} seats):")
    for c in cols:
        r = np.corrcoef(ps[f"{c}_ex"], ps["adj"])[0, 1]
        print(f"    {c:<8} r {r:+.4f}  z {r * np.sqrt(len(ps) - 3):+.1f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--ctx", nargs="+", required=True)
    ap.add_argument("--leverage", required=True)
    a = ap.parse_args()
    global FINAL
    m = pd.read_csv(a.metrics, sep="\t", dtype={f"n{i}": str for i in range(4)}, keep_default_na=False, na_values=[])
    FINAL = m.set_index("path")[["s0", "s1", "s2", "s3"]]
    d = load(a.npz)
    print("== context and placement leverage")
    d = add_context(d, a.ctx, a.leverage)
    print("== tolerance bands")
    d = add_category_weights(add_types(add_variants(d)))
    allcols = [c for f in FAMILIES.values() for c in f] + ["hu_reg", "hu_tol"]
    d = add_excess(d, allcols)
    print(f"decisions {len(d):,}; games by source: " +
          ", ".join(f"{k} {v:,}" for k, v in d.groupby("source")["game"].nunique().items()))
    hh = d["source"] == "houou"
    for c in allcols:
        print(f"  {c:<8} houou mean {d.loc[hh, c].mean():.4f}; share from late-game decisions "
              f"{d.loc[hh & d.late, c].sum() / max(d.loc[hh, c].sum(), 1e-12):.1%} (late = {d.loc[hh, 'late'].mean():.1%} of decisions)")

    z = kyoku_validation(d, m, allcols)
    fam = min(FAMILIES, key=lambda f: min(z[c] for c in FAMILIES[f]) if len(FAMILIES[f]) == 1 else max(z[c] for c in FAMILIES[f]))
    cols = FAMILIES[fam]
    print(f"  -> primary variant family: {fam} ({', '.join(cols)}); per-judge families use the weaker judge's z")

    print("\n== calibration (kyoku without a kan, not all-last): mean excess by source")
    comp = (d["kan"] == 0) & (d["last"] == 0)
    J = {}
    for c in cols:
        gh = per_game(d, f"{c}_ex", comp & (d.source == "houou"))
        J[c] = float("nan")
        for src in ("throne", "jade"):
            gs = per_game(d, f"{c}_ex", comp & (d.source == src))
            if len(gs) < 2:
                print(f"  {c}: no {src} games yet - calibration pending, thresholds unavailable")
                continue
            dd, se, na, nb = diff(gs, gh)
            print(f"  {c}: {src} - houou = {dd:+.5f} ± {1.96 * se:.5f} ({na}/{nb} games)")
            if src == "jade":
                J[c] = dd
    print("  thresholds: rule 1 J/3: " + ", ".join(f"{c} {J[c] / 3:.5f}" for c in cols) +
          "; rule 4 J/5: " + ", ".join(f"{c} {J[c] / 5:.5f}" for c in cols))

    sc = pd.read_csv(a.scores, sep="\t", usecols=["path", "m2_mean", "m2_weakest"])
    mm = m[["path", "year", "final_spread", "gap12", "gap34", "mean_spread", "busted"]].merge(sc, on="path")
    hg = d[hh][["game", "rel"]].drop_duplicates().merge(mm, left_on="rel", right_on="path")
    print(f"\n== rule 1 on {len(hg):,} houou games: worst quartile minus best quartile of mean excess "
          f"(positive = the metric's 'worse' games really contain more error)")
    tests = [("final_spread", "M1 final spread (small = close = best)"), ("gap34", "M1 one-low gap 3rd-4th"),
             ("gap12", "M1 one-high gap 1st-2nd"), ("mean_spread", "M1 mean spread at kyoku starts"),
             ("m2_mean", "M2 player strength (lower avg rank = best)"), ("m2_weakest", "M2 weakest player (exploratory)")]
    passed = []
    for col, label in tests:
        q = hg[col].rank(pct=True, method="first")
        best, worst = set(hg.loc[q <= .25, "game"]), set(hg.loc[q > .75, "game"])
        ok, lines = True, []
        for c in cols:
            for view, mask in (("all", None), ("late", d["late"])):
                g = per_game(d, f"{c}_ex", mask)
                dd, se, _, _ = diff(g[g.index.isin(worst)], g[g.index.isin(best)])
                lines.append(f"    {c:<8} {view:<4} {dd:+.5f} ± {1.96 * se:.5f}  (z {dd / se:+.1f})")
                if view == "all":
                    ok &= (dd >= J[c] / 3) and (dd - 1.96 * se > 0)
        g = per_game(d, "v4_hit", d["e1"])
        dd, se, _, _ = diff(g[g.index.isin(best)], g[g.index.isin(worst)])
        lines.append(f"    v1 East-1 agreement best - worst: {100 * dd:+.2f} ± {196 * se:.2f} pp")
        print(f"  {label}: {'PASS' if ok else 'fail'}")
        print("\n".join(lines))
        if ok:
            passed.append(col)

    print("\n== exploratory: how much do players differ? (M3, per-name mean excess in the sample)")
    names = m.set_index("path")[["n0", "n1", "n2", "n3"]]
    for c in cols:
        ps = d[hh].groupby(["game", "seat"]).agg(v=(f"{c}_ex", "mean"), rel=("rel", "first")).reset_index()
        ps["name"] = [names.at[r, f"n{s}"] if r in names.index else None for r, s in zip(ps["rel"], ps["seat"])]
        multi = ps.groupby("name")["v"].agg(["mean", "count", "var"])
        multi = multi[multi["count"] >= 3]
        tau2 = multi["mean"].var() - (multi["var"] / multi["count"]).mean()
        print(f"  {c}: seat-level SD {ps['v'].std():.5f}; players with >= 3 seats: {len(multi)}; between-player SD "
              f"of true mean ≈ {np.sqrt(max(tau2, 0)):.5f} (J = {J[c]:.5f})")
    if len(cols) == 2:
        g0, g1 = per_game(d[hh], f"{cols[0]}_ex"), per_game(d[hh], f"{cols[1]}_ex")
        gm = (g0["sum"] / g0["count"]).to_frame("a").join((g1["sum"] / g1["count"]).rename("b"))
        print(f"  judge agreement on game-level mean excess: Pearson {gm.corr().iloc[0, 1]:.3f}")

    print("\n== the user's view on late-game decisions: close (final spread < 25th pct) minus lopsided (> 75th pct)")
    q = hg["final_spread"].rank(pct=True, method="first")
    close, lop = set(hg.loc[q <= .25, "game"]), set(hg.loc[q > .75, "game"])
    for c in cols:
        r1 = diff(*(lambda g: (g[g.index.isin(close)], g[g.index.isin(lop)]))(per_game(d, c, d["late"])))
        r2 = diff(*(lambda g: (g[g.index.isin(close)], g[g.index.isin(lop)]))(per_game(d, f"{c}_ex", d["late"])))
        print(f"  {c}: raw {r1[0]:+.5f} ± {1.96 * r1[1]:.5f}; situation-adjusted {r2[0]:+.5f} ± {1.96 * r2[1]:.5f}")

    for col in passed:
        full = pd.read_csv(a.scores, sep="\t", usecols=["path", col])
        cut = full[col].nsmallest(N_SELECT).max()
        sel = set(hg.loc[hg[col] <= cut, "game"])
        v11 = set(hg.loc[hg["year"] >= 2019, "game"])
        print(f"\n== rule 4 for {col}: selected {N_SELECT:,} minus v11's set")
        for c in cols:
            g = per_game(d, f"{c}_ex")
            dd, se, _, _ = diff(g[g.index.isin(sel)], g[g.index.isin(v11)])
            print(f"  {c}: {dd:+.5f} ± {1.96 * se:.5f} (need <= {-J[c] / 5:.5f})")


if __name__ == "__main__":
    main()
