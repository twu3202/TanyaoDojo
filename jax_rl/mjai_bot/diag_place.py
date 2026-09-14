"""顺位差距定位:点数流与"把点数变成顺位点"的能力,分开量。

**动机(2026-09-14)**。λ=0.1 的 368,000 步快照在 s14000 上攻 +5 / 守 −4 点/局(diag_gap),
按点数流几乎追平 v4,avg_pt 却仍是 −0.77。diag_gap 的攻/守只算和了得点与放铳失点,
不含被自摸、立直押金、流局罚符,也完全不看顺位。所以要回答:差距是在**点数**上,还是在**把点数换成顺位**上。

**两个拆分**(我们的座位 vs 同局三个 mortal-v4 座位的平均):
  1. 点数流按阶段:东场 / 南 1-3 局 / 南四及以后(all-last),含一切收支(以 start_kyoku 的权威 scores 为准)。
  2. 顺位点按"进入南四时的名次 r"拆:
       E_ours[pt] − E_v4[pt] = Σ_r (P_ours(r) − P_v4(r))·m_v4(r)   ← 进南四时的位置优势(前面打出来的)
                             + Σ_r P_ours(r)·(m_ours(r) − m_v4(r))  ← 南四的兑现能力(同名次进场,谁收尾更好)
     未进入南四就结束(被飞)的对局单列一类 r=0。两项之和是恒等式。
     因为四席顺位点之和为 0,我们与 v4 座位均值之差 = 4/3 × avg_pt;输出统一乘 3/4 折回 avg_pt 口径,两项相加即 avg_pt。

**自检**:按局重算的 avg_pt 必须与 group_ci / run_eval 一致(终局分沿用 group_ci.final_scores,含立直押金与残留供托)。

用法: python diag_place.py --log-dir DIR --seed-lo 14000 --seed-hi 15999 --expect-games 8000 [--dump x.json]
      python diag_place.py --compare a.json b.json ...
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re

from group_ci import final_scores, PTS, SUFFIX_SEAT

NAME_RE = re.compile(r"(\d+)_(\d+)_([a-d])\.json\.gz$")
PHASES = ("east", "south13", "alllast")


def rank_of(sc, seat):
    return sorted(range(4), key=lambda i: (-sc[i], i)).index(seat)


def scan(path):
    """返回 dict: 各阶段起点分数 + 终局分数。"""
    marks = {}          # 'south' -> S1 开局分, 'alllast' -> 首个南四开局分
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if '"start_kyoku"' not in line:
                continue
            ev = json.loads(line)
            bk, ky = ev["bakaze"], int(ev["kyoku"])
            if bk != "E" and "south" not in marks:
                marks["south"] = list(ev["scores"])
            if (bk == "S" and ky == 4 or bk not in ("E", "S")) and "alllast" not in marks:
                marks["alllast"] = list(ev["scores"])
    return marks, final_scores(path)


def blank():
    return dict(n=0, pt=0.0, rank=[0, 0, 0, 0], pts={p: 0.0 for p in PHASES},
                entry_n=[0] * 5, entry_pt=[0.0] * 5,     # entry 下标: 0=未进南四, 1..4=进南四时名次
                games=[])                                # 仅我们的座位: (seed, entry, final_rank)


def add(acc, seat, marks, fin, seed=None):
    r = rank_of(fin, seat)
    acc["n"] += 1
    acc["pt"] += PTS[r]
    acc["rank"][r] += 1
    start = [25000] * 4
    s_south = marks.get("south", fin)
    s_all = marks.get("alllast", fin if "south" in marks else s_south)
    acc["pts"]["east"] += s_south[seat] - start[seat]
    acc["pts"]["south13"] += s_all[seat] - s_south[seat]
    acc["pts"]["alllast"] += fin[seat] - s_all[seat]
    e = rank_of(marks["alllast"], seat) + 1 if "alllast" in marks else 0
    acc["entry_n"][e] += 1
    acc["entry_pt"][e] += PTS[r]
    if seed is not None:
        acc["games"].append((seed, e, r))


def split_ci(d):
    """两项的组层 95% CI:m_v4(r) 视为固定(由 3 倍局数估出),我们每局贡献
    conv_i = pt_i − m_v4(e_i),entry_i = m_v4(e_i) − E_v4,按 seed 组(复式 4 局)求均值再求 SE。"""
    import numpy as np
    o, v = d["ours"], d["v4"]
    mv = [v["entry_pt"][e] / v["entry_n"][e] if v["entry_n"][e] else 0.0 for e in range(5)]
    ev = v["pt"] / v["n"]
    by = {}
    for seed, e, r in o["games"]:
        c, en = PTS[r] - mv[e], mv[e] - ev
        a = by.setdefault(seed, [0.0, 0.0, 0])
        a[0] += c; a[1] += en; a[2] += 1
    g = np.array([[a[0] / a[2], a[1] / a[2]] for a in by.values()]) * 0.75
    se = g.std(0, ddof=1) / np.sqrt(len(g))
    return 1.96 * se[1], 1.96 * se[0]


def by_entry(d):
    o, v = d["ours"], d["v4"]
    lines = [f"  {'进南四名次':<8}{'我们占比':>8}{'v4占比':>8}{'我们终局pt':>10}{'v4终局pt':>10}{'兑现差':>8}{'贡献(avg_pt)':>12}"]
    lab = ["未进南四", "1位", "2位", "3位", "4位"]
    for e in range(5):
        if not o["entry_n"][e] or not v["entry_n"][e]:
            continue
        po, pv = o["entry_n"][e] / o["n"], v["entry_n"][e] / v["n"]
        mo, mv = o["entry_pt"][e] / o["entry_n"][e], v["entry_pt"][e] / v["entry_n"][e]
        lines.append(f"  {lab[e]:<9}{po:>8.1%}{pv:>8.1%}{mo:>10.2f}{mv:>10.2f}{mo-mv:>+8.2f}{po*(mo-mv)*0.75:>+12.3f}")
    return "\n".join(lines)


def report(name, d):
    o, v = d["ours"], d["v4"]
    no, nv = o["n"], v["n"]
    s = 0.75
    avg = o["pt"] / no
    rk = lambda a, i: a["rank"][i] / a["n"] * 100
    pts = {p: (o["pts"][p] / no - v["pts"][p] / nv) for p in PHASES}
    entry = conv = 0.0
    for e in range(5):
        if v["entry_n"][e] == 0:
            continue
        mv = v["entry_pt"][e] / v["entry_n"][e]
        po, pv = o["entry_n"][e] / no, v["entry_n"][e] / nv
        entry += (po - pv) * mv
        if o["entry_n"][e]:
            conv += po * (o["entry_pt"][e] / o["entry_n"][e] - mv)
    ci = f" ±{split_ci(d)[0]:.2f} ±{split_ci(d)[1]:.2f}" if o.get("games") else ""
    return (f"{name:<34}{no:>6} {avg:+7.3f} | {rk(o,0)-rk(v,0):+6.2f} {rk(o,3)-rk(v,3):+6.2f} |"
            f" {pts['east']:+6.0f} {pts['south13']:+6.0f} {pts['alllast']:+6.0f} {sum(pts.values()):+6.0f} |"
            f" {entry*s:+7.3f} {conv*s:+7.3f}{ci}")


HEADER = (f"{'snapshot':<34}{'局数':>5} {'avg_pt':>7} | {'1位pp':>6} {'4位pp':>6} |"
          f" {'东场':>5} {'南1-3':>5} {'南4+':>5} {'合计':>5} |  {'进南四位置':>6} {'南四兑现':>6}\n"
          + " " * 48 + "(相对同局 v4 座位, 点/局对局)          (折成 avg_pt, 两项和=avg_pt)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir")
    ap.add_argument("--seed-lo", type=int, default=14000)
    ap.add_argument("--seed-hi", type=int, default=15999)
    ap.add_argument("--expect-games", type=int, default=0)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--compare", nargs="*")
    args = ap.parse_args()

    if args.compare:
        print(HEADER)
        ds = [(os.path.basename(p).removesuffix(".json"), json.load(open(p))) for p in args.compare]
        for name, d in ds:
            print(report(name, d))
        for name, d in ds:
            print(f"\n{name}\n{by_entry(d)}")
        return

    ours, v4 = blank(), blank()
    for p in sorted(glob.glob(os.path.join(args.log_dir, "*.json.gz"))):
        m = NAME_RE.search(os.path.basename(p))
        if not m or not (args.seed_lo <= int(m.group(1)) <= args.seed_hi):
            continue
        seat = SUFFIX_SEAT[m.group(3)]
        marks, fin = scan(p)
        add(ours, seat, marks, fin, seed=int(m.group(1)))
        for i in range(4):
            if i != seat:
                add(v4, i, marks, fin)
    if args.expect_games and ours["n"] != args.expect_games:
        raise SystemExit(f"局数 {ours['n']:,} ≠ 预期 {args.expect_games:,}")
    d = {"ours": ours, "v4": v4}
    print(f"avg_pt(按局重算) = {ours['pt'] / ours['n']:+.3f}   ← 须与 group_ci 一致")
    print(HEADER)
    print(report(os.path.basename(args.log_dir.rstrip('/')), d))
    if args.dump:
        json.dump(d, open(args.dump, "w"), indent=1)


if __name__ == "__main__":
    main()
