"""被飞(点数 < 0 提前终局)定位 —— diag_place 显示差距主要落在"没打到南四就结束"的约 22% 对局里。

量三件事(我们的座位 vs 同局 mortal-v4 座位的每席平均):
  1. 提前终局的对局里:谁被飞(终局分 < 0)、谁拿第四;我们作为"飞人者"(本局和了且有人 < 0)的比例。
  2. 条件被飞率:某局开局时点数处于低分段(< 8000 / 8000-15000),这一局结束时被飞的概率。
     这是"已经低分之后怎么活下来",排除了"有多常掉到低分"的差异。
  3. 各低分段的进入频率(每局对局有多少个"开局低分"的局),看是"更常掉进低分"还是"低分时更容易死"。
终局分沿用 group_ci.final_scores(含立直押金);被飞以终局分 < 0 判定。

用法: python diag_bust.py --log-dir DIR --seed-lo 14000 --seed-hi 15999 --expect-games 8000 [--dump x.json]
      python diag_bust.py --compare a.json b.json ...
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re

from group_ci import final_scores, SUFFIX_SEAT

NAME_RE = re.compile(r"(\d+)_(\d+)_([a-d])\.json\.gz$")
BANDS = ((-10**9, 8000, "<8k"), (8000, 15000, "8-15k"))


def scan(path):
    """返回 (是否提前终局, 终局分, 每局开局分数列表[含每局结束分数])。"""
    starts, early = [], True
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if '"start_kyoku"' not in line:
                continue
            ev = json.loads(line)
            starts.append(list(ev["scores"]))
            if ev["bakaze"] == "S" and int(ev["kyoku"]) == 4 or ev["bakaze"] not in ("E", "S"):
                early = False
    fin = final_scores(path)
    return early, fin, starts + [fin]


def blank():
    return dict(n=0, early=0, early_bust=0, early_4th=0, bust=0,
                band_n={b[2]: 0 for b in BANDS}, band_bust={b[2]: 0 for b in BANDS})


def add(acc, seat, early, fin, seq):
    acc["n"] += 1
    rank = sorted(range(4), key=lambda i: (-fin[i], i)).index(seat)
    if fin[seat] < 0:
        acc["bust"] += 1
    if early:
        acc["early"] += 1
        acc["early_bust"] += int(fin[seat] < 0)
        acc["early_4th"] += int(rank == 3)
    for k in range(len(seq) - 1):
        s0, s1 = seq[k][seat], seq[k + 1][seat]
        for lo, hi, lab in BANDS:
            if lo <= s0 < hi:
                acc["band_n"][lab] += 1
                acc["band_bust"][lab] += int(s1 < 0)


def report(name, d):
    o, v = d["ours"], d["v4"]
    f = lambda a, k: a[k] / a["n"] * 100
    cols = [f"{name:<36}{o['n']:>6}",
            f"被飞 {f(o,'bust'):5.2f}/{f(v,'bust'):5.2f}%",
            f"提前终局中拿4位 {o['early_4th']/max(o['early'],1)*100:5.1f}/{v['early_4th']/max(v['early'],1)*100:5.1f}%"]
    for _, _, lab in BANDS:
        no, nv = o["band_n"][lab] / o["n"], v["band_n"][lab] / v["n"]
        bo = o["band_bust"][lab] / max(o["band_n"][lab], 1) * 100
        bv = v["band_bust"][lab] / max(v["band_n"][lab], 1) * 100
        cols.append(f"{lab}: 每对局{no:.2f}/{nv:.2f}局 该局被飞{bo:5.2f}/{bv:5.2f}%")
    return "  ".join(cols)


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
        print("(我们/v4 每席)")
        for p in args.compare:
            print(report(os.path.basename(p).removesuffix(".json"), json.load(open(p))))
        return
    ours, v4 = blank(), blank()
    for p in sorted(glob.glob(os.path.join(args.log_dir, "*.json.gz"))):
        m = NAME_RE.search(os.path.basename(p))
        if not m or not (args.seed_lo <= int(m.group(1)) <= args.seed_hi):
            continue
        seat = SUFFIX_SEAT[m.group(3)]
        early, fin, seq = scan(p)
        add(ours, seat, early, fin, seq)
        for i in range(4):
            if i != seat:
                add(v4, i, early, fin, seq)
    if args.expect_games and ours["n"] != args.expect_games:
        raise SystemExit(f"局数 {ours['n']:,} ≠ 预期 {args.expect_games:,}")
    d = {"ours": ours, "v4": v4}
    print(f"提前终局 {ours['early']/ours['n']:.1%}")
    print(report(os.path.basename(args.log_dir.rstrip('/')), d))
    if args.dump:
        json.dump(d, open(args.dump, "w"), indent=1)


if __name__ == "__main__":
    main()
