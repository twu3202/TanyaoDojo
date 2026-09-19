"""提前终局(有人被飞)对局的拆解:是"被飞时打法有问题",还是"到那一刻我们本来就落后"?

100k 顺位拆解显示:缺口几乎全在"没打到南四就结束"的对局(-1.1 ~ -1.3 avg_pt),而攻守点数流已追平 v4。
但阶段点数流又显示我们在东场与南 1-3 一路落后、靠南四找补 —— 若如此,提前终局只是**剥夺了找补机会**,
真正的问题在中前盘,而不是被飞本身。本脚本把提前终局的对局按"谁被飞"拆开,并报那一刻的分数,用来区分这两种解释:
  · 我们被飞的对局:我们的顺位点几乎必然是 -135,看占比;
  · 别人被飞的对局:我们没被飞,顺位由当时分数决定 —— 若这里也明显差,说明是"到那时就落后",与被飞行为无关。
另报所有对局在"南四开始前"的分数差,作为中前盘落后的直接读数。

用法: diag_early.py --log-dir DIR --seed-lo .. --seed-hi .. --expect-games N [--dump x.json]
      diag_early.py --compare a.json b.json ...
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


def scan(path):
    pre_s4, early = None, True
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if '"start_kyoku"' not in line:
                continue
            ev = json.loads(line)
            if (ev["bakaze"] == "S" and int(ev["kyoku"]) == 4) or ev["bakaze"] not in ("E", "S"):
                if pre_s4 is None:
                    pre_s4 = list(ev["scores"])
                early = False
    return early, pre_s4, final_scores(path)


def blank():
    return dict(n=0, pre_n=0, pre_score=0, early_self=0, early_self_pt=0.0,
                early_opp=0, early_opp_pt=0.0, early_opp_score=0, late_pt=0.0, late_n=0)


def add(acc, seat, early, pre_s4, fin):
    acc["n"] += 1
    pt = PTS[sorted(range(4), key=lambda i: (-fin[i], i)).index(seat)]
    if pre_s4 is not None:
        acc["pre_n"] += 1
        acc["pre_score"] += pre_s4[seat]
    if early:
        if fin[seat] < 0:
            acc["early_self"] += 1; acc["early_self_pt"] += pt
        else:
            acc["early_opp"] += 1; acc["early_opp_pt"] += pt; acc["early_opp_score"] += fin[seat]
    else:
        acc["late_n"] += 1; acc["late_pt"] += pt


def report(name, d):
    o, v = d["ours"], d["v4"]
    f = lambda a, k, m: a[k] / max(a[m], 1)
    return (f"{name:<34}{o['n']:>7}"
            f" | 自己被飞 {o['early_self']/o['n']:>6.2%}/{v['early_self']/v['n']:>6.2%}"
            f" | 别人被飞局 占比 {o['early_opp']/o['n']:>6.2%} 我们 pt {f(o,'early_opp_pt','early_opp'):>+7.2f}"
            f" v4 {f(v,'early_opp_pt','early_opp'):>+7.2f} 差 {f(o,'early_opp_pt','early_opp')-f(v,'early_opp_pt','early_opp'):>+6.2f}"
            f" 终局分 {f(o,'early_opp_score','early_opp'):>7.0f}/{f(v,'early_opp_score','early_opp'):>7.0f}"
            f" | 打满局 pt {f(o,'late_pt','late_n'):>+6.2f}/{f(v,'late_pt','late_n'):>+6.2f}"
            f" | 南四前分数 {f(o,'pre_score','pre_n'):>7.0f}/{f(v,'pre_score','pre_n'):>7.0f}"
            f" 差 {f(o,'pre_score','pre_n')-f(v,'pre_score','pre_n'):>+6.0f}")


HEADER = ("snapshot                            局数 | 被飞率 我们/v4 | 别人被飞的对局(我们没被飞) | 打满南四的对局 | 进南四前的分数")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir"); ap.add_argument("--seed-lo", type=int, default=10000)
    ap.add_argument("--seed-hi", type=int, default=99999); ap.add_argument("--expect-games", type=int, default=0)
    ap.add_argument("--dump"); ap.add_argument("--compare", nargs="*")
    args = ap.parse_args()
    if args.compare:
        print(HEADER)
        for p in args.compare:
            print(report(os.path.basename(p).removesuffix(".json"), json.load(open(p))))
        return
    ours, v4 = blank(), blank()
    for p in sorted(glob.glob(os.path.join(args.log_dir, "*.json.gz"))):
        m = NAME_RE.search(os.path.basename(p))
        if not m or not (args.seed_lo <= int(m.group(1)) <= args.seed_hi):
            continue
        seat = SUFFIX_SEAT[m.group(3)]
        early, pre_s4, fin = scan(p)
        add(ours, seat, early, pre_s4, fin)
        for i in range(4):
            if i != seat:
                add(v4, i, early, pre_s4, fin)
    if args.expect_games and ours["n"] != args.expect_games:
        raise SystemExit(f"局数 {ours['n']:,} ≠ 预期 {args.expect_games:,}")
    d = {"ours": ours, "v4": v4}
    print(HEADER); print(report(os.path.basename(args.log_dir.rstrip("/")), d))
    if args.dump:
        json.dump(d, open(args.dump, "w"), indent=1)


if __name__ == "__main__":
    main()
