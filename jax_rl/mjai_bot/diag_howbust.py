"""被飞是怎么发生的:放铳 / 被自摸 / 流局罚符 / 立直棒。

100k 拆解显示我们的缺口几乎全在"自己被飞"(比 v4 多 0.4~0.8pp, 每次 -135 点),而打满南四的对局我们是赢的。
要决定该改什么(少推?少立直?低分时别鸣?),先看压垮那一局的是哪种支出:
  deal_in  = 我们放铳(hora 的 target 是我们)
  tsumo    = 别人自摸我们付
  ryukyoku = 流局罚符
只统计"把我们打到 < 0 的那一局"。同时报那一局开局时我们的点数。

用法: diag_howbust.py --log-dir DIR --seed-lo .. --seed-hi .. [--dump x.json] | --compare a.json b.json
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
from collections import defaultdict

from group_ci import SUFFIX_SEAT

NAME_RE = re.compile(r"(\d+)_(\d+)_([a-d])\.json\.gz$")


def scan(path, seat):
    """返回 (是否被飞, 致命那局的类型, 该局开局我们的点数) —— 逐局重算分数, 与 group_ci 同口径(含立直押金)。"""
    sc = None; reaches = [0] * 4; kind = None; start = None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ev = json.loads(line)
            t = ev.get("type")
            if t == "start_kyoku":
                sc = list(ev["scores"]); reaches = [0] * 4; start = sc[seat]
            elif t == "reach_accepted":
                reaches[int(ev["actor"])] += 1
            elif t in ("hora", "ryukyoku") and sc is not None:
                d = ev.get("deltas", [0] * 4)
                for i in range(4):
                    sc[i] += int(d[i])
                if t == "hora":
                    a, tg = int(ev["actor"]), int(ev["target"])
                    k = "deal_in" if (tg == seat and a != seat) else ("tsumo" if a != seat and a == tg else "other")
                else:
                    k = "ryukyoku"
                if sc[seat] - 1000 * reaches[seat] < 0:
                    return True, k, start
    return False, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir"); ap.add_argument("--seed-lo", type=int, default=10000)
    ap.add_argument("--seed-hi", type=int, default=99999); ap.add_argument("--dump"); ap.add_argument("--compare", nargs="*")
    args = ap.parse_args()
    if args.compare:
        print(f"  {'snapshot':<26}{'对局':>8}{'被飞率':>8}  " + "  ".join(f"{k:>16}" for k in ("放铳", "被自摸", "流局罚符")) + "   致命局开局分")
        for p in args.compare:
            d = json.load(open(p)); o, v = d["ours"], d["v4"]
            row = f"  {os.path.basename(p).removesuffix('.json'):<26}{o['n']:>8,}{o['bust']/o['n']:>8.2%}  "
            for k in ("deal_in", "tsumo", "ryukyoku"):
                row += f"{o['kind'].get(k,0)/o['n']:>7.2%}/{v['kind'].get(k,0)/v['n']:>7.2%}  "
            row += f"  {o['start_sum']/max(o['bust'],1):>7.0f}/{v['start_sum']/max(v['bust'],1):>7.0f}"
            print(row)
        return
    acc = {w: dict(n=0, bust=0, kind=defaultdict(int), start_sum=0) for w in ("ours", "v4")}
    for p in sorted(glob.glob(os.path.join(args.log_dir, "*.json.gz"))):
        m = NAME_RE.search(os.path.basename(p))
        if not m or not (args.seed_lo <= int(m.group(1)) <= args.seed_hi):
            continue
        our = SUFFIX_SEAT[m.group(3)]
        for i in range(4):
            a = acc["ours"] if i == our else acc["v4"]
            a["n"] += 1
            busted, kind, start = scan(p, i)
            if busted:
                a["bust"] += 1; a["kind"][kind] += 1; a["start_sum"] += start
    d = {w: dict(n=a["n"], bust=a["bust"], kind=dict(a["kind"]), start_sum=a["start_sum"]) for w, a in acc.items()}
    print(json.dumps({w: {k: v for k, v in x.items() if k != "kind"} | {"kind": x["kind"]} for w, x in d.items()}, ensure_ascii=False))
    if args.dump:
        json.dump(d, open(args.dump, "w"))


if __name__ == "__main__":
    main()
