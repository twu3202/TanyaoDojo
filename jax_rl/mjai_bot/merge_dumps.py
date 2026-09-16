"""把多段的 gapdump / placedump / bustdump 按计数相加, 合成更大样本的机制读数(各量都是计数或和, 可直接相加)。
用法: merge_dumps.py <输出名> <段名1> <段名2> ...   (段名 = 各 dump 目录里的文件名去掉 .json)"""
import json, sys, os
R = os.path.expanduser("~/Projects/better_mortal/runs")
out, parts = sys.argv[1], sys.argv[2:]

def add(a, b):
    if isinstance(a, dict):
        return {k: add(a[k], b[k]) for k in a}
    if isinstance(a, list):
        if a and isinstance(a[0], list):          # placedump 的 games: [(seed, entry, rank), ...] 拼接
            return a + b
        if not a and b and isinstance(b[0], list):
            return b
        return [x + y for x, y in zip(a, b)] if len(a) == len(b) and (not a or not isinstance(a[0], list)) else a + b
    return a + b

for kind in ("gapdump", "placedump", "bustdump"):
    acc = None
    for p in parts:
        j = json.load(open(f"{R}/{kind}/{p}.json"))
        acc = j if acc is None else add(acc, j)
    json.dump(acc, open(f"{R}/{kind}/{out}.json", "w"))
    n = acc.get("n_games") or acc["ours"]["n"]
    print(f"{kind}/{out}.json  games={n}")
