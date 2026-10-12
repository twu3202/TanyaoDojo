"""Placement leverage by situation (data-quality study v5, patches/data_quality/README.md). No model.

For every seat at the start of every kyoku in the pool: how much do points won or lost in this kyoku move the seat's
final placement pt? Within each situation stratum (phase E1..S4 / West, all-last, rank at kyoku start, gap to the
player above, gap to the player below), leverage = slope of final pt on the kyoku's point change, in pt per 1,000
points. A runaway leader's kyoku hardly moves its final pt (low leverage); a close all-last fight moves it a lot.

pt = Tenhou houou [90, 45, 0, -135] (the project's evaluation pt), ties by seat order from the first dealer.

    python placement_leverage.py DATA_ROOT OUT.tsv [--procs 6]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

PT = [90, 45, 0, -135]
EDGES = [1, 4000, 8000, 16000]


def bucket(x):
    return int(np.digitize(x, EDGES))


def one(path):
    try:
        op = gzip.open if path.endswith(".gz") else open
        starts, scores, sticks = [], None, 0
        with op(path, "rt", encoding="utf-8") as f:
            for ln in f:
                if '"start_kyoku"' in ln:
                    e = json.loads(ln)
                    scores = list(e["scores"])
                    ph = {"E": 0, "S": 4, "W": 8, "N": 8}[e["bakaze"]] + (e["kyoku"] - 1 if e["bakaze"] in "ES" else 0)
                    starts.append((min(ph, 8), list(scores)))
                    sticks = e.get("kyotaku", 0)
                elif '"reach_accepted"' in ln:
                    scores[json.loads(ln)["actor"]] -= 1000
                    sticks += 1
                elif '"hora"' in ln or '"ryukyoku"' in ln:
                    e = json.loads(ln)
                    if e.get("deltas"):
                        scores = [s + d for s, d in zip(scores, e["deltas"])]
                    if e["type"] == "hora":
                        sticks = 0
        if not starts:
            return None
        order = sorted(range(4), key=lambda i: (-scores[i], i))
        final = list(scores)
        final[order[0]] += 1000 * sticks
        pt = [0] * 4
        for r, i in enumerate(order):
            pt[i] = PT[r]
        rows = []
        for k, (ph, s) in enumerate(starts):
            nxt = starts[k + 1][1] if k + 1 < len(starts) else final
            last = int(k == len(starts) - 1)
            o = sorted(range(4), key=lambda i: (-s[i], i))
            for seat in range(4):
                r = o.index(seat)
                above = s[o[r - 1]] - s[seat] if r > 0 else 0
                below = s[seat] - s[o[r + 1]] if r < 3 else 0
                rows.append(((ph, last, r + 1, bucket(above), bucket(below)), (nxt[seat] - s[seat]) / 1000.0, pt[seat]))
        return rows
    except Exception as e:  # noqa: BLE001
        print(f"skip {path}: {e}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    fl = [os.path.join(a.root, y, f) for y in sorted(os.listdir(a.root)) if os.path.isdir(os.path.join(a.root, y))
          for f in sorted(os.listdir(os.path.join(a.root, y))) if ".mjson" in f]
    acc = defaultdict(lambda: np.zeros(5))     # n, sum d, sum d^2, sum pt, sum d*pt
    with Pool(a.procs) as pool:
        for i, rows in enumerate(pool.imap_unordered(one, fl, chunksize=500)):
            if rows:
                for key, dlt, p in rows:
                    acc[key] += (1, dlt, dlt * dlt, p, dlt * p)
            if (i + 1) % 500000 == 0:
                print(f"{i + 1:,} / {len(fl):,}", file=sys.stderr, flush=True)
    with open(a.out, "w") as f:
        f.write("phase\tlast\trank\tab\tbb\tn\tmean_delta\tsd_delta\tmean_pt\tleverage\n")
        for key in sorted(acc):
            n, sd, sd2, sp, sdp = acc[key]
            var = sd2 / n - (sd / n) ** 2
            lev = (sdp / n - (sd / n) * (sp / n)) / var if var > 0 and n >= 50 else float("nan")
            f.write("\t".join(map(str, key)) + f"\t{int(n)}\t{sd / n:.4f}\t{np.sqrt(max(var, 0)):.4f}\t{sp / n:.3f}\t{lev:.4f}\n")
    print(f"done: {len(acc)} strata", file=sys.stderr)


if __name__ == "__main__":
    main()
