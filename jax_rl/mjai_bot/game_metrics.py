"""Per-game metrics for the data-quality study (2026-10-11, patches/data_quality/README.md).

One pass over Tenhou houou mjai logs (.mjson / .mjson.gz). For every game it records who played, how the scores
developed and how the game ended; the per-player strength and the quality validation are computed from this table
later (quality_validate.py). Nothing here uses a model.

Scores are tracked through the log: start_kyoku sets them, reach_accepted takes 1000, hora / ryukyoku add their
deltas (which already include the riichi sticks and honba). Sticks left on the table at the end go to the top player,
as on Tenhou. Ties in final score rank by seat order from the first dealer (Tenhou rule; kyoku_first is 0 here).

    python game_metrics.py DATA_ROOT OUT.tsv [--procs 8]
Output columns: see COLS.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from multiprocessing import Pool

COLS = ["path", "year", "n0", "n1", "n2", "n3", "s0", "s1", "s2", "s3", "r0", "r1", "r2", "r3",
        "n_kyoku", "south", "final_spread", "mean_spread", "max_spread", "spread_at_s1",
        "gap12", "gap34", "busted", "e1_spread0", "ok"]
TYPES = ('"start_game"', '"start_kyoku"', '"reach_accepted"', '"hora"', '"ryukyoku"', '"end_game"')


def one(path: str):
    rel = path
    try:
        op = gzip.open if path.endswith(".gz") else open
        with op(path, "rt", encoding="utf-8") as f:
            lines = [ln for ln in f if any(t in ln for t in TYPES)]
        names = None
        scores = None
        spreads = []
        n_kyoku = 0
        south = 0
        sticks = 0
        spread_at_s1 = -1
        ended = False
        for ln in lines:
            ev = json.loads(ln)
            t = ev["type"]
            if t == "start_game":
                names = ev.get("names") or ["?"] * 4
            elif t == "start_kyoku":
                scores = list(ev["scores"])
                sticks = ev.get("kyotaku", 0)
                n_kyoku += 1
                sp = max(scores) - min(scores)
                spreads.append(sp)
                if ev["bakaze"] == "S":
                    if not south:
                        spread_at_s1 = sp
                    south = 1
            elif t == "reach_accepted":
                scores[ev["actor"]] -= 1000
                sticks += 1
            elif t in ("hora", "ryukyoku"):
                d = ev.get("deltas")
                if d:
                    for i in range(4):
                        scores[i] += d[i]
                if t == "hora":
                    sticks = 0
            elif t == "end_game":
                ended = True
        if names is None or scores is None or not ended:
            return None
        order = sorted(range(4), key=lambda i: (-scores[i], i))
        final = list(scores)
        final[order[0]] += 1000 * sticks
        ranks = [0] * 4
        for r, i in enumerate(order):
            ranks[i] = r + 1
        fs = sorted(final, reverse=True)
        year = os.path.basename(os.path.dirname(path))
        return [rel, year, *[n.replace("\t", " ") for n in names], *final, *ranks,
                n_kyoku, south, fs[0] - fs[3], round(sum(spreads) / len(spreads)), max(spreads), spread_at_s1,
                fs[0] - fs[1], fs[2] - fs[3], int(fs[3] < 0), spreads[0], 1]
    except Exception as e:  # noqa: BLE001 - one bad log must not stop the pass
        print(f"skip {path}: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def files(root: str):
    for y in sorted(os.listdir(root)):
        d = os.path.join(root, y)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if ".mjson" in f:
                yield os.path.join(d, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    fl = list(files(a.root))
    print(f"{len(fl):,} logs", file=sys.stderr, flush=True)
    n = bad = 0
    with Pool(a.procs) as pool, open(a.out + ".tmp", "w", encoding="utf-8") as out:
        out.write("\t".join(COLS) + "\n")
        for row in pool.imap(one, fl, chunksize=500):
            n += 1
            if row is None:
                bad += 1
                continue
            row[0] = os.path.relpath(row[0], a.root)
            out.write("\t".join(map(str, row)) + "\n")
            if n % 200000 == 0:
                print(f"{n:,} / {len(fl):,}  ({bad} skipped)", file=sys.stderr, flush=True)
    os.replace(a.out + ".tmp", a.out)
    print(f"done: {n - bad:,} games, {bad} skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
