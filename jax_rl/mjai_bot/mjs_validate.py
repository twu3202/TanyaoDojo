"""Full libriichi replay check of converted logs (2026-10-06, new-data arm): every game through
GameplayLoader.load_log (the parse + encode path offline training uses), then per seat: every logged action is
legal under its mask, the kyoku index is consistent with the GRP features, and the samples of the kyoku listed in
start_game `drop_kyoku` (repaired jade games, see mjs_convert.py), which training skips, are counted.

    python mjs_validate.py --list FILES.txt --out RESULT.jsonl.gz [--procs 8]
Run from a Mortal tree (libriichi.so importable). Prints pass/fail counts by reason and decision totals.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import multiprocessing as mp
import time

import numpy as np

_loader = None


def init():
    global _loader
    from libriichi.dataset import GameplayLoader
    _loader = GameplayLoader(version=4, oracle=False, player_names=[], excludes=[])


def check(path):
    rec = {"path": path}
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            raw = f.read()
        games = _loader.load_log(raw)
    except Exception as e:  # noqa: BLE001
        rec.update(status="fail", reason="load: " + str(e).splitlines()[0][:120])
        return rec
    if len(games) != 4:
        rec.update(status="fail", reason=f"{len(games)} seats")
        return rec
    drop = set(json.loads(raw.split("\n", 1)[0]).get("drop_kyoku", []))
    n_all = n_used = 0
    for g in games:
        actions = np.asarray(g.take_actions())
        masks = np.stack(g.take_masks()) if len(actions) else np.zeros((0, 46), bool)
        at_kyoku = np.frombuffer(bytes(g.take_at_kyoku()), dtype=np.uint8)   # Vec<u8> arrives as bytes
        n_kyoku = len(g.take_grp().take_feature())
        if len(actions) and not masks[np.arange(len(actions)), actions].all():
            rec.update(status="fail", reason="action outside mask")
            return rec
        if len(at_kyoku) and at_kyoku.max() >= n_kyoku:
            rec.update(status="fail", reason="kyoku index beyond GRP features")
            return rec
        n_all += len(actions)
        n_used += int(sum(1 for x in at_kyoku if x not in drop))
    rec.update(status="ok", decisions=n_all, used=n_used, n_kyoku=n_kyoku)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    paths = [ln.strip() for ln in open(args.list) if ln.strip()]
    t0 = time.time()
    st = collections.Counter()
    dec = collections.Counter()
    with mp.Pool(args.procs, initializer=init) as pool, gzip.open(args.out, "wt") as out:
        for i, rec in enumerate(pool.imap_unordered(check, paths, chunksize=64)):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            st["ok" if rec["status"] == "ok" else "fail: " + rec["reason"]] += 1
            if rec["status"] == "ok":
                dec["decisions"] += rec["decisions"]
                dec["used"] += rec["used"]
            if (i + 1) % 50000 == 0:
                print(f"  {i + 1:,}/{len(paths):,}  {time.time() - t0:.0f}s  {dict(st.most_common(4))}", flush=True)
    print(json.dumps({"files": len(paths), "status": dict(st.most_common()), "decisions": dec["decisions"],
                      "used_decisions": dec["used"], "seconds": round(time.time() - t0, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
