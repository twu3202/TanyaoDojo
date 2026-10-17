"""Pre-scan an offline training file list with the trainer's own reader (GameplayLoader.load_gz_log_files, one file at
a time, as with file_batch_size = 1): files it cannot read, files with a logged action that the mask marks illegal
(train.py asserts `masks[range(batch_size), actions].all()` and the whole run dies), and the number of training
samples, so a run can be sized to exactly one pass.

    python scan_logs.py --index FILE_INDEX.pth --out DIR [--procs 12]
Writes DIR/bad.tsv (path, reason), DIR/counts.tsv (path, samples) and prints the totals.
Run with Mortal's python dir on PYTHONPATH (libriichi).
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool

import numpy as np

_loader = None


def check(path):
    global _loader
    if _loader is None:
        from libriichi.dataset import GameplayLoader
        _loader = GameplayLoader(version=4, oracle=False, player_names=[], excludes=[])
    try:
        data = _loader.load_gz_log_files([path])
    except Exception as e:  # noqa: BLE001
        return path, 0, "unreadable: " + str(e).splitlines()[0][:200]
    n, bad = 0, None
    for file in data:
        for game in file:
            a, m = np.asarray(game.take_actions()), np.asarray(game.take_masks(), bool)
            n += len(a)
            if len(a) and not m[np.arange(len(a)), a].all():
                j = int(np.flatnonzero(~m[np.arange(len(a)), a])[0])
                bad = f"illegal action {int(a[j])} at sample {j} of seat {int(game.take_player_id())}"
    return path, n, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    import torch
    files = torch.load(a.index, weights_only=True)["file_list"]
    os.makedirs(a.out, exist_ok=True)
    total, nbad = 0, 0
    with Pool(a.procs) as pool, open(os.path.join(a.out, "counts.tsv"), "w") as fc, \
            open(os.path.join(a.out, "bad.tsv"), "w") as fb:
        for i, (p, n, bad) in enumerate(pool.imap(check, files, chunksize=64)):
            if bad:
                fb.write(f"{p}\t{bad}\n")
                fb.flush()
                nbad += 1
            else:
                total += n
            fc.write(f"{p}\t{n}\n")
            if (i + 1) % 50000 == 0:
                print(f"{i + 1:,} / {len(files):,} files, {nbad} bad", flush=True)
    print(f"files {len(files):,}, bad {nbad}, samples in good files {total:,}, "
          f"steps per pass at batch 512 = {total // 512:,}")


if __name__ == "__main__":
    main()
