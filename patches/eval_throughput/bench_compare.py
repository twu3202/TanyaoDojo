"""Scores one throughput benchmark run (bench_eval.sh): games/s, identity with the reference logs, resource use.
    python bench_compare.py OUT REF LO GROUPS NAME NPROC GAMES T0 T1 RC
A game is identical when its decompressed mjai log equals the reference byte for byte once the per-call timing field
"eval_time_ns" is removed (it is the only field that differs between two runs of the same layout: 1,200 of 1,200
games matched on 10-19, q_values included). The per-group placement
points of the run and of the reference are compared too, so a layout that changes a few near-tie decisions shows
how far the readout moves, not just that it moved."""
import gzip, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from group_ci import final_scores, PTS, SUFFIX_SEAT  # noqa: E402

out, ref, lo, groups, name, nproc, games, t0, t1, rc = sys.argv[1:11]
lo, groups, t0, t1 = int(lo), int(groups), float(t0), float(t1)


TIMING = re.compile(rb'"eval_time_ns":\d+,?')


def read(path):
    with gzip.open(path, "rb") as f:
        return TIMING.sub(b"", f.read())


def pt(path, suf):
    sc, _ = final_scores(path)
    order = sorted(range(4), key=lambda i: (-sc[i], i))
    return PTS[order.index(SUFFIX_SEAT[suf])]


same = missing = 0
dif = []
for s in range(lo, lo + groups):
    a, b = [], []
    for suf in "abcd":
        fn = f"{s}_20260711_{suf}.json.gz"
        p, q = os.path.join(out, "logs", fn), os.path.join(ref, fn)
        if not os.path.exists(p):
            missing += 1
            continue
        same += read(p) == read(q)
        a.append(pt(p, suf))
        b.append(pt(q, suf))
    if len(a) == 4:
        dif.append(np.mean(a) - np.mean(b))
n = 4 * groups - missing
wall = t1 - t0
smi = np.loadtxt(os.path.join(out, "smi.csv"), delimiter=",", ndmin=2) if os.path.getsize(os.path.join(out, "smi.csv")) else np.zeros((1, 2))
load = np.loadtxt(os.path.join(out, "load.txt"), ndmin=1)
d = np.array(dif) if dif else np.zeros(1)
print(f"{name:<14} nproc {nproc} games/iter {games:>5}  rc {rc}  {n:,} games in {wall:6.0f} s = {n / wall:5.2f} games/s  "
      f"identical {same}/{n}  pt(run - ref) {d.mean():+.3f} (max |group| {np.abs(d).max():.1f})  "
      f"gpu util {smi[:, 0].mean():.0f}% mem max {smi[:, 1].max():.0f} MiB  load {load.mean():.1f}")
