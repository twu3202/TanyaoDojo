"""多段合并配对: merge_segs.py A1.npz B1.npz [A2.npz B2.npz ...] -> 各段与合并后 B 的 avg_pt、B - A 配对(组层 CI)。
merge12k.py 的推广(12k = 两段, 28k = 三段, 100k = 更多段)。"""
import sys
import numpy as np
args = sys.argv[1:]
assert len(args) % 2 == 0
D, BB, AA = [], [], []
for fa, fb in zip(args[::2], args[1::2]):
    a, b = np.load(fa), np.load(fb)
    c = np.intersect1d(a["seed"], b["seed"])
    ga = a["mean"][np.searchsorted(a["seed"], c)]
    gb = b["mean"][np.searchsorted(b["seed"], c)]
    D.append(gb - ga); BB.append(gb); AA.append(ga)
    lab = f"s{c.min()}-{c.max()}"
    se = (gb - ga).std(ddof=1) / np.sqrt(len(c))
    print(f"  {lab:13s} 组数 {len(c):5d}  B {gb.mean():+.3f}  A {ga.mean():+.3f}  B-A {np.mean(gb-ga):+.3f} ± {1.96*se:.3f}  z={np.mean(gb-ga)/se:+.2f}")
d, bb, aa = map(np.concatenate, (D, BB, AA))
se, seb, sea = (x.std(ddof=1) / np.sqrt(len(x)) for x in (d, bb, aa))
print(f"  合并 {len(d)*4:,} 局({len(d)} 组): B = {bb.mean():+.3f} ± {1.96*seb:.3f}   A = {aa.mean():+.3f} ± {1.96*sea:.3f}")
print(f"  B - A = {d.mean():+.3f} ± {1.96*se:.3f}   z = {d.mean()/se:+.2f}   r = {np.corrcoef(aa, bb)[0,1]:+.3f}")
