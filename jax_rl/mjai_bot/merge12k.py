"""12k 合并配对: merge12k.py A_s10000.npz B_s10000.npz A_s14000.npz B_s14000.npz -> 报告 B 的 12k avg_pt 与 B - A。
两段按牌局组拼接(1000 + 2000 组), 组层 CI。与 read12k.sh 里内联的算法相同。"""
import sys
import numpy as np


def seg(fa, fb):
    a, b = np.load(fa), np.load(fb)
    c = np.intersect1d(a["seed"], b["seed"])
    gb = b["mean"][np.searchsorted(b["seed"], c)]
    ga = a["mean"][np.searchsorted(a["seed"], c)]
    return gb - ga, gb, ga


d1, b1, a1 = seg(sys.argv[1], sys.argv[2])
d2, b2, a2 = seg(sys.argv[3], sys.argv[4])
for lab, d, bb, aa in (("s10000", d1, b1, a1), ("s14000", d2, b2, a2),
                       ("12k", np.concatenate([d1, d2]), np.concatenate([b1, b2]), np.concatenate([a1, a2]))):
    se = d.std(ddof=1) / np.sqrt(len(d))
    seb = bb.std(ddof=1) / np.sqrt(len(bb))
    print(f"  {lab:7s} 组数 {len(d):5d}  B avg_pt = {bb.mean():+.3f} ± {1.96*seb:.3f}   (A {aa.mean():+.3f})"
          f"   B - A = {d.mean():+.3f} ± {1.96*se:.3f}  z = {d.mean()/se:+.2f}  r = {np.corrcoef(aa, bb)[0, 1]:+.3f}")
