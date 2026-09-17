"""两个快照在同一批 100k 牌(seeds 10000-34999)上的配对: 各自把分段 npz 按 seed 拼起来, 按 seed 对齐求差。
用法: pair100k.py <A 前缀> <B 前缀>   (前缀如 tpt_tpt_s408000, 读 evaldump/<前缀>_s*_gpufp32.npz)"""
import glob, sys
import numpy as np
import os
D = os.path.expanduser("~/Projects/better_mortal/runs/evaldump/")

def load(prefix):
    seeds, means = [], []
    for f in sorted(glob.glob(f"{D}{prefix}_s[0-9]*_gpufp32.npz")):
        z = np.load(f); seeds.append(z["seed"]); means.append(z["mean"])
    s, m = np.concatenate(seeds), np.concatenate(means)
    u, idx = np.unique(s, return_index=True)
    assert len(u) == len(s), f"{prefix}: 有重复 seed"
    return u, m[idx]

(sa, ma), (sb, mb) = load(sys.argv[1]), load(sys.argv[2])
c = np.intersect1d(sa, sb)
a, b = ma[np.searchsorted(sa, c)], mb[np.searchsorted(sb, c)]
for lab, x in ((sys.argv[1], a), (sys.argv[2], b)):
    print(f"  {lab:20s} {len(x)*4:,} 局  avg_pt = {x.mean():+.3f} ± {1.96*x.std(ddof=1)/np.sqrt(len(x)):.3f}")
d = b - a; se = d.std(ddof=1) / np.sqrt(len(d))
print(f"  配对 B - A = {d.mean():+.3f} ± {1.96*se:.3f}   z = {d.mean()/se:+.2f}   r = {np.corrcoef(a, b)[0,1]:+.3f}   (seeds {c.min()}-{c.max()}, {len(c)} 组)")
