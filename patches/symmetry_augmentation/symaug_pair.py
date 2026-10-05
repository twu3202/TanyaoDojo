"""Paired comparison on shared seeds: symaug (v11 recipe + symmetry augmentation, S) vs v11 (same recipe, V) and
B1 (v11 recipe at 256x54, B).    python symaug_pair.py LO HI [LO HI ...]   (dumps are read from this script's directory)
Pre-registered (patches/symmetry_augmentation/README.md): 12k S - V < 0 stops; 100k S - V >= +0.5 with z >= 2 passes."""
import glob, os, sys
import numpy as np
D = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get("SYMTAG", "symaug_final")
PAT = {"S": f"{TAG}_s*_gpufp32.npz", "V": "v11_manual_s*.npz", "B": "b1_b1final_s*_gpufp32.npz"}
NAME = {"S": f"symaug ({TAG})", "V": "v11", "B": "B1 256x54"}
m = {}
for k, pat in PAT.items():
    m[k] = {}
    for f in sorted(glob.glob(f"{D}/{pat}")):
        z = np.load(f)
        for s, v in zip(z["seed"], z["mean"]):
            m[k].setdefault(int(s), float(v))
r = [int(x) for x in sys.argv[1:]]
keep = lambda s: any(lo <= s <= hi for lo, hi in zip(r[::2], r[1::2]))
print(f"seeds {sys.argv[1:]}, pt/hanchan vs mortal_v4, 95% CI (seed-group, 4 rotations per group)")
for a, b in (("S", "V"), ("S", "B"), ("B", "V")):
    seeds = sorted(s for s in set(m[a]) & set(m[b]) if keep(s))
    n = len(seeds)
    if n < 2:
        print(f"  {NAME[a]} - {NAME[b]}: {n} shared groups")
        continue
    va, vb = (np.array([m[k][s] for s in seeds]) for k in (a, b))
    d = va - vb
    se = d.std(ddof=1) / np.sqrt(n)
    print(f"  {NAME[a] + ' - ' + NAME[b]:<32} {d.mean():+.3f} +- {1.96 * se:.3f}  z {d.mean() / se:+.2f}  r {np.corrcoef(va, vb)[0, 1]:+.3f}"
          f"   [{n:,} groups = {4 * n:,} games each; {NAME[a]} {va.mean():+.3f} +- {1.96 * va.std(ddof=1) / np.sqrt(n):.3f}, "
          f"{NAME[b]} {vb.mean():+.3f}]")
