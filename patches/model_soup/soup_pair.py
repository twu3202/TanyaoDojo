"""Paired readouts for the model-soup probe (patches/model_soup/README.md).

Per-group means come from the GPU fp32 dumps (evaldump, and runs/s1pair for the 408k-line continuations). A seed
present in two dumps of one model must agree; the script says so if not. "mean of N" is the per-group mean of the
ingredients' outcomes, on the groups all of them have.
"""
import glob
import os

import numpy as np

R = os.path.expanduser("~/Projects/better_mortal/runs")
DIRS = [f"{R}/evaldump", f"{R}/s1pair"]
TAGS = {
    "SV": "xfer_soupv_s160000", "A160k": "xfer_v11s2_s160000", "C160k": "xfer_v11c2_s160000",
    "F160k": "xfer_v11f_s160000", "R160k": "xfer_v11r_s160000", "FS160k": "xfer_v11fs_s160000",
    "A80k": "xfer_v11_s80000",
    "ST": "xfer_soupt_s448000", "408k": "tpt_tpt_s408000", "h448k": "tpt_tpt_s448000", "S1-448k": "stage1_s448000",
    "M1c0.3-448k": "anc30_s448000", "M1sel-448k": "m1sel_s448000", "C'final": "xfer_hist_s278600",
}
m = {}
for k, tag in TAGS.items():
    d = {}
    for D in DIRS:
        for f in sorted(glob.glob(f"{D}/{tag}_s*_gpufp32.npz")):
            if "part" in f:
                continue
            z = np.load(f)
            for s, v in zip(z["seed"].tolist(), z["mean"].tolist()):
                if s in d and abs(d[s] - v) > 1e-9:
                    print(f"  WARNING {k}: seed {s} differs between dumps ({d[s]} vs {v})")
                d[int(s)] = float(v)
    m[k] = d
pooled = {s: (m["A160k"][s] + m["C160k"][s]) / 2 for s in set(m["A160k"]) & set(m["C160k"])}
m["pooled control"] = pooled


def mean_of(keys):
    seeds = set.intersection(*(set(m[k]) for k in keys))
    return {s: float(np.mean([m[k][s] for k in keys])) for s in seeds}


m["mean of 5"] = mean_of(["A160k", "C160k", "F160k", "R160k", "FS160k"])
m["mean of 4"] = mean_of(["h448k", "S1-448k", "M1c0.3-448k", "M1sel-448k"])


def show(a, b=None):
    seeds = sorted(set(m[a]) & (set(m[b]) if b else set(m[a])))
    label = f"{a} vs v4" if b is None else f"{a} - {b}"
    if len(seeds) < 2:
        print(f"  {label:<28} (no common groups)")
        return
    y = np.array([m[a][s] - (m[b][s] if b else 0.0) for s in seeds])
    h = 1.96 * y.std(ddof=1) / np.sqrt(len(y))
    print(f"  {label:<28} {len(seeds):>6,} groups  {y.mean():+.3f} +- {h:.3f}  z {y.mean() / (h / 1.96):+.2f}")


print("v11 line (soup of five 160k arms from A80k):")
for b in (None, "pooled control", "mean of 5", "A160k", "C160k", "F160k", "R160k", "FS160k", "A80k", "408k"):
    show("SV", b)
show("mean of 5", "pooled control")
print("408k line (soup of four 448k continuations from 408k):")
for b in (None, "408k", "mean of 4", "h448k", "S1-448k", "M1c0.3-448k", "M1sel-448k", "C'final"):
    show("ST", b)
show("mean of 4", "408k")
