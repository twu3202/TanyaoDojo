"""Online transfer test: paired readouts on shared seed groups (README.md).
    python xfer_pair.py BARM BBASE STEP LO HI [LO HI ...]
BARM = the second arm (q25 or q12one), BBASE = the dump tag of its offline base (q25_final or q12one_final), STEP = the
snapshot (40000 or 80000). Every model must cover every seed of the requested ranges, and a seed dumped twice must
carry the same value, else the script prints INCOMPLETE / CONFLICT and exits 2.
Primary: B - A at the same step. The interaction (B - A online) - (B - A offline) is the per-group difference of the
four vectors, so its CI is the joint one, not two CIs subtracted."""
import glob, os, sys
import numpy as np

R = os.environ.get("XFER_RUNS", os.path.expanduser("~/Projects/better_mortal/runs"))   # XFER_RUNS: test fixtures only
barm, bbase, step = sys.argv[1], sys.argv[2], sys.argv[3]
r = [int(x) for x in sys.argv[4:]]
want = sorted(s for lo, hi in zip(r[::2], r[1::2]) for s in range(lo, hi + 1))
PAT = {
    "v11": f"{R}/symaug/pair/v11_manual_s*.npz",
    "Bbase": f"{R}/evaldump/{bbase}_s*_gpufp32.npz",
    "A": f"{R}/evaldump/xfer_v11_s{step}_s*_gpufp32.npz",
    "B": f"{R}/evaldump/xfer_{barm}_s{step}_s*_gpufp32.npz",
    "408k": f"{R}/evaldump/tpt_tpt_s408000_s*_gpufp32.npz",
}
NAME = {"v11": "v11 (offline)", "Bbase": f"{bbase} (offline)", "A": f"v11 online {step}", "B": f"{barm} online {step}",
        "408k": "tpt 408k"}
m, bad = {}, False
for k, pat in PAT.items():
    m[k] = {}
    for f in sorted(glob.glob(pat)):
        z = np.load(f)
        for s, v in zip(z["seed"], z["mean"]):
            s, v = int(s), float(v)
            if s in m[k] and m[k][s] != v:
                print(f"CONFLICT: {NAME[k]} seed {s} dumped twice with different values ({f})")
                bad = True
            m[k].setdefault(s, v)
have = {k: [s for s in want if s in m[k]] for k in m}
for k in m:
    if len(have[k]) != len(want):
        print(f"INCOMPLETE: {NAME[k]} has {len(have[k]):,} of the {len(want):,} requested groups")
        bad = True
if bad:
    sys.exit(2)
V = {k: np.array([m[k][s] for s in want]) for k in m}
n = len(want)


def show(label, d):
    se = d.std(ddof=1) / np.sqrt(n)
    print(f"  {label:<44} {d.mean():+.3f} +- {1.96 * se:.3f}  z {d.mean() / se:+.2f}")


print(f"seeds {sys.argv[4:]}: {n:,} groups = {4 * n:,} games per model; pt/hanchan, 95% CI over seed groups")
print("vs mortal_v4:")
for k in ("v11", "Bbase", "A", "B", "408k"):
    show(NAME[k], V[k])
print("paired:")
show(f"PRIMARY  {NAME['B']} - {NAME['A']}", V["B"] - V["A"])
show(f"offline  {NAME['Bbase']} - {NAME['v11']}", V["Bbase"] - V["v11"])
show("interaction (online diff - offline diff)", (V["B"] - V["A"]) - (V["Bbase"] - V["v11"]))
show(f"recipe   {NAME['A']} - {NAME['v11']}", V["A"] - V["v11"])
show(f"gain     {NAME['B']} - {NAME['Bbase']}", V["B"] - V["Bbase"])
show(f"{NAME['A']} - tpt 408k", V["A"] - V["408k"])
show(f"{NAME['B']} - tpt 408k", V["B"] - V["408k"])
