#!/bin/bash
# Calibrated base, after the collapse (patches/calibrated_base/README.md, "Online result"): diagnostic evaluation.
# 1,000 groups each (seeds 10000-10999, sacred protocol, GPU fp32) of the Cal-QL base, its online snapshot at 8,000
# batches, v11 itself, and A80k's own snapshot at 8,000 batches; mechanism readouts; Q-gap / flip readout of all four.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/xfer_tools/cq_diag.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/cq_chain.log
W=/mnt/c/Users/asus; ST=$W/GPU_STATUS_mortal.txt; MM=$B/Mortal/mortal; D=$R/evaldump
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
status() {
  printf '%s\r\n' "mortal GPU status (TanyaoDojo session), $(date '+%F %T %Z')" "$@" \
    "To book: create C:\\Users\\asus\\GPU_BOOK_<name>.txt (one line: who and what; add a line not_before=YYYY-MM-DD HH:MM to defer it)." \
    > $ST.new && mv $ST.new $ST
}
idle() {
  status "State: FREE (idle) since $(date +%T): ${1:-the Cal-QL diagnostic evaluation has ended and no mortal GPU work is queued.}" \
    "An active booking (GPU_BOOK_<name>.txt, once past its not_before) gets the card within a minute; this file then shows the hold." \
    "This file is rewritten before any new mortal GPU work starts."
}
echo $$ > $T/cq_diag.pid
status "State: BUSY - diagnostic evaluation of the collapsed Cal-QL arm (4 x 1,000 groups, about 50 minutes from $(date +%H:%M)). Exclusive: do not start GPU work." \
  "Lending: a booking fires at each evaluation call (about 12 minutes apart)." \
  "This file is rewritten when it ends."
say "diagnostic evaluation of the collapse: CQ s8000, A s8000, CQ base, v11 base (1,000 groups each)"
mkdir -p $R/xfer_v11base/archive
[ -f $R/xfer_v11base/archive/xfer_v11base_s0.pth ] || cp $R/symaug/v11_best.pth $R/xfer_v11base/archive/xfer_v11base_s0.pth
[ -f $R/xfer_v11cq/archive/xfer_v11cq_s0.pth ] || cp $R/v11cq/v11cq_final.pth $R/xfer_v11cq/archive/xfer_v11cq_s0.pth
for a in "v11cq 8000" "v11 8000" "v11cq 0" "v11base 0"; do
  set -- $a; bash $T/xfer_eval.sh $1 $2 seg 10000 10 || say "diag eval $1 s$2 FAILED"
done
$PY - "$D" > $T/cq_diag.txt 2>&1 <<'PY'
import glob, json, os, sys
import numpy as np
D = sys.argv[1]; R = os.path.expanduser("~/Projects/better_mortal/runs")
tags = {"CQ8k": "xfer_v11cq_s8000", "A8k": "xfer_v11_s8000", "CQbase": "xfer_v11cq_s0", "v11": "xfer_v11base_s0",
        "A80k": "xfer_v11_s80000"}
m = {}
for k, t in tags.items():
    z = np.load(f"{D}/{t}_s10000_gpufp32.npz"); m[k] = dict(zip(z["seed"].tolist(), z["mean"].tolist()))
def show(a, b=None):
    s = sorted(set(m[a]) & (set(m[b]) if b else set(m[a])))
    y = np.array([m[a][i] - (m[b][i] if b else 0) for i in s]); h = 1.96 * y.std(ddof=1) / np.sqrt(len(y))
    print(f"  {a + (' - ' + b if b else ' vs v4'):<18} {len(s):>5,} groups  {y.mean():+.3f} +- {h:.3f}  z {y.mean() / (h / 1.96):+.2f}")
print("seeds 10000-10999:")
for a, b in (("CQ8k", None), ("A8k", None), ("CQbase", None), ("v11", None), ("A80k", None),
             ("CQ8k", "A8k"), ("CQbase", "v11"), ("CQ8k", "CQbase"), ("A8k", "v11")):
    show(a, b)
print("mechanism (vs the v4 seats in the same games):")
for k, t in tags.items():
    tag = f"{t}_s10000"
    g = json.load(open(f"{R}/gapdump/{tag}.json")); p = json.load(open(f"{R}/placedump/{tag}.json")); b = json.load(open(f"{R}/bustdump/{tag}.json"))
    o, th = g["ours"], g["theirs"]; r = lambda d, x: d[x] / d["kyoku"]
    att = r(o, "win_pts") - r(th, "win_pts"); dfn = -(r(o, "deal_in_pts") - r(th, "deal_in_pts"))
    n = p["ours"]["n"]; f4 = p["ours"]["rank"][3] / n; f4v = (n - p["ours"]["rank"][3]) / (3 * n)
    print(f"  {k:<7} n={g['n_games']} call {100*r(o,'naki'):.2f} ({100*r(th,'naki'):.2f}) win {100*(r(o,'win')-r(th,'win')):+.2f} "
          f"dealin {100*(r(o,'deal_in')-r(th,'deal_in')):+.2f} riichi {100*(r(o,'riichi')-r(th,'riichi')):+.2f} "
          f"net {att+dfn:+.0f} ({att:+.0f} / {dfn:+.0f}) bust {100*b['ours']['bust']/b['ours']['n']:.2f} 4th {100*(f4-f4v):+.2f}")
PY
say "diag pairings: $(tr -s ' ' < $T/cq_diag.txt | tr '\n' '|')"
cd $MM && DEVICE=cuda:0 PYTHONPATH=$MM $PY $T/gap_readout.py --files "$B/data/houou/2026/*.mjson*" --n-files 40 \
  v11=$R/symaug/v11_best.pth CQbase=$R/v11cq/v11cq_final.pth A8k=$R/xfer_v11/archive/xfer_v11_s8000.pth \
  CQ8k=$R/xfer_v11cq/archive/xfer_v11cq_s8000.pth A80k=$R/xfer_v11/archive/xfer_v11_s80000.pth > $T/cq_diag_gap.txt 2>&1
say "diag Q-gap: $(tr -s ' ' < $T/cq_diag_gap.txt | tr '\n' '|')"
idle; say "CQ_DIAG_DONE"
