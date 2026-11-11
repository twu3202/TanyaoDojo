#!/bin/bash
# Pessimism dose, after the drop (README, "Online result"): exploratory, not pre-registered.
# How much of A2_80k - A80k is the base and how much the online stage? A 12k of the weight-0.2 base itself
# (s10000 + s14000, 3,000 groups, sacred protocol, GPU fp32), paired on the same seeds with v11's existing 12k
# (xfer_v11base_s0, from the Cal-QL diagnostics) and with both arms' 80k snapshots.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/xfer_tools/a2_diag.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/a2_chain.log
W=/mnt/c/Users/asus; ST=$W/GPU_STATUS_mortal.txt; D=$R/evaldump
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
status() {
  printf '%s\r\n' "mortal GPU status (TanyaoDojo session), $(date '+%F %T %Z')" "$@" \
    "To book: create C:\\Users\\asus\\GPU_BOOK_<name>.txt (one line: who and what; add a line not_before=YYYY-MM-DD HH:MM to defer it)." \
    > $ST.new && mv $ST.new $ST
}
idle() {
  status "State: FREE (idle) since $(date +%T): ${1:-the pessimism-dose base evaluation has ended and no mortal GPU work is queued.}" \
    "An active booking (GPU_BOOK_<name>.txt, once past its not_before) gets the card within a minute; this file then shows the hold." \
    "This file is rewritten before any new mortal GPU work starts."
}
mortal_busy() { pgrep -f "^$PY .*(train|train_pg|run_eval|client|server|critic_diag2?|fs_check)\.py" > /dev/null; }
echo $$ > $T/a2_diag.pid
status "State: BUSY - exploratory 12k of the pessimism-dose base (3,000 groups, about 40 minutes from $(date +%H:%M)). Exclusive: do not start GPU work." \
  "Lending: a booking fires at each evaluation call (about 25 minutes apart)." "This file is rewritten when it ends."
say "exploratory: 12k of the weight-0.2 base (xfer_v11a02base_s0)"
mkdir -p $R/xfer_v11a02base/archive
[ -f $R/xfer_v11a02base/archive/xfer_v11a02base_s0.pth ] || cp $R/v11a02/v11a02_final.pth $R/xfer_v11a02base/archive/xfer_v11a02base_s0.pth
say "base md5 $(md5sum $R/xfer_v11a02base/archive/xfer_v11a02base_s0.pth | cut -c1-8)"
bash $T/xfer_eval.sh v11a02base 0 12k || say "a2_diag 12k FAILED"
$PY - "$D" > $T/a2_diag.txt 2>&1 <<'PY'
import sys
import numpy as np
D = sys.argv[1]
def load(t):
    m = {}
    for s in (10000, 14000):
        z = np.load(f"{D}/{t}_s{s}_gpufp32.npz"); m.update(zip(z["seed"].tolist(), z["mean"].tolist()))
    return m
m = {k: load(t) for k, t in (("a02base", "xfer_v11a02base_s0"), ("v11", "xfer_v11base_s0"),
                              ("A2_80k", "xfer_v11a02_s80000"), ("A80k", "xfer_v11_s80000"))}
s = sorted(set.intersection(*(set(d) for d in m.values())))
def show(name, y):
    h = 1.96 * y.std(ddof=1) / np.sqrt(len(y))
    print(f"  {name:<44} {len(y):>5,} groups  {y.mean():+.3f} +- {h:.3f}  z {y.mean() / (h / 1.96):+.2f}")
v = {k: np.array([d[i] for i in s]) for k, d in m.items()}
for k in v:
    show(f"{k} vs v4", v[k])
show("a02base - v11 (the base)", v["a02base"] - v["v11"])
show("A2_80k - a02base (stage gain, dose arm)", v["A2_80k"] - v["a02base"])
show("A80k - v11 (stage gain, control)", v["A80k"] - v["v11"])
show("stage-gain difference", (v["A2_80k"] - v["a02base"]) - (v["A80k"] - v["v11"]))
show("A2_80k - A80k (same seeds)", v["A2_80k"] - v["A80k"])
PY
say "a2_diag: $(tr -s ' ' < $T/a2_diag.txt | tr '\n' '|')"
cd /mnt/d/Better_mortal/jax_rl/mjai_bot && {
  for t in v11a02base_s0 v11base_s0; do
    [ -f $R/gapdump/xfer_${t}_f3k.json ] || $PY merge_dumps.py xfer_${t}_f3k xfer_${t}_s10000 xfer_${t}_s14000 > /dev/null || say "merge $t failed"
  done
  $PY - > $T/a2_diag_mech.txt 2>&1 <<'PY'
import json, os
R = os.path.expanduser("~/Projects/better_mortal/runs")
for tag in ("xfer_v11base_s0_f3k", "xfer_v11a02base_s0_f3k", "xfer_v11_s80000_f3k", "xfer_v11a02_s80000_f3k"):
    try:
        g = json.load(open(f"{R}/gapdump/{tag}.json")); p = json.load(open(f"{R}/placedump/{tag}.json")); b = json.load(open(f"{R}/bustdump/{tag}.json"))
    except OSError:
        print(f"  {tag}: missing"); continue
    o, t = g["ours"], g["theirs"]
    r = lambda d, k: d[k] / d["kyoku"]
    att = r(o, "win_pts") - r(t, "win_pts"); dfn = -(r(o, "deal_in_pts") - r(t, "deal_in_pts"))
    n = p["ours"]["n"]; f4 = p["ours"]["rank"][3] / n; f4v = (n - p["ours"]["rank"][3]) / (3 * n)
    print(f"  {tag:<24} n={g['n_games']} call {100*r(o,'naki'):.2f} ({100*r(t,'naki'):.2f}) win {100*(r(o,'win')-r(t,'win')):+.2f} "
          f"dealin {100*(r(o,'deal_in')-r(t,'deal_in')):+.2f} riichi {100*(r(o,'riichi')-r(t,'riichi')):+.2f} "
          f"net {att+dfn:+.0f} ({att:+.0f} / {dfn:+.0f}) bust {100*b['ours']['bust']/b['ours']['n']:.2f} / {100*b['v4']['bust']/b['v4']['n']:.2f} "
          f"4th {100*(f4-f4v):+.2f}")
PY
}
say "a2_diag mechanism: $(tr -s ' ' < $T/a2_diag_mech.txt | tr '\n' '|')"
idle; say "A2_DIAG_DONE"
END=$(( $(date +%s) + 86400 ))
while [ $(date +%s) -lt $END ]; do
  mortal_busy && { say "mortal GPU work started: lending loop ends"; exit 0; }
  n0=$(ls $W/GPU_BOOK_*.txt 2>/dev/null | wc -l)
  if [ $n0 -gt 0 ]; then
    bash $T/gpu_share.sh >> $LOG 2>&1
    [ $(ls $W/GPU_BOOK_*.txt 2>/dev/null | wc -l) -lt $n0 ] && { idle; say "hold over; status: FREE (idle)"; }
  fi
  sleep 60
done
say "lending loop ends after a day"
