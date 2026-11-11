#!/bin/bash
# P2 phase 1 follow-up, evaluation only (README.md, "Follow-up: the phase 1 trajectory"). Exploratory and post hoc:
# it reopens no gate and starts no training. It asks whether rl1_best's aggressive profile appears anywhere along
# phase 1, in particular at U40k, the snapshot that July's selection rule (best.pth, the best test_play reading)
# picked in this run, the way rl1_best was picked in July.
#   0. waits for an idle GPU (no mortal work, no guest hold) and rewrites GPU_STATUS_mortal.txt
#   1. checks that best.pth and the U40k archive carry the same weights
#   2. 12k (seeds 10000-10999, 14000-15999) of U8k, U16k, U24k, U40k, U48k, U56k; U32k and U64k exist
#   3. U40k extended to the 16,000 groups (seeds 10000-25999)
#   4. pairings and mechanism on common groups, then the card is lent to bookings for a day
# Log runs/xfer_tools/u1_traj.log; markers U1_TRAJ_DONE / U1_TRAJ_STOPPED; results u1_traj.txt.
# Stop: kill $(cat runs/xfer_tools/u1_traj.pid). Re-runnable: finished segments are skipped.
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/u1_traj.log
D=$R/evaldump; DMIN=${DMIN:-10}; W=/mnt/c/Users/asus; ST=$W/GPU_STATUS_mortal.txt; SP=$R/xfer_u1
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
dfree() { df -BG --output=avail /mnt/d | tail -1 | tr -dc 0-9; }
status() {
  printf '%s\r\n' "mortal GPU status (TanyaoDojo session), $(date '+%F %T %Z')" "$@" \
    "To book: create C:\\Users\\asus\\GPU_BOOK_<name>.txt (one line: who and what; add a line not_before=YYYY-MM-DD HH:MM to defer it)." \
    > $ST.new && mv $ST.new $ST
}
idle() {
  status "State: FREE (idle) since $(date +%T): the P2 trajectory evaluation has ended and no mortal GPU work is queued." \
    "An active booking (GPU_BOOK_<name>.txt, once past its not_before) gets the card within a minute; this file then shows the hold." \
    "This file is rewritten before any new mortal GPU work starts."
}
stopped() { say "$1: U1_TRAJ_STOPPED"; idle; exit 1; }
lend_and_exit() {
  END=$(( $(date +%s) + 86400 ))
  while [ $(date +%s) -lt $END ]; do
    pgrep -f "^$PY .*(train|train_pg|run_eval|client|server)\.py" > /dev/null && { say "mortal GPU work started: lending loop ends"; exit 0; }
    n0=$(ls $W/GPU_BOOK_*.txt 2>/dev/null | wc -l)
    if [ $n0 -gt 0 ]; then
      bash $T/gpu_share.sh >> $LOG 2>&1
      [ $(ls $W/GPU_BOOK_*.txt 2>/dev/null | wc -l) -lt $n0 ] && { idle; say "hold over; status: FREE (idle)"; }
    fi
    sleep 60
  done
  say "lending loop ends after a day"; exit 0
}
echo $$ > $T/u1_traj.pid
say "waiting for an idle GPU"
n=0
while [ $n -lt 3 ]; do
  if pgrep -f "^$PY .*(train|train_pg|run_eval|client|server|ev_map)\.py" > /dev/null || pgrep -f "gpu_share\.sh" > /dev/null \
     || ls $W/GPU_BUSY_*.txt > /dev/null 2>&1
  then n=0; else n=$((n + 1)); fi
  sleep 60
done
status "State: BUSY - evaluation only (P2 phase 1 trajectory: six snapshots on 12k, then U40k on 16,000 groups), about 6.5 hours from $(date +%H:%M). Exclusive: do not start GPU work." \
  "Lending: a booking fires at the next evaluation call (calls come at most ~35 min apart)." \
  "This file is rewritten when the chain ends."
$PY - $SP/best.pth $SP/archive/xfer_u1_s40000.pth >> $LOG 2>&1 <<'PY' || stopped "best.pth check FAILED"
import sys, torch
a = torch.load(sys.argv[1], weights_only=True, map_location="cpu"); b = torch.load(sys.argv[2], weights_only=True, map_location="cpu")
same = all(torch.equal(a[n][k], b[n][k]) for n in ("mortal", "current_dqn") for k in a[n])
print(f"best.pth steps {a.get('steps')} vs archive s40000 steps {b.get('steps')}: weights {'IDENTICAL' if same else 'DIFFER'}")
PY
for k in 40000 8000 16000 24000 48000 56000; do
  [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free before U$((k / 1000))k"
  bash $T/xfer_eval.sh u1 $k 12k || stopped "U$((k / 1000))k 12k FAILED"
done
say "12k trajectory done"
for s in 11000 12000 13000 $(seq 16000 1000 25000); do
  [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free before U40k s$s"
  bash $T/xfer_eval.sh u1 40000 seg $s 10 || stopped "U40k segment s$s FAILED"
done
say "U40k 16,000 groups done"
cd /mnt/d/Better_mortal/jax_rl/mjai_bot || stopped "no mjai_bot dir"
for p in xfer_u1_s8000 xfer_u1_s16000 xfer_u1_s24000 xfer_u1_s32000 xfer_u1_s40000 xfer_u1_s48000 xfer_u1_s56000 xfer_u1_s64000 \
         xfer_hist_s60000 xfer_v11_s80000; do
  $PY merge_dumps.py ${p}_12k ${p}_s10000 ${p}_s14000 > /dev/null 2>&1 || say "merge 12k $p FAILED"
done
segs=""; for s in 10000 11000 12000 13000 14000 $(seq 16000 1000 25000); do segs="$segs xfer_u1_s40000_s$s"; done
$PY merge_dumps.py xfer_u1_s40000_f16k $segs > /dev/null 2>&1 || say "merge 16k U40k FAILED"
$PY - "$D" > $T/u1_traj.txt 2>&1 <<'PY'
import glob, json, os, sys
import numpy as np
D = sys.argv[1]; R = os.path.expanduser("~/Projects/better_mortal/runs")
def load(p):
    d = {}
    for f in glob.glob(f"{D}/{p}_s*_gpufp32.npz"):
        z = np.load(f); d.update({int(s): float(v) for s, v in zip(z["seed"], z["mean"])})
    return d
U = {k: load(f"xfer_u1_s{k}000") for k in (8, 16, 24, 32, 40, 48, 56, 64)}
ref = {"A80k": load("xfer_v11_s80000"), "rl1_best": load("xfer_hist_s60000"), "C' final": load("xfer_hist_s278600")}
twelve = set(range(10000, 11000)) | set(range(14000, 16000))
def show(label, a, b=None, only=None):
    seeds = set(a) & (set(b) if b is not None else set(a))
    if only is not None:
        seeds &= only
    seeds = sorted(seeds)
    if len(seeds) < 2:
        print(f"  {label:<26} missing"); return
    y = np.array([a[s] - (b[s] if b is not None else 0.0) for s in seeds])
    h = 1.96 * y.std(ddof=1) / np.sqrt(len(y))
    print(f"  {label:<26} {len(seeds):>6,} groups  {y.mean():+.3f} +- {h:.3f}  z {y.mean() / (h / 1.96):+.2f}")
print("Strength on the 12k groups (3,000), every snapshot paired with U64k on the same groups")
for k in (8, 16, 24, 32, 40, 48, 56, 64):
    show(f"U{k}k vs v4", U[k], only=twelve)
for k in (8, 16, 24, 32, 40, 48, 56):
    show(f"U{k}k - U64k", U[k], U[64], only=twelve)
print("U40k (best.pth, July's selection rule) on all common groups")
show("U40k vs v4", U[40])
for n, b in [("U64k", U[64])] + list(ref.items()):
    show(f"U40k - {n}", U[40], b)
print("Mechanism against the v4 seats in the same games")
for tag in [f"xfer_u1_s{k}000_12k" for k in (8, 16, 24, 32, 40, 48, 56, 64)] + \
           ["xfer_hist_s60000_12k", "xfer_v11_s80000_12k", "xfer_u1_s40000_f16k", "xfer_u1_s64000_f16k", "xfer_hist_s60000_all"]:
    try:
        g = json.load(open(f"{R}/gapdump/{tag}.json")); pl = json.load(open(f"{R}/placedump/{tag}.json")); bu = json.load(open(f"{R}/bustdump/{tag}.json"))
    except OSError:
        print(f"  {tag}: missing"); continue
    o, t = g["ours"], g["theirs"]
    r = lambda d, k: d[k] / d["kyoku"]
    att = r(o, "win_pts") - r(t, "win_pts"); dfn = -(r(o, "deal_in_pts") - r(t, "deal_in_pts"))
    n = pl["ours"]["n"]; f4 = pl["ours"]["rank"][3] / n; f4v = (n - pl["ours"]["rank"][3]) / (3 * n)
    print(f"  {tag:<24} n={g['n_games']} call {100*r(o,'naki'):.2f} ({100*r(t,'naki'):.2f}) win {100*(r(o,'win')-r(t,'win')):+.2f} "
          f"dealin {100*(r(o,'deal_in')-r(t,'deal_in')):+.2f} riichi {100*(r(o,'riichi')-r(t,'riichi')):+.2f} "
          f"net {att+dfn:+.0f} ({att:+.0f} / {dfn:+.0f}) bust {100*bu['ours']['bust']/bu['ours']['n']:.2f} / {100*bu['v4']['bust']/bu['v4']['n']:.2f} "
          f"4th {100*(f4-f4v):+.2f}")
PY
say "results: $(tr '\n' ' ' < $T/u1_traj.txt)"
idle; say "U1_TRAJ_DONE"; lend_and_exit
