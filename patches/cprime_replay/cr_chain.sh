#!/bin/bash
# P1 short: the C' jump step replayed from its own start (README.md). Chosen by the user on 11-07.
#   0. waits for an idle GPU (no mortal work, no guest hold) and rewrites GPU_STATUS_mortal.txt
#   1. arm cr: runs/xfer_cr from C392k's full July state (runs/cprime_C392k_0723_1921.pth, md5 7a679fb4: steps
#      215,800, Adam moments, scheduler at 5e-6), port 5598. Anchor and base opponent = rl1_best (July's BASE.pth,
#      md5 b9c0ff1f). July's config: reward [6,4,2,0], LR 5e-6 (what the restored scheduler ran in July), Boltzmann
#      temperature 0.1. The config may differ from v11s2's only in pts, peak, final and the anchor weight.
#      Pool base,base,v4,C392k: July's spec base,base,v4,snap, with snap = C392k as in the 09-13 restoration.
#   2. lambda 0.5 up to step 240,000 (where July's trainer read lambda 0.2), then lambda 0.2 up to 278,600 (July's stop)
#   3. 12k of CR240k, the 16,000 groups of CR278.6k, pairings, mechanism and the verdict set in advance
#   4. the card is marked FREE and lent to bookings for a day
# Log runs/xfer_tools/cr_chain.log; markers CR_CHAIN_DONE / CR_CHAIN_STOPPED; results cr_final.txt, cr_mech.txt.
# Re-runnable: a set-up arm is not set up again, a reached stop is not trained again, finished segments are skipped.
# Stop: kill $(cat runs/xfer_tools/cr_chain.pid), then bash runs/xfer_tools/xfer_helpers.sh stop runs/xfer_cr.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/xfer_tools/cr_chain.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/cr_chain.log
D=$R/evaldump; C=$B/configs; DMIN=${DMIN:-10}; W=/mnt/c/Users/asus; ST=$W/GPU_STATUS_mortal.txt
SRC=$R/cprime_C392k_0723_1921.pth; ANC=/mnt/d/Better_mortal/weights_backup/rl1_best_155k.pth
ARM=cr; SP=$R/xfer_cr; CFG=$C/online_selfplay_xfer_cr.toml
POOLSPEC="base,base,v4,$SRC"
CK1=$SP/archive/xfer_cr_s240000.pth; CK2=$SP/archive/xfer_cr_s278600.pth
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
dfree() { df -BG --output=avail /mnt/d | tail -1 | tr -dc 0-9; }
norm() { sed -E -e "s/xfer_$1/ARM/g" -e 's/55[0-9]{2}/PORT/g' -e 's/ *#.*$//' $C/online_selfplay_xfer_$1.toml | grep -v '^\s*$'; }
status() {
  printf '%s\r\n' "mortal GPU status (TanyaoDojo session), $(date '+%F %T %Z')" "$@" \
    "To book: create C:\\Users\\asus\\GPU_BOOK_<name>.txt (one line: who and what; add a line not_before=YYYY-MM-DD HH:MM to defer it)." \
    > $ST.new && mv $ST.new $ST
}
idle() {
  status "State: FREE (idle) since $(date +%T): the C' replay chain has ended and no mortal GPU work is queued." \
    "An active booking (GPU_BOOK_<name>.txt, once past its not_before) gets the card within a minute; this file then shows the hold." \
    "This file is rewritten before any new mortal GPU work starts."
}
stopped() { say "$1: CR_CHAIN_STOPPED"; idle; exit 1; }
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
echo $$ > $T/cr_chain.pid
say "waiting for an idle GPU"
n=0
while [ $n -lt 3 ]; do
  if pgrep -f "^$PY .*(train|train_pg|run_eval|client|server|ev_map)\.py" > /dev/null || pgrep -f "gpu_share\.sh" > /dev/null \
     || ls $W/GPU_BUSY_*.txt > /dev/null 2>&1
  then n=0; else n=$((n + 1)); fi
  sleep 60
done
status "State: BUSY - online RL training (C' replay: arm cr from C392k, 24,200 batches at lambda 0.5 then 38,600 at lambda 0.2, about 6.5 hours from $(date +%H:%M)), then evaluation (about 4 hours). Exclusive: do not start GPU work." \
  "Lending: a booking fires at an arm start or at an evaluation call, never during training." \
  "This file is rewritten at each phase and when the chain ends."
# ---------------- set-up ----------------
if [ ! -e $SP/state.pth ]; then
  [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free"
  [ "$(md5sum < $SRC | cut -c1-8)" = 7a679fb4 ] || stopped "C392k md5 mismatch"
  [ "$(md5sum < $ANC | cut -c1-8)" = b9c0ff1f ] || stopped "rl1_best md5 mismatch"
  say "arm cr from $SRC, anchor $ANC, D: $(dfree) GB free"
  PTS="[6.0, 4.0, 2.0, 0.0]" ANCHOR=0.5 bash $T/xfer_setup.sh cr $SRC 5598 >> $LOG 2>&1 || stopped "setup cr FAILED"
  # July's BASE.pth was rl1_best: the anchor and the base opponents
  cp $ANC $SP/BASE.pth.new && mv $SP/BASE.pth.new $SP/BASE.pth
  sed -i -e "s/^peak = .*/peak = 5e-6                # C' replay: July's effective LR (its restored scheduler; its config said 1e-5)/" \
         -e "s/^final = .*/final = 5e-6/" $CFG
  echo "BASE.pth replaced by rl1_best: $(md5sum < $SP/BASE.pth | cut -c1-32)" >> $SP/manifest.txt
  say "BASE.pth = rl1_best $(md5sum < $SP/BASE.pth | cut -c1-8); config peak/final 5e-6"
fi
$PY - $SP/state.pth >> $LOG 2>&1 <<'PY' || stopped "start state check FAILED"
import sys, torch
s = torch.load(sys.argv[1], weights_only=True, map_location="cpu")
sc = s["scheduler"]; lr = [g["lr"] for g in s["optimizer"]["param_groups"]]
print(f"state.pth: steps {s['steps']}, online {s['config']['control']['online']}, scheduler peak/final {sc['peak']}/{sc['final']}, optimizer lr {lr}")
assert s["config"]["control"]["online"] and sc["peak"] == sc["final"] == 5e-6 and all(x == 5e-6 for x in lr)
PY
want() {   # want <weight>: the config may differ from v11s2's only in these lines
  d=$(diff <(norm v11s2) <(norm cr) | grep "^[<>]")
  ok='^> (pts = \[6\.0, 4\.0, 2\.0, 0\.0\]|peak = 5e-6|final = 5e-6|weight = '"$1"')$'
  nexp=4; [ "$1" = 0.2 ] && nexp=3
  [ "$(echo "$d" | grep -c '^<')" = $nexp ] && [ "$(echo "$d" | grep -c '^>')" = $nexp ] && [ "$(echo "$d" | grep -c -E "$ok")" = $nexp ] \
    || stopped "config (weight $1) differs from v11s2 beyond the July recipe lines: $(echo "$d" | tr '\n' ' ')"
  say "config vs v11s2 (weight $1): $(echo "$d" | tr '\n' ' ')"
}
# ---------------- lambda 0.5, 215,800 -> 240,000 ----------------
if [ ! -f $CK1 ]; then
  want 0.5
  START_AFTER=215800 STOP_AT=240000 POOL=$POOLSPEC bash $T/xfer_arm.sh cr
  [ -f $CK1 ] || stopped "lambda 0.5 phase did not reach 240,000"
fi
say "lambda 0.5 phase trained to 240,000"
# ---------------- lambda 0.2, 240,000 -> 278,600 ----------------
if [ ! -f $CK2 ]; then
  awk '/^weight = /{print "weight = 0.2   # C'"'"' replay: July read lambda 0.2 at step 240,000"; next} {print}' $CFG > $CFG.new && mv $CFG.new $CFG
  want 0.2
  START_AFTER=240000 STOP_AT=278600 POOL=$POOLSPEC bash $T/xfer_arm.sh cr
  [ -f $CK2 ] || stopped "lambda 0.2 phase did not reach 278,600"
fi
say "trained to 278,600; $($PY $T/xfer_reuse.py $SP 2>/dev/null | tail -1)"
# ---------------- evaluation ----------------
status "State: BUSY - evaluation of the C' replay (CR240k 12k, CR278.6k 16,000 groups), about 4 hours from $(date +%H:%M). Exclusive: do not start GPU work." \
  "Lending: a booking fires at the next evaluation call (calls come at most ~35 min apart)." \
  "This file is rewritten when the chain ends."
bash $T/xfer_eval.sh cr 240000 12k || stopped "CR240k 12k FAILED"
bash $T/xfer_eval.sh cr 278600 12k || stopped "CR278.6k 12k FAILED"
for s in 11000 12000 13000 $(seq 16000 1000 25000); do
  [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free before CR278.6k s$s"
  bash $T/xfer_eval.sh cr 278600 seg $s 10 || stopped "CR278.6k segment s$s FAILED"
done
$PY - "$D" > $T/cr_final.txt 2>&1 <<'PY'
import glob, sys
import numpy as np
D = sys.argv[1]
def load(p):
    d = {}
    for f in glob.glob(f"{D}/{p}_s*_gpufp32.npz"):
        z = np.load(f); d.update({int(s): float(v) for s, v in zip(z["seed"], z["mean"])})
    return d
m = {k: load(p) for k, p in {"CR278.6k": "xfer_cr_s278600", "CR240k": "xfer_cr_s240000", "C392k": "xfer_hist_s215800",
     "C' final": "xfer_hist_s278600", "rl1_best": "xfer_hist_s60000", "408k": "tpt_tpt_s408000", "U40k": "xfer_u1_s40000"}.items()}
twelve = set(range(10000, 11000)) | set(range(14000, 16000)); allg = set(range(10000, 26000)); new = allg - twelve
def show(label, a, b=None, only=allg, tag=None):
    S = sorted(set(m[a]) & (set(m[b]) if b else set(m[a])) & only)
    if len(S) < 2:
        print(f"  {label:<40} missing"); return
    y = np.array([m[a][s] - (m[b][s] if b else 0.0) for s in S]); h = 1.96 * y.std(ddof=1) / np.sqrt(len(y)); z = y.mean() / (h / 1.96)
    print(f"  {label:<40} {len(S):>6,} groups  {y.mean():+.3f} +- {h:.3f}  z {z:+.2f}")
    if tag:
        print(f"{tag} {y.mean():.4f} {z:.3f}")
show("CR278.6k - C392k (PRIMARY)", "CR278.6k", "C392k", tag="PRIMARY")
show("CR278.6k - C392k, 13,000 new groups", "CR278.6k", "C392k", only=new)
show("history: C' final - C392k", "C' final", "C392k")
show("CR278.6k - C' final", "CR278.6k", "C' final")
show("CR278.6k vs v4", "CR278.6k")
show("CR278.6k - rl1_best", "CR278.6k", "rl1_best")
show("CR278.6k - 408k", "CR278.6k", "408k")
show("CR278.6k - U40k", "CR278.6k", "U40k")
show("CR240k - C392k (lambda 0.5 part), 12k", "CR240k", "C392k", only=twelve)
show("CR278.6k - CR240k (lambda 0.2 part), 12k", "CR278.6k", "CR240k", only=twelve)
PY
say "pairings: $(grep -v '^PRIMARY' $T/cr_final.txt | tr '\n' ' ')"
cd /mnt/d/Better_mortal/jax_rl/mjai_bot || stopped "no mjai_bot dir"
segs=""; for s in 10000 11000 12000 13000 14000 $(seq 16000 1000 25000); do segs="$segs xfer_cr_s278600_s$s"; done
$PY merge_dumps.py xfer_cr_s278600_f16k $segs > /dev/null 2>&1 || say "merge 16k CR278.6k FAILED"
$PY merge_dumps.py xfer_cr_s240000_12k xfer_cr_s240000_s10000 xfer_cr_s240000_s14000 > /dev/null 2>&1 || say "merge 12k CR240k FAILED"
for p in xfer_cr_s278600 xfer_hist_s215800 xfer_hist_s278600; do
  $PY merge_dumps.py ${p}_12k ${p}_s10000 ${p}_s14000 > /dev/null 2>&1 || say "merge 12k $p FAILED"
done
$PY - > $T/cr_mech.txt 2>&1 <<'PY'
import json, os
R = os.path.expanduser("~/Projects/better_mortal/runs")
for tag in ["xfer_cr_s278600_f16k", "xfer_hist_s278600_all", "xfer_hist_s215800_all",
            "xfer_cr_s240000_12k", "xfer_cr_s278600_12k", "xfer_hist_s215800_12k", "xfer_hist_s278600_12k"]:
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
say "mechanism: $(tr '\n' ' ' < $T/cr_mech.txt)"
read -r mean z <<< "$(grep '^PRIMARY' $T/cr_final.txt | cut -d' ' -f2-)"
v=$($PY -c "import sys; m, z = map(float, sys.argv[1:3]); print('REPRODUCED' if m >= 0.7 and z >= 2 else 'NOT_REPRODUCED' if m <= 0.3 else 'UNDECIDED')" "${mean:-nan}" "${z:-nan}")
say "verdict: CR278.6k - C392k ${mean:-?} (z ${z:-?}): $v"
idle; say "CR_CHAIN_DONE"; lend_and_exit
