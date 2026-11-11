#!/bin/bash
# Second round of the C' step replay (README.md, "Second round"). Chosen by the user on 11-07, after the first replay
# (arm cr) did not reproduce the step.
#   0. waits for an idle GPU (no mortal work, no guest hold) and rewrites GPU_STATUS_mortal.txt
#   1. arm cr2: the first replay again, on new random self-play seeds (each worker draws secrets.randbits(64)).
#      runs/xfer_cr2, port 5600: C392k's full July state, anchor and base opponents rl1_best, reward [6,4,2,0],
#      LR 5e-6, temperature 0.1, pool base,base,v4,C392k on 4 workers; lambda 0.5 to 240,000, then 0.2 to 278,600
#   2. 12k of CR2_240k, the 16,000 groups of CR2_278.6k
#   3. waits (up to 48 hours, lending the card) for arm ct: the same step on trx with July's topology (7 workers,
#      loader 0, RTX 6000 Ada). A Mac relay puts its archives in runs/xfer_ct/archive with READY, or STOPPED.
#   4. 12k of CT240k, the 16,000 groups of CT278.6k, on this card like every other snapshot of the comparison
#   5. pairings, mechanism and the verdicts set in advance; then the card is marked FREE and lent for a day
# Log runs/xfer_tools/cr2_chain.log; markers CR2_CHAIN_DONE / CR2_CHAIN_STOPPED; results cr2_final.txt, cr2_mech.txt.
# Re-runnable: a set-up arm is not set up again, a reached stop is not trained again, finished segments are skipped.
# Stop: kill $(cat runs/xfer_tools/cr2_chain.pid), then bash runs/xfer_tools/xfer_helpers.sh stop runs/xfer_cr2.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/xfer_tools/cr2_chain.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/cr2_chain.log
D=$R/evaldump; C=$B/configs; DMIN=${DMIN:-10}; W=/mnt/c/Users/asus; ST=$W/GPU_STATUS_mortal.txt
SRC=$R/cprime_C392k_0723_1921.pth; ANC=/mnt/d/Better_mortal/weights_backup/rl1_best_155k.pth
ARM=cr2; SP=$R/xfer_cr2; CFG=$C/online_selfplay_xfer_cr2.toml; CTA=$R/xfer_ct/archive
POOLSPEC="base,base,v4,$SRC"
CK1=$SP/archive/xfer_cr2_s240000.pth; CK2=$SP/archive/xfer_cr2_s278600.pth
SEGS16="11000 12000 13000 $(seq 16000 1000 25000)"
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
dfree() { df -BG --output=avail /mnt/d | tail -1 | tr -dc 0-9; }
norm() { sed -E -e "s/xfer_$1/ARM/g" -e 's/55[0-9]{2}|5600/PORT/g' -e 's/ *#.*$//' $C/online_selfplay_xfer_$1.toml | grep -v '^\s*$'; }
status() {
  printf '%s\r\n' "mortal GPU status (TanyaoDojo session), $(date '+%F %T %Z')" "$@" \
    "To book: create C:\\Users\\asus\\GPU_BOOK_<name>.txt (one line: who and what; add a line not_before=YYYY-MM-DD HH:MM to defer it)." \
    > $ST.new && mv $ST.new $ST
}
idle() {
  status "State: FREE (idle) since $(date +%T): ${1:-the second replay chain of the C-prime step has ended and no mortal GPU work is queued.}" \
    "An active booking (GPU_BOOK_<name>.txt, once past its not_before) gets the card within a minute; this file then shows the hold." \
    "This file is rewritten before any new mortal GPU work starts."
}
stopped() { say "$1: CR2_CHAIN_STOPPED"; idle; exit 1; }
lend_once() {   # one look at the bookings, as the lending loop does
  local n0; n0=$(ls $W/GPU_BOOK_*.txt 2>/dev/null | wc -l)
  if [ $n0 -gt 0 ]; then
    bash $T/gpu_share.sh >> $LOG 2>&1
    [ $(ls $W/GPU_BOOK_*.txt 2>/dev/null | wc -l) -lt $n0 ] && { idle "$1"; say "hold over; status: FREE (idle)"; }
  fi
}
lend_and_exit() {
  END=$(( $(date +%s) + 86400 ))
  while [ $(date +%s) -lt $END ]; do
    pgrep -f "^$PY .*(train|train_pg|run_eval|client|server)\.py" > /dev/null && { say "mortal GPU work started: lending loop ends"; exit 0; }
    lend_once ""; sleep 60
  done
  say "lending loop ends after a day"; exit 0
}
echo $$ > $T/cr2_chain.pid
say "waiting for an idle GPU"
n=0
while [ $n -lt 3 ]; do
  if pgrep -f "^$PY .*(train|train_pg|run_eval|client|server|ev_map)\.py" > /dev/null || pgrep -f "gpu_share\.sh" > /dev/null \
     || ls $W/GPU_BUSY_*.txt > /dev/null 2>&1
  then n=0; else n=$((n + 1)); fi
  sleep 60
done
status "State: BUSY - online RL training (second C' replay: arm cr2 from C392k, 24,200 batches at lambda 0.5 then 38,600 at lambda 0.2, about 7.5 hours from $(date +%H:%M)), then evaluation (about 4.5 hours), then the trx arm's evaluation (about 4.5 hours). Exclusive: do not start GPU work." \
  "Lending: a booking fires at an arm start or at an evaluation call, never during training." \
  "This file is rewritten at each phase and when the chain ends."
# ---------------- set-up ----------------
if [ ! -e $SP/state.pth ]; then
  [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free"
  [ "$(md5sum < $SRC | cut -c1-8)" = 7a679fb4 ] || stopped "C392k md5 mismatch"
  [ "$(md5sum < $ANC | cut -c1-8)" = b9c0ff1f ] || stopped "rl1_best md5 mismatch"
  say "arm cr2 from $SRC, anchor $ANC, D: $(dfree) GB free"
  PTS="[6.0, 4.0, 2.0, 0.0]" ANCHOR=0.5 bash $T/xfer_setup.sh cr2 $SRC 5600 >> $LOG 2>&1 || stopped "setup cr2 FAILED"
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
want() {   # want <weight>: the config must equal the first replay's at the same weight, apart from the arm and port
  d=$(diff <(norm cr) <(norm cr2) | grep "^[<>]")
  [ "$1" = 0.5 ] && d=$(diff <(norm cr | sed 's/^weight = 0.2$/weight = 0.5/') <(norm cr2) | grep "^[<>]")
  [ -z "$d" ] || stopped "config (weight $1) differs from the first replay's: $(echo "$d" | tr '\n' ' ')"
  say "config (weight $1) equals the first replay's at the same weight"
}
# ---------------- lambda 0.5, 215,800 -> 240,000 ----------------
if [ ! -f $CK1 ]; then
  want 0.5
  START_AFTER=215800 STOP_AT=240000 POOL=$POOLSPEC bash $T/xfer_arm.sh cr2
  [ -f $CK1 ] || stopped "lambda 0.5 phase did not reach 240,000"
fi
say "lambda 0.5 phase trained to 240,000"
# ---------------- lambda 0.2, 240,000 -> 278,600 ----------------
if [ ! -f $CK2 ]; then
  awk '/^weight = /{print "weight = 0.2   # C'"'"' replay: July read lambda 0.2 at step 240,000"; next} {print}' $CFG > $CFG.new && mv $CFG.new $CFG
  want 0.2
  START_AFTER=240000 STOP_AT=278600 POOL=$POOLSPEC bash $T/xfer_arm.sh cr2
  [ -f $CK2 ] || stopped "lambda 0.2 phase did not reach 278,600"
fi
say "trained to 278,600; $($PY $T/xfer_reuse.py $SP 2>/dev/null | tail -1)"
# ---------------- evaluation of cr2 ----------------
status "State: BUSY - evaluation of the second C' replay (CR2_240k 12k, CR2_278.6k 16,000 groups), about 4.5 hours from $(date +%H:%M). Exclusive: do not start GPU work." \
  "Lending: a booking fires at the next evaluation call (calls come at most ~35 min apart)." \
  "This file is rewritten at each phase and when the chain ends."
bash $T/xfer_eval.sh cr2 240000 12k || stopped "CR2_240k 12k FAILED"
bash $T/xfer_eval.sh cr2 278600 12k || stopped "CR2_278.6k 12k FAILED"
for s in $SEGS16; do
  [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free before CR2_278.6k s$s"
  bash $T/xfer_eval.sh cr2 278600 seg $s 10 || stopped "CR2_278.6k segment s$s FAILED"
done
say "cr2 evaluated"
# ---------------- the trx arm ----------------
CT=skip
WAIT_END=$(( $(date +%s) + 48 * 3600 ))
idle "the card waits for the trx arm of the C' replay (its archives come by a relay), then evaluates it (about 4.5 hours)."
say "waiting for the trx arm (runs/xfer_ct/archive/READY or STOPPED, at most 48 hours)"
until [ -f $CTA/READY ] || [ -f $CTA/STOPPED ] || [ $(date +%s) -ge $WAIT_END ]; do
  lend_once "the card waits for the trx arm of the C' replay, then evaluates it (about 4.5 hours)."; sleep 60
done
if [ -f $CTA/READY ]; then
  ok=1
  for k in 240000 278600; do
    want_md5=$(grep "^xfer_ct_s$k.pth " $CTA/READY | cut -d' ' -f2)
    [ -n "$want_md5" ] && [ "$(md5sum < $CTA/xfer_ct_s$k.pth | cut -c1-32)" = "$want_md5" ] || { ok=0; say "trx archive s$k md5 mismatch or missing"; }
  done
  [ $ok = 1 ] && CT=eval
else
  say "no trx arm to evaluate ($([ -f $CTA/STOPPED ] && echo 'it stopped' || echo '48 hours passed'))"
fi
if [ $CT = eval ]; then
  n=0   # a guest test may hold the card now
  while [ $n -lt 3 ]; do
    if pgrep -f "^$PY .*(train|train_pg|run_eval|client|server|ev_map)\.py" > /dev/null || pgrep -f "gpu_share\.sh" > /dev/null \
       || ls $W/GPU_BUSY_*.txt > /dev/null 2>&1
    then n=0; else n=$((n + 1)); fi
    sleep 60
  done
  status "State: BUSY - evaluation of the trx arm of the C' replay (CT240k 12k, CT278.6k 16,000 groups), about 4.5 hours from $(date +%H:%M). Exclusive: do not start GPU work." \
    "Lending: a booking fires at the next evaluation call (calls come at most ~35 min apart)." \
    "This file is rewritten when the chain ends."
  mkdir -p $R/xfer_ct
  bash $T/xfer_eval.sh ct 240000 12k || stopped "CT240k 12k FAILED"
  bash $T/xfer_eval.sh ct 278600 12k || stopped "CT278.6k 12k FAILED"
  for s in $SEGS16; do
    [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free before CT278.6k s$s"
    bash $T/xfer_eval.sh ct 278600 seg $s 10 || stopped "CT278.6k segment s$s FAILED"
  done
  say "ct evaluated"
fi
# ---------------- pairings ----------------
$PY - "$D" > $T/cr2_final.txt 2>&1 <<'PY'
import glob, sys
import numpy as np
D = sys.argv[1]
def load(p):
    d = {}
    for f in glob.glob(f"{D}/{p}_s*_gpufp32.npz"):
        z = np.load(f); d.update({int(s): float(v) for s, v in zip(z["seed"], z["mean"])})
    return d
m = {k: load(p) for k, p in {"CR2": "xfer_cr2_s278600", "CR2_240k": "xfer_cr2_s240000", "CT": "xfer_ct_s278600",
     "CT240k": "xfer_ct_s240000", "CR": "xfer_cr_s278600", "CR240k": "xfer_cr_s240000", "C392k": "xfer_hist_s215800",
     "C' final": "xfer_hist_s278600", "rl1_best": "xfer_hist_s60000", "408k": "tpt_tpt_s408000"}.items()}
common = set(m["CR"]) & set(m["CR2"])
m["CR+CR2 mean"] = {s: (m["CR"][s] + m["CR2"][s]) / 2 for s in common}
twelve = set(range(10000, 11000)) | set(range(14000, 16000)); allg = set(range(10000, 26000)); new = allg - twelve
def show(label, a, b=None, only=allg, tag=None):
    S = sorted(set(m[a]) & (set(m[b]) if b else set(m[a])) & only)
    if len(S) < 2:
        print(f"  {label:<44} missing"); return
    y = np.array([m[a][s] - (m[b][s] if b else 0.0) for s in S]); h = 1.96 * y.std(ddof=1) / np.sqrt(len(y)); z = y.mean() / (h / 1.96)
    print(f"  {label:<44} {len(S):>6,} groups  {y.mean():+.3f} +- {h:.3f}  z {z:+.2f}")
    if tag:
        print(f"{tag} {y.mean():.4f} {z:.3f}")
show("CR2 - C392k (PRIMARY, second 5060 Ti draw)", "CR2", "C392k", tag="PRIMARY_CR2")
show("CT - C392k (PRIMARY, trx July topology)", "CT", "C392k", tag="PRIMARY_CT")
show("CR2 - C392k, 13,000 new groups", "CR2", "C392k", only=new)
show("CT - C392k, 13,000 new groups", "CT", "C392k", only=new)
show("first replay: CR - C392k", "CR", "C392k")
show("history: C' final - C392k", "C' final", "C392k")
show("CR2 - CR (two 5060 Ti draws)", "CR2", "CR")
show("CT - mean of CR and CR2 (topology + hardware)", "CT", "CR+CR2 mean")
show("CT - C' final", "CT", "C' final")
show("CR2 - C' final", "CR2", "C' final")
show("CT vs v4", "CT"); show("CR2 vs v4", "CR2")
show("CT - 408k", "CT", "408k"); show("CT - rl1_best", "CT", "rl1_best")
show("CR2_240k - C392k (lambda 0.5 part), 12k", "CR2_240k", "C392k", only=twelve)
show("CR2 - CR2_240k (lambda 0.2 part), 12k", "CR2", "CR2_240k", only=twelve)
show("CT240k - C392k (lambda 0.5 part), 12k", "CT240k", "C392k", only=twelve)
show("CT - CT240k (lambda 0.2 part), 12k", "CT", "CT240k", only=twelve)
PY
say "pairings: $(grep -v '^PRIMARY' $T/cr2_final.txt | tr '\n' ' ')"
cd /mnt/d/Better_mortal/jax_rl/mjai_bot || stopped "no mjai_bot dir"
for a in cr2 ct; do
  segs=""; for s in 10000 11000 12000 13000 14000 $(seq 16000 1000 25000); do segs="$segs xfer_${a}_s278600_s$s"; done
  $PY merge_dumps.py xfer_${a}_s278600_f16k $segs > /dev/null 2>&1 || say "merge 16k $a FAILED"
  $PY merge_dumps.py xfer_${a}_s240000_12k xfer_${a}_s240000_s10000 xfer_${a}_s240000_s14000 > /dev/null 2>&1 || say "merge 12k ${a}240k FAILED"
done
$PY - > $T/cr2_mech.txt 2>&1 <<'PY'
import json, os
R = os.path.expanduser("~/Projects/better_mortal/runs")
for tag in ["xfer_hist_s215800_all", "xfer_cr_s278600_f16k", "xfer_cr2_s278600_f16k", "xfer_ct_s278600_f16k",
            "xfer_hist_s278600_all", "xfer_cr_s240000_12k", "xfer_cr2_s240000_12k", "xfer_ct_s240000_12k"]:
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
say "mechanism: $(tr '\n' ' ' < $T/cr2_mech.txt)"
for a in CR2 CT; do
  read -r mean z <<< "$(grep "^PRIMARY_$a " $T/cr2_final.txt | cut -d' ' -f2-)"
  v=$($PY -c "import sys; m, z = map(float, sys.argv[1:3]); print('REPRODUCED' if m >= 0.7 and z >= 2 else 'NOT_REPRODUCED' if m <= 0.3 else 'UNDECIDED')" "${mean:-nan}" "${z:-nan}" 2>/dev/null)
  say "verdict $a: $a - C392k ${mean:-?} (z ${z:-?}): ${v:-MISSING}"
done
idle; say "CR2_CHAIN_DONE"; lend_and_exit
