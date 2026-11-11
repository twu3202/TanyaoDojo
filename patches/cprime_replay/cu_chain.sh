#!/bin/bash
# Successor of cr_chain.sh (README.md, "Successor: the same step from U40k"). Chosen by the user on 11-07, before the
# replay's result. It trains only if the replay's verdict is REPRODUCED or UNDECIDED.
#   0. waits for CR_CHAIN_DONE in cr_chain.log (CR_CHAIN_STOPPED or NOT_REPRODUCED: exits, nothing trained), then for
#      an idle GPU (no mortal work, no guest hold), and rewrites GPU_STATUS_mortal.txt
#   1. arm cu: runs/xfer_cu from U40k (runs/xfer_u1/archive/xfer_u1_s40000.pth, md5 b6fefbe2: P2 phase 1's best.pth,
#      steps 40,000, scheduler at 5e-6), anchored to itself, port 5599, with exactly the replay's recipe: reward
#      [6,4,2,0], LR 5e-6, Boltzmann temperature 0.1. Pool base,base,v4,base: the replay's spec with the start in the
#      snap seat, and the start is the base here.
#   2. lambda 0.5 for 24,200 batches (to 64,200), then lambda 0.2 for 38,600 (to 102,800): the replay's lengths
#   3. 12k of CU64.2k, the 16,000 groups of CU102.8k, pairings, mechanism and the verdict set in advance
#   4. the card is marked FREE and lent to bookings for a day
# Log runs/xfer_tools/cu_chain.log; markers CU_CHAIN_DONE / CU_CHAIN_STOPPED / CU_CHAIN_SKIPPED; results cu_final.txt,
# cu_mech.txt. Stop: kill $(cat runs/xfer_tools/cu_chain.pid), then bash runs/xfer_tools/xfer_helpers.sh stop runs/xfer_cu.
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/cu_chain.log
D=$R/evaldump; C=$B/configs; DMIN=${DMIN:-10}; W=/mnt/c/Users/asus; ST=$W/GPU_STATUS_mortal.txt
SRC=$R/xfer_u1/archive/xfer_u1_s40000.pth
ARM=cu; SP=$R/xfer_cu; CFG=$C/online_selfplay_xfer_cu.toml
POOLSPEC="base,base,v4,base"
CK1=$SP/archive/xfer_cu_s64200.pth; CK2=$SP/archive/xfer_cu_s102800.pth
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
dfree() { df -BG --output=avail /mnt/d | tail -1 | tr -dc 0-9; }
norm() { sed -E -e "s/xfer_$1/ARM/g" -e 's/55[0-9]{2}/PORT/g' -e 's/ *#.*$//' $C/online_selfplay_xfer_$1.toml | grep -v '^\s*$'; }
status() {
  printf '%s\r\n' "mortal GPU status (TanyaoDojo session), $(date '+%F %T %Z')" "$@" \
    "To book: create C:\\Users\\asus\\GPU_BOOK_<name>.txt (one line: who and what; add a line not_before=YYYY-MM-DD HH:MM to defer it)." \
    > $ST.new && mv $ST.new $ST
}
idle() {
  status "State: FREE (idle) since $(date +%T): the U40k step chain has ended and no mortal GPU work is queued." \
    "An active booking (GPU_BOOK_<name>.txt, once past its not_before) gets the card within a minute; this file then shows the hold." \
    "This file is rewritten before any new mortal GPU work starts."
}
stopped() { say "$1: CU_CHAIN_STOPPED"; idle; exit 1; }
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
echo $$ > $T/cu_chain.pid
say "waiting for the replay's verdict (cr_chain.log)"
CRL=$T/cr_chain.log
until grep -q -E 'CR_CHAIN_(DONE|STOPPED)' $CRL 2>/dev/null; do sleep 120; done
grep -q CR_CHAIN_STOPPED $CRL && { say "the replay stopped: CU_CHAIN_SKIPPED"; exit 0; }
v=$(grep -a '^.* verdict: ' $CRL | tail -1)
say "replay $v"
case "$v" in
  *": REPRODUCED"|*": UNDECIDED") ;;
  *) say "the replay did not reproduce the step: nothing trained, CU_CHAIN_SKIPPED"; exit 0;;
esac
say "waiting for an idle GPU"
n=0
while [ $n -lt 3 ]; do
  if pgrep -f "^$PY .*(train|train_pg|run_eval|client|server|ev_map)\.py" > /dev/null || pgrep -f "gpu_share\.sh" > /dev/null \
     || ls $W/GPU_BUSY_*.txt > /dev/null 2>&1
  then n=0; else n=$((n + 1)); fi
  sleep 60
done
status "State: BUSY - online RL training (the C' step from U40k: arm cu, 24,200 batches at lambda 0.5 then 38,600 at lambda 0.2, about 6.5 hours from $(date +%H:%M)), then evaluation (about 4 hours). Exclusive: do not start GPU work." \
  "Lending: a booking fires at an arm start or at an evaluation call, never during training." \
  "This file is rewritten at each phase and when the chain ends."
# ---------------- set-up ----------------
if [ ! -e $SP/state.pth ]; then
  [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free"
  [ "$(md5sum < $SRC | cut -c1-8)" = b6fefbe2 ] || stopped "U40k md5 mismatch"
  say "arm cu from $SRC (anchored to itself), D: $(dfree) GB free"
  PTS="[6.0, 4.0, 2.0, 0.0]" ANCHOR=0.5 bash $T/xfer_setup.sh cu $SRC 5599 >> $LOG 2>&1 || stopped "setup cu FAILED"
  sed -i -e "s/^peak = .*/peak = 5e-6                # the replay's recipe (July's effective LR); U40k's scheduler is at 5e-6 too/" \
         -e "s/^final = .*/final = 5e-6/" $CFG
  say "config peak/final 5e-6"
fi
$PY - $SP/state.pth >> $LOG 2>&1 <<'PY' || stopped "start state check FAILED"
import sys, torch
s = torch.load(sys.argv[1], weights_only=True, map_location="cpu")
sc = s["scheduler"]; lr = [g["lr"] for g in s["optimizer"]["param_groups"]]
print(f"state.pth: steps {s['steps']}, online {s['config']['control']['online']}, scheduler peak/final {sc['peak']}/{sc['final']}, optimizer lr {lr}")
assert s["config"]["control"]["online"] and s["steps"] == 40000 and sc["peak"] == sc["final"] == 5e-6 and all(x == 5e-6 for x in lr)
PY
want() {   # want <weight>: the config may differ from v11s2's only in these lines
  d=$(diff <(norm v11s2) <(norm cu) | grep "^[<>]")
  ok='^> (pts = \[6\.0, 4\.0, 2\.0, 0\.0\]|peak = 5e-6|final = 5e-6|weight = '"$1"')$'
  nexp=4; [ "$1" = 0.2 ] && nexp=3
  [ "$(echo "$d" | grep -c '^<')" = $nexp ] && [ "$(echo "$d" | grep -c '^>')" = $nexp ] && [ "$(echo "$d" | grep -c -E "$ok")" = $nexp ] \
    || stopped "config (weight $1) differs from v11s2 beyond the July recipe lines: $(echo "$d" | tr '\n' ' ')"
  say "config vs v11s2 (weight $1): $(echo "$d" | tr '\n' ' ')"
}
# ---------------- lambda 0.5, 40,000 -> 64,200 ----------------
if [ ! -f $CK1 ]; then
  want 0.5
  START_AFTER=40000 STOP_AT=64200 POOL=$POOLSPEC bash $T/xfer_arm.sh cu
  [ -f $CK1 ] || stopped "lambda 0.5 phase did not reach 64,200"
fi
say "lambda 0.5 phase trained to 64,200"
# ---------------- lambda 0.2, 64,200 -> 102,800 ----------------
if [ ! -f $CK2 ]; then
  awk '/^weight = /{print "weight = 0.2   # as in the replay: lambda 0.2 after 24,200 batches"; next} {print}' $CFG > $CFG.new && mv $CFG.new $CFG
  want 0.2
  START_AFTER=64200 STOP_AT=102800 POOL=$POOLSPEC bash $T/xfer_arm.sh cu
  [ -f $CK2 ] || stopped "lambda 0.2 phase did not reach 102,800"
fi
say "trained to 102,800; $($PY $T/xfer_reuse.py $SP 2>/dev/null | tail -1)"
# ---------------- evaluation ----------------
status "State: BUSY - evaluation of the U40k step (CU64.2k 12k, CU102.8k 16,000 groups), about 4 hours from $(date +%H:%M). Exclusive: do not start GPU work." \
  "Lending: a booking fires at the next evaluation call (calls come at most ~35 min apart)." \
  "This file is rewritten when the chain ends."
bash $T/xfer_eval.sh cu 64200 12k || stopped "CU64.2k 12k FAILED"
bash $T/xfer_eval.sh cu 102800 12k || stopped "CU102.8k 12k FAILED"
for s in 11000 12000 13000 $(seq 16000 1000 25000); do
  [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free before CU102.8k s$s"
  bash $T/xfer_eval.sh cu 102800 seg $s 10 || stopped "CU102.8k segment s$s FAILED"
done
$PY - "$D" > $T/cu_final.txt 2>&1 <<'PY'
import glob, sys
import numpy as np
D = sys.argv[1]
def load(p):
    d = {}
    for f in glob.glob(f"{D}/{p}_s*_gpufp32.npz"):
        z = np.load(f); d.update({int(s): float(v) for s, v in zip(z["seed"], z["mean"])})
    return d
m = {k: load(p) for k, p in {"CU102.8k": "xfer_cu_s102800", "CU64.2k": "xfer_cu_s64200", "U40k": "xfer_u1_s40000",
     "CR278.6k": "xfer_cr_s278600", "C392k": "xfer_hist_s215800", "C' final": "xfer_hist_s278600", "408k": "tpt_tpt_s408000",
     "A160k": "xfer_v11s2_s160000", "C160k": "xfer_v11c2_s160000"}.items()}
m["pooled control"] = {s: (m["A160k"][s] + m["C160k"][s]) / 2 for s in set(m["A160k"]) & set(m["C160k"])}
twelve = set(range(10000, 11000)) | set(range(14000, 16000)); allg = set(range(10000, 26000))
def show(label, a, b=None, only=allg, tag=None):
    S = sorted(set(m[a]) & (set(m[b]) if b else set(m[a])) & only)
    if len(S) < 2:
        print(f"  {label:<40} missing"); return
    y = np.array([m[a][s] - (m[b][s] if b else 0.0) for s in S]); h = 1.96 * y.std(ddof=1) / np.sqrt(len(y)); z = y.mean() / (h / 1.96)
    print(f"  {label:<40} {len(S):>6,} groups  {y.mean():+.3f} +- {h:.3f}  z {z:+.2f}")
    if tag:
        print(f"{tag} {y.mean():.4f} {z:.3f}")
show("CU102.8k - U40k (PRIMARY)", "CU102.8k", "U40k", tag="PRIMARY")
show("replay: CR278.6k - C392k", "CR278.6k", "C392k")
show("history: C' final - C392k", "C' final", "C392k")
show("CU102.8k vs v4", "CU102.8k")
show("CU102.8k - CR278.6k", "CU102.8k", "CR278.6k")
show("CU102.8k - C' final", "CU102.8k", "C' final")
show("CU102.8k - pooled control", "CU102.8k", "pooled control")
show("CU102.8k - 408k", "CU102.8k", "408k")
show("CU64.2k - U40k (lambda 0.5 part), 12k", "CU64.2k", "U40k", only=twelve)
show("CU102.8k - CU64.2k (lambda 0.2 part), 12k", "CU102.8k", "CU64.2k", only=twelve)
PY
say "pairings: $(grep -v '^PRIMARY' $T/cu_final.txt | tr '\n' ' ')"
cd /mnt/d/Better_mortal/jax_rl/mjai_bot || stopped "no mjai_bot dir"
segs=""; for s in 10000 11000 12000 13000 14000 $(seq 16000 1000 25000); do segs="$segs xfer_cu_s102800_s$s"; done
$PY merge_dumps.py xfer_cu_s102800_f16k $segs > /dev/null 2>&1 || say "merge 16k CU102.8k FAILED"
$PY merge_dumps.py xfer_cu_s64200_12k xfer_cu_s64200_s10000 xfer_cu_s64200_s14000 > /dev/null 2>&1 || say "merge 12k CU64.2k FAILED"
for p in xfer_cu_s102800 xfer_u1_s40000; do
  $PY merge_dumps.py ${p}_12k ${p}_s10000 ${p}_s14000 > /dev/null 2>&1 || say "merge 12k $p FAILED"
done
$PY - > $T/cu_mech.txt 2>&1 <<'PY'
import json, os
R = os.path.expanduser("~/Projects/better_mortal/runs")
for tag in ["xfer_cu_s102800_f16k", "xfer_u1_s40000_f16k", "xfer_cr_s278600_f16k", "xfer_hist_s278600_all",
            "xfer_cu_s64200_12k", "xfer_cu_s102800_12k", "xfer_u1_s40000_12k"]:
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
say "mechanism: $(tr '\n' ' ' < $T/cu_mech.txt)"
read -r mean z <<< "$(grep '^PRIMARY' $T/cu_final.txt | cut -d' ' -f2-)"
v=$($PY -c "import sys; m, z = map(float, sys.argv[1:3]); print('HOLDS' if m >= 0.7 and z >= 2 else 'FAILS' if m <= 0.3 else 'UNDECIDED')" "${mean:-nan}" "${z:-nan}")
say "verdict: CU102.8k - U40k ${mean:-?} (z ${z:-?}): $v"
idle; say "CU_CHAIN_DONE"; lend_and_exit
