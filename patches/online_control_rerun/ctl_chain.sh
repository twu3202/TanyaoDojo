#!/bin/bash
# Control rerun (README.md): the v11 line's stage 2 control A160k (runs/xfer_v11s2) trained a second time, with the
# same start, config and code, to measure how much of its +0.28 over A80k is the recipe and how much its draw.
#   0. waits for an idle GPU (no trainer / worker / eval) and rewrites GPU_STATUS_mortal.txt
#   1. runs/xfer_v11c2 from runs/xfer_v11/archive/xfer_v11_s80000.pth, port 5595; the generated config must equal
#      v11s2's (paths and port aside)
#   2. 80,000 batches (80k -> 160k), as v11s2
#   3. the fixed 16,000 groups A160k has (seeds 10000-25999): 12k first, then the other 13 segments; no sequential
#      stop, this is an estimate, not a screen
#   4. paired readouts (C160k - A80k, - A160k, the pooled control, the four dropped variants against C160k and the
#      pooled control) and mechanism merges (12k and 16k)
#   5. the chain ends; GPU_STATUS_mortal.txt says FREE (idle)
# Log runs/xfer_tools/ctl_chain.log; markers CTL_CHAIN_DONE / CTL_CHAIN_STOPPED. Re-runnable: a set-up arm is not set
# up again, a trained arm is not trained again, and xfer_eval.sh skips finished segments. Every phase and segment
# needs DMIN (10) GB free on D:.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/xfer_tools/ctl_chain.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/ctl_chain.log
D=$R/evaldump; ARM=v11c2; SP=$R/xfer_v11c2; SRC=$R/xfer_v11/archive/xfer_v11_s80000.pth; DMIN=${DMIN:-10}
C=$B/configs; CFG=$C/online_selfplay_xfer_$ARM.toml; CK=$SP/archive/xfer_${ARM}_s160000.pth
W=/mnt/c/Users/asus; ST=$W/GPU_STATUS_mortal.txt
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
dfree() { df -BG --output=avail /mnt/d | tail -1 | tr -dc 0-9; }
norm() { sed -E -e "s/xfer_$1/ARM/g" -e 's/55[0-9]{2}/PORT/g' $C/online_selfplay_xfer_$1.toml; }
status() {
  printf '%s\r\n' "mortal GPU status (TanyaoDojo session), $(date '+%F %T %Z')" "$@" \
    "To book: create C:\\Users\\asus\\GPU_BOOK_<name>.txt (one line: who and what; add a line not_before=YYYY-MM-DD HH:MM to defer it)." \
    > $ST.new && mv $ST.new $ST
}
stopped() { say "$1: CTL_CHAIN_STOPPED"; status "State: FREE (idle) since $(date +%T): the control-rerun chain stopped ($1)."; exit 1; }
echo $$ > $T/ctl_chain.pid
say "waiting for an idle GPU"
idle=0
while [ $idle -lt 3 ]; do
  if pgrep -f "(mainline_next|anc_chain|pg_chain)\.sh" > /dev/null || pgrep -f "^$PY .*(train|train_pg|run_eval|client|server|ev_map)\.py" > /dev/null
  then idle=0; else idle=$((idle + 1)); fi
  sleep 60
done
status "State: BUSY - online RL training (control rerun, arm v11c2: 80,000 batches from A80k, about 9 hours from $(date +%H:%M)), then evaluation (16,000 groups, about 3 hours). Exclusive: do not start GPU work." \
  "Lending: a booking fires at the arm start or at each evaluation call (calls come at most ~35 min apart), never during training." \
  "This file is rewritten when the chain ends."
if [ -e $SP/state.pth ]; then
  say "arm $ARM already set up ($(head -1 $SP/manifest.txt)), D: $(dfree) GB free"
else
  [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free"
  say "GPU idle; arm $ARM from $SRC, D: $(dfree) GB free"
  bash $T/xfer_setup.sh $ARM $SRC 5595 >> $LOG 2>&1 || stopped "setup FAILED"
fi
if [ ! -f $CK ]; then
  d=$(diff <(norm v11s2) <(norm $ARM) | grep "^[<>]")
  [ -z "$d" ] || stopped "config differs from v11s2: $(echo "$d" | tr '\n' ' ')"
  say "config = v11s2's (paths and port aside)"
  STOP_AT=160000 bash $T/xfer_arm.sh $ARM
  [ -f $CK ] || stopped "arm did not reach 160k"
fi
say "arm trained; $(grep -a '^overall' $SP/arm.log | tr '\n' ' ')"
pair() {
  $PY - "$D" <<'PY'
import glob, sys
import numpy as np
D = sys.argv[1]
pats = {"C160k": "xfer_v11c2_s160000", "A160k": "xfer_v11s2_s160000", "A80k": "xfer_v11_s80000",
        "F160k": "xfer_v11f_s160000", "R160k": "xfer_v11r_s160000", "W200k": "xfer_v11w_s200000",
        "P160k": "xfer_p1_s160000", "408k": "tpt_tpt_s408000"}
m = {}
for k, p in pats.items():
    d = {}
    for f in glob.glob(f"{D}/{p}_s*_gpufp32.npz"):
        z = np.load(f)
        d.update({int(s): float(v) for s, v in zip(z["seed"], z["mean"])})
    m[k] = d
# pooled control: the mean of the two control runs, on the seeds both played
m["pooled"] = {s: (m["A160k"][s] + m["C160k"][s]) / 2 for s in set(m["A160k"]) & set(m["C160k"])}
def show(label, a, b=None):
    seeds = sorted(set(m[a]) & (set(m[b]) if b else set(m[a])))
    if len(seeds) < 2:
        return
    y = np.array([m[a][s] - (m[b][s] if b else 0.0) for s in seeds])
    h = 1.96 * y.std(ddof=1) / np.sqrt(len(y))
    print(f"  {label:<26} {len(seeds):>6,} groups  {y.mean():+.3f} +- {h:.3f}  z {y.mean() / (h / 1.96):+.2f}")
show("C160k vs v4", "C160k")
show("C160k - A80k", "C160k", "A80k")
show("A160k - A80k (same seeds)", "A160k", "A80k")
show("C160k - A160k", "C160k", "A160k")
show("pooled control - A80k", "pooled", "A80k")
for v in ("F160k", "R160k", "W200k", "P160k"):
    show(f"{v} - C160k", v, "C160k")
    show(f"{v} - pooled control", v, "pooled")
show("C160k - tpt 408k", "C160k", "408k")
PY
}
mech() {
  cd /mnt/d/Better_mortal/jax_rl/mjai_bot || return
  S16="s10000 s11000 s12000 s13000 s14000 $(seq -f 's%g' 16000 1000 25000 | tr '\n' ' ')"
  for p in xfer_v11c2_s160000 xfer_v11s2_s160000; do
    $PY merge_dumps.py ${p}_f16k $(for s in $S16; do echo ${p}_$s; done) > /dev/null || echo "merge ${p}_f16k failed"
  done
  $PY merge_dumps.py xfer_v11c2_s160000_f3k xfer_v11c2_s160000_s10000 xfer_v11c2_s160000_s14000 > /dev/null || echo "merge f3k failed"
  $PY - <<'PY'
import json, os
R = os.path.expanduser("~/Projects/better_mortal/runs")
for tag in ("xfer_v11_s80000_f3k", "xfer_v11s2_s160000_f3k", "xfer_v11c2_s160000_f3k", "xfer_v11s2_s160000_f16k", "xfer_v11c2_s160000_f16k"):
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
status "State: BUSY - evaluation of the control rerun (arm v11c2, 16,000 groups in 14 calls, about 3 hours from $(date +%H:%M)). Exclusive: do not start GPU work." \
  "Lending: a booking fires at the next evaluation call (calls come at most ~35 min apart)." \
  "This file is rewritten when the chain ends."
[ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free before the 12k"
bash $T/xfer_eval.sh $ARM 160000 12k || stopped "12k FAILED"
pair > $T/ctl_12k.txt 2>&1; say "12k: $(tr '\n' ' ' < $T/ctl_12k.txt)"
for s in 11000 12000 13000 $(seq 16000 1000 25000); do
  [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free before s$s"
  bash $T/xfer_eval.sh $ARM 160000 seg $s 10 || stopped "segment s$s FAILED"
done
pair > $T/ctl_final.txt 2>&1; say "pairings: $(tr '\n' ' ' < $T/ctl_final.txt)"
mech > $T/ctl_mech.txt 2>&1; say "mechanism: $(tr '\n' ' ' < $T/ctl_mech.txt)"
$PY $T/xfer_reuse.py $SP | tail -1 >> $LOG 2>&1
idle() {
  status "State: FREE (idle) since $(date +%T): the control-rerun chain has ended and no mortal GPU work is queued." \
    "An active booking (GPU_BOOK_<name>.txt, once past its not_before) gets the card within a minute; this file then shows the hold." \
    "This file is rewritten before any new mortal GPU work starts."
}
idle; say "CTL_CHAIN_DONE"
# nothing is queued after this chain and gpu_share.sh runs only at a boundary, so lend the idle card here for a day
# (as pg_free_watch.sh did), until new mortal GPU work starts
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
say "lending loop ends after a day"
