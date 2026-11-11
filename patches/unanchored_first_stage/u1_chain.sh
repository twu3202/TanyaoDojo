#!/bin/bash
# P2, short version (README.md): does an unanchored first stage from v11 grow rl1_best's aggressive profile, and does
# anchored training then repair it the way C' did?
#   0. waits for an idle GPU (no mortal work, no guest hold) and rewrites GPU_STATUS_mortal.txt
#   1. phase 1, arm u1: runs/xfer_u1 from v11 (runs/symaug/v11_best.pth, md5 2ae5581c), port 5596, July's B1 recipe:
#      no anchor, LR 5e-6, Boltzmann temperature 0.05, pts [6,4,2,0]. The config may differ from v11s2's only in
#      exactly these five lines (paths and port aside). 64,000 batches (rl1_best was taken at 60,000).
#   2. 12k of U32k (trajectory), the 16,000 groups of U64k (seeds 10000-25999), pairings and mechanism
#   3. gate, set in advance: U64k attack >= +10 points per kyoku AND riichi >= +1.0 pp (both against the v4 seats,
#      16,000 groups) AND U64k - A80k >= -0.5. Met -> phase 2; not met -> the chain ends.
#   4. phase 2, arm u1a: runs/xfer_u1a from the U64k archive, anchored to it, with the replay control's recipe
#      (config = v11s2's), 80,000 batches (64k -> 144k); the 16,000 groups of U144k, pairings and mechanism
#   5. the card is marked FREE and lent to bookings for a day
# Log runs/xfer_tools/u1_chain.log; markers U1_CHAIN_DONE / U1_CHAIN_STOPPED. Re-runnable: set-up arms are not set up
# again, trained arms not trained again, finished segments skipped. Stop: kill $(cat runs/xfer_tools/u1_chain.pid),
# then bash runs/xfer_tools/xfer_helpers.sh stop runs/xfer_u1 (or xfer_u1a) if an arm is training.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/xfer_tools/u1_chain.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/u1_chain.log
D=$R/evaldump; C=$B/configs; DMIN=${DMIN:-10}; W=/mnt/c/Users/asus; ST=$W/GPU_STATUS_mortal.txt
SRC=$R/symaug/v11_best.pth
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
dfree() { df -BG --output=avail /mnt/d | tail -1 | tr -dc 0-9; }
norm() { sed -E -e "s/xfer_$1/ARM/g" -e 's/55[0-9]{2}/PORT/g' -e 's/ *#.*$//' $C/online_selfplay_xfer_$1.toml | grep -v '^\s*$'; }
status() {
  printf '%s\r\n' "mortal GPU status (TanyaoDojo session), $(date '+%F %T %Z')" "$@" \
    "To book: create C:\\Users\\asus\\GPU_BOOK_<name>.txt (one line: who and what; add a line not_before=YYYY-MM-DD HH:MM to defer it)." \
    > $ST.new && mv $ST.new $ST
}
idle() {
  status "State: FREE (idle) since $(date +%T): the unanchored-first-stage chain (P2) has ended and no mortal GPU work is queued." \
    "An active booking (GPU_BOOK_<name>.txt, once past its not_before) gets the card within a minute; this file then shows the hold." \
    "This file is rewritten before any new mortal GPU work starts."
}
stopped() { say "$1: U1_CHAIN_STOPPED"; idle; exit 1; }
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
echo $$ > $T/u1_chain.pid
say "waiting for an idle GPU"
n=0
while [ $n -lt 3 ]; do
  # busy: mortal GPU work, a guest hold (gpu_share.sh waiting) or a guest's GPU_BUSY_ file
  if pgrep -f "^$PY .*(train|train_pg|run_eval|client|server|ev_map)\.py" > /dev/null || pgrep -f "gpu_share\.sh" > /dev/null \
     || ls $W/GPU_BUSY_*.txt > /dev/null 2>&1
  then n=0; else n=$((n + 1)); fi
  sleep 60
done
# ---------------- phase 1: unanchored, July's B1 recipe ----------------
ARM=u1; SP=$R/xfer_u1; CFG=$C/online_selfplay_xfer_u1.toml; CK1=$SP/archive/xfer_u1_s64000.pth
status "State: BUSY - online RL training (P2 phase 1: arm u1, unanchored, 64,000 batches from v11, about 7.5 hours from $(date +%H:%M)), then evaluation (about 4 hours); a phase 2 of about 12 hours may follow. Exclusive: do not start GPU work." \
  "Lending: a booking fires at an arm start or at an evaluation call, never during training." \
  "This file is rewritten at each phase and when the chain ends."
if [ ! -e $SP/state.pth ]; then
  [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free"
  [ "$(md5sum < $SRC | cut -c1-8)" = 2ae5581c ] || stopped "v11 base md5 mismatch"
  say "arm u1 from $SRC, D: $(dfree) GB free"
  PTS="[6.0, 4.0, 2.0, 0.0]" ANCHOR=0 bash $T/xfer_setup.sh u1 $SRC 5596 >> $LOG 2>&1 || stopped "setup u1 FAILED"
  sed -i -e "s/^peak = .*/peak = 5e-6                # P2: July B1 recipe (rl1)/" -e "s/^final = .*/final = 5e-6/" \
         -e "s/^boltzmann_temp = .*/boltzmann_temp = 0.05      # P2: July B1 recipe (rl1), upstream value/" $CFG
fi
if [ ! -f $CK1 ]; then
  d=$(diff <(norm v11s2) <(norm u1) | grep "^[<>]")
  want='^> (pts = \[6\.0, 4\.0, 2\.0, 0\.0\]|weight = 0|peak = 5e-6|final = 5e-6|boltzmann_temp = 0\.05)$'
  [ "$(echo "$d" | grep -c '^<')" = 5 ] && [ "$(echo "$d" | grep -c '^>')" = 5 ] && [ "$(echo "$d" | grep -c -E "$want")" = 5 ] \
    || stopped "phase 1 config differs from v11s2 beyond the five recipe lines: $(echo "$d" | tr '\n' ' ')"
  say "phase 1 config vs v11s2: $(echo "$d" | tr '\n' ' ')"
  STOP_AT=64000 bash $T/xfer_arm.sh u1
  [ -f $CK1 ] || stopped "phase 1 did not reach 64k"
fi
say "phase 1 trained; $(grep -a '^overall' $SP/arm.log | tr '\n' ' ')"
pair() {   # pair <label> <model pattern>... printed against the reference lines
  $PY - "$D" "$@" <<'PY'
import glob, sys
import numpy as np
D = sys.argv[1]; which = sys.argv[2]
pats = {"U32k": "xfer_u1_s32000", "U64k": "xfer_u1_s64000", "U144k": "xfer_u1a_s144000", "A80k": "xfer_v11_s80000",
        "A160k": "xfer_v11s2_s160000", "C160k": "xfer_v11c2_s160000", "rl1_best": "xfer_hist_s60000",
        "C' final": "xfer_hist_s278600", "408k": "tpt_tpt_s408000", "v11": "v11_v11"}
m = {}
for k, p in pats.items():
    d = {}
    for f in glob.glob(f"{D}/{p}_s*_gpufp32.npz"):
        z = np.load(f)
        d.update({int(s): float(v) for s, v in zip(z["seed"], z["mean"])})
    m[k] = d
m["pooled control"] = {s: (m["A160k"][s] + m["C160k"][s]) / 2 for s in set(m["A160k"]) & set(m["C160k"])}
def show(label, a, b=None):
    seeds = sorted(set(m[a]) & (set(m[b]) if b else set(m[a])))
    if len(seeds) < 2:
        return None
    y = np.array([m[a][s] - (m[b][s] if b else 0.0) for s in seeds])
    h = 1.96 * y.std(ddof=1) / np.sqrt(len(y)); z = y.mean() / (h / 1.96)
    print(f"  {label:<30} {len(seeds):>6,} groups  {y.mean():+.3f} +- {h:.3f}  z {z:+.2f}")
    return y.mean()
if which == "gate":
    r = show("U64k - A80k", "U64k", "A80k"); print(f"GATE_STRENGTH {r if r is not None else 'nan'}"); sys.exit(0)
if which == "phase1":
    for k in ("U32k", "U64k"):
        show(f"{k} vs v4", k)
    for b in ("A80k", "rl1_best", "v11", "C' final", "408k"):
        show(f"U64k - {b}", "U64k", b)
    show("U32k - A80k", "U32k", "A80k")
else:
    show("U144k vs v4", "U144k")
    show("U144k - U64k (the anchored stage)", "U144k", "U64k")
    show("pooled control - A80k (same recipe, A80k start)", "pooled control", "A80k")
    for b in ("pooled control", "A160k", "C' final", "408k"):
        show(f"U144k - {b}", "U144k", b)
PY
}
mech() {   # mech <tag>...: merge the 16 segments of each tag, then print the readout of each merged tag
  cd /mnt/d/Better_mortal/jax_rl/mjai_bot || return
  for p in "$@"; do
    segs=""; for s in 10000 11000 12000 13000 14000 $(seq 16000 1000 25000); do segs="$segs ${p}_s$s"; done
    [ -f $R/gapdump/${p}_s10000.json ] && $PY merge_dumps.py ${p}_f16k $segs > /dev/null
  done
  $PY - "$@" <<'PY'
import json, os, sys
R = os.path.expanduser("~/Projects/better_mortal/runs")
for p in sys.argv[1:]:
    tag = p if p.endswith("_all") or p.endswith("_100k") else p + "_f16k"
    try:
        g = json.load(open(f"{R}/gapdump/{tag}.json")); pl = json.load(open(f"{R}/placedump/{tag}.json")); b = json.load(open(f"{R}/bustdump/{tag}.json"))
    except OSError:
        print(f"  {tag}: missing"); continue
    o, t = g["ours"], g["theirs"]
    r = lambda d, k: d[k] / d["kyoku"]
    att = r(o, "win_pts") - r(t, "win_pts"); dfn = -(r(o, "deal_in_pts") - r(t, "deal_in_pts"))
    n = pl["ours"]["n"]; f4 = pl["ours"]["rank"][3] / n; f4v = (n - pl["ours"]["rank"][3]) / (3 * n)
    ri = 100 * (r(o, "riichi") - r(t, "riichi"))
    print(f"  {tag:<24} n={g['n_games']} call {100*r(o,'naki'):.2f} ({100*r(t,'naki'):.2f}) win {100*(r(o,'win')-r(t,'win')):+.2f} "
          f"dealin {100*(r(o,'deal_in')-r(t,'deal_in')):+.2f} riichi {ri:+.2f} "
          f"net {att+dfn:+.0f} ({att:+.0f} / {dfn:+.0f}) bust {100*b['ours']['bust']/b['ours']['n']:.2f} / {100*b['v4']['bust']/b['v4']['n']:.2f} "
          f"4th {100*(f4-f4v):+.2f}")
    print(f"MECH {tag} {att:.2f} {ri:.3f}")
PY
}
ev16() {   # ev16 <arm> <step>: the 16,000 groups, 12k first
  bash $T/xfer_eval.sh $1 $2 12k || stopped "$1 s$2 12k FAILED"
  for s in 11000 12000 13000 $(seq 16000 1000 25000); do
    [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free before $1 s$2 s$s"
    bash $T/xfer_eval.sh $1 $2 seg $s 10 || stopped "$1 s$2 segment s$s FAILED"
  done
}
status "State: BUSY - evaluation of P2 phase 1 (U32k 12k, U64k 16,000 groups), about 4 hours from $(date +%H:%M). Exclusive: do not start GPU work." \
  "Lending: a booking fires at the next evaluation call (calls come at most ~35 min apart)." \
  "This file is rewritten at each phase and when the chain ends."
bash $T/xfer_eval.sh u1 32000 12k || stopped "U32k 12k FAILED"
ev16 u1 64000
pair phase1 > $T/u1_phase1.txt 2>&1; say "phase 1 pairings: $(tr '\n' ' ' < $T/u1_phase1.txt)"
mech xfer_u1_s64000 xfer_hist_s60000_all xfer_v11_s80000_100k > $T/u1_mech1.txt 2>&1
say "phase 1 mechanism: $(grep -v '^MECH' $T/u1_mech1.txt | tr '\n' ' ')"
read -r att ri <<< "$(grep '^MECH xfer_u1_s64000_f16k' $T/u1_mech1.txt | cut -d' ' -f3-)"
str=$(pair gate | grep GATE_STRENGTH | cut -d' ' -f2)
gate=$($PY -c "import sys; a, r, s = map(float, sys.argv[1:4]); print('MET' if a >= 10 and r >= 1.0 and s >= -0.5 else 'NOT_MET')" "${att:-nan}" "${ri:-nan}" "${str:-nan}")
say "gate: attack ${att:-?} (>= +10), riichi ${ri:-?} pp (>= +1.0), U64k - A80k ${str:-?} (>= -0.5): $gate"
$PY $T/xfer_reuse.py $SP | tail -1 >> $LOG 2>&1
if [ "$gate" != MET ]; then idle; say "gate not met: no phase 2"; say "U1_CHAIN_DONE"; lend_and_exit; fi
# ---------------- phase 2: anchored to U64k with the replay control's recipe ----------------
ARM=u1a; SP=$R/xfer_u1a; CFG=$C/online_selfplay_xfer_u1a.toml; CK2=$SP/archive/xfer_u1a_s144000.pth
status "State: BUSY - online RL training (P2 phase 2: arm u1a, anchored to U64k, 80,000 batches, about 9 hours from $(date +%H:%M)), then evaluation (about 3 hours). Exclusive: do not start GPU work." \
  "Lending: a booking fires at an arm start or at an evaluation call, never during training." \
  "This file is rewritten when the chain ends."
if [ ! -e $SP/state.pth ]; then
  [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free before phase 2"
  bash $T/xfer_setup.sh u1a $CK1 5597 >> $LOG 2>&1 || stopped "setup u1a FAILED"
  # train.py restores the LR scheduler's own peak / final from an online state, overriding the config. U64k carries
  # phase 1's 5e-6, so set them to the control's 1e-5 here, or phase 2 would silently train at half the control's LR.
  $PY - $SP/state.pth >> $LOG 2>&1 <<'PY' || stopped "LR reset of u1a state.pth FAILED"
import os, sys, torch
p = sys.argv[1]; s = torch.load(p, weights_only=True, map_location="cpu")
old = (s["scheduler"]["peak"], s["scheduler"]["final"])
s["scheduler"]["peak"] = s["scheduler"]["final"] = 1e-5
s["scheduler"]["_last_lr"] = [1e-5 for _ in s["scheduler"]["_last_lr"]]
for g in s["optimizer"]["param_groups"]:
    g["lr"] = 1e-5
torch.save(s, p + ".new"); os.replace(p + ".new", p)
print("u1a state.pth: scheduler peak/final", old, "-> 1e-5 (the control's LR)")
PY
fi
if [ ! -f $CK2 ]; then
  d=$(diff <(norm v11s2) <(norm u1a) | grep "^[<>]")
  [ -z "$d" ] || stopped "phase 2 config differs from v11s2: $(echo "$d" | tr '\n' ' ')"
  say "phase 2 config = v11s2's (paths and port aside), state = U64k"
  STOP_AT=144000 bash $T/xfer_arm.sh u1a
  [ -f $CK2 ] || stopped "phase 2 did not reach 144k"
fi
say "phase 2 trained; $(grep -a '^overall' $SP/arm.log | tr '\n' ' ')"
status "State: BUSY - evaluation of P2 phase 2 (U144k, 16,000 groups), about 3 hours from $(date +%H:%M). Exclusive: do not start GPU work." \
  "Lending: a booking fires at the next evaluation call." "This file is rewritten when the chain ends."
ev16 u1a 144000
pair phase2 > $T/u1_phase2.txt 2>&1; say "phase 2 pairings: $(tr '\n' ' ' < $T/u1_phase2.txt)"
mech xfer_u1a_s144000 xfer_u1_s64000 xfer_hist_s278600_all > $T/u1_mech2.txt 2>&1
say "phase 2 mechanism: $(grep -v '^MECH' $T/u1_mech2.txt | tr '\n' ' ')"
$PY $T/xfer_reuse.py $SP | tail -1 >> $LOG 2>&1
idle; say "U1_CHAIN_DONE"
lend_and_exit
