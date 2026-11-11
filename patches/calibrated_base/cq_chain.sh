#!/bin/bash
# Calibrated offline base (patches/calibrated_base/README.md): queued after the model-soup chain.
#   1. waits for SOUP_CHAIN_DONE / SOUP_CHAIN_STOPPED; a file runs/xfer_tools/CQ_SKIP skips this chain
#   2. offline fine-tune of v11 (runs/symaug/v11_best.pth, 1,220,000 steps) with Cal-QL ([cql] calibrated = true),
#      v11's offline recipe otherwise, for 40,000 steps -> runs/v11cq/v11cq_final.pth (snapshots every 10,000)
#   3. gate: the median top-2 Q gap on 40 Houou files must fall to <= 0.5 x v11's, else the chain stops
#   4. the control's first online stage from it: arm v11cq, the XFER recipe exactly as A80k (config checked equal),
#      80,000 batches
#   5. 12k of CQ80k, then SPRT CQ80k - A80k (0 vs +1, alpha 0.025, beta 0.2, a look per 1,000 groups, cap 16,000)
#   6. paired and mechanism readouts, Q-gap readout of CQ80k vs A80k, reuse, CQ_CHAIN_DONE, lends the card a day
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/xfer_tools/cq_chain.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/cq_chain.log
D=$R/evaldump; W=/mnt/c/Users/asus; ST=$W/GPU_STATUS_mortal.txt; DMIN=${DMIN:-10}; C=$B/configs
MM=$B/Mortal/mortal; OFF=$R/v11cq; OCFG=$C/local_train_v11cq.toml; SRC=$R/symaug/v11_best.pth
ARM=v11cq; SP=$R/xfer_v11cq; CK=$SP/archive/xfer_${ARM}_s80000.pth
S0=1220000; STOP=1260000; TRAIN_MD5=5df8e75e23617b9f5370913d18e4ec29; SRC_MD5=2ae5581c3b8be41af8b1b1f88ded6b89
HOUOU="$B/data/houou/2026/*.mjson*"
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
dfree() { df -BG --output=avail /mnt/d | tail -1 | tr -dc 0-9; }
status() {
  printf '%s\r\n' "mortal GPU status (TanyaoDojo session), $(date '+%F %T %Z')" "$@" \
    "To book: create C:\\Users\\asus\\GPU_BOOK_<name>.txt (one line: who and what; add a line not_before=YYYY-MM-DD HH:MM to defer it)." \
    > $ST.new && mv $ST.new $ST
}
idle() {
  status "State: FREE (idle) since $(date +%T): ${1:-the calibrated-base chain has ended and no mortal GPU work is queued.}" \
    "An active booking (GPU_BOOK_<name>.txt, once past its not_before) gets the card within a minute; this file then shows the hold." \
    "This file is rewritten before any new mortal GPU work starts."
}
stopped() { say "$1: CQ_CHAIN_STOPPED"; idle "the calibrated-base chain stopped ($1)."; exit 1; }
mortal_busy() { pgrep -f "^$PY .*(train|train_pg|run_eval|client|server|critic_diag2?|fs_check)\.py" > /dev/null; }
last_step() { grep -a -o "total steps: [0-9,]*" $OFF/train.log 2>/dev/null | tail -1 | tr -dc 0-9; }
echo $$ > $T/cq_chain.pid

# 1. wait for the soup chain
say "waiting for the model-soup chain to end"
until grep -qE 'SOUP_CHAIN_(DONE|STOPPED)' $T/soup_chain.log 2>/dev/null; do sleep 120; done
[ -e $T/CQ_SKIP ] && { say "CQ_SKIP present: not run: CQ_CHAIN_SKIPPED"; exit 0; }
say "soup chain ended; waiting for an idle card"
idle=0; n=0
while [ $idle -lt 3 ]; do
  if mortal_busy || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] \
     || compgen -G "$W/GPU_BUSY_*.txt" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
  n=$((n + 1)); [ $n -gt 720 ] && stopped "card not idle for 12 hours"
  sleep 60
done
[ -e $T/CQ_SKIP ] && { say "CQ_SKIP present: not run: CQ_CHAIN_SKIPPED"; exit 0; }
[ "$(md5sum $MM/train.py | cut -d' ' -f1)" = $TRAIN_MD5 ] || stopped "train.py is not the Cal-QL build $TRAIN_MD5"
[ "$(md5sum $SRC | cut -d' ' -f1)" = $SRC_MD5 ] || stopped "v11_best.pth md5 differs"

# 2. offline fine-tune with Cal-QL
if [ ! -f $OFF/v11cq_final.pth ]; then
  status "State: BUSY - offline fine-tune of v11 with Cal-QL (40,000 steps, about 1.7 hours from $(date +%H:%M)), then online RL (80,000 batches, about 9 hours) and evaluation (up to 3 hours). Exclusive: do not start GPU work." \
    "Lending: a booking fires at the online arm's start or at each evaluation call, never during training." \
    "This file is rewritten when the chain ends."
  [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free"
  mkdir -p $OFF $OFF/tb $OFF/test_play $OFF/1v3
  if [ ! -f $OFF/mortal.pth ]; then
    cp $SRC $OFF/mortal.pth
    cp $R/v11long/file_index.pth $OFF/file_index.pth
    sed -e "s#/runs/v11long/#/runs/v11cq/#g" -e "s#^name = 'v11long'#name = 'v11cq'#" $C/local_train_v11long.toml \
      | awk '{print} /^\[cql\]/{print "calibrated = true          # Cal-QL (patches/calibrated_base): untaken actions are not pushed below the data return"}' > $OCFG
    d=$(diff <(sed -e 's#/runs/v11long/#/runs/X/#g' -e "s/'v11long'/'X'/" $C/local_train_v11long.toml) \
             <(sed -e 's#/runs/v11cq/#/runs/X/#g' -e "s/'v11cq'/'X'/" $OCFG) | grep '^[<>]')
    [ "$d" = "> calibrated = true          # Cal-QL (patches/calibrated_base): untaken actions are not pushed below the data return" ] \
      || stopped "offline config differs from v11long's beyond the Cal-QL line: $d"
    say "offline run $OFF from $SRC (steps $S0, LR at its 1e-5 floor); config = v11long's + calibrated = true"
  fi
  # unit check of the Cal-QL term: no gradient into actions whose Q is below the data return
  ( cd $MM && $PY - >> $LOG 2>&1 <<'PY'
import torch
q = torch.tensor([[1.0, 0.2, -0.5, -float("inf")]], requires_grad=True)
masks = torch.tensor([[True, True, True, False]]); g = torch.tensor([0.5]); a = 0
q_cal = torch.where(masks, torch.maximum(q, g[:, None]), q)
loss = q_cal.logsumexp(-1).mean() - q[0, a]
loss.backward()
gr = q.grad[0].tolist()
assert gr[1] == 0 and gr[2] == 0 and gr[0] != 0 and gr[3] == 0, gr
print("Cal-QL unit check ok: grad", [round(x, 4) for x in gr])
PY
  ) || stopped "Cal-QL unit check failed"
  nrs=0; snap_done=""
  cd $MM
  while true; do
    MORTAL_CFG=$OCFG setsid $PY train.py >> $OFF/train.log 2>&1 < /dev/null &
    pid=$!; echo $pid > $OFF/train.pid
    say "offline train.py pid $pid started (attempt $nrs, last logged step $(last_step))"
    while kill -0 $pid 2>/dev/null; do
      st=$(last_step); st=${st:-0}
      for s in 1230000 1240000 1250000; do
        if [ "$st" -ge $s ] && ! echo "$snap_done" | grep -q $s; then
          sleep 20; cp $OFF/mortal.pth $OFF/v11cq_s$s.pth; snap_done="$snap_done $s"
          say "snapshot v11cq_s$s.pth (checkpoint steps $($PY -c "import torch;print(torch.load('$OFF/v11cq_s$s.pth',weights_only=True,map_location='cpu')['steps'])"))"
        fi
      done
      if [ "$st" -ge $STOP ]; then
        sleep 20
        cp $OFF/mortal.pth $OFF/v11cq_final.pth
        # the trainer plays its 1,000 test games against v4 right after this save; keep them as the sanity readout
        n0=$(grep -ac 'avg pt' $OFF/train.log); w=0
        while [ $(grep -ac 'avg pt' $OFF/train.log) -le $n0 ] && [ $w -lt 900 ] && kill -0 $pid 2>/dev/null; do sleep 15; w=$((w + 15)); done
        say "reached $st (checkpoint steps $($PY -c "import torch;print(torch.load('$OFF/v11cq_final.pth',weights_only=True,map_location='cpu')['steps'])")) -> v11cq_final.pth; stopping pid $pid"
        kill -TERM -$pid 2>/dev/null || kill -TERM $pid; sleep 60
        kill -0 $pid 2>/dev/null && { kill -KILL -$pid 2>/dev/null || kill -KILL $pid; }
        break 2
      fi
      [ "$st" -ge $((STOP - 4000)) ] && sleep 10 || sleep 60
    done
    nrs=$((nrs + 1))
    say "offline trainer exited at logged step $(last_step); tail: $(tail -n 40 $OFF/train.log | tr '\r' '\n' | grep -v 'TRAIN:' | tail -n 4 | tr '\n' ' ')"
    [ $nrs -gt 3 ] && stopped "offline trainer crashed $nrs times"
    sleep 60
  done
  say "offline test_play (vs v4, 1,000 games each): $(grep -a 'avg pt' $OFF/train.log | awk '{print $NF}' | tr '\n' ' ')"
  lastpt=$(grep -a 'avg pt' $OFF/train.log | tail -1 | awk '{print $NF}')
  [ -n "$lastpt" ] && awk -v x=$lastpt 'BEGIN{exit !(x < -8)}' && stopped "offline test_play $lastpt < -8: the calibrated base is broken"
fi

# 3. gate: Q-gap readout
cd $MM
DEVICE=cuda:0 PYTHONPATH=$MM $PY $T/gap_readout.py --files "$HOUOU" --n-files 40 v11=$SRC \
  $(for s in 1230000 1240000 1250000; do [ -f $OFF/v11cq_s$s.pth ] && echo "cq_s$s=$OFF/v11cq_s$s.pth"; done) \
  cq_final=$OFF/v11cq_final.pth > $T/cq_gap_offline.txt 2>&1 || stopped "gap readout failed"
say "Q-gap readout: $(tr -s ' ' < $T/cq_gap_offline.txt | tr '\n' '|')"
ratio=$(awk '/^  v11 /{for(i=1;i<=NF;i++) if($i=="gap"){g0=$(i+1)}} /^  cq_final /{for(i=1;i<=NF;i++) if($i=="gap"){g1=$(i+1)}} END{if(g0>0) printf "%.3f", g1/g0}' $T/cq_gap_offline.txt)
say "median top-2 gap, cq_final / v11 = $ratio"
[ -n "$ratio" ] || stopped "gap ratio not parsed"
if ! awk -v r=$ratio 'BEGIN{exit !(r <= 0.5)}'; then
  # addendum (README, 11-10 20:40): the large-gap gate, used only when relaunched with CQ_GATE=large
  alt=$($PY - $T/cq_gap_offline.txt <<'PY'
import re, sys
rows = {}
for line in open(sys.argv[1]):
    m = re.match(r"\s+(\S+)\s+steps .*call ([\d.]+).*p90 ([\d.]+);", line)
    if m:
        rows[m.group(1)] = (float(m.group(2)), float(m.group(3)))
c0, p0 = rows["v11"]; c1, p1 = rows["cq_final"]
print(f"{c1 / c0:.3f} {p1 / p0:.3f}")
PY
)
  say "gate NOT MET: the gap fell to $ratio of v11's (needs <= 0.5); call-available and p90 ratios: ${alt:-unparsed}"
  if [ "${CQ_GATE:-median}" = large ] && [ -n "$alt" ] && echo $alt | awk '{exit !($1 <= 0.5 && $2 <= 0.5)}'; then
    say "large-gap gate met (call-available and p90 gaps both <= 0.5 x v11's): the online stage runs (README addendum)"
    gate_large=1
  else
    idle; say "CQ_CHAIN_DONE (gate not met)"; exit 0
  fi
fi
[ -n "${gate_large:-}" ] || say "gate met (ratio $ratio <= 0.5)"

# 4. the control's first online stage from the calibrated base
norm() { sed -E -e "s/xfer_$1/ARM/g" -e 's/55[0-9]{2}|560[0-9]/PORT/g' $C/online_selfplay_xfer_$1.toml; }
if [ ! -e $SP/state.pth ]; then
  [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free before the arm"
  bash $T/xfer_setup.sh $ARM $OFF/v11cq_final.pth 5602 >> $LOG 2>&1 || stopped "setup FAILED"
fi
if [ ! -f $CK ]; then
  d=$(diff <(norm v11) <(norm $ARM) | grep "^[<>]")
  [ -z "$d" ] || stopped "online config differs from A80k's (xfer_v11): $(echo "$d" | tr '\n' ' ')"
  say "online config identical to A80k's (xfer_v11) up to paths and port"
  status "State: BUSY - online RL training (arm v11cq: 80,000 batches from the Cal-QL base, about 9 hours from $(date +%H:%M)), then evaluation (up to 16,000 groups, at most about 3 hours). Exclusive: do not start GPU work." \
    "Lending: a booking fires at the arm start or at each evaluation call (calls come at most ~35 min apart), never during training." \
    "This file is rewritten when the chain ends."
  STOP_AT=80000 bash $T/xfer_arm.sh $ARM >> $LOG 2>&1
  [ -f $CK ] || stopped "arm did not reach 80k"
  say "arm trained; $($PY $T/xfer_reuse.py $SP | tail -1)"
fi

# 5. 12k, then SPRT against A80k
status "State: BUSY - evaluation of the Cal-QL arm (v11cq, up to 16,000 groups in at most 14 calls, from $(date +%H:%M)). Exclusive: do not start GPU work." \
  "Lending: a booking fires at the next evaluation call (calls come at most ~35 min apart)." \
  "This file is rewritten when the chain ends."
[ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free before the 12k"
bash $T/xfer_eval.sh $ARM 80000 12k || stopped "12k FAILED"
seq_look() {
  $PY $T/seqtest.py "$D/xfer_${ARM}_s80000_s*_gpufp32.npz" "$D/xfer_v11_s80000_s*_gpufp32.npz" \
      --delta0 0 --delta1 1 --alpha 0.025 --beta 0.2 --look 1000 --nmax 16000 > $T/cq_seq.txt 2>&1
  echo $?
}
verdict=3
rc=$(seq_look); say "look at 1,000 groups: exit $rc, $(tail -2 $T/cq_seq.txt | tr -s " \n" " ")"; [ "$rc" = 3 ] || verdict=$rc
for s in 11000 12000 13000 $(seq 16000 1000 25000); do
  [ "$verdict" = 3 ] || break
  [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free before segment s$s"
  bash $T/xfer_eval.sh $ARM 80000 seg $s 10 || stopped "segment s$s FAILED"
  hi=$((s + 999))
  rc=$(seq_look); say "look through s$hi: exit $rc, $(tail -2 $T/cq_seq.txt | tr -s " \n" " ")"; [ "$rc" = 3 ] || verdict=$rc
done
case $verdict in
  0) say "SPRT: promote";;
  1) say "SPRT: drop at $hi";;
  4) say "SPRT: undecided at the 16,000-group cap; read the fixed 16k";;
  *) say "SPRT: exit $verdict";;
esac

# 6. readouts
$PY - "$D" > $T/cq_final.txt 2>&1 <<'PY'
import glob, sys
import numpy as np
D = sys.argv[1]
pats = {"CQ80k": "xfer_v11cq_s80000", "A80k": "xfer_v11_s80000", "A160k": "xfer_v11s2_s160000",
        "C160k": "xfer_v11c2_s160000", "408k": "tpt_tpt_s408000", "U64k": "xfer_u1_s64000"}
m = {}
for k, p in pats.items():
    d = {}
    for f in glob.glob(f"{D}/{p}_s*_gpufp32.npz"):
        z = np.load(f); d.update({int(s): float(v) for s, v in zip(z["seed"], z["mean"])})
    m[k] = d
def show(label, a, b=None):
    seeds = sorted(set(m[a]) & (set(m[b]) if b else set(m[a])))
    if len(seeds) < 2:
        print(f"  {label:<20} (no common groups)"); return
    y = np.array([m[a][s] - (m[b][s] if b else 0.0) for s in seeds])
    h = 1.96 * y.std(ddof=1) / np.sqrt(len(y))
    print(f"  {label:<20} {len(seeds):>6,} groups  {y.mean():+.3f} +- {h:.3f}  z {y.mean() / (h / 1.96):+.2f}")
show("CQ80k vs v4", "CQ80k")
for b in ("A80k", "A160k", "C160k", "U64k", "408k"):
    show(f"CQ80k - {b}", "CQ80k", b)
PY
say "pairings: $(tr -s ' ' < $T/cq_final.txt | tr '\n' '|')"
cd /mnt/d/Better_mortal/jax_rl/mjai_bot && {
  segs=$(ls $R/gapdump | grep -oE "^xfer_${ARM}_s80000_s[0-9]+\.json$" | sed 's/\.json$//' | tr '\n' ' ')
  $PY merge_dumps.py xfer_${ARM}_s80000_fall $segs > /dev/null || say "merge fall failed"
  $PY merge_dumps.py xfer_${ARM}_s80000_f3k xfer_${ARM}_s80000_s10000 xfer_${ARM}_s80000_s14000 > /dev/null || say "merge f3k failed"
  $PY - > $T/cq_mech.txt 2>&1 <<'PY'
import json, os
R = os.path.expanduser("~/Projects/better_mortal/runs")
for tag in ("xfer_v11_s80000_f3k", "xfer_v11cq_s80000_f3k", "xfer_v11_s80000_100k", "xfer_v11cq_s80000_fall"):
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
say "mechanism: $(tr -s ' ' < $T/cq_mech.txt | tr '\n' '|')"
cd $MM && DEVICE=cuda:0 PYTHONPATH=$MM $PY $T/gap_readout.py --files "$HOUOU" --n-files 40 \
  A80k=$R/xfer_v11/archive/xfer_v11_s80000.pth CQ80k=$CK > $T/cq_gap_online.txt 2>&1
say "online Q-gap readout: $(tr -s ' ' < $T/cq_gap_online.txt | tr '\n' '|')"
idle; say "CQ_CHAIN_DONE"
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
