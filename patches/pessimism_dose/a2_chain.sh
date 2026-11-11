#!/bin/bash
# Pessimism dose, second round (patches/pessimism_dose/README.md "Second round"): two more fine-tune doses of v11.
#   1. waits for an idle card; a file runs/xfer_tools/A2_SKIP skips this chain
#   2. offline fine-tunes of v11 (runs/symaug/v11_best.pth, 1,220,000 steps), v11's offline recipe with the
#      conservative weight min_q_weight 5 -> 0.2 (runs/v11a02), then -> 0 (runs/v11a00); 20,000 steps each
#   3. gate per dose: p90 top-2 Q gap <= 0.7 x v11's (40 files), greedy mean Q <= v11's + 0.5 (10 files), the
#      trainer's test games vs v4 not below -8; the largest passing weight goes online
#   4. the control's first online stage from it, exactly as A80k (config checked equal), port 5603
#   5. 12k, then SPRT A2_80k - A80k (0 vs +1, alpha 0.025, beta 0.2, a look per 1,000 groups, cap 16,000)
#   6. paired and mechanism readouts, Q-gap and Q-level readouts, A2_CHAIN_DONE, lends the card a day
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/xfer_tools/a2_chain.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/a2_chain.log
D=$R/evaldump; W=/mnt/c/Users/asus; ST=$W/GPU_STATUS_mortal.txt; DMIN=${DMIN:-10}; C=$B/configs
MM=$B/Mortal/mortal; SRC=$R/symaug/v11_best.pth; A1F=$R/v11a1/v11a1_final.pth
S0=1220000; STOP=1240000; TRAIN_MD5=5df8e75e23617b9f5370913d18e4ec29; SRC_MD5=2ae5581c3b8be41af8b1b1f88ded6b89
HOUOU="$B/data/houou/2026/*.mjson*"
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
dfree() { df -BG --output=avail /mnt/d | tail -1 | tr -dc 0-9; }
status() {
  printf '%s\r\n' "mortal GPU status (TanyaoDojo session), $(date '+%F %T %Z')" "$@" \
    "To book: create C:\\Users\\asus\\GPU_BOOK_<name>.txt (one line: who and what; add a line not_before=YYYY-MM-DD HH:MM to defer it)." \
    > $ST.new && mv $ST.new $ST
}
idle() {
  status "State: FREE (idle) since $(date +%T): ${1:-the pessimism-dose chain (second round) has ended and no mortal GPU work is queued.}" \
    "An active booking (GPU_BOOK_<name>.txt, once past its not_before) gets the card within a minute; this file then shows the hold." \
    "This file is rewritten before any new mortal GPU work starts."
}
stopped() { say "$1: A2_CHAIN_STOPPED"; idle "the pessimism-dose chain (second round) stopped ($1)."; exit 1; }
mortal_busy() { pgrep -f "^$PY .*(train|train_pg|run_eval|client|server|critic_diag2?|fs_check)\.py" > /dev/null; }
last_step() { grep -a -o "total steps: [0-9,]*" $OFF/train.log 2>/dev/null | tail -1 | tr -dc 0-9; }
echo $$ > $T/a2_chain.pid

# 1. wait for an idle card
[ -e $T/A2_SKIP ] && { say "A2_SKIP present: not run: A2_CHAIN_SKIPPED"; exit 0; }
say "waiting for an idle card"
idle=0; n=0
while [ $idle -lt 3 ]; do
  if mortal_busy || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] \
     || compgen -G "$W/GPU_BUSY_*.txt" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
  n=$((n + 1)); [ $n -gt 720 ] && stopped "card not idle for 12 hours"
  sleep 60
done
[ -e $T/A2_SKIP ] && { say "A2_SKIP present: not run: A2_CHAIN_SKIPPED"; exit 0; }
[ "$(md5sum $MM/train.py | cut -d' ' -f1)" = $TRAIN_MD5 ] || stopped "train.py is not the Cal-QL build $TRAIN_MD5"
[ "$(md5sum $SRC | cut -d' ' -f1)" = $SRC_MD5 ] || stopped "v11_best.pth md5 differs"
[ -f $A1F ] || stopped "round one's final snapshot $A1F is missing"

# 2. offline fine-tunes: weight 0.2, then weight 0
offline() {  # NAME WT
  NAME=$1; WT=$2; OFF=$R/v11$NAME; OCFG=$C/local_train_v11$NAME.toml
  if [ ! -f $OFF/v11${NAME}_final.pth ]; then
    status "State: BUSY - offline fine-tune of v11 with conservative weight $WT (20,000 steps, about 50 minutes from $(date +%H:%M)); two doses in all, then possibly online RL (80,000 batches, about 9 hours) and evaluation (up to 3 hours). Exclusive: do not start GPU work." \
      "Lending: a booking fires at the online arm's start or at each evaluation call, never during training." \
      "This file is rewritten when the chain ends."
    [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free"
    mkdir -p $OFF $OFF/tb $OFF/test_play $OFF/1v3
    if [ ! -f $OFF/mortal.pth ]; then
      cp $SRC $OFF/mortal.pth
      cp $R/v11long/file_index.pth $OFF/file_index.pth
      sed -e "s#/runs/v11long/#/runs/v11$NAME/#g" -e "s#^name = 'v11long'#name = 'v11$NAME'#" \
          -e "s|^min_q_weight = 5\$|min_q_weight = $WT          # pessimism dose, round two (patches/pessimism_dose): v11 used 5|" \
          $C/local_train_v11long.toml > $OCFG
      d=$(diff <(sed -e 's#/runs/v11long/#/runs/X/#g' -e "s/'v11long'/'X'/" $C/local_train_v11long.toml) \
               <(sed -e "s#/runs/v11$NAME/#/runs/X/#g" -e "s/'v11$NAME'/'X'/" $OCFG) | grep '^[<>]' | tr '\n' '|')
      [ "$d" = "< min_q_weight = 5|> min_q_weight = $WT          # pessimism dose, round two (patches/pessimism_dose): v11 used 5|" ] \
        || stopped "offline config differs from v11long's beyond the conservative weight: $d"
      grep -q '^calibrated' $OCFG && stopped "offline config has a Cal-QL line"
      say "offline run $OFF from $SRC (steps $S0, LR at its 1e-5 floor); config = v11long's with min_q_weight $WT"
    fi
    nrs=0; snap_done=""
    cd $MM
    while true; do
      MORTAL_CFG=$OCFG setsid $PY train.py >> $OFF/train.log 2>&1 < /dev/null &
      pid=$!; echo $pid > $OFF/train.pid
      say "offline train.py pid $pid started (attempt $nrs, last logged step $(last_step))"
      while kill -0 $pid 2>/dev/null; do
        st=$(last_step); st=${st:-0}
        for s in 1230000; do
          if [ "$st" -ge $s ] && ! echo "$snap_done" | grep -q $s; then
            sleep 20; cp $OFF/mortal.pth $OFF/v11${NAME}_s$s.pth; snap_done="$snap_done $s"
            say "snapshot v11${NAME}_s$s.pth (checkpoint steps $($PY -c "import torch;print(torch.load('$OFF/v11${NAME}_s$s.pth',weights_only=True,map_location='cpu')['steps'])"))"
          fi
        done
        if [ "$st" -ge $STOP ]; then
          sleep 20
          cp $OFF/mortal.pth $OFF/v11${NAME}_final.pth
          # the trainer plays its 1,000 test games against v4 right after this save; keep them as the sanity readout
          n0=$(grep -ac 'avg pt' $OFF/train.log); w=0
          while [ $(grep -ac 'avg pt' $OFF/train.log) -le $n0 ] && [ $w -lt 900 ] && kill -0 $pid 2>/dev/null; do sleep 15; w=$((w + 15)); done
          say "reached $st (checkpoint steps $($PY -c "import torch;print(torch.load('$OFF/v11${NAME}_final.pth',weights_only=True,map_location='cpu')['steps'])")) -> v11${NAME}_final.pth; stopping pid $pid"
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
    say "$NAME offline test_play (vs v4, 1,000 games each): $(grep -a 'avg pt' $OFF/train.log | awk '{print $NF}' | tr '\n' ' ')"
  fi
}
offline a02 0.2
offline a00 0.0

# 3. gate per dose: Q-gap and Q-level readouts
cd $MM
DEVICE=cuda:0 PYTHONPATH=$MM $PY $T/gap_readout.py --files "$HOUOU" --n-files 40 v11=$SRC a1_final=$A1F \
  a02_s1230000=$R/v11a02/v11a02_s1230000.pth a02_final=$R/v11a02/v11a02_final.pth \
  a00_s1230000=$R/v11a00/v11a00_s1230000.pth a00_final=$R/v11a00/v11a00_final.pth > $T/a2_gap_offline.txt 2>&1 \
  || stopped "gap readout failed"
say "Q-gap readout: $(tr -s ' ' < $T/a2_gap_offline.txt | tr '\n' '|')"
DEVICE=cuda:0 PYTHONPATH=$MM $PY $T/q_levels.py --files "$HOUOU" --n-files 10 v11=$SRC a1_final=$A1F \
  a02_final=$R/v11a02/v11a02_final.pth a00_final=$R/v11a00/v11a00_final.pth > $T/a2_qlev_offline.txt 2>&1 \
  || stopped "Q-level readout failed"
say "Q-level readout: $(grep -v -e Warning -e 'others =' $T/a2_qlev_offline.txt | tr -s ' ' | tr '\n' '|')"
ARM=""
for NAME in a02 a00; do
  g=$($PY - $T/a2_gap_offline.txt $T/a2_qlev_offline.txt $NAME <<'PY'
import re, sys
p90 = {m.group(1): float(m.group(2)) for m in (re.match(r"\s+(\S+)\s+steps .*p90 ([\d.]+);", l) for l in open(sys.argv[1])) if m}
lev = {m.group(1): float(m.group(2)) for m in (re.match(r"\s+(\S+)\s+all:\s+greedy ([-+\d.]+)", l) for l in open(sys.argv[2])) if m}
k = sys.argv[3] + "_final"
print(f"{p90[k] / p90['v11']:.3f} {lev[k] - lev['v11']:+.3f}")
PY
)
  [ -n "$g" ] || stopped "gate readouts for $NAME not parsed"
  set -- $g
  tp=$(grep -a 'avg pt' $R/v11$NAME/train.log | tail -1 | awk '{print $NF}')
  say "gate $NAME: p90 ratio $1 (needs <= 0.7), greedy mean Q minus v11's $2 (needs <= +0.5), last test games vs v4 ${tp:-none} (needs >= -8)"
  if awk -v r=$1 -v l=$2 -v t=${tp:--99} 'BEGIN{exit !(r <= 0.7 && l <= 0.5 && t >= -8)}'; then
    [ -z "$ARM" ] && ARM=v11$NAME
  fi
done
[ -n "$ARM" ] || { say "gate NOT MET for both doses: a fine-tune of v11 cannot build the arm"; idle; say "A2_CHAIN_DONE (gate not met)"; exit 0; }
BASE=$R/$ARM/${ARM}_final.pth; SP=$R/xfer_$ARM; CK=$SP/archive/xfer_${ARM}_s80000.pth
say "gate met: $ARM ($BASE) goes online"

# 4. the control's first online stage from the chosen base
norm() { sed -E -e "s/xfer_$1/ARM/g" -e 's/55[0-9]{2}|560[0-9]/PORT/g' $C/online_selfplay_xfer_$1.toml; }
if [ ! -e $SP/state.pth ]; then
  [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free before the arm"
  bash $T/xfer_setup.sh $ARM $BASE 5603 >> $LOG 2>&1 || stopped "setup FAILED"
fi
if [ ! -f $CK ]; then
  d=$(diff <(norm v11) <(norm $ARM) | grep "^[<>]")
  [ -z "$d" ] || stopped "online config differs from A80k's (xfer_v11): $(echo "$d" | tr '\n' ' ')"
  say "online config identical to A80k's (xfer_v11) up to paths and port"
  status "State: BUSY - online RL training (arm $ARM: 80,000 batches from the pessimism-dose base, about 9 hours from $(date +%H:%M)), then evaluation (up to 16,000 groups, at most about 3 hours). Exclusive: do not start GPU work." \
    "Lending: a booking fires at the arm start or at each evaluation call (calls come at most ~35 min apart), never during training." \
    "This file is rewritten when the chain ends."
  STOP_AT=80000 bash $T/xfer_arm.sh $ARM >> $LOG 2>&1
  [ -f $CK ] || stopped "arm did not reach 80k"
  say "arm trained; $($PY $T/xfer_reuse.py $SP | tail -1)"
fi

# 5. 12k, then SPRT against A80k
status "State: BUSY - evaluation of the pessimism-dose arm ($ARM, up to 16,000 groups in at most 14 calls, from $(date +%H:%M)). Exclusive: do not start GPU work." \
  "Lending: a booking fires at the next evaluation call (calls come at most ~35 min apart)." \
  "This file is rewritten when the chain ends."
[ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free before the 12k"
bash $T/xfer_eval.sh $ARM 80000 12k || stopped "12k FAILED"
seq_look() {
  $PY $T/seqtest.py "$D/xfer_${ARM}_s80000_s*_gpufp32.npz" "$D/xfer_v11_s80000_s*_gpufp32.npz" \
      --delta0 0 --delta1 1 --alpha 0.025 --beta 0.2 --look 1000 --nmax 16000 > $T/a2_seq.txt 2>&1
  echo $?
}
verdict=3
rc=$(seq_look); say "look at 1,000 groups: exit $rc, $(tail -2 $T/a2_seq.txt | tr -s " \n" " ")"; [ "$rc" = 3 ] || verdict=$rc
for s in 11000 12000 13000 $(seq 16000 1000 25000); do
  [ "$verdict" = 3 ] || break
  [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free before segment s$s"
  bash $T/xfer_eval.sh $ARM 80000 seg $s 10 || stopped "segment s$s FAILED"
  hi=$((s + 999))
  rc=$(seq_look); say "look through s$hi: exit $rc, $(tail -2 $T/a2_seq.txt | tr -s " \n" " ")"; [ "$rc" = 3 ] || verdict=$rc
done
case $verdict in
  0) say "SPRT: promote";;
  1) say "SPRT: drop at $hi";;
  4) say "SPRT: undecided at the 16,000-group cap; read the fixed 16k";;
  *) say "SPRT: exit $verdict";;
esac

# 6. readouts
$PY - "$D" "$ARM" > $T/a2_final.txt 2>&1 <<'PY'
import glob, sys
import numpy as np
D = sys.argv[1]
pats = {"A2_80k": f"xfer_{sys.argv[2]}_s80000", "A80k": "xfer_v11_s80000", "A160k": "xfer_v11s2_s160000",
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
show("A2_80k vs v4", "A2_80k")
for b in ("A80k", "A160k", "C160k", "U64k", "408k"):
    show(f"A2_80k - {b}", "A2_80k", b)
PY
say "pairings: $(tr -s ' ' < $T/a2_final.txt | tr '\n' '|')"
cd /mnt/d/Better_mortal/jax_rl/mjai_bot && {
  segs=$(ls $R/gapdump | grep -oE "^xfer_${ARM}_s80000_s[0-9]+\.json$" | sed 's/\.json$//' | tr '\n' ' ')
  $PY merge_dumps.py xfer_${ARM}_s80000_fall $segs > /dev/null || say "merge fall failed"
  $PY merge_dumps.py xfer_${ARM}_s80000_f3k xfer_${ARM}_s80000_s10000 xfer_${ARM}_s80000_s14000 > /dev/null || say "merge f3k failed"
  $PY - "$ARM" > $T/a2_mech.txt 2>&1 <<'PY'
import json, os, sys
R = os.path.expanduser("~/Projects/better_mortal/runs")
a = sys.argv[1]
for tag in ("xfer_v11_s80000_f3k", f"xfer_{a}_s80000_f3k", "xfer_v11_s80000_100k", f"xfer_{a}_s80000_fall"):
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
say "mechanism: $(tr -s ' ' < $T/a2_mech.txt | tr '\n' '|')"
cd $MM && DEVICE=cuda:0 PYTHONPATH=$MM $PY $T/gap_readout.py --files "$HOUOU" --n-files 40 \
  A80k=$R/xfer_v11/archive/xfer_v11_s80000.pth A2_80k=$CK > $T/a2_gap_online.txt 2>&1
say "online Q-gap readout: $(tr -s ' ' < $T/a2_gap_online.txt | tr '\n' '|')"
DEVICE=cuda:0 PYTHONPATH=$MM $PY $T/q_levels.py --files "$HOUOU" --n-files 10 \
  A80k=$R/xfer_v11/archive/xfer_v11_s80000.pth A2_80k=$CK > $T/a2_qlev_online.txt 2>&1
say "online Q-level readout: $(tr -s ' ' < $T/a2_qlev_online.txt | tr '\n' '|')"
idle; say "A2_CHAIN_DONE"
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
