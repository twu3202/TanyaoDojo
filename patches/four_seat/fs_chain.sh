#!/bin/bash
# Four-seat self-play, variant S2 (README.md): the v11 line's stage 2 from A80k with three workers playing four-seat
# self-play (the trainee in every seat, all seats trained, one rotation per seed) and one worker 1v3 against v4, against
# the two control runs of the same stage (A160k = runs/xfer_v11s2, C160k = runs/xfer_v11c2).
#   0. waits for an idle card (no mortal GPU work, no compute apps, no guest GPU_BUSY_*) and rewrites GPU_STATUS_mortal.txt
#   1. runs/xfer_v11fs from runs/xfer_v11/archive/xfer_v11_s80000.pth, port 5597; the generated config must differ
#      from v11s2's only in capacity (4000 -> 1750) and the trainer's loader workers (num_workers 8 -> 4, throughput
#      only), paths and port aside. Two starts on 11-10 (01:02 with file_batch_size 4, 01:21 with 8 loader workers)
#      took WSL under the memory watch's line at the trainer's start: a four-seat file holds 4.3x a 1v3 file's samples
#   2. 80,000 batches (80k -> 160k), pool self,v4,self,self; a memory watch stops the arm if WSL has under 1.5 GB free
#   3. 12k of FS160k; SPRT FS160k - pooled control (the mean of A160k and C160k per group; 0 vs +1, alpha 0.025,
#      beta 0.2), looks per 1,000 groups from seed 10000, capped at the 16,000 groups both controls have
#   4. paired readouts (FS160k vs v4, - pooled, - A160k, - C160k, - A80k, - tpt 408k), mechanism readouts, reuse
#   5. the chain ends; GPU_STATUS_mortal.txt says FREE (idle); the idle card is lent for a day
# Log runs/xfer_tools/fs_chain.log; markers FS_CHAIN_DONE / FS_CHAIN_STOPPED. Re-runnable: a set-up arm is not set up
# again, a trained arm is not trained again, and xfer_eval.sh skips finished segments.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/xfer_tools/fs_chain.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/fs_chain.log
D=$R/evaldump; DP=$R/evaldump_pooled; ARM=v11fs; SP=$R/xfer_v11fs; SRC=$R/xfer_v11/archive/xfer_v11_s80000.pth
DMIN=${DMIN:-10}; C=$B/configs; CFG=$C/online_selfplay_xfer_$ARM.toml; CK=$SP/archive/xfer_${ARM}_s160000.pth
W=/mnt/c/Users/asus; ST=$W/GPU_STATUS_mortal.txt
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
dfree() { df -BG --output=avail /mnt/d | tail -1 | tr -dc 0-9; }
norm() { sed -E -e "s/xfer_$1/ARM/g" -e 's/55[0-9]{2}/PORT/g' $C/online_selfplay_xfer_$1.toml; }
status() {
  printf '%s\r\n' "mortal GPU status (TanyaoDojo session), $(date '+%F %T %Z')" "$@" \
    "To book: create C:\\Users\\asus\\GPU_BOOK_<name>.txt (one line: who and what; add a line not_before=YYYY-MM-DD HH:MM to defer it)." \
    > $ST.new && mv $ST.new $ST
}
idle() {
  status "State: FREE (idle) since $(date +%T): ${1:-the four-seat chain has ended and no mortal GPU work is queued.}" \
    "An active booking (GPU_BOOK_<name>.txt, once past its not_before) gets the card within a minute; this file then shows the hold." \
    "This file is rewritten before any new mortal GPU work starts."
}
stopped() { say "$1: FS_CHAIN_STOPPED"; idle "the four-seat chain stopped ($1)."; exit 1; }
mortal_busy() { pgrep -f "^$PY .*(train|train_pg|run_eval|client|server|critic_diag2?|fs_check)\.py" > /dev/null; }
echo $$ > $T/fs_chain.pid
say "waiting for an idle card"
idle=0; n=0
while [ $idle -lt 3 ]; do
  if mortal_busy || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] \
     || compgen -G "$W/GPU_BUSY_*.txt" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
  n=$((n + 1)); [ $n -gt 720 ] && stopped "card not idle for 12 hours"
  sleep 60
done
status "State: BUSY - online RL training (four-seat self-play arm v11fs: 80,000 batches from A80k, about 9 hours from $(date +%H:%M)), then evaluation (up to 16,000 groups, at most about 3 hours). Exclusive: do not start GPU work." \
  "Lending: a booking fires at the arm start or at each evaluation call (calls come at most ~35 min apart), never during training." \
  "This file is rewritten when the chain ends."
if [ -e $SP/state.pth ]; then
  say "arm $ARM already set up ($(head -1 $SP/manifest.txt)), D: $(dfree) GB free"
else
  [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free"
  say "card idle; arm $ARM from $SRC, D: $(dfree) GB free"
  bash $T/xfer_setup.sh $ARM $SRC 5597 >> $LOG 2>&1 || stopped "setup FAILED"
  awk '/^\[/{sec=$0}
       sec=="[online.server]" && /^capacity = /{print "capacity = 1750            # four-seat (patches/four_seat): ~3,150 files = ~1.13 M samples per drain, as the control"; next}
       sec=="[dataset]" && /^num_workers = /{print "num_workers = 4            # four-seat: 8 loader workers holding four-seat files ran WSL out of memory (11-10)"; next}
       {print}' $CFG > $CFG.new && mv $CFG.new $CFG
  echo "four-seat: capacity 1750, loader num_workers 4; pool self,v4,self,self; player.py $(md5sum $B/Mortal/mortal/player.py | cut -c1-12), run_selfplay_m1.sh $(md5sum $B/scripts/run_selfplay_m1.sh | cut -c1-12)" >> $SP/manifest.txt
fi
if [ ! -f $CK ]; then
  d=$(diff <(norm v11s2) <(norm $ARM) | grep "^[<>]")
  if [ "$(echo "$d" | grep -c .)" != 4 ] || [ "$(echo "$d" | grep -c -E '^> (capacity = 1750|num_workers = 4)( |$)')" != 2 ] \
     || [ "$(echo "$d" | grep -c -E '^< (capacity = 4000|num_workers = 8)( |$)')" != 2 ]; then
    stopped "config differs from v11s2 beyond capacity and num_workers: $(echo "$d" | tr '\n' ' ')"
  fi
  say "config vs v11s2: $(echo "$d" | tr -s ' ' | tr '\n' ' ')"
  # memory watch: WSL dying takes every process with it; stop the arm (state.pth is kept) if under 1 GB is free
  # (4 GB of swap behind it); under 3 GB, the largest processes are logged
  ( while [ ! -f $CK ]; do
      a=$(free -m | awk 'NR==2{print $7}')
      if [ -f $SP/pids ] && [ "$a" -lt 3000 ] && [ $((SECONDS % 60)) -lt 5 ]; then
        echo "$(date +%T) $a top: $(ps -eo rss,args --sort=-rss | sed -n '2,6p' | awk '{printf "%d MB %s %s | ", $1/1024, $2, $3}')" >> $SP/mem.log
      fi
      if [ -f $SP/pids ] && [ "$a" -lt 1000 ]; then
        say "WSL memory $a MB free: stopping the arm"; bash $T/xfer_helpers.sh stop $SP >> $LOG 2>&1; break
      fi
      [ -f $SP/pids ] && [ $((SECONDS % 60)) -lt 5 ] && echo "$(date +%T) $a" >> $SP/mem.log
      sleep 5
    done ) &
  MW=$!
  # MINBUF 1750: the trainer starts once the buffer holds ~600k samples, as the control's 4,000 one-seat files do
  MINBUF=1750 POOL=self,v4,self,self STOP_AT=160000 bash $T/xfer_arm.sh $ARM
  kill $MW 2>/dev/null
  [ -f $CK ] || stopped "arm did not reach 160k"
fi
say "arm trained; $(grep -a '^overall' $SP/arm.log | tr '\n' ' '); WSL memory min $(sort -k2 -n $SP/mem.log 2>/dev/null | head -1 | cut -d' ' -f2) MB free"
# the pooled control: per group, the mean of the two control runs (A160k and C160k, same seeds 10000-25999)
mkdir -p $DP
$PY - $D $DP <<'PY' >> $LOG 2>&1
import glob, os, sys
import numpy as np
D, DP = sys.argv[1], sys.argv[2]
n = 0
for fa in sorted(glob.glob(f"{D}/xfer_v11s2_s160000_s*_gpufp32.npz")):
    fc = fa.replace("xfer_v11s2_", "xfer_v11c2_")
    if not os.path.exists(fc):
        continue
    a, c = np.load(fa), np.load(fc)
    ma = dict(zip(a["seed"].tolist(), a["mean"].tolist())); mc = dict(zip(c["seed"].tolist(), c["mean"].tolist()))
    seeds = np.array(sorted(set(ma) & set(mc)))
    np.savez(fa.replace(D, DP).replace("xfer_v11s2_", "xfer_pooledctl_"), seed=seeds,
             mean=np.array([(ma[s] + mc[s]) / 2 for s in seeds]))
    n += len(seeds)
print(f"pooled control: {n} groups")
PY
pair() {
  $PY - "$D" "$DP" <<'PY'
import glob, sys
import numpy as np
D, DP = sys.argv[1], sys.argv[2]
pats = {"FS160k": f"{D}/xfer_v11fs_s160000", "pooled": f"{DP}/xfer_pooledctl_s160000", "A160k": f"{D}/xfer_v11s2_s160000",
        "C160k": f"{D}/xfer_v11c2_s160000", "A80k": f"{D}/xfer_v11_s80000", "408k": f"{D}/tpt_tpt_s408000"}
m = {}
for k, p in pats.items():
    d = {}
    for f in glob.glob(f"{p}_s*_gpufp32.npz"):
        z = np.load(f)
        d.update({int(s): float(v) for s, v in zip(z["seed"], z["mean"])})
    m[k] = d
def show(label, a, b=None):
    seeds = sorted(set(m[a]) & (set(m[b]) if b else set(m[a])))
    if len(seeds) < 2:
        return
    y = np.array([m[a][s] - (m[b][s] if b else 0.0) for s in seeds])
    h = 1.96 * y.std(ddof=1) / np.sqrt(len(y))
    print(f"  {label:<26} {len(seeds):>6,} groups  {y.mean():+.3f} +- {h:.3f}  z {y.mean() / (h / 1.96):+.2f}")
show("FS160k vs v4", "FS160k")
show("FS160k - pooled control", "FS160k", "pooled")
show("FS160k - A160k", "FS160k", "A160k")
show("FS160k - C160k", "FS160k", "C160k")
show("FS160k - A80k", "FS160k", "A80k")
show("pooled control - A80k", "pooled", "A80k")
show("FS160k - tpt 408k", "FS160k", "408k")
PY
}
mech() {
  cd /mnt/d/Better_mortal/jax_rl/mjai_bot || return
  segs=$(ls $R/gapdump | grep -oE "^xfer_v11fs_s160000_s[0-9]+\.json$" | sed 's/\.json$//' | tr '\n' ' ')
  $PY merge_dumps.py xfer_v11fs_s160000_fall $segs > /dev/null || echo "merge fall failed"
  $PY merge_dumps.py xfer_v11fs_s160000_f3k xfer_v11fs_s160000_s10000 xfer_v11fs_s160000_s14000 > /dev/null || echo "merge f3k failed"
  $PY - <<'PY'
import json, os
R = os.path.expanduser("~/Projects/better_mortal/runs")
for tag in ("xfer_v11_s80000_f3k", "xfer_v11s2_s160000_f3k", "xfer_v11c2_s160000_f3k", "xfer_v11fs_s160000_f3k",
            "xfer_v11s2_s160000_f16k", "xfer_v11c2_s160000_f16k", "xfer_v11fs_s160000_fall"):
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
status "State: BUSY - evaluation of the four-seat arm (v11fs, up to 16,000 groups in at most 14 calls, from $(date +%H:%M)). Exclusive: do not start GPU work." \
  "Lending: a booking fires at the next evaluation call (calls come at most ~35 min apart)." \
  "This file is rewritten when the chain ends."
[ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free before the 12k"
bash $T/xfer_eval.sh $ARM 160000 12k || stopped "12k FAILED"
pair > $T/fs_12k.txt 2>&1; say "12k: $(tr '\n' ' ' < $T/fs_12k.txt)"
seq_look() {
  $PY $T/seqtest.py "$D/xfer_${ARM}_s160000_s*_gpufp32.npz" "$DP/xfer_pooledctl_s160000_s*_gpufp32.npz" \
      --delta0 0 --delta1 1 --alpha 0.025 --beta 0.2 --look 1000 --nmax 16000 > $T/fs_seq.txt 2>&1
  echo $?
}
verdict=3; hi=10999
rc=$(seq_look); say "look at 1,000 groups: exit $rc, $(tail -2 $T/fs_seq.txt | tr -s " \n" " ")"; [ "$rc" = 3 ] || verdict=$rc
for s in 11000 12000 13000 $(seq 16000 1000 25000); do
  [ "$verdict" = 3 ] || [ "$verdict" = 0 ] || break      # drop or cap: stop playing; promote: play to the cap
  [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free before s$s"
  bash $T/xfer_eval.sh $ARM 160000 seg $s 10 || stopped "segment s$s FAILED"
  if [ $s = 13000 ]; then hi=15999; else hi=$((s + 999)); fi
  if [ "$verdict" = 3 ]; then
    rc=$(seq_look); say "look through s$hi: exit $rc, $(tail -2 $T/fs_seq.txt | tr -s " \n" " ")"; [ "$rc" = 3 ] || verdict=$rc
  fi
done
case $verdict in
  0) say "SPRT: promote";;
  1) say "SPRT: drop at $hi";;
  4) say "SPRT: undecided at the 16,000-group cap; read the fixed 16k";;
  *) say "SPRT: exit $verdict";;
esac
pair > $T/fs_final.txt 2>&1; say "pairings: $(tr '\n' ' ' < $T/fs_final.txt)"
mech > $T/fs_mech.txt 2>&1; say "mechanism: $(tr '\n' ' ' < $T/fs_mech.txt)"
$PY $T/xfer_reuse.py $SP | tail -1 >> $LOG 2>&1
idle; say "FS_CHAIN_DONE"
# nothing is queued after this chain and gpu_share.sh runs only at a boundary, so lend the idle card here for a day
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
