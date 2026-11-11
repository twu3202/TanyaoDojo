#!/bin/bash
# Reward-schedule test (README.md): the v11 line's stage 2 repeated with the old training reward [6, 4, 2, 0] of the
# historical C' stage instead of the eval-proportional [2.390, 1.195, 0, -3.585], against the control already played
# (runs/xfer_v11s2, A160k).
#   0. waits for the main-line loop to stop (MAINLINE_STOP, after ml10), then for an idle GPU
#   1. runs/xfer_v11r from runs/xfer_v11/archive/xfer_v11_s80000.pth, set up as xfer_v11s2 except pts; the generated
#      config must differ from v11s2's only in the pts line (paths and port aside); 80,000 -> 160,000 batches
#   2. 12k of the 120k and 160k snapshots; SPRT R160k - A160k (0 vs +1, alpha 0.025, beta 0.2), looks per 1,000
#      groups from seed 10000, capped at the 16,000 groups A160k has
#   3. paired readouts: R160k - A160k, R160k - A80k (the control gained +0.28 +- 0.52), R160k - tpt 408k, R160k vs v4
#   4. promote: the old-reward line continues by mainline_next.sh; otherwise the chain ends
# Log runs/xfer_tools/rew_chain.log; markers REW_CHAIN_DONE / REW_CHAIN_STOPPED. Re-runnable: a trained arm is not set
# up again and xfer_eval.sh skips finished segments. Every segment needs DMIN (10) GB free on D:.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/xfer_tools/rew_chain.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/rew_chain.log
D=$R/evaldump; ARM=v11r; SRC=$R/xfer_v11/archive/xfer_v11_s80000.pth; OLD="[6, 4, 2, 0]"; DMIN=${DMIN:-10}
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
dfree() { df -BG --output=avail /mnt/d | tail -1 | tr -dc 0-9; }
echo $$ > $T/rew_chain.pid
say "waiting for the main-line loop to stop"
until grep -a -q -E "MAINLINE_STOP found|MAINLINE_STOPPED" $T/mainline.log 2>/dev/null && ! pgrep -f mainline_next.sh > /dev/null; do
  sleep 120
done
idle=0
while [ $idle -lt 3 ]; do
  if pgrep -f "^$PY .*(train|run_eval|client|server|ev_map)\.py" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
  sleep 60
done
if grep -a -q " XFER_ARM_DONE$" $R/xfer_$ARM/arm.log 2>/dev/null; then
  say "resuming: arm $ARM already trained, D: $(dfree) GB free"
else
  [ "$(dfree)" -ge $DMIN ] || { say "D: below $DMIN GB free: REW_CHAIN_STOPPED"; exit 1; }
  if [ -e $R/xfer_$ARM/state.pth ]; then
    # set up by an earlier run of this chain but not trained to 160k: xfer_setup.sh refuses an existing run
    say "GPU idle; arm $ARM already set up (state.pth $(md5sum < $R/xfer_$ARM/state.pth | cut -c1-12)), D: $(dfree) GB free"
  else
    say "GPU idle; arm $ARM from $SRC with pts $OLD, D: $(dfree) GB free"
    PTS="$OLD" bash $T/xfer_setup.sh $ARM $SRC 5588 >> $LOG 2>&1 || { say "setup FAILED: REW_CHAIN_STOPPED"; exit 1; }
  fi
  norm() { sed -E -e "s/xfer_$1/ARM/g" -e 's/55[0-9]{2}/PORT/g' $B/configs/online_selfplay_xfer_$1.toml; }
  d=$(diff <(norm v11s2) <(norm $ARM) | grep "^[<>]")
  if [ "$(echo "$d" | grep -c -v -E '^[<>] pts = ')" != 0 ] || [ "$(echo "$d" | wc -l)" != 2 ]; then
    say "config differs from v11s2 beyond pts: $(echo "$d" | tr '\n' ' '): REW_CHAIN_STOPPED"; exit 1
  fi
  say "config vs v11s2: $(echo "$d" | tr '\n' ' ')"
  STOP_AT=160000 bash $T/xfer_arm.sh $ARM
  grep -a -q " XFER_ARM_DONE$" $R/xfer_$ARM/arm.log || { say "arm did not reach 160k: REW_CHAIN_STOPPED"; exit 1; }
fi
say "arm trained; $(grep -a '^overall' $R/xfer_$ARM/arm.log | tail -1)"
pair() {
  $PY - "$D" <<'PY'
import glob, sys
import numpy as np
D = sys.argv[1]
pats = {"R160k": "xfer_v11r_s160000", "R120k": "xfer_v11r_s120000", "A160k": "xfer_v11s2_s160000",
        "A80k": "xfer_v11_s80000", "408k": "tpt_tpt_s408000"}
m = {}
for k, p in pats.items():
    d = {}
    for f in glob.glob(f"{D}/{p}_s*_gpufp32.npz"):
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
for k in ("R120k", "R160k"):
    show(f"{k} vs v4", k)
show("R160k - A160k", "R160k", "A160k")
show("R160k - A80k", "R160k", "A80k")
show("A160k - A80k (same seeds)", "A160k", "A80k")
show("R160k - tpt 408k", "R160k", "408k")
PY
}
for st in 120000 160000; do bash $T/xfer_eval.sh $ARM $st 12k; done
pair > $T/rew_12k.txt 2>&1; say "12k: $(tr '\n' ' ' < $T/rew_12k.txt)"
seq_look() {
  $PY $T/seqtest.py "$D/xfer_${ARM}_s160000_s*_gpufp32.npz" "$D/xfer_v11s2_s160000_s*_gpufp32.npz" \
      --delta0 0 --delta1 1 --alpha 0.025 --beta 0.2 --look 1000 --nmax 16000 > $T/rew_seq.txt 2>&1
  echo $?
}
verdict=3; hi=10999
rc=$(seq_look); say "look at 1,000 groups: exit $rc, $(tail -2 $T/rew_seq.txt | tr -s " \n" " ")"; [ "$rc" = 3 ] || verdict=$rc
for s in 11000 12000 13000 $(seq 16000 1000 25000); do
  [ "$verdict" = 3 ] || [ "$verdict" = 0 ] || break      # drop or cap: stop playing; promote: play to the cap
  [ "$(dfree)" -ge $DMIN ] || { say "D: below $DMIN GB free before s$s: REW_CHAIN_STOPPED"; exit 1; }
  bash $T/xfer_eval.sh $ARM 160000 seg $s 10 || { say "segment s$s FAILED: REW_CHAIN_STOPPED"; exit 1; }
  if [ $s = 13000 ]; then hi=15999; else hi=$((s + 999)); fi
  if [ "$verdict" = 3 ]; then
    rc=$(seq_look); say "look through s$hi: exit $rc, $(tail -2 $T/rew_seq.txt | tr -s " \n" " ")"; [ "$rc" = 3 ] || verdict=$rc
  fi
done
case $verdict in
  0) say "SPRT: promote";;
  1) say "SPRT: drop at $hi";;
  4) say "SPRT: undecided at the 16,000-group cap; read the fixed 16k";;
  *) say "SPRT: exit $verdict";;
esac
pair > $T/rew_final.txt 2>&1; say "pairings: $(tr '\n' ' ' < $T/rew_final.txt)"
$PY $T/xfer_reuse.py $R/xfer_$ARM | tail -1 >> $LOG 2>&1
say "REW_CHAIN_DONE"
if [ "$verdict" = 0 ]; then
  rm -f $T/MAINLINE_STOP; echo "v11r 160000 1000000000 $OLD" > $T/mainline.txt
  say "promoted: the old-reward line continues by mainline_next.sh from $(cat $T/mainline.txt)"
  exec bash $T/mainline_next.sh
fi
say "not promoted: chain ends, GPU free"
