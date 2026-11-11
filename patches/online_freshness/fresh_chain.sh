#!/bin/bash
# Sample-freshness test (README.md): the v11 line's stage 2 repeated with every fresh sample trained on once
# (num_epochs = 1) instead of about 1.8 times, against the control already played (runs/xfer_v11s2, A160k).
#   1. runs/xfer_v11f from runs/xfer_v11/archive/xfer_v11_s80000.pth, set up exactly as xfer_v11s2 (re-anchored to
#      A80k, which is also the 'base' opponents; v4 in the pool) except num_epochs = 1; 80,000 -> 160,000 batches
#   2. 12k of the 120k and 160k snapshots; screening of F160k - A160k by SPRT 0 vs +1 (alpha 0.025, beta 0.2), looks
#      per 1,000 groups from seed 10000, capped at the 16,000 groups A160k has
#   3. paired readouts: F160k - A160k, F160k - A80k (the control gained +0.28 +- 0.52), F160k - tpt 408k, F160k vs v4
#   4. the default follow-on: mainline_next.sh from F160k (num_epochs 1) if the SPRT promoted, else from A240k
# Log runs/xfer_tools/fresh_chain.log; markers FRESH_CHAIN_DONE / FRESH_CHAIN_STOPPED.
# Re-runnable: a trained arm is not set up again, and xfer_eval.sh skips the segments already done.
# The WSL disk lives on D:, and D: filling up took WSL down (2026-10-29), so every segment first checks that D: has
# DMIN (10) GB free and stops cleanly otherwise.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/xfer_tools/fresh_chain.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/fresh_chain.log
D=$R/evaldump; ARM=v11f; SRC=$R/xfer_v11/archive/xfer_v11_s80000.pth; DMIN=${DMIN:-10}
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
dfree() { df -BG --output=avail /mnt/d | tail -1 | tr -dc 0-9; }
echo $$ > $T/fresh_chain.pid
idle=0
while [ $idle -lt 3 ]; do
  if pgrep -f "^$PY .*(train|run_eval|client|server|ev_map)\.py" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
  sleep 60
done
if grep -a -q " XFER_ARM_DONE$" $R/xfer_$ARM/arm.log 2>/dev/null; then
  say "resuming: arm $ARM already trained, D: $(dfree) GB free"
else
  say "GPU idle; arm $ARM from $SRC with num_epochs 1"
  EPOCHS=1 bash $T/xfer_setup.sh $ARM $SRC 5587 >> $LOG 2>&1 || { say "setup FAILED: FRESH_CHAIN_STOPPED"; exit 1; }
  STOP_AT=160000 bash $T/xfer_arm.sh $ARM
  grep -a -q " XFER_ARM_DONE$" $R/xfer_$ARM/arm.log || { say "arm did not reach 160k: FRESH_CHAIN_STOPPED"; exit 1; }
fi
say "arm trained; $(grep -a '^overall' $R/xfer_$ARM/arm.log | tail -1)"
pair() {
  $PY - "$D" <<'PY'
import glob, sys
import numpy as np
D = sys.argv[1]
pats = {"F160k": "xfer_v11f_s160000", "F120k": "xfer_v11f_s120000", "A160k": "xfer_v11s2_s160000",
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
for k in ("F120k", "F160k"):
    show(f"{k} vs v4", k)
show("F160k - A160k", "F160k", "A160k")
show("F160k - A80k", "F160k", "A80k")
show("A160k - A80k (same seeds)", "A160k", "A80k")
show("F160k - tpt 408k", "F160k", "408k")
PY
}
for st in 120000 160000; do bash $T/xfer_eval.sh $ARM $st 12k; done
pair > $T/fresh_12k.txt 2>&1; say "12k: $(tr '\n' ' ' < $T/fresh_12k.txt)"
seq_look() {
  $PY $T/seqtest.py "$D/xfer_${ARM}_s160000_s*_gpufp32.npz" "$D/xfer_v11s2_s160000_s*_gpufp32.npz" \
      --delta0 0 --delta1 1 --alpha 0.025 --beta 0.2 --look 1000 --nmax 16000 > $T/fresh_seq.txt 2>&1
  echo $?
}
verdict=3; hi=10999
rc=$(seq_look); say "look at 1,000 groups: exit $rc, $(tail -2 $T/fresh_seq.txt | tr -s " \n" " ")"; [ "$rc" = 3 ] || verdict=$rc
for s in 11000 12000 13000 $(seq 16000 1000 25000); do
  [ "$verdict" = 3 ] || [ "$verdict" = 0 ] || break      # drop or cap: stop playing; promote: play to the cap
  [ "$(dfree)" -ge $DMIN ] || { say "D: below $DMIN GB free before s$s: FRESH_CHAIN_STOPPED"; exit 1; }
  bash $T/xfer_eval.sh $ARM 160000 seg $s 10 || { say "segment s$s FAILED: FRESH_CHAIN_STOPPED"; exit 1; }
  if [ $s = 13000 ]; then hi=15999; else hi=$((s + 999)); fi
  if [ "$verdict" = 3 ]; then
    rc=$(seq_look); say "look through s$hi: exit $rc, $(tail -2 $T/fresh_seq.txt | tr -s " \n" " ")"; [ "$rc" = 3 ] || verdict=$rc
  fi
done
case $verdict in
  0) say "SPRT: promote";;
  1) say "SPRT: drop at $hi";;
  4) say "SPRT: undecided at the 16,000-group cap; read the fixed 16k";;
  *) say "SPRT: exit $verdict";;
esac
pair > $T/fresh_final.txt 2>&1; say "pairings: $(tr '\n' ' ' < $T/fresh_final.txt)"
$PY $T/xfer_reuse.py $R/xfer_$ARM >> $LOG 2>&1
say "FRESH_CHAIN_DONE"
if [ "$verdict" = 0 ]; then echo "v11f 160000 1" > $T/mainline.txt; else echo "v11s3 240000 1000000000" > $T/mainline.txt; fi
say "default follow-on: mainline_next.sh from $(cat $T/mainline.txt)"
exec bash $T/mainline_next.sh
