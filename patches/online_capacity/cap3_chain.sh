#!/bin/bash
# Online capacity test, stage 2 control (README.md): the v11 arm (192 x 40) continued from 80k to 160k batches exactly
# as B1's arm was in cap2_chain.sh: re-anchored to its own 80k snapshot, which also replaces the offline base as the
# 'base' opponents; v4 stays in the pool.
#   1. wait for CAP2_CHAIN_DONE / _STOPPED, then for an idle GPU
#   2. runs/xfer_v11s2 from runs/xfer_v11/archive/xfer_v11_s80000.pth (full online state), 80,000 -> 160,000 batches
#   3. 12k of the 120k and 160k snapshots; screening of the stage gain A160k - A80k by SPRT 0 vs +1 (alpha 0.025,
#      beta 0.2), one look per 1,000 groups from seed 10000 (A80k has the full 100k); a promote is confirmed by the
#      rest of the 100k
#   4. paired readouts on everything played: A160k - A80k, A160k - 408k, C160k - A160k (capacity at equal steps)
# Log runs/xfer_tools/cap3_chain.log; markers CAP3_CHAIN_DONE / CAP3_CHAIN_STOPPED as the last word of a line.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/xfer_tools/cap3_chain.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/cap3_chain.log
D=$R/evaldump; ARM=v11s2; SRC=$R/xfer_v11/archive/xfer_v11_s80000.pth
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
echo $$ > $T/cap3_chain.pid
say "waiting for stage 2 of arm C (cap2_chain)"
until grep -a -q -E " CAP2_CHAIN_(DONE|STOPPED)$" $T/cap2_chain.log 2>/dev/null; do sleep 120; done
[ -f $SRC ] || { say "no $SRC: CAP3_CHAIN_STOPPED"; exit 1; }
idle=0
while [ $idle -lt 3 ]; do
  if pgrep -f "^$PY .*(train|run_eval|client|server|ev_map)\.py" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
  sleep 60
done
say "GPU idle; stage 2 from $SRC"
bash $T/xfer_setup.sh $ARM $SRC 5585 >> $LOG 2>&1 || { say "setup FAILED: CAP3_CHAIN_STOPPED"; exit 1; }
STOP_AT=160000 bash $T/xfer_arm.sh $ARM
grep -a -q " XFER_ARM_DONE$" $R/xfer_$ARM/arm.log || { say "stage 2 did not reach 160k: CAP3_CHAIN_STOPPED"; exit 1; }
say "stage 2 trained; $(grep -a '^overall' $R/xfer_$ARM/arm.log | tail -1)"
pair() {   # paired readouts over every seed that all listed models cover
  $PY - "$D" <<'PY'
import glob, sys
import numpy as np
D = sys.argv[1]
pats = {"A160k": "xfer_v11s2_s160000", "A120k": "xfer_v11s2_s120000", "A80k": "xfer_v11_s80000",
        "408k": "tpt_tpt_s408000", "C160k": "xfer_b1s2_s160000"}
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
for k in ("A120k", "A160k"):
    show(f"{k} vs v4", k)
show("A160k - A80k", "A160k", "A80k")
show("A160k - tpt 408k", "A160k", "408k")
show("C160k - A160k", "C160k", "A160k")
PY
}
for st in 120000 160000; do bash $T/xfer_eval.sh $ARM $st 12k; done
pair > $T/cap3_12k.txt 2>&1; say "12k: $(tr '\n' ' ' < $T/cap3_12k.txt)"
seq_look() {
  $PY $T/seqtest.py "$D/xfer_${ARM}_s160000_s*_gpufp32.npz" "$D/xfer_v11_s80000_s*_gpufp32.npz" \
      --delta0 0 --delta1 1 --alpha 0.025 --beta 0.2 --look 1000 > $T/cap3_seq.txt 2>&1
  echo $?
}
verdict=3; hi=10999
rc=$(seq_look); say "look at 1,000 groups: exit $rc, $(tail -2 $T/cap3_seq.txt | tr -s " \n" " ")"; [ "$rc" = 3 ] || verdict=$rc
for s in 11000 12000 13000 $(seq 16000 1000 34000); do
  [ "$verdict" = 1 ] && break                       # drop: stop playing
  bash $T/xfer_eval.sh $ARM 160000 seg $s 10 || { say "segment s$s FAILED: CAP3_CHAIN_STOPPED"; exit 1; }
  if [ $s = 13000 ]; then hi=15999; else hi=$((s + 999)); fi
  if [ "$verdict" = 3 ]; then
    rc=$(seq_look); say "look through s$hi: exit $rc, $(tail -2 $T/cap3_seq.txt | tr -s " \n" " ")"; [ "$rc" = 3 ] || verdict=$rc
  fi
done
case $verdict in
  0) say "SPRT: promote (confirmed on the full 100k above)";;
  1) say "SPRT: drop at $hi";;
  4) say "SPRT: undecided at the cap; read the fixed 100k";;
  *) say "SPRT: exit $verdict";;
esac
pair > $T/cap3_final.txt 2>&1; say "pairings: $(tr '\n' ' ' < $T/cap3_final.txt)"
say "CAP3_CHAIN_DONE"
