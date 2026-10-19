#!/bin/bash
# Online transfer test on the 5060 Ti (README.md), queued behind the q12one evaluation chain:
#   1. wait for Q12ONE_CHAIN_DONE / _STOPPED, then for no trainer or evaluator of this venv for 5 minutes;
#      runs/xfer_tools/HOLD (if present) holds the start
#   2. arm B's base = q12one_final if its 100k vs v4 is above q25_final's on the same 25,000 groups, else q25_final
#   3. arm A (v11) 0 -> 80,000 batches, 12k of its 40k and 80k snapshots; then arm B the same way; 12k pairings
#   4. the other 88k of both 80k snapshots, 100k pairing
# Log runs/xfer_tools/chain.log; markers XFER_CHAIN_DONE / XFER_CHAIN_STOPPED as the last word of a line.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/xfer_tools/xfer_chain.sh > /dev/null 2>&1 &)
# Cancel before it starts training: kill $(cat runs/xfer_tools/chain.pid); after: also xfer_helpers.sh stop <run dir>.
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/chain.log
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
echo $$ > $T/chain.pid
say "chain waiting for the q12one chain"
until grep -a -q -E " Q12ONE_CHAIN_(DONE|STOPPED)$" $R/q12one/chain.log 2>/dev/null; do sleep 300; done
idle=0
while [ $idle -lt 5 ]; do
  if [ -f $T/HOLD ] || pgrep -f "^$PY .*(train|run_eval|client|server)\.py" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
  sleep 60
done
pick=$($PY - <<'PY'
import glob, os
import numpy as np
D = os.path.expanduser("~/Projects/better_mortal/runs/evaldump")
want = set(range(10000, 35000))
out = {}
for tag in ("q25_final", "q12one_final"):
    m = {}
    for f in glob.glob(f"{D}/{tag}_s*_gpufp32.npz"):
        z = np.load(f)
        for s, v in zip(z["seed"], z["mean"]):
            m.setdefault(int(s), float(v))
    if want <= set(m):
        out[tag] = float(np.mean([m[s] for s in sorted(want)]))
best = "q12one_final" if out.get("q12one_final", -1e9) > out.get("q25_final", -1e9) else "q25_final"
print(best, " ".join(f"{k}={v:+.3f}" for k, v in out.items()))
PY
)
set -- $pick; BBASE=$1; BARM=${BBASE%_final}
say "arm B = $BARM (100k vs v4: ${pick#* })"
case $BARM in q25) BPATH=$R/q25/q25_final.pth;; q12one) BPATH=$R/q12one/q12one_final.pth;; esac

run_arm() {   # ARM BASE PORT
  bash $T/xfer_setup.sh $1 $2 $3 >> $LOG 2>&1 || { say "setup $1 FAILED: XFER_CHAIN_STOPPED"; exit 1; }
  bash $T/xfer_arm.sh $1
  if ! grep -a -q " XFER_ARM_DONE$" $R/xfer_$1/arm.log; then say "arm $1 did not reach 80k: XFER_CHAIN_STOPPED"; exit 1; fi
  say "arm $1 trained; $(grep -a '^overall' $R/xfer_$1/arm.log | tail -1)"
  bash $T/xfer_eval.sh $1 40000 12k
  bash $T/xfer_eval.sh $1 80000 12k
}
run_arm v11 $R/symaug/v11_best.pth 5581
run_arm $BARM $BPATH 5582
for st in 40000 80000; do
  $PY $T/xfer_pair.py $BARM $BBASE $st 10000 10999 14000 15999 > $T/xfer_12k_s$st.txt 2>&1
  say "12k at $st: $(tr '\n' ' ' < $T/xfer_12k_s$st.txt)"
done
bash $T/xfer_eval.sh v11 80000 rest
bash $T/xfer_eval.sh $BARM 80000 rest
$PY $T/xfer_pair.py $BARM $BBASE 80000 10000 34999 > $T/xfer_100k.txt 2>&1
say "100k: $(tr '\n' ' ' < $T/xfer_100k.txt)"
say "XFER_CHAIN_DONE"
