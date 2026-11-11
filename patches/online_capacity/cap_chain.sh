#!/bin/bash
# Online capacity test on the 5060 Ti (README.md): the online transfer test's recipe from B1 (256 x 54), compared
# with the v11 arm (192 x 40) that already ran it.
#   1. arm b1: xfer_setup (the config takes the base's own [resnet] size), 0 -> 80,000 batches, 12k of 40k and 80k
#   2. screening of b1 80k - v11 80k by the sequential test (SPRT 0 vs +1, alpha 0.025, beta 0.2), one look per
#      1,000 groups in ascending seed order from 10000; a promote is confirmed by the rest of the 100k
#   3. pairings on everything played (xfer_pair.py)
# Log runs/xfer_tools/cap_chain.log; markers CAP_CHAIN_DONE / CAP_CHAIN_STOPPED as the last word of a line.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/xfer_tools/cap_chain.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/cap_chain.log
D=$R/evaldump; ARM=b1; BBASE=b1_b1final
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
echo $$ > $T/cap_chain.pid
idle=0
while [ $idle -lt 3 ]; do
  if pgrep -f "^$PY .*(train|run_eval|client|server|ev_map)\.py" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
  sleep 60
done
say "GPU idle; arm $ARM from $R/b1_256x54/b1_final.pth"
bash $T/xfer_setup.sh $ARM $R/b1_256x54/b1_final.pth 5583 >> $LOG 2>&1 || { say "setup FAILED: CAP_CHAIN_STOPPED"; exit 1; }
bash $T/xfer_arm.sh $ARM
grep -a -q " XFER_ARM_DONE$" $R/xfer_$ARM/arm.log || { say "arm did not reach 80k: CAP_CHAIN_STOPPED"; exit 1; }
say "arm trained; $(grep -a '^overall' $R/xfer_$ARM/arm.log | tail -1)"
for st in 40000 80000; do
  bash $T/xfer_eval.sh $ARM $st 12k
  $PY $T/xfer_pair.py $ARM $BBASE $st 10000 10999 14000 15999 > $T/cap_12k_s$st.txt 2>&1
  say "12k at $st: $(tr '\n' ' ' < $T/cap_12k_s$st.txt)"
done
seq_look() {
  $PY $T/seqtest.py "$D/xfer_${ARM}_s80000_s*_gpufp32.npz" "$D/xfer_v11_s80000_s*_gpufp32.npz" \
      --delta0 0 --delta1 1 --alpha 0.025 --beta 0.2 --look 1000 > $T/cap_seq.txt 2>&1
  echo $?
}
verdict=3; hi=10999
rc=$(seq_look); say "look at 1,000 groups: exit $rc, $(tail -2 $T/cap_seq.txt | tr -s " \n" " ")"; [ "$rc" = 3 ] || verdict=$rc
for s in 11000 12000 13000 $(seq 16000 1000 34000); do
  [ "$verdict" = 1 ] && break                       # drop: stop playing
  bash $T/xfer_eval.sh $ARM 80000 seg $s 10 || { say "segment s$s FAILED: CAP_CHAIN_STOPPED"; exit 1; }
  if [ $s = 13000 ]; then hi=15999; else hi=$((s + 999)); fi
  if [ "$verdict" = 3 ]; then
    rc=$(seq_look); say "look through s$hi: exit $rc, $(tail -2 $T/cap_seq.txt | tr -s " \n" " ")"; [ "$rc" = 3 ] || verdict=$rc
  fi
done
case $verdict in
  0) say "SPRT: promote (confirmed on the full 100k above)";;
  1) say "SPRT: drop at $hi";;
  4) say "SPRT: undecided at the cap; read the fixed 100k";;
  *) say "SPRT: exit $verdict";;
esac
$PY $T/xfer_pair.py $ARM $BBASE 80000 10000 $hi > $T/cap_final.txt 2>&1
say "pairing 10000-$hi: $(tr '\n' ' ' < $T/cap_final.txt)"
say "CAP_CHAIN_DONE"
