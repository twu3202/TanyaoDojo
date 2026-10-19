#!/bin/bash
# First throughput round on the 5060 Ti, in the GPU gap between the q12one evaluation and the online transfer test.
# Holds the transfer chain with runs/xfer_tools/HOLD for at most about 75 minutes (removed on any exit), waits for
# Q12ONE_CHAIN_DONE / _STOPPED and 3 idle minutes, then runs the baseline layouts below, one at a time.
# Log runs/bench_eval/chain.log, results runs/bench_eval/summary.txt; marker BENCH_CHAIN_DONE.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/bench_eval/tools/bench_chain.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; BD=$R/bench_eval; T=$BD/tools; PY=$B/.venv/bin/python
HOLD=$R/xfer_tools/HOLD; LOG=$BD/chain.log
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
mkdir -p $BD
echo $$ > $BD/chain.pid
echo "bench_eval chain $$ ($(date '+%F %T'))" > $HOLD
trap 'rm -f $HOLD; say "HOLD removed"' EXIT
say "holding the transfer chain; waiting for the q12one chain"
until grep -a -q -E " Q12ONE_CHAIN_(DONE|STOPPED)$" $R/q12one/chain.log 2>/dev/null; do sleep 120; done
idle=0
while [ $idle -lt 3 ]; do
  if pgrep -f "^$PY .*(train|run_eval|client|server)\.py" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
  sleep 60
done
say "GPU idle; baseline round"
# name nproc games-per-iter. b0 = today's layout (also the run-to-run determinism check against the 10-15 logs);
# b1/b2 = the same 100-seed batches in 2 / 4 concurrent processes; b3/b4 = wider batches.
for c in "b0_1x400 1 400" "b1_2x400 2 400" "b2_4x400 4 400" "b3_1x1600 1 1600" "b4_2x800 2 800"; do
  set -- $c
  TIMEOUT=1200 bash $T/bench_eval.sh $1 $2 $3 >> $LOG 2>&1
  say "$(tail -1 $BD/summary.txt)"
done
say "BENCH_CHAIN_DONE"
