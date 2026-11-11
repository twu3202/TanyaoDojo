#!/bin/bash
# Second throughput round on the 5060 Ti (README.md), in the GPU gap after the online transfer test.
# One change per candidate against b2 (4 x 400). CPU load in round 0 exceeded the 20 threads, so this round caps the
# per-process CPU threads and tries 8 processes; b2r repeats b2 to measure the run-to-run timing spread that a
# candidate must beat. Waits for 3 idle minutes first; log runs/bench_eval/chain.log, marker BENCH_ROUND1_DONE.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/bench_eval/tools/bench_round1.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; BD=$R/bench_eval; T=$BD/tools; PY=$B/.venv/bin/python
LOG=$BD/chain.log
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
echo $$ > $BD/chain.pid
idle=0
while [ $idle -lt 3 ]; do
  if pgrep -f "^$PY .*(train|run_eval|client|server)\.py" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
  sleep 60
done
say "GPU idle; round 1"
# name | env | nproc games-per-iter | groups. RAYON_NUM_THREADS caps libriichi's arena pool (rayon par_iter),
# OMP/MKL cap torch's CPU threads. The 800-group pair compares 8 processes with 4 on the same, longer workload.
for c in "b2r_4x400||4 400|400" \
         "b5_4x400_rayon5|RAYON_NUM_THREADS=5|4 400|400" \
         "b6_4x400_omp1|OMP_NUM_THREADS=1 MKL_NUM_THREADS=1|4 400|400" \
         "b7_4x400_rayon5_omp1|RAYON_NUM_THREADS=5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1|4 400|400" \
         "b2w_4x400_800g||4 400|800" \
         "b8_8x400_800g||8 400|800"; do
  IFS='|' read -r name envs layout groups <<< "$c"
  env $envs NGROUPS=$groups TIMEOUT=1500 bash $T/bench_eval.sh $name $layout >> $LOG 2>&1
  say "$(tail -1 $BD/summary.txt)"
done
say "BENCH_ROUND1_DONE"
