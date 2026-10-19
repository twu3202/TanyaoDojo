#!/bin/bash
# Evaluation throughput benchmark (README.md): one fixed workload, one candidate layout, games/s plus an identity
# check against the reference logs of the same games.
#   bash bench_eval.sh NAME NPROC GAMES_PER_ITER
# Workload: q25_final (infer copy) vs mortal_v4, duplicate 1v3, seeds 10000-10399 (400 groups = 1,600 games),
# seed_key 20260711, GPU fp32 / AMP off, challenger name q25_final: exactly the games of the q25 100k evaluation
# (runs/eval_q25_final_s10000_gpufp32, 2026-10-15), so every log can be compared byte for byte.
# NPROC processes split the seeds into contiguous blocks; GAMES_PER_ITER is run_eval's --games (batch width).
set -u
NAME=$1; NP=$2; G=$3; NGROUPS=${NGROUPS:-400}; LO=10000
B=$HOME/Projects/better_mortal; R=$B/runs; BD=$R/bench_eval; T=$BD/tools; PY=$B/.venv/bin/python
E=/mnt/d/Better_mortal/jax_rl/mjai_bot
CK=$R/q25/q25_final_infer.pth; REF=$R/eval_q25_final_s10000_gpufp32
per=$((NGROUPS / NP))
if [ $((per * NP)) -ne $NGROUPS ] || [ $(( (per * 4) % G )) -ne 0 ]; then echo "!! $NGROUPS groups do not split into $NP x $G"; exit 2; fi
iters=$(( per * 4 / G ))
OUT=$BD/$NAME; rm -rf $OUT; mkdir -p $OUT/logs
( while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits >> $OUT/smi.csv 2>/dev/null; sleep 2; done ) &
SMI=$!
( while true; do cut -d' ' -f1 /proc/loadavg >> $OUT/load.txt; sleep 5; done ) &
LD=$!
t0=$(date +%s.%N)
cd $E
pids=()
for p in $(seq 0 $((NP - 1))); do
  s=$((LO + p * per))
  timeout ${TIMEOUT:-1500} $PY run_eval.py $CK --challenger-type mortal --challenger-name q25_final --device cuda:0 --amp off \
      --games $G --iters $iters --seed-start $s --log-dir $OUT/logs < /dev/null > $OUT/p$p.out 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
t1=$(date +%s.%N)
kill $SMI $LD 2>/dev/null
$PY $T/bench_compare.py $OUT $REF $LO $NGROUPS "$NAME" $NP $G $t0 $t1 $rc | tee -a $BD/summary.txt
