#!/bin/bash
# X1 on the 5060 Ti (research/2026-11-03_rl_infra_survey, section 8): the inference-architecture variants developed
# on trx, timed on the real card in the GPU gap after the anchor-schedule chain. Workload = bench_eval.sh's:
# q25_final (infer copy) vs mortal_v4, 1v3, seeds 10000-10399 (4 processes x 400 games), GPU fp32, RAYON 5 / OMP 1.
#   t1     single process, variant timed: the share of wall time spent in inference (Amdahl)
#   base   4 x 400 as today (b7); base2 repeats it for the timing spread
#   pinned 4 x 400, pinned host buffers + async copies (games identical)
#   (graph, CUDA graphs on 64-wide buckets, is left out: on trx it was slower than base in 4 x 400 (10.66 vs
#   11.10 games/s) and reserved 5.6-9.2 GiB per process, more than 4 processes can have on this 8 GB card)
#   free   one inference server + 4 arena clients, requests merged as they arrive (games not reproducible)
# Waits for ANC_CHAIN_DONE / ANC_CHAIN_STOPPED in anc_chain.log, then 3 idle GPU minutes; if the chain promoted, the
# main-line loop keeps the GPU busy and this waits on. Log runs/bench_eval/x1.log, marker X1_5060_DONE.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/bench_eval/tools/x1_5060.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; BD=$R/bench_eval; X=$BD/x1; PY=$B/.venv/bin/python; LOG=$BD/x1.log
E=/mnt/d/Better_mortal/jax_rl/mjai_bot; CK=$R/q25/q25_final_infer.pth
export MORTAL_DIR=$B/Mortal/mortal MORTAL_V4=$B/baseline/mortal_v4.pth RAYON_NUM_THREADS=5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
echo $$ > $BD/x1.pid
say "waiting for the anchor-schedule chain to end, then an idle GPU"
until grep -a -q -E "ANC_CHAIN_DONE|ANC_CHAIN_STOPPED" $R/xfer_tools/anc_chain.log 2>/dev/null; do sleep 300; done
idle=0
while [ $idle -lt 3 ]; do
  if pgrep -f "mainline_next\.sh|anc_chain\.sh|rew_chain\.sh" > /dev/null \
     || pgrep -f "^$PY .*(train|run_eval|client|server|ev_map|x1_bench)\.py" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
  sleep 60
done
say "GPU idle; X1 on the 5060 Ti"
cd $E
h() { (cd $1 && for f in $(ls); do zcat $f | sed -E 's/"eval_time_ns":[0-9]+,?//g'; done | md5sum | cut -c1-12); }
stamp() {  # stamp <tag> <t0> <t1>
  local n; n=$(ls $X/$1/logs | wc -l)
  say "$1: $n games in $(echo "$3 - $2" | bc) s = $(echo "scale=3; $n / ($3 - $2)" | bc) games/s;" \
      "GPU $(awk '{s+=$1} END {printf "%.0f", s/NR}' $X/$1/gpu.txt)%; hash $(h $X/$1/logs);" \
      "$(grep -h -oE 'peak reserved [0-9.]+ GiB' $X/$1/*.out | tr '\n' ' ')"
}
four() {  # four <tag> <variant>: 4 processes x 400 games, contiguous seed blocks
  local tag=$1 ps=() t0 t1 gp
  rm -rf $X/$tag; mkdir -p $X/$tag/logs
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -lms 1000 > $X/$tag/gpu.txt & gp=$!
  t0=$(date +%s.%N)
  for p in 0 1 2 3; do
    $PY x1_bench.py $CK --variant $2 --challenger-name q25_final --games 400 --seed-start $((10000 + p * 100)) \
        --log-dir $X/$tag/logs < /dev/null > $X/$tag/p$p.out 2>&1 & ps+=($!)
  done
  for p in "${ps[@]}"; do wait $p; done
  t1=$(date +%s.%N); kill $gp; stamp $tag $t0 $t1
}
serve_free() {  # one inference server, 4 arena clients
  local tag=free sock=/tmp/x1_free.sock ps=() t0 t1 gp sp
  rm -rf $X/$tag; mkdir -p $X/$tag/logs
  $PY x1_server.py $CK --clients 4 --mode free --socket $sock < /dev/null > $X/$tag/server.out 2>&1 & sp=$!
  until [ -S $sock ]; do sleep 0.5; done
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -lms 1000 > $X/$tag/gpu.txt & gp=$!
  t0=$(date +%s.%N)
  for p in 0 1 2 3; do
    $PY x1_bench.py $CK --variant client --client-id $p --socket $sock --challenger-name q25_final --games 400 \
        --seed-start $((10000 + p * 100)) --log-dir $X/$tag/logs < /dev/null > $X/$tag/p$p.out 2>&1 & ps+=($!)
  done
  for p in "${ps[@]}"; do wait $p; done
  t1=$(date +%s.%N); wait $sp; kill $gp; stamp $tag $t0 $t1
  say "  $(grep -h '^server ' $X/$tag/server.out | tr '\n' ' ')"
}
mkdir -p $X/t1/logs
$PY x1_bench.py $CK --variant timed --challenger-name q25_final --games 400 --log-dir $X/t1/logs < /dev/null > $X/t1/t1.out 2>&1
say "t1: $(grep -v INFO $X/t1/t1.out | tr -s ' \n' ' ')"
four base base
four pinned pinned
serve_free
four base2 base
say "X1_5060_DONE"
