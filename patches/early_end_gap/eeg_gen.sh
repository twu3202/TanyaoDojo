#!/bin/bash
# Early-end gap diagnosis (README.md): fresh tpt 408k games against mortal_v4 on a diagnostic seed key, as raw
# material for state-filtered EV maps and as an independent replication of 408k's level and early-end deficit.
# Duplicate 1v3, GPU fp32 / AMP off, seeds 10000-34999 (25,000 groups = 100k games) on key 20261021. Not the sacred
# key, so these games never enter a record. Five segments of 5,000 groups, each split over NPROC (4) processes by
# whole 100-seed iterations (the 4 x 400 layout of patches/eval_throughput); every segment gets group_ci --strict
# and the mechanism readouts. Waits for the throughput round (BENCH_ROUND1_DONE) and two idle GPU minutes.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/eeg/tools/eeg_gen.sh > /dev/null 2>&1 &)
# Log runs/eeg/gen.log, marker EEG_GEN_DONE.
set -u
KEY=${KEY:-20261021}
B=$HOME/Projects/better_mortal; R=$B/runs; O=$R/eeg; PY=$B/.venv/bin/python; T=$R/xfer_tools
E=/mnt/d/Better_mortal/jax_rl/mjai_bot; CK=$R/tpt_tpt_s408000.pth; NAME=tpt408k
LOG=$O/gen.log
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
mkdir -p $O/dump $O/gapdump $O/placedump $O/bustdump
echo $$ > $O/gen.pid
say "waiting for the throughput round"
until grep -a -q "BENCH_ROUND1_DONE" $R/bench_eval/chain.log 2>/dev/null; do sleep 60; done
idle=0
while [ $idle -lt 2 ]; do
  if pgrep -f "^$PY .*(train|run_eval|client|server)\.py" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
  sleep 60
done
say "start: $NAME md5 $(md5sum $CK | cut -c1-8), key $KEY, NPROC ${NPROC:-4} ${EEG_ENV:-}"
cd $E
fail=0
for s in 10000 15000 20000 25000 30000; do
  it=50; hi=$((s + it * 100 - 1))
  [ -f $O/done_s$s ] && continue
  L=$O/logs_s$s; rm -rf "$L"; mkdir -p "$L"
  NP=${NPROC:-4}; off=0; pids=()
  for p in $(seq 0 $((NP - 1))); do
    k=$(( it / NP + (p < it % NP ? 1 : 0) )); [ $k -eq 0 ] && continue
    env ${EEG_ENV:-} $PY run_eval.py "$CK" --challenger-type mortal --challenger-name $NAME --device cuda:0 --amp off \
        --games 400 --iters $k --seed-start $((s + off * 100)) --seed-key $KEY --log-dir "$L" \
        < /dev/null > "$L.out.$p" 2>&1 &
    pids+=($!); off=$((off + k))
  done
  for pid in "${pids[@]}"; do wait $pid; done
  cat "$L".out.* > "$L.out"; rm -f "$L".out.*
  $PY $T/group_ci.py --strict --log-dir "$L" --seed-lo $s --seed-hi $hi --dump "$O/dump/s$s.npz" \
      < /dev/null > "$L.ci" 2>&1
  rc=$?
  for t in gap place bust; do
    $PY diag_$t.py --log-dir "$L" --seed-lo $s --seed-hi $hi --expect-games $((it * 400)) \
        --dump "$O/${t}dump/s$s.json" < /dev/null > "$L.$t" 2>&1
  done
  if [ $rc = 0 ] && grep -a -q "正确口径" "$L.ci"; then touch $O/done_s$s; say "s$s-$hi done: $(grep -a '正确口径' "$L.ci" | head -1)"
  else say "s$s group_ci FAILED (exit $rc), see $L.ci"; fail=1; fi
done
[ $fail = 0 ] && say "EEG_GEN_DONE" || say "EEG_GEN_INCOMPLETE"
