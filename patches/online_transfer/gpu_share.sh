#!/bin/bash
# Lending the card to another session's short test (2026-11-03: the local-TTS session), only at a boundary:
# xfer_arm.sh runs this before it starts an arm and xfer_eval.sh before each evaluation call, so a running arm is never
# shared or paused (evaluation results do not depend on when they run).
#   Booking: the guest creates C:\Users\asus\GPU_BOOK_<name>.txt (one line: who and what). A line
#   "not_before=YYYY-MM-DD HH:MM" defers it: until then the booking is ignored. At the next boundary with an active
#   booking the arm or evaluation waits and the card is free. The guest creates C:\Users\asus\GPU_BUSY_<name>.txt while it uses the card and
#   deletes it when done; the hold ends then, after HOLD_IDLE (45) minutes if no BUSY file appears, and after HOLD_MAX
#   (90) in any case. The active booking files are removed when the hold ends.
#   Co-located load, on request: while a BUSY file exists, C:\Users\asus\GPU_LOAD_please.txt starts rounds of the
#   bench_eval workload (q25_final vs v4, 4 x 400 games, GPU fp32, RAYON 5 / OMP 1, seeds 10000-10399), one line per
#   round in C:\Users\asus\GPU_LOAD_mortal.log (games/s, exit code, log hash, peak reserved). Deleting the request
#   stops the load after the current round; when the hold ends a running round is killed.
#   Status for the guest in C:\Users\asus\GPU_STATUS_mortal.txt; GPU memory and use every 10 s in GPU_SHARE_nvsmi.csv.
# Without a booking it returns at once. Output goes to the caller's log. SHARE_DIR / SHARE_E / SHARE_LD override the
# Windows folder, the x1_bench folder and the round folder, and SHARE_DRY skips the busy-card check (for the dry run).
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; PY=$B/.venv/bin/python; W=${SHARE_DIR:-/mnt/c/Users/asus}
E=${SHARE_E:-/mnt/d/Better_mortal/jax_rl/mjai_bot}; CK=$R/q25/q25_final_infer.pth; LD=${SHARE_LD:-$R/bench_eval/share}
HOLD_IDLE=${HOLD_IDLE:-45}; HOLD_MAX=${HOLD_MAX:-90}
ST=$W/GPU_STATUS_mortal.txt; LREQ=$W/GPU_LOAD_please.txt; LLOG=$W/GPU_LOAD_mortal.log
now=$(date +%s); active=()
for f in "$W"/GPU_BOOK_*.txt; do
  [ -f "$f" ] || continue
  nb=$(tr -d '\r' < "$f" | sed -n 's/^not_before=//p' | head -1)
  if [ -n "$nb" ]; then t=$(date -d "$nb" +%s 2>/dev/null) || t=0; [ "$t" -gt "$now" ] && continue; fi
  active+=("$f")
done
[ ${#active[@]} -gt 0 ] || exit 0
if [ -z "${SHARE_DRY:-}" ] && pgrep -f "^$PY .*(train|train_pg|client|run_eval)\.py" > /dev/null; then   # never lend a busy card
  echo "$(date +%m-%d_%T) gpu_share: booking active but mortal GPU work is running: no hold"; exit 0
fi
export MORTAL_DIR=$B/Mortal/mortal MORTAL_V4=$B/baseline/mortal_v4.pth RAYON_NUM_THREADS=5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
say() { echo "$(date +%m-%d_%T) gpu_share: $*"; }
status() { printf '%s\r\n' "mortal GPU status (TanyaoDojo session), $(date '+%F %T %Z')" "$@" > $ST.new && mv $ST.new $ST; }
guest_on() { compgen -G "$W/GPU_BUSY_*.txt" > /dev/null; }
h() { (cd $1 && for f in $(ls); do zcat $f | sed -E 's/"eval_time_ns":[0-9]+,?//g'; done | md5sum | cut -c1-12); }
round() {  # round <dir>: one 4 x 400 round, one line on stdout
  local d=$1 ps=() t0 t1 n rc=0
  rm -rf $d; mkdir -p $d/logs
  t0=$(date +%s.%N)
  for p in 0 1 2 3; do
    (cd $E && exec $PY x1_bench.py $CK --variant base --challenger-name q25_final --games 400 \
        --seed-start $((10000 + p * 100)) --log-dir $d/logs) < /dev/null > $d/p$p.out 2>&1 & ps+=($!)
  done
  for p in "${ps[@]}"; do wait $p || rc=$?; done
  t1=$(date +%s.%N); n=$(ls $d/logs | wc -l)
  printf '%s\r\n' "$(date '+%F %T') $(basename $d): $n games in $(echo "$t1 - $t0" | bc) s = $(echo "scale=3; $n / ($t1 - $t0)" | bc) games/s (solo on this card: about 5.8); exit $rc; hash $(h $d/logs); $(grep -h -oE 'peak reserved [0-9.]+ GiB' $d/*.out | tr '\n' ' ')"
}
loader() { local i=0; while [ -e $LREQ ] && guest_on; do i=$((i + 1)); round $LD/r$i >> $LLOG; done; }
book=$(cat "${active[@]}" | tr -d '\r' | tr '\n' ' ')
start=$(date +%s); seen=0; lp=  # loader pid (never 0: kill 0 would hit the whole process group)
say "booked by: $book; arm waits (guest marker within $HOLD_IDLE min, at most $HOLD_MAX min)"
status "State: FREE for the booking ($book) since $(date +%T)." \
  "Create C:\\Users\\asus\\GPU_BUSY_<name>.txt while you use the card and delete it when done; that ends the hold." \
  "No BUSY file by $(date -d @$((start + HOLD_IDLE * 60)) +%T): the hold ends. Hard end $(date -d @$((start + HOLD_MAX * 60)) +%T)." \
  "Co-located load: with your BUSY file present, create C:\\Users\\asus\\GPU_LOAD_please.txt; one line per 4 x 400 round" \
  "in C:\\Users\\asus\\GPU_LOAD_mortal.log; delete the request to stop after the round."
rm -rf $LD; mkdir -p $LD
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 10 > $W/GPU_SHARE_nvsmi.csv 2>&1 &
np=$!
while :; do
  el=$(( ($(date +%s) - start) / 60 ))
  if guest_on; then
    [ $seen = 1 ] || { seen=1; say "guest on at $el min: $(cd $W && ls GPU_BUSY_*.txt | tr '\n' ' ')"; }
    if [ -e $LREQ ] && { [ -z "$lp" ] || ! kill -0 $lp 2>/dev/null; }; then loader & lp=$!; say "co-located load started at $el min"; fi
  else
    [ $seen = 1 ] && { say "guest done at $el min"; break; }
    [ $el -ge $HOLD_IDLE ] && { say "no guest marker after $el min"; break; }
  fi
  [ $el -ge $HOLD_MAX ] && { say "hold cap of $HOLD_MAX min reached"; break; }
  sleep 20
done
if [ -n "$lp" ] && kill -0 $lp 2>/dev/null; then
  kill $lp 2>/dev/null; pkill -f "x1_bench.py .*--log-dir $LD/" 2>/dev/null; wait $lp 2>/dev/null
  printf '%s\r\n' "$(date '+%F %T') load stopped: the hold ended" >> $LLOG
fi
kill $np 2>/dev/null
rm -f "${active[@]}" $LREQ
say "hold over after $(( ($(date +%s) - start) / 60 )) min; load rounds: $(ls $LD | wc -l)"
status "State: BUSY since $(date +%T): mortal training or evaluation (exclusive). Do not start GPU work;" \
  "ask the TanyaoDojo session, or book the next boundary with a GPU_BOOK_<name>.txt file (not_before=... to defer)."
