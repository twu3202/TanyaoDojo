#!/bin/bash
# 11-04: the PG probe chain is the last mortal job queued on the 5060 Ti (the main line is stopped). gpu_share.sh runs
# only at a boundary (an arm start or an evaluation call), so if the chain ends before a booking's not_before, nothing
# would ever lend the card. This waits for the chain to end, marks the card idle in GPU_STATUS_mortal.txt, and runs
# gpu_share.sh once a booking is active, until 11-04 23:00. Stop it with: kill $(cat runs/xfer_tools/pg_free_watch.pid)
set -u
B=$HOME/Projects/better_mortal; T=$B/runs/xfer_tools; PY=$B/.venv/bin/python; W=/mnt/c/Users/asus
ST=$W/GPU_STATUS_mortal.txt; CLOG=$T/pg_chain.log; END=$(date -d '2026-11-04 23:00' +%s)
echo $$ > $T/pg_free_watch.pid
say() { echo "$(date +%m-%d_%T) pg_free_watch: $*"; }
busy() { pgrep -f "^$PY .*(train|train_pg|server|client|run_eval)\.py" > /dev/null; }
books() { ls $W/GPU_BOOK_*.txt 2>/dev/null | wc -l; }
idle() {
  printf '%s\r\n' "mortal GPU status (TanyaoDojo session), $(date '+%F %T %Z')" \
    "State: FREE (idle) since $(date +%T): the policy-gradient probe chain has ended and no mortal GPU work is queued." \
    "An active booking (GPU_BOOK_<name>.txt, once past its not_before) gets the card within a minute; this file then shows the hold." \
    "This file is rewritten before any new mortal GPU work starts." > $ST.new && mv $ST.new $ST
}
say "waiting for pg_chain to end"
until grep -a -qE "PG_CHAIN_(DONE|STOPPED)" $CLOG || ! pgrep -f "bash $T/pg_chain.sh" > /dev/null; do sleep 60; done
sleep 60
say "pg_chain ended: $(tail -1 $CLOG | cut -c1-120)"
busy && { say "mortal GPU work still running: status left alone"; exit 0; }
idle; say "status: FREE (idle)"
while [ $(date +%s) -lt $END ] && [ $(books) -gt 0 ]; do
  n0=$(books); bash $T/gpu_share.sh
  if [ $(books) -lt $n0 ]; then busy || idle; say "hold over; status: FREE (idle)"; fi
  sleep 60
done
say "done ($(books) booking files left)"
