#!/bin/bash
# Critic upper-bound diagnostic on the 5060 Ti (patches/critic_diag/README.md), chosen by the user on 2026-11-09.
#   1. serves any active booking first (a boundary, as every chain does), then waits for an idle card;
#   2. trains the six critics (critic_diag.py train), CD_TOTAL batches as pre-registered;
#   3. writes the report (critic_diag.py report) to runs/critic_diag/report.txt;
#   4. marks the card idle and lends it for a day.
# Log runs/xfer_tools/cd_chain.log, markers CD_CHAIN_DONE / CD_CHAIN_STOPPED.
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; W=/mnt/c/Users/asus
ST=$W/GPU_STATUS_mortal.txt; OUT=$R/critic_diag; LOG=$T/cd_chain.log
export MORTAL_CFG=$OUT/config.toml CD_TOTAL=${CD_TOTAL:?set CD_TOTAL} CD_VAL_EVERY=${CD_VAL_EVERY:?set CD_VAL_EVERY}
export CD_WORKERS=${CD_WORKERS:-4}
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
status() {
  printf '%s\r\n' "mortal GPU status (TanyaoDojo session), $(date '+%F %T %Z')" "$@" \
    "To book: create C:\\Users\\asus\\GPU_BOOK_<name>.txt (one line: who and what; add a line not_before=YYYY-MM-DD HH:MM to defer it)." \
    > $ST.new && mv $ST.new $ST
}
idle() {
  status "State: FREE (idle) since $(date +%T): ${1:-the critic diagnostic has ended and no mortal GPU work is queued.}" \
    "An active booking (GPU_BOOK_<name>.txt, once past its not_before) gets the card within a minute; this file then shows the hold." \
    "This file is rewritten before any new mortal GPU work starts."
}
lend_once() {
  local n0; n0=$(ls $W/GPU_BOOK_*.txt 2>/dev/null | wc -l)
  if [ $n0 -gt 0 ]; then
    bash $T/gpu_share.sh >> $LOG 2>&1
    [ $(ls $W/GPU_BOOK_*.txt 2>/dev/null | wc -l) -lt $n0 ] && { idle "$1"; say "hold over; status: FREE (idle)"; }
  fi
}
mortal_busy() { pgrep -f "^$PY .*(train|train_pg|run_eval|client|server|critic_diag)\.py" > /dev/null; }
echo $$ > $T/cd_chain.pid
say "cd_chain start: CD_TOTAL $CD_TOTAL, CD_VAL_EVERY $CD_VAL_EVERY, workers $CD_WORKERS, critic_diag.py $(md5sum $B/Mortal/mortal/critic_diag.py | cut -c1-12)"
lend_once "the critic diagnostic (about 2.5 hours) starts after this hold."
n=0
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && ! compgen -G "$W/GPU_BUSY_*.txt" > /dev/null \
      && ! mortal_busy; do
  n=$((n + 1)); [ $n -gt 720 ] && { say "card not idle for 12 hours: CD_CHAIN_STOPPED"; idle; exit 1; }
  lend_once "the critic diagnostic waits for an idle card."; sleep 60
done
eta=$(date -d "+$(( CD_TOTAL * 10 / 21 / 60 + 20 )) minutes" '+%m-%d %H:%M')   # ~2.1 batches/s
status "State: BUSY since $(date +%T): critic diagnostic (offline value fitting on old self-play games), until about $eta." \
  "Bookings are served when it ends; it cannot be paused in the middle."
say "training (ETA $eta)"
cd $B/Mortal/mortal
$PY critic_diag.py train $OUT > $OUT/train.log 2>&1
rc=$?
if [ $rc -ne 0 ] || ! grep -q CD_TRAIN_DONE $OUT/train.log; then
  say "train exited $rc: $(grep -v Warning $OUT/train.log | tail -2 | tr '\n' ' ' | cut -c1-300): CD_CHAIN_STOPPED"
  idle "the critic diagnostic stopped with an error."; exit 1
fi
say "trained: $(grep TEST $OUT/train.log | tail -1 | cut -c1-300)"
$PY critic_diag.py report $OUT > $OUT/report.txt 2>&1 || say "report exited $?"
idle; say "CD_CHAIN_DONE"
END=$(( $(date +%s) + 86400 ))
while [ $(date +%s) -lt $END ]; do
  mortal_busy && { say "mortal GPU work started: lending loop ends"; exit 0; }
  lend_once ""; sleep 60
done
say "lending loop ends after a day"
