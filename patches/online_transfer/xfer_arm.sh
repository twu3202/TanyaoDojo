#!/bin/bash
# Online transfer test: trains one arm (prepared by xfer_setup.sh) from step 0 to STOP_AT (80,000 batches = 10,000
# Adam updates), then stops it. Topology of the M1 run on the 5060 Ti: server + 4 GPU workers + trainer, the trainer
# only after the buffer holds MINBUF (4,000) files, so its first drain is not a few hundred games ground on for 4,000
# batches. Opponent pool base, v4, base, base (base = the arm's own offline start). Snapshots every 8,000 steps.
#   bash xfer_arm.sh ARM        (writes XFER_ARM_DONE or XFER_ARM_STOPPED as the last word of a line of SP/arm.log)
set -u
ARM=$1; STOP_AT=${STOP_AT:-80000}
B=$HOME/Projects/better_mortal; R=$B/runs; SP=$R/xfer_$ARM; T=$R/xfer_tools; PY=$B/.venv/bin/python
CFG=$B/configs/online_selfplay_xfer_$ARM.toml; LOG=$SP/arm.log
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
[ -f $SP/manifest.txt ] || { echo "!! run xfer_setup.sh first"; exit 1; }
say "arm $ARM: starting, stop at $STOP_AT steps"
SP=$SP CFG=$CFG MINBUF=4000 GATE_MIN=60 bash $B/scripts/run_selfplay_m1.sh --resume --workers 4 --pool base,v4,base,base >> $LOG 2>&1
if ! [ "$(sed -n 2p $SP/pids 2>/dev/null)" -gt 0 ] 2>/dev/null; then
  bash $T/xfer_helpers.sh stop $SP >> $LOG 2>&1; say "trainer did not start: XFER_ARM_STOPPED"; exit 1
fi
: > $SP/helper_pids
setsid env EVERY=8000 bash $T/xfer_helpers.sh archiver $SP xfer_$ARM < /dev/null > /dev/null 2>&1 & echo $! >> $SP/helper_pids
setsid env MINBUF=4000 bash $T/xfer_helpers.sh guard $SP $CFG < /dev/null > /dev/null 2>&1 & echo $! >> $SP/helper_pids
setsid $PY $T/xfer_collapse_guard.py $SP < /dev/null > $SP/collapse_guard.out 2>&1 & echo $! >> $SP/helper_pids
say "helpers $(tr '\n' ' ' < $SP/helper_pids)"
stop_helpers() { for p in $(cat $SP/helper_pids); do kill -TERM -"$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null; done; }
CK=$SP/archive/xfer_${ARM}_s${STOP_AT}.pth
until [ -f $CK ]; do
  if [ ! -f $SP/pids ]; then
    stop_helpers; say "run stopped before $STOP_AT (collapse guard or by hand): XFER_ARM_STOPPED"; exit 1
  fi
  sleep 60
done
sleep 30
stop_helpers
bash $T/xfer_helpers.sh stop $SP >> $LOG 2>&1
$PY $T/xfer_reuse.py $SP >> $LOG 2>&1
say "arm $ARM reached $STOP_AT: XFER_ARM_DONE"
