#!/bin/bash
# Online transfer test: the per-run helpers of the M1 run (runs/m1_calanchor), with the run directory as a parameter.
#   bash xfer_helpers.sh archiver SP PREFIX     # copy state.pth every EVERY (8,000) steps into SP/archive, steps checked
#   bash xfer_helpers.sh guard SP CFG           # restart the trainer if it disappears (after the buffer gate)
#   bash xfer_helpers.sh stop SP                # stop server / trainer / workers by PID and process group only
# Never pattern-kill train.py: other runs on this machine use it too.
set -u
B=$HOME/Projects/better_mortal; PY=$B/.venv/bin/python; MM=$B/Mortal/mortal
what=$1; SP=$2
case $what in
archiver)
  PREFIX=$3; A=$SP/archive; mkdir -p $A; LOG=$SP/archiver.log; EVERY=${EVERY:-8000}; last=${START_AFTER:-0}
  echo "$(date +%m-%d_%H:%M:%S) STEP ARCHIVER START every=$EVERY after=$last prefix=$PREFIX" >> $LOG
  while true; do
    sleep 30
    n=$(tr -d '\r' < $SP/logs/trainer.log | grep -a -oE "total steps: [0-9,]+" | tail -1 | grep -oE "[0-9,]+$" | tr -d ,)
    [ -z "$n" ] && continue
    if [ $((n % EVERY)) -eq 0 ] && [ "$n" -gt "$last" ]; then
      f=$A/${PREFIX}_s${n}.pth
      for try in 1 2 3; do
        cp $SP/state.pth $f.tmp && got=$($PY -c "import torch,sys; s=torch.load(sys.argv[1],weights_only=True,map_location='cpu'); assert 'mortal' in s; print(s['steps'])" $f.tmp 2>/dev/null) && break
        sleep 20
      done
      note=""; [ "$got" != "$n" ] && note=" MISMATCH(file_steps=$got)"
      # a mismatched copy is kept under its own name so the stop rule (which waits for _s<n>.pth) never fires on it
      if [ -z "$note" ]; then mv $f.tmp $f; else mv $f.tmp $A/${PREFIX}_s${n}_mismatch${got}.pth; fi
      echo "$(date +%m-%d_%H:%M:%S) archived $(basename $f) steps=$(printf "%'d" $n) try=${try}${note}" >> $LOG
      last=$n
    fi
  done;;
guard)
  CFG=$3; LOG=$SP/trainer_guard.log; MINBUF=${MINBUF:-4000}
  echo "$(date +%m-%d_%H:%M:%S) GUARD START cfg=$CFG minbuf=$MINBUF" >> $LOG
  miss=0
  while true; do
    sleep 30
    t=$(sed -n 2p $SP/pids 2>/dev/null)
    # the trainer is the process group led by line 2 of pids (train.py respawns its child inside that group)
    if [ -n "$t" ] && pgrep -g "$t" > /dev/null 2>&1; then miss=0; continue; fi
    [ -f $SP/pids ] || exit 0                       # the run was stopped on purpose
    miss=$((miss+1)); [ $miss -lt 3 ] && continue   # gone for 90 s before it counts
    echo "$(date +%m-%d_%H:%M:%S) ALERT trainer gone, buffer=$(ls $SP/buffer | wc -l), restarting" >> $LOG
    for i in $(seq 1 240); do [ "$(ls $SP/buffer | wc -l)" -ge "$MINBUF" ] && break; sleep 15; done
    cd $MM
    setsid env MORTAL_CFG=$CFG $PY train.py >> $SP/logs/trainer.log 2>&1 < /dev/null &
    NEW=$!
    { sed -n 1p $SP/pids; echo $NEW; sed -n '3,$p' $SP/pids; } > $SP/pids.new && mv $SP/pids.new $SP/pids
    echo "$(date +%m-%d_%H:%M:%S) restarted trainer pid=$NEW buffer=$(ls $SP/buffer | wc -l)" >> $LOG
    miss=0; sleep 300
  done;;
stop)
  PIDF=$SP/pids
  [ -f "$PIDF" ] || { echo "[xfer-stop] no $PIDF, nothing running"; exit 0; }
  for p in $(cat "$PIDF"); do kill -TERM -"$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null; done
  sleep 6
  for p in $(cat "$PIDF"); do kill -0 "$p" 2>/dev/null && { kill -KILL -"$p" 2>/dev/null || kill -KILL "$p" 2>/dev/null; }; done
  sleep 2
  alive=0; for p in $(cat "$PIDF"); do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
  echo "[xfer-stop] processes of this run still alive: $alive; state.pth $(stat -c '%s %y' $SP/state.pth 2>/dev/null)"
  mv "$PIDF" "$PIDF.stopped.$(date +%m%d_%H%M%S)";;
*) echo "unknown helper $what"; exit 1;;
esac
