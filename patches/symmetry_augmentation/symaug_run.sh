#!/bin/bash
# symaug offline training: train.py from the separate Mortal_symaug tree, restarted after a crash (resumes from
# mortal.pth, saved every 400 steps; the data epoch is reshuffled on restart, as with B1), at most 5 times. Stops at
# 1.22M steps (= v11 best point) and keeps ${NAME}_final.pth. NAME = symaug (192x40, 5060 Ti) or symaug256 (256x54,
# trx; or mjs, the new-data arm, with TREE=Mortal_mjs/mortal): run dir runs/$NAME, config configs/local_train_$NAME.toml,
# code tree $TREE (default Mortal_symaug/mortal), stop step $STOP. Logs: runs/$NAME/{train,run}.log.
# Start: (NAME=symaug setsid nohup bash ~/Projects/better_mortal/runs/symaug/symaug_run.sh > /dev/null 2>&1 &)
# Stop:  kill -TERM -$(cat runs/$NAME/run.pid); kill -TERM -$(cat runs/$NAME/train.pid)   (process groups)
set -u
NAME=${NAME:-symaug}
ROOT=$HOME/Projects/better_mortal; T=${TREE:-$ROOT/Mortal_symaug/mortal}; R=$ROOT/runs/$NAME
CFG=$ROOT/configs/local_train_$NAME.toml; PY=$ROOT/.venv/bin/python
STOP=${STOP:-1220000}; LOG=$R/run.log; MAXRESTART=5
say() { echo "$(date +%m-%d_%H:%M:%S) $*" >> $LOG; }
last_step() { grep -a -o "total steps: [0-9,]*" $R/train.log 2>/dev/null | tail -1 | tr -dc 0-9; }
echo $$ > $R/run.pid
n=0
cd $T
while true; do
  MORTAL_CFG=$CFG setsid $PY train.py >> $R/train.log 2>&1 < /dev/null &
  pid=$!; echo $pid > $R/train.pid
  say "train.py pid $pid started (attempt $n, last logged step $(last_step))"
  while kill -0 $pid 2>/dev/null; do
    st=$(last_step); st=${st:-0}
    if [ "$st" -ge $STOP ]; then
      sleep 20   # "total steps" is logged just before torch.save of that step; the next save is 400 steps later
      got=$($PY -c "import torch;print(torch.load('$R/mortal.pth',weights_only=True,map_location='cpu')['steps'])")
      cp $R/mortal.pth $R/${NAME}_final.pth
      say "reached $st (checkpoint steps=$got) -> ${NAME}_final.pth; stopping pid $pid"
      kill -TERM -$pid 2>/dev/null || kill -TERM $pid; sleep 60
      kill -0 $pid 2>/dev/null && { say "still alive, SIGKILL"; kill -KILL -$pid 2>/dev/null || kill -KILL $pid; }
      say SYMAUG_TRAIN_DONE
      exit 0
    fi
    [ "$st" -ge $((STOP - 4000)) ] && sleep 10 || sleep 300
  done
  n=$((n+1))
  say "CRASH? pid $pid exited at logged step $(last_step); tail of train.log:"
  tail -n 40 $R/train.log | tr '\r' '\n' | grep -v "TRAIN:\|it/s\]" | tail -n 8 >> $LOG
  [ $n -gt $MAXRESTART ] && { say "too many restarts ($n), giving up"; exit 1; }
  sleep 60
done
