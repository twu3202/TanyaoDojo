#!/bin/bash
# 等 GPU 空出 >5GB 连续 3 次探测 → 跑 10 万局 final 对 v4 → 聚合
VENV=$HOME/Projects/better_mortal/.venv/bin/activate
RUN=$HOME/Projects/better_mortal/runs/eval_final
CFG=$HOME/Projects/better_mortal/configs/local_eval_final.toml
LOG=$RUN/eval_100k.log
mkdir -p $RUN
source $VENV
cd $HOME/Projects/better_mortal/Mortal/mortal
echo "[gpuwatch] waiting for >5GB free GPU... $(date)"
streak=0
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -gt 5000 ]; then
    streak=$((streak+1))
    echo "[gpuwatch] free=${free}MiB streak=$streak $(date)"
    [ $streak -ge 3 ] && break
  else
    streak=0
  fi
  sleep 30
done
echo "[gpuwatch] GPU free, launching eval $(date)"
MORTAL_CFG=$CFG python one_vs_three.py > $LOG 2>&1
echo "[gpuwatch] eval done $(date)"
python $HOME/Projects/better_mortal/scripts/aggregate_eval.py $LOG >> $LOG 2>&1
touch $RUN/eval_done.marker
echo "[gpuwatch] all done $(date)"
