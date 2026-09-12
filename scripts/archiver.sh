#!/bin/bash
# 续跑 C'(λ=0.2)期间:每 2 小时把 state.pth 按时间戳另存(不滚动覆盖,供逐档复式评测),
# 拷贝后用 torch.load 校验,半截文件则 60 秒后重拷;trainer 挂了只报警,不自动重启。
S=$HOME/Projects/better_mortal/runs/selfplay
A=$S/archive_resume; mkdir -p $A
PY=$HOME/Projects/better_mortal/.venv/bin/python
LOG=$S/archiver.log
echo "$(date +%m-%d_%H:%M:%S) ARCHIVER START" >> $LOG
while true; do
  sleep 7200
  ts=$(date +%m%d_%H%M); f=$A/state_${ts}.pth
  for try in 1 2 3; do
    cp $S/state.pth $f && $PY -c "import torch,sys; s=torch.load(sys.argv[1],weights_only=True,map_location='cpu'); assert 'mortal' in s" $f 2>/dev/null && break
    sleep 60
  done
  st=$(grep -a -oE "total steps: [0-9,]+" $S/logs/trainer.log | tail -1 | grep -oE "[0-9,]+$")
  echo "$(date +%m-%d_%H:%M:%S) archived $(basename $f) steps=${st} try=${try}" >> $LOG
  tp=$(sed -n 2p $S/pids)
  kill -0 $tp 2>/dev/null || echo "$(date +%m-%d_%H:%M:%S) ALERT: trainer pid=$tp 不在了" >> $LOG
done
