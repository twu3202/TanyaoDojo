#!/bin/bash
# 按步数归档 state.pth(每 8,000 步一份, 恰在 test_play 间隙 state.pth 静止时拷贝), 供逐档复式评测。
# 2026-09-14 trainer 数据加载改多进程后训练变快数倍, 按墙钟每 2 小时一档会跨 3-4 万步, 筛选分辨率不够。
# 拷贝后用 torch.load 核对 steps 字段与目标一致, 不一致(已被下一次存盘覆盖)则如实记录实际步数。
S=$HOME/Projects/better_mortal/runs/selfplay
A=$S/archive_resume; mkdir -p $A
PY=$HOME/Projects/better_mortal/.venv/bin/python
LOG=$S/archiver.log
EVERY=${EVERY:-8000}
last=${START_AFTER:-0}
echo "$(date +%m-%d_%H:%M:%S) STEP ARCHIVER START every=$EVERY after=$last prefix=${PREFIX:-state}" >> $LOG
while true; do
  sleep 30
  n=$(tr -d '\r' < $S/logs/trainer.log | grep -a -oE "total steps: [0-9,]+" | tail -1 | grep -oE "[0-9,]+$" | tr -d ,)
  [ -z "$n" ] && continue
  if [ $((n % EVERY)) -eq 0 ] && [ "$n" -gt "$last" ]; then
    f=$A/${PREFIX:-state}_s${n}.pth
    for try in 1 2 3; do
      cp $S/state.pth $f.tmp && got=$($PY -c "import torch,sys; s=torch.load(sys.argv[1],weights_only=True,map_location='cpu'); assert 'mortal' in s; print(s['steps'])" $f.tmp 2>/dev/null) && break
      sleep 20
    done
    mv $f.tmp $f
    note=""; [ "$got" != "$n" ] && note=" MISMATCH(file_steps=$got)"
    echo "$(date +%m-%d_%H:%M:%S) archived $(basename $f) steps=$(printf "%'d" $n) try=${try}${note}" >> $LOG
    last=$n
  fi
done
