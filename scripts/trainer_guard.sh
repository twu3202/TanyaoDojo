#!/bin/bash
# trainer 守护(2026-09-16): trainer 进程消失就自动拉起。
# 起因: 09-14 16:51 trainer 在 376,000 步重生子进程时 CUDA OOM 崩溃退出(当时评测站 8 个探针分片在抢显存),
#       没有守护进程, server 与 7 个 worker 空转了两天, 白白损失约 48 小时训练。
# 存活判据锚定到 venv python 的绝对路径: 只写 "train.py" 会命中我自己 ssh 过来的监控命令行(其中含 train.py),
#       守护因此连续 5 分钟以为 trainer 还活着(2026-09-16 实发)。父或子任一存活即视为健康。
S=/home/twu/Projects/better_mortal/runs/selfplay
R=/home/twu/Projects/better_mortal
MM=$R/Mortal/mortal
PY=$R/.venv/bin/python
CFG=${CFG:-$R/configs/online_selfplay_tenhoupt.toml}
LOG=$S/trainer_guard.log
MINBUF=${MINBUF:-6000}
echo "$(date +%m-%d_%H:%M:%S) GUARD START cfg=$CFG minbuf=$MINBUF" >> $LOG
miss=0
while true; do
  sleep 30
  if pgrep -f "^/home/twu/Projects/better_mortal/.venv/bin/python .*train\.py" > /dev/null; then miss=0; continue; fi
  miss=$((miss+1)); [ $miss -lt 3 ] && continue      # 连续三次(90 秒)都没有才算死
  echo "$(date +%m-%d_%H:%M:%S) ALERT trainer 不在, buffer=$(ls $S/buffer | wc -l), 准备拉起" >> $LOG
  for i in $(seq 1 240); do [ "$(ls $S/buffer | wc -l)" -ge "$MINBUF" ] && break; sleep 15; done
  cd $MM
  setsid env MORTAL_CFG=$CFG $PY train.py >> $S/logs/trainer.log 2>&1 < /dev/null &
  NEW=$!
  { sed -n 1p $S/pids; echo $NEW; sed -n '3,$p' $S/pids; } > $S/pids.new && mv $S/pids.new $S/pids
  echo "$(date +%m-%d_%H:%M:%S) 已拉起 trainer pid=$NEW buffer=$(ls $S/buffer | wc -l)" >> $LOG
  miss=0; sleep 300
done
