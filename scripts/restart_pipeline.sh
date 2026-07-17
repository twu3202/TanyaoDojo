#!/bin/bash
# 自包含的流水线干净重启(自身也应 setsid 运行,避免 ssh 掉线中断):
# 清掉旧 orchestrator/下载/eval_watcher 残留 → 只启动 queue_train.sh(含训练+评测)。
ROOT=$HOME/Projects/better_mortal
LOG=$ROOT/runs/restart.log
mkdir -p $ROOT/runs
exec >> $LOG 2>&1
echo "=== restart $(date) ==="
for pat in "queue_train.sh" "dl_server_data.sh" "dl_year" "dl_all_parallel" "eval_watcher.sh" "tenhou-to-mjai"; do
  for p in $(pgrep -f "$pat"); do kill -9 "$p" 2>/dev/null; done
done
sleep 5
echo "after kill: queue=$(pgrep -f queue_train.sh|wc -l) curl=$(pgrep -f tenhou-to-mjai|wc -l) evalw=$(pgrep -f eval_watcher.sh|wc -l)"
cd $ROOT
setsid bash scripts/queue_train.sh < /dev/null > /dev/null 2>&1 &
sleep 6
echo "relaunched: queue=$(pgrep -f queue_train.sh|wc -l)"
echo "=== restart done $(date) ==="
