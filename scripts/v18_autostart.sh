#!/bin/bash
# 看门狗:v5手动评测完成 + 18年数据到齐 → 自动起 v18 训练(v5配方×全18年)
# setsid 脱离会话运行。run_track.sh 会自动 训练→评测。
ROOT=$HOME/Projects/better_mortal
LOG=$ROOT/runs/v18_autostart.log
mkdir -p $ROOT/runs
exec >> $LOG 2>&1
echo "==== v18 autostart watcher $(date) ===="
echo "[wait] v5 手动评测完成 marker..."
while [ ! -f $ROOT/runs/v5_manual_eval_done.marker ]; do sleep 120; done
echo "[ok] v5 评测已完成 $(date)"
echo "[wait] 18年数据到齐(>2M局)..."
while true; do
  n=$(find $ROOT/data/houou -name '*.mjson.gz' 2>/dev/null | wc -l)
  echo "  当前 $n 局 $(date)"
  [ "$n" -gt 2000000 ] && break
  sleep 300
done
echo "[ok] 数据到齐:$n 局,启动 v18 训练 $(date)"
setsid bash $ROOT/scripts/run_track.sh v18 $ROOT/configs/local_train_v18.toml < /dev/null > /dev/null 2>&1 &
sleep 8
echo "[ok] v18 track 已启动:$(pgrep -f "run_track.sh v18"|wc -l) $(date)"
