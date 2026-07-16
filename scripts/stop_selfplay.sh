#!/bin/bash
# Track C 自对弈 —— 每晚优雅停止。只按 PID/进程组杀, 绝不 pattern-kill train.py
# (v18 等其它训练也用 train.py, pattern-kill 会误杀!)。state.pth 保留供次夜 --resume。
set -u
ROOT=$HOME/Projects/better_mortal
SP=$ROOT/runs/selfplay
PIDF=$SP/pids
STATE=$SP/state.pth

if [ ! -f "$PIDF" ]; then
  echo "[selfplay-stop] 无 $PIDF, 没有在跑的 selfplay。"; exit 0
fi

echo "[selfplay-stop] 优雅停止(SIGTERM 进程组)..."
# 每个角色都是 setsid 起的 → 自身即进程组长; 杀负 PID = 杀整组(含 trainer 自旋的子进程)。
for p in $(cat "$PIDF"); do
  kill -TERM -"$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null
done
sleep 6
# 仍存活的强杀
for p in $(cat "$PIDF"); do
  if kill -0 "$p" 2>/dev/null; then
    kill -KILL -"$p" 2>/dev/null || kill -KILL "$p" 2>/dev/null
  fi
done
sleep 2

# 校验(只看本任务 PID, 不 pattern 匹配)
alive=0
for p in $(cat "$PIDF"); do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
echo "[selfplay-stop] 本任务残留存活进程: $alive"
echo "[selfplay-stop] state.pth: $(ls -la $STATE 2>/dev/null | awk '{print $5, $6, $7, $8}' || echo none)"
mv "$PIDF" "$PIDF.stopped.$(cat $SP/.stopseq 2>/dev/null || echo 0)" 2>/dev/null || rm -f "$PIDF"
echo "[selfplay-stop] 完成。次夜续跑: bash run_selfplay.sh --resume"
