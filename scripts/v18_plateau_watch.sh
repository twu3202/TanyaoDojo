#!/bin/bash
# v18 floor-LR plateau watchdog (server-side, detached).
# Fires ONLY on the known pathology: steps past max_steps (LR floored) AND best.pth
# not improving for >1.5h AND v18 still training. Writes an alert marker, then exits.
# Lightweight: it only ALERTS (marker), never auto-stops.
R=$HOME/Projects/better_mortal
MAXSTEPS=1240000      # v18 config max_steps (LR reaches floor here)
STALE=5400           # best.pth stale threshold = 1.5h
LOG=$R/runs/v18_plateau_watch.log
ALERT=$R/runs/v18_plateau_alert.marker
rm -f "$ALERT"
echo "==== v18 plateau watch start $(date) (maxsteps=$MAXSTEPS stale=${STALE}s) ====" >> "$LOG"
while true; do
  # exit if v18 training no longer running (finished / stopped) or already evaluated
  if ! pgrep -f "[r]un_track.sh v18" >/dev/null 2>&1; then
    echo "[$(date)] v18 run_track gone -> watcher exit" >> "$LOG"; break
  fi
  if [ -f "$R/runs/v18_eval_done.marker" ]; then
    echo "[$(date)] v18 already evaluated -> watcher exit" >> "$LOG"; break
  fi
  steps=$(grep -a "total steps" "$R/runs/track_v18.log" 2>/dev/null | tail -1 | sed -E 's/.*total steps: ([0-9,]+).*/\1/' | tr -d ',')
  bmt=$(stat -c %Y "$R/runs/v18/best.pth" 2>/dev/null || echo 0)
  now=$(date +%s)
  age=$(( now - bmt ))
  echo "[$(date)] steps=${steps:-?} best_age=${age}s" >> "$LOG"
  if [ -n "$steps" ] && [ "$steps" -ge "$MAXSTEPS" ] 2>/dev/null && [ "$age" -ge "$STALE" ]; then
    echo "[$(date)] PLATEAU DETECTED: steps=$steps past max_steps, best.pth stale ${age}s" >> "$LOG"
    printf "v18 平台期空转: 步数=%s (已过 max_steps=%s, LR触底), best.pth 已停更 %d 分钟。建议按 v11 同款: 停 v18 进程组 + 手动起 100k 评测。检测于 %s\n" \
      "$steps" "$MAXSTEPS" "$((age/60))" "$(date)" > "$ALERT"
    break
  fi
  sleep 600
done
echo "==== v18 plateau watch end $(date) ====" >> "$LOG"
