#!/bin/bash
# Track C 自对弈在线 RL —— 每晚分段启动/续跑。单机: server(CPU) + N worker + trainer(GPU)。
# 用法:
#   首夜(热启动基座): bash run_selfplay.sh --base /path/to/v18orv5_best.pth [--workers 4] [--pool base,base,v4,snap] [--hours 8]
#   之后每夜(续跑):    bash run_selfplay.sh --resume [--workers 4] [--pool base,base,v4,snap] [--hours 8]
#   每夜结束停止:       bash stop_selfplay.sh
#
# 设计要点:
#  - state.pth 每 save_every 步存盘(含 optimizer/scheduler/steps/best_perf)→ 续跑无损。
#  - 首夜 --base 把基座权重拷成 state.pth + BASE.pth; online 加载"只取权重、全新 optimizer"(train.py:113)。
#  - 续跑 --resume 直接从 state.pth 恢复(权重+optimizer 全量)。
#  - resume 护栏: --base 若发现 state.pth 已存在则拒绝, 避免覆盖训练进度。
#  - worker 默认 CPU 生成(不抢 trainer 的 GPU); --worker-device cuda:0 可改 GPU 生成换吞吐(会拖慢 trainer)。
#  - ⭐ 对手池(防 "Mortal killer" 反制过拟合): --pool 逗号表按序循环分给 N 个 worker,
#      成员: base=冻结基座(BASE.pth) | v4=mortal_v4 | snap=trainee最新快照 | /绝对路径=任意ckpt。
#      每夜启动时 state.pth→snapshot.pth 原子刷新 → "每晚重启"天然就是快照对手的刷新节奏
#      (TrainPlayer 启动时加载对手, 进程存续期内对手不变)。默认 pool=base(官方行为)。
#  - 所有角色 setsid 脱离进程, PID 记入 pids 文件; 停止只按 PID/进程组(绝不 pattern-kill train.py, 避免误杀 v18)。
set -u
ROOT=$HOME/Projects/better_mortal
MM=$ROOT/Mortal/mortal
PY=$ROOT/.venv/bin/python
SP=$ROOT/runs/selfplay
CFG=$ROOT/configs/online_selfplay.toml
PIDF=$SP/pids
LOGD=$SP/logs
STATE=$SP/state.pth
SNAP=$SP/snapshot.pth
mkdir -p $SP $LOGD $SP/buffer $SP/drain

WORKERS=4; HOURS=0; MODE=""; BASE=""; WDEV=cpu; POOLSPEC="base"
while [ $# -gt 0 ]; do case "$1" in
  --base)          MODE=base; BASE="$2"; shift 2;;
  --resume)        MODE=resume; shift;;
  --workers)       WORKERS="$2"; shift 2;;
  --worker-device) WDEV="$2"; shift 2;;
  --pool)          POOLSPEC="$2"; shift 2;;
  --hours)         HOURS="$2"; shift 2;;
  *) echo "unknown arg: $1"; exit 1;;
esac; done

# ---- 已有实例保护 ----
if [ -f $PIDF ]; then
  for p in $(cat $PIDF 2>/dev/null); do
    if kill -0 "$p" 2>/dev/null; then
      echo "!! 已有 selfplay 进程存活(PID $p, 见 $PIDF)。先跑 stop_selfplay.sh"; exit 1
    fi
  done
fi

# ---- hot-start vs resume(resume 护栏)----
if [ "$MODE" = base ]; then
  [ -f "$BASE" ] || { echo "!! base ckpt 不存在: $BASE"; exit 1; }
  if [ -f "$STATE" ]; then
    echo "!! state.pth 已存在 —— 拒绝用 --base 覆盖已有进度。"
    echo "   续跑请用 --resume; 若确实要从头重来, 先手动: mv $STATE $STATE.bak"; exit 1
  fi
  cp "$BASE" "$STATE"
  cp "$BASE" "$SP/BASE.pth"   # 冻结基座留档: 对手池 base 成员 + 配置 baseline.train 默认指它
  echo "[selfplay] 首夜热启动: base=$BASE → state.pth (online 只取权重+全新 optimizer)"
elif [ "$MODE" = resume ]; then
  [ -f "$STATE" ] || { echo "!! state.pth 不存在, 无法 --resume。首夜请用 --base <ckpt>"; exit 1; }
  echo "[selfplay] 续跑: 从现有 state.pth 恢复(权重+optimizer+steps)"
else
  echo "用法: --base <ckpt>(首夜) 或 --resume(之后)。见脚本头注释。"; exit 1
fi

# ---- 对手池解析 + 快照刷新 ----
IFS=',' read -ra POOL <<< "$POOLSPEC"
[ ${#POOL[@]} -ge 1 ] || { echo "!! --pool 为空"; exit 1; }
case ",$POOLSPEC," in *,snap,*)
  # 每夜启动时用当前 state.pth 原子刷新快照对手(首夜 state=基座, snap 等价 base)
  cp "$STATE" "$SNAP.tmp" && mv "$SNAP.tmp" "$SNAP"
  echo "[selfplay] snapshot.pth 已刷新(<- 当前 state.pth)";;
esac
resolve_opp() {  # 池成员 -> ckpt 路径
  case "$1" in
    base) echo "$SP/BASE.pth";;
    v4)   echo "$ROOT/baseline/mortal_v4.pth";;
    snap) echo "$SNAP";;
    /*)   echo "$1";;
    *)    echo "";;
  esac
}
for m in "${POOL[@]}"; do
  p=$(resolve_opp "$m")
  [ -n "$p" ] || { echo "!! 未知池成员: $m (可用: base|v4|snap|/绝对路径)"; exit 1; }
  [ -f "$p" ] || { echo "!! 池成员 $m 的 ckpt 不存在: $p"; exit 1; }
done

: > $PIDF
cd $MM

# ---- server(CPU, 轻)----
setsid env MORTAL_CFG=$CFG $PY server.py > $LOGD/server.log 2>&1 < /dev/null &
echo $! >> $PIDF
echo "[selfplay] server 启动 (PID $!)"; sleep 5

# ---- trainer(GPU; online main() 会自旋重生子进程)----
setsid env MORTAL_CFG=$CFG $PY train.py > $LOGD/trainer.log 2>&1 < /dev/null &
echo $! >> $PIDF
echo "[selfplay] trainer 启动 (PID $!, GPU)"; sleep 6

# ---- N workers(每个按对手池循环分配对手, 各自独立配置)----
for i in $(seq 1 $WORKERS); do
  m=${POOL[$(( (i-1) % ${#POOL[@]} ))]}
  OPP=$(resolve_opp "$m")
  WCFG_I=$SP/worker_${i}.toml
  # 生成 worker 配置: ①(可选)device→cpu ②baseline.train 指向该 worker 的对手
  if [ "$WDEV" = cpu ]; then
    sed 's#cuda:0#cpu#g' $CFG | sed -E "s#^state_file = .*BASE\.pth.*#state_file = '${OPP}'#" > $WCFG_I
  else
    sed -E "s#^state_file = .*BASE\.pth.*#state_file = '${OPP}'#" $CFG > $WCFG_I
  fi
  setsid env MORTAL_CFG=$WCFG_I TRAIN_PLAY_PROFILE=default $PY client.py > $LOGD/worker_$i.log 2>&1 < /dev/null &
  echo $! >> $PIDF
  echo "[selfplay] worker $i 启动 (PID $!, device=$WDEV, 对手=$m -> $OPP)"
done
echo "[selfplay] 全部就位。对手池: $POOLSPEC  PIDs: $PIDF  日志: $LOGD"

# ---- 可选: 到点自动优雅停(每夜时长预算)----
if [ "$HOURS" != 0 ]; then
  secs=$(awk "BEGIN{print int($HOURS*3600)}")
  setsid bash -c "sleep $secs; bash $ROOT/scripts/stop_selfplay.sh" < /dev/null > $LOGD/autostop.log 2>&1 &
  echo "[selfplay] 将在 ${HOURS}h 后自动优雅停止(见 $LOGD/autostop.log)"
fi
