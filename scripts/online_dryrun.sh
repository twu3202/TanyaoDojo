#!/bin/zsh
# Track C 完整三件套 dry-run(Mac 隔离,CPU)。启动 server+trainer+worker 跑数分钟,
# 验证:自对弈→提交buffer→trainer drain训练→推新参数 这个环能转起来。
set -u
PY=/opt/anaconda3/envs/mj-mortal/bin/python
CFG=/Users/r/HMM/Better_mortal/configs/mac_online_dryrun.toml
MM=/Users/r/HMM/Better_mortal/Mortal/mortal
D=/tmp/mortal_dryrun/online_dry
rm -rf $D && mkdir -p $D
# 热启动:v1_best 作为初始 state(trainer 会加载其权重,online模式不加载其optimizer)
cp /Users/r/HMM/Better_mortal/weights_backup/v1_best.pth $D/state.pth

cd $MM
echo "[dry] starting server..."
MORTAL_CFG=$CFG $PY server.py > $D/server.log 2>&1 &
SV=$!
sleep 4
echo "[dry] starting trainer (online)..."
MORTAL_CFG=$CFG $PY train.py > $D/trainer.log 2>&1 &
TR=$!
sleep 6
echo "[dry] starting worker..."
MORTAL_CFG=$CFG $PY client.py > $D/worker.log 2>&1 &
WK=$!

echo "[dry] running ~6min (server=$SV trainer=$TR worker=$WK)..."
sleep 360

echo "[dry] stopping all..."
kill -9 $WK $TR $SV 2>/dev/null
pkill -9 -f "[c]lient.py"; pkill -9 -f "[s]erver.py"; pkill -9 -f "[t]rain.py.*" 2>/dev/null
# 只杀本 dry-run 的(Mac 本地,服务器训练在远端不受影响)
sleep 2
echo "[dry] === server.log tail ==="; tail -6 $D/server.log
echo "[dry] === worker.log tail ==="; grep -aE "rankings|submitted|param" $D/worker.log | tail -6
echo "[dry] === trainer.log tail ==="; grep -aE "drain|param|loss|steps|submitted|Error|Traceback|device" $D/trainer.log | tail -12
echo "[dry] === buffer/drain state ==="; ls $D/drain 2>/dev/null | wc -l | xargs echo "drain files:"; ls $D/train_play 2>/dev/null | wc -l | xargs echo "generated logs:"
