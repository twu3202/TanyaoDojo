#!/bin/bash
# 2026-09-17 对手池实验: 训练对手换成与评测相同的 mortal_v4(单杠杆)
# 起因: 09-16 23:18 之后 310 轮自对弈里只有 52 轮(16.8%)打 v4, 66% 打 rl1_best(BASE)、17% 打 C392k —— 两者选立直约 43.8%,
#       评测对手 v4 是 39.0%。与"奖励口径 ≠ 评测口径"同类:训练分布 ≠ 评测分布。放松锚后打法一路往被动滑(副露 30.6 → 27.3)
#       与"对着激进对手多弃牌划算"一致。
# 设计: 从 tpt_s408000 分叉, 只改对手池 → 6 个 worker 全部打 v4; server capacity 4000 → 5000 使每次 drain 仍为 9,600(6×800×2 轮)。
#       锚 C' final / λ=0.1 / 天凤口径 pts / loader 12 / test_play 800 与 408k→448k 那段完全相同。
# 判据(先写下): 448,000 步做 100k, 与 tpt_s448000(同起点、只差对手池)在同一批 100k 牌上配对。
#   配对 z >= +2 → 对手池是杠杆, 以后训练对手一律 v4;  |z| < 2 → 记录, 看机制;  z <= -2 → 回退。
#   机制预言: 副露率不像 tpt 448k 那样掉到 27.3(应保持 ≥ 30), 和了差不到 -0.72, 被飞差 < +0.43pp、"未进南四"行 > -1.19。
#   止损: 选立直 >= 43% 或 放铳差 >= +0.8pp 或 drain 连续两次 < 8000(数据量混杂)。
S=/home/twu/Projects/better_mortal/runs/selfplay
R=/home/twu/Projects/better_mortal
MM=$R/Mortal/mortal
PY=$R/.venv/bin/python
E=/home/twu/evalsta
CFGT=$R/configs/online_selfplay_tenhoupt.toml
CFGV=$R/configs/online_selfplay_tpt_v4pool.toml
V4=$R/baseline/mortal_v4.pth

echo "== $(date +%T) 停旧分支"
for p in $(ps -eo pid,args | awk '$2=="bash" && ($3 ~ /trainer_guard\.sh$/ || $3 ~ /srv_evalwatch\.sh$/ || $3 ~ /archiver_steps\.sh$/ || $3 ~ /probe_par\.sh$/) {print $1}'); do kill $p && echo "  stopped daemon $p"; done
pkill -f "[d]iag_disagree.py"
TRN=$(sed -n 2p $S/pids); SRV=$(sed -n 1p $S/pids)
old_steps=$(tr -d '\r' < $S/logs/trainer.log | grep -a -oE "total steps: [0-9,]+" | tail -1)
kill -TERM -$TRN 2>/dev/null; kill -TERM $TRN 2>/dev/null
pkill -f "^/home/twu/Projects/better_mortal/.venv/bin/python .*train\.py"
pkill -f "^/home/twu/Projects/better_mortal/.venv/bin/python client\.py"
sleep 8
cp $S/state.pth $S/archive_resume/state_tpt_end_$(date +%m%d_%H%M).pth
echo "  tpt 分支末态已存($old_steps)"
kill $SRV; sleep 3
pgrep -af "^/home/twu/Projects/better_mortal/.venv/bin/python (server|client|.*train)\.py" && { echo "!! 还有残留进程, 中止"; exit 1; }

echo "== 配置"
awk '/^\[/ {sec=$0}
 sec=="[online.server]" && /^capacity/ {print "capacity = 5000             # 2026-09-17 对手池实验: 6 个 v4 worker × 800 局 × 2 轮 = 每次 drain 9,600(与此前相同)"; next}
 {print}' $CFGT > $CFGV
sed -i '1i # 对手池实验(2026-09-17): 与 online_selfplay_tenhoupt.toml 只差 [online.server] capacity(4000→5000, 保持 drain 9,600);\n# 对手换成 v4 在 worker 配置 worker_v4_N.toml 里([baseline.train] state_file = mortal_v4.pth)。' $CFGV
diff $CFGT $CFGV
for i in 1 2 3 4 5 6; do
  sed "s#train_play_w3#train_play_v4_w$i#" $S/worker_3.toml > $S/worker_v4_$i.toml
  mkdir -p $S/train_play_v4_w$i
done
grep -H -A3 "^\[baseline.train\]" $S/worker_v4_1.toml | grep state_file
grep -H "^device" $S/worker_v4_*.toml | grep -c cuda

echo "== 恢复 408k 并冷启动"
cp $S/archive_resume/tpt_s408000.pth $S/state.pth
$PY -c "import torch; print('state.pth steps =', torch.load('$S/state.pth', weights_only=True, map_location='cpu')['steps'])"
cd $MM
setsid env MORTAL_CFG=$CFGV $PY server.py > $S/logs/server.log 2>&1 < /dev/null &
SRV=$!; sleep 5
setsid env MORTAL_CFG=$CFGV $PY train.py > $S/logs/trainer.boot_v4p.log 2>&1 < /dev/null &
B=$!
for i in $(seq 1 60); do grep -a -q "param has been submitted" $S/logs/trainer.boot_v4p.log && break; sleep 3; done
sleep 3; kill -TERM -$B 2>/dev/null; kill -TERM $B 2>/dev/null; sleep 4
pkill -KILL -f "^/home/twu/Projects/better_mortal/.venv/bin/python .*train\.py"
grep -a -E "loaded:|param has been submitted" $S/logs/trainer.boot_v4p.log
W=()
for i in 1 2 3 4 5 6; do
  setsid env MORTAL_CFG=$S/worker_v4_$i.toml TRAIN_PLAY_PROFILE=default $PY client.py > $S/logs/worker_v4_$i.log 2>&1 < /dev/null &
  W+=($!)
done
echo "  workers: ${W[*]}   等 buffer 攒满(>= 9000 且 90 秒不再增长)"
prev=-1; stable=0; t0=$(date +%s)
for i in $(seq 1 120); do
  n=$(ls $S/buffer | wc -l)
  if [ "$n" -eq "$prev" ] && [ "$n" -ge 9000 ]; then stable=$((stable+1)); else stable=0; fi
  [ $stable -ge 3 ] && break
  prev=$n; sleep 30
done
echo "  $(date +%T) buffer=$(ls $S/buffer | wc -l)(用时 $(( ($(date +%s)-t0)/60 )) 分钟)"
grep -a "total buffer size" $S/logs/server.log | tail -3 | cut -c12-19,60-

mv $S/logs/trainer.log $S/logs/trainer.tpt.log
setsid env MORTAL_CFG=$CFGV $PY train.py > $S/logs/trainer.log 2>&1 < /dev/null &
TRN=$!
{ echo $SRV; echo $TRN; printf "%s\n" "${W[@]}"; } > $S/pids
for i in $(seq 1 60); do grep -a -q "file list size" $S/logs/trainer.log && break; sleep 5; done
tr -d '\r' < $S/logs/trainer.log | grep -a -E "anchor enabled|loaded:|file list size|total steps" | tail -4

CFG=$CFGV setsid nohup bash $S/trainer_guard.sh > /dev/null 2>&1 < /dev/null &
PREFIX=v4p START_AFTER=408000 setsid nohup bash $S/archiver_steps.sh > /dev/null 2>&1 < /dev/null &
echo v4p > $E/tag.txt; : > $E/full12k.txt; echo 3 > $E/K
cd $E && PREFIX=v4p FROM=416000 setsid nohup bash srv_evalwatch.sh > /dev/null 2>&1 < /dev/null &
sleep 2
ps -eo pid,args | grep -E "[t]rainer_guard|[a]rchiver_steps|[s]rv_evalwatch" | cut -c1-70
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo V4POOL_DONE $(date +%T)
