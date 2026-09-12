#!/bin/bash
# 川麻 P3 观测 + 容量 256x10,补 {观测} x {容量} 2x2 的缺格。
# 用法: run_p3cap.sh <stage>    stage=a 从零到 5033 万;stage=b 续到 1.51 亿
#
# 立直线优先:这个跑在服务器上只是因为立直线正在本机 CPU 上评测、暂时不需要 GPU。
# 立直线一出结果就让位 —— 停机只需 kill,快照按 block 滚动落盘。
set -eu
STAGE="${1:?stage a|b}"
R=$HOME/Better_mortal/runs/sichuan_p3
mkdir -p "$R"
cd ~/Better_mortal

COMMON="seed=0 dual_c=3.0 obs_disc=True channels=256 blocks=10 mem_fraction=0.95"

if [ "$STAGE" = a ]; then
  nohup ~/mahjax_env/bin/python -m sichuan.ppo_sichuan $COMMON \
    total_timesteps=50331648 snapshot_every_blocks=64 log_every_blocks=32 \
    save_path="$R/p3cap.pkl" > $HOME/p3cap_a.log 2>&1 &
else
  nohup ~/mahjax_env/bin/python -m sichuan.ppo_sichuan $COMMON \
    init_from="$R/p3cap.pkl" steps_done=50331648 total_timesteps=100663296 \
    snapshot_every_blocks=128 log_every_blocks=64 \
    save_path="$R/p3cap_long.pkl" > $HOME/p3cap_b.log 2>&1 &
fi
echo "p3cap stage=$STAGE pid=$!"
