#!/bin/bash
# p3cap(P3 观测 + 256x10)的 seed 复制:A 段 5033 万 -> 自动接 B 段 1.51 亿,两段结构与 seed=0 逐项相同。
# 用法: run_p3cap_seed.sh <seed>
# 这是**抢跑**:容量判据评测正在服务器 CPU 上跑。过线则 seed 本来就要补;不过线则浪费的是本来空转的 GPU。
set -eu
S="${1:?seed}"
R=$HOME/Better_mortal/runs/sichuan_p3
mkdir -p "$R"
PY=$HOME/mahjax_env/bin/python
COMMON="seed=$S dual_c=3.0 obs_disc=True channels=256 blocks=10 mem_fraction=0.95"
A="cd $HOME/Better_mortal && $PY -m sichuan.ppo_sichuan $COMMON total_timesteps=50331648 snapshot_every_blocks=384 log_every_blocks=64 save_path=$R/p3cap_s$S.pkl"
B="cd $HOME/Better_mortal && $PY -m sichuan.ppo_sichuan $COMMON init_from=$R/p3cap_s$S.pkl steps_done=50331648 total_timesteps=100663296 snapshot_every_blocks=768 log_every_blocks=128 save_path=$R/p3cap_s${S}_long.pkl"
nohup bash -c "$A && $B" > $HOME/p3cap_s$S.log 2>&1 < /dev/null &
echo "p3cap seed=$S pid=$!"
