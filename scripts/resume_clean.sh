#!/bin/bash
# 续跑 qc_clean(牌山 RNG + GAE 差一位都修正后, 从未污染的 bc_v2_ft_g186 起的干净线)。
#
# 每次让出 GPU 再恢复都会换一个 save 前缀(否则带序号快照会被覆盖), 于是"日志步数 + 偏移"
# 的记账很容易算错。所以把累计偏移写进 ~/qc_clean_offset, 由 eval_once.sh 读取,
# 不再硬编码在评测脚本里。
#
# 用法: ~/resume_clean.sh <累计已跑步数> <本次 save 前缀>
#   例:  ~/resume_clean.sh 344981504 qc_clean3
set -eu
DONE="$1"                    # 恢复前累计步数(取自上一段日志最后一行 steps= + 其偏移)
PREFIX="$2"
BUDGET=1500000000
REST=$((BUDGET - DONE))
[ "$REST" -gt 0 ] || { echo "预算已用完 (DONE=$DONE >= $BUDGET)"; exit 1; }

# 上一段的最新滚动 ckpt 作为起点
PREV=$(ls -t $HOME/jax_rl_lean/qc_clean*.pkl 2>/dev/null | grep -vE '\.b[0-9]+\.pkl$|\.critic$' | head -1)
[ -n "$PREV" ] || { echo "找不到起点 ckpt"; exit 1; }
echo "$DONE" > $HOME/qc_clean_offset

cd ~/jax_rl_lean
nohup ~/mahjax_env/bin/python ppo_qcritic.py \
  round_mode=half reward_mode=shaped grp_path=$HOME/jax_rl_lean/grp.pkl obs_base=v2 \
  num_envs=8192 num_steps=32 updates_per_jit=4 update_epochs=1 minibatch_size=4096 \
  mem_fraction=0.95 channels=256 blocks=10 critic_channels=128 critic_blocks=6 \
  critic_warmup_blocks=0 gamma=1.0 gae_lambda=1.0 clip_eps=0.02 lr=2e-5 \
  ent_coef=0.01 mag_coef=0.20 log_every_blocks=8 \
  pretrained_model_path="$PREV" \
  critic_pretrained_path="$PREV.critic" \
  magnet_model_path=$HOME/jax_rl_lean/bc_v2_ft_g186.pkl \
  league_pool=$HOME/jax_rl_lean/bc_v2_ft_g186.pkl self_pool_slots=2 \
  snapshot_every_blocks=64 total_timesteps=$REST save_model=True \
  save_path=$HOME/jax_rl_lean/$PREFIX.pkl \
  ckpt_every_blocks=8 snap_ckpt_every_blocks=64 \
  > $HOME/qc_clean.log 2>&1 &
echo "qc_clean 续跑 pid=$! 起点=$(basename $PREV) 累计=$DONE 本段=$REST 前缀=$PREFIX"
