#!/bin/bash
# srv3:牌山 RNG 补丁 + GAE 差一位修正后的证伪段。
#
# 设计:与 srv2 严格配对 —— 同一起点(qc_shaped_srv.pkl = b1408 = 14.77 亿步)、
# 同样步数(403,701,760,即 srv2 实跑的那一段)、其余超参逐字相同。
# 唯一差异是两个修正:
#   (1) mahjax _init 播种 rng_key(此前第 2 局起牌山恒为常数,half 模式下 88.9% 的步中招)
#   (2) ppo_qcritic 逐座位 GAE 的盘间重置改用 t+1 的 is_new_episode(此前差一位)
# 若 srv3 仍与 srv2 同样横盘,则 bug 不是塑形版不涨的原因,该假设可从清单划掉。
#
# 日志落 ~/ 不落 /tmp —— 立直线的等价迁移一直没做,这次顺手做掉。
set -eu
cd ~/jax_rl_lean
OFF=1477443584                      # 起点已跑步数;本段日志 steps 需加此偏移
SEG=403701760                       # 与 srv2 等长
nohup ~/mahjax_env/bin/python ppo_qcritic.py \
  round_mode=half reward_mode=shaped grp_path=$HOME/jax_rl_lean/grp.pkl obs_base=v2 \
  num_envs=8192 num_steps=32 updates_per_jit=4 update_epochs=1 minibatch_size=4096 \
  mem_fraction=0.85 channels=256 blocks=10 critic_channels=128 critic_blocks=6 \
  critic_warmup_blocks=0 gamma=1.0 gae_lambda=1.0 clip_eps=0.02 lr=2e-5 \
  ent_coef=0.01 mag_coef=0.20 \
  pretrained_model_path=$HOME/jax_rl_lean/qc_shaped_srv.pkl \
  critic_pretrained_path=$HOME/jax_rl_lean/qc_shaped_srv.pkl.critic \
  magnet_model_path=$HOME/jax_rl_lean/bc_v2_ft_g186.pkl \
  league_pool=$HOME/jax_rl_lean/bc_v2_ft_g186.pkl self_pool_slots=2 \
  snapshot_every_blocks=64 total_timesteps=$SEG save_model=True \
  save_path=$HOME/jax_rl_lean/qc_fixed_srv3.pkl \
  ckpt_every_blocks=8 snap_ckpt_every_blocks=64 \
  > $HOME/qc_fixed_srv3.log 2>&1 &
echo "srv3 started pid=$! offset=$OFF seg=$SEG log=~/qc_fixed_srv3.log"
