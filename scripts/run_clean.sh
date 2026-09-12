#!/bin/bash
# qc_clean:牌山 RNG 与 GAE 差一位都修好之后,从**未受污染的 BC 基座**重跑塑形版 RL。
#
# 为什么不从 qc_shaped_srv.pkl(b1408)续:自 2026-08-23 把 round_mode 改成 half 起,
# placement(9.98亿)/ 本机塑形(1.01亿)/ srv(14.77亿)/ srv2(4.04亿)全部跑在
# "第 2..9 局牌山恒为同一组 8 副牌"的数据上(half 模式下 88.9% 的步)。那些权重是
# 记住固定牌局的产物,不能当地基。bc_v2_ft_g186 训自天凤牌谱、不经 mahjax,是干净的,
# 且它就是记分册里 -4.66 ± 0.535 的冠军基座。
# critic 同理不载入污染版,走默认 16 块预热从零训。
#
# 超参与坏数据那轮**逐字相同**,唯一差异是两个修正,这样"bug 是不是元凶"才判得干净:
#   b512  对照:坏数据线 12k = -3.788 ± 1.54
#   b1408 对照:坏数据线 12k = -4.043 ± 1.54
# 日志落 ~/ 不落 /tmp。
set -eu
cd ~/jax_rl_lean
nohup ~/mahjax_env/bin/python ppo_qcritic.py \
  round_mode=half reward_mode=shaped grp_path=$HOME/jax_rl_lean/grp.pkl obs_base=v2 \
  num_envs=8192 num_steps=32 updates_per_jit=4 update_epochs=1 minibatch_size=4096 \
  mem_fraction=0.85 channels=256 blocks=10 critic_channels=128 critic_blocks=6 \
  gamma=1.0 gae_lambda=1.0 clip_eps=0.02 lr=2e-5 \
  ent_coef=0.01 mag_coef=0.20 \
  pretrained_model_path=$HOME/jax_rl_lean/bc_v2_ft_g186.pkl \
  magnet_model_path=$HOME/jax_rl_lean/bc_v2_ft_g186.pkl \
  league_pool=$HOME/jax_rl_lean/bc_v2_ft_g186.pkl self_pool_slots=2 \
  snapshot_every_blocks=64 total_timesteps=1500000000 save_model=True \
  save_path=$HOME/jax_rl_lean/qc_clean.pkl \
  ckpt_every_blocks=8 snap_ckpt_every_blocks=64 \
  > $HOME/qc_clean.log 2>&1 &
echo "qc_clean started pid=$! init=bc_v2_ft_g186 (未污染) budget=1.5e9"
