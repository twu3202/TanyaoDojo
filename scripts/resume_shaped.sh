#!/bin/bash
# 从 09-03 12:23 的 ckpt(block 1408 / 14.77 亿步)续跑塑形版到 30 亿步。
# 快照另存 srv2 前缀,避免覆盖 b64..b1408 的历史快照。
set -eu
cd ~/jax_rl_lean
OFF=1477443584                      # 已跑步数,续跑日志的 steps 需加此偏移
REST=$((3000000000 - OFF))
nohup ~/mahjax_env/bin/python ppo_qcritic.py   round_mode=half reward_mode=shaped grp_path=$HOME/jax_rl_lean/grp.pkl obs_base=v2   num_envs=8192 num_steps=32 updates_per_jit=4 update_epochs=1 minibatch_size=4096   mem_fraction=0.85 channels=256 blocks=10 critic_channels=128 critic_blocks=6   critic_warmup_blocks=0 gamma=1.0 gae_lambda=1.0 clip_eps=0.02 lr=2e-5   ent_coef=0.01 mag_coef=0.20   pretrained_model_path=$HOME/jax_rl_lean/qc_shaped_srv.pkl   critic_pretrained_path=$HOME/jax_rl_lean/qc_shaped_srv.pkl.critic   magnet_model_path=$HOME/jax_rl_lean/bc_v2_ft_g186.pkl   league_pool=$HOME/jax_rl_lean/bc_v2_ft_g186.pkl self_pool_slots=2   snapshot_every_blocks=64 total_timesteps=$REST save_model=True   save_path=$HOME/jax_rl_lean/qc_shaped_srv2.pkl   ckpt_every_blocks=8 snap_ckpt_every_blocks=64   > /tmp/qc_shaped_srv2.log 2>&1 &
echo "resumed pid=$! offset=$OFF remaining=$REST"
