#!/bin/bash
# Phase 1 一键启动(GPU 模式下运行):
#   1) GPU 冒烟:v4 vs v4 32 局复式
#   2) GRP 全量训练(上限 4 小时,checkpoint 每 2000 步滚动保存)
#   3) 主训练 v1(40x192,离线 CQL,2019-2026 数据,1 epoch 约 3-5 天)
# 用法: setsid /root/better_mortal/scripts/run_phase1.sh < /dev/null > /root/autodl-tmp/runs/phase1.log 2>&1 &
set -e
PY=/root/miniconda3/bin/python
cd /root/better_mortal/Mortal/mortal
mkdir -p /root/autodl-tmp/runs/eval_smoke_gpu /root/autodl-tmp/runs/grp_full /root/autodl-tmp/runs/v1

echo "==== [1/3] GPU eval smoke $(date) ===="
MORTAL_CFG=/root/better_mortal/configs/remote_eval_smoke_gpu.toml $PY one_vs_three.py

echo "==== [2/3] GRP full training (4h cap) $(date) ===="
timeout 4h env MORTAL_CFG=/root/better_mortal/configs/remote_grp_full.toml $PY train_grp.py || true
ls -la /root/autodl-tmp/runs/grp_full/grp.pth

echo "==== [3/3] main training v1 $(date) ===="
MORTAL_CFG=/root/better_mortal/configs/remote_train_v1.toml $PY train.py
echo "==== phase1 done $(date) ===="
