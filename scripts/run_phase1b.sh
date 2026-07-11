#!/bin/bash
set -e
PY=/root/miniconda3/bin/python
cd /root/better_mortal/Mortal/mortal
echo "==== [2/3] GRP full training (4h cap) $(date) ===="
timeout 4h env MORTAL_CFG=/root/better_mortal/configs/remote_grp_full.toml $PY train_grp.py || true
ls -la /root/autodl-tmp/runs/grp_full/grp.pth
echo "==== [3/3] main training v1 $(date) ===="
MORTAL_CFG=/root/better_mortal/configs/remote_train_v1.toml $PY train.py
echo "==== phase1 done $(date) ===="
