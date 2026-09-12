#!/bin/bash
# 川麻容量臂(P3 观测 + 256x10)的预注册判据:直接对打同观测的 128x6(p3obs),n=3000。
# 判据(跑前写死):1.51 亿档 z>=2 且为正 -> "感知饱和后容量开始付钱",补 seed;否则容量在新观测下仍为零。
# 5033 万档一并测,只看形状、不作判据。评测是 CPU 绑定,服务器 48 核空着。
set -eu
cd ~/Better_mortal
R=$HOME/Better_mortal/runs/sichuan_p3
export JAX_PLATFORMS=cpu
PY=$HOME/mahjax_env/bin/python
nohup $PY -m sichuan.eval_h2h "$R/p3cap_long.pkl" "$HOME/p3obs_long.pkl" 3000 \
  > $HOME/h2h_cap_long.log 2>&1 < /dev/null &
echo "cap_long pid=$!"
nohup $PY -m sichuan.eval_h2h "$R/p3cap.pkl" "$HOME/p3obs.pkl" 3000 \
  > $HOME/h2h_cap_a.log 2>&1 < /dev/null &
echo "cap_a pid=$!"
