#!/bin/bash
# 在服务器上按神圣协议评测价值线 C'(锚定值微调,从 rl1_best 重基)的 best.pth(07-24)。
# 与本机 rl1_best 评测用同一批牌局 seeds 10000-10999,跑完当场 dump 逐牌局组,供配对比较。
# 冠军走 CPU、不开 AMP,与项目历史评测口径一致;服务器 48 核。
set -eu
B=$HOME/Projects/better_mortal
T=$HOME/Better_mortal/jax_rl/mjai_bot
LOG=$B/runs/eval_cprime_best
mkdir -p "$LOG" "$B/runs/evaldump"
export MORTAL_DIR=$B/Mortal/mortal MORTAL_V4=$B/baseline/mortal_v4.pth
PY=$B/.venv/bin/python
A="cd $T && $PY run_eval.py $B/runs/selfplay/best.pth --challenger-type mortal --challenger-name cprime_best --device cpu --games 400 --iters 10 --seed-start 10000 --log-dir $LOG"
D="cd $T && $PY group_ci.py --log-dir $LOG --seed-lo 10000 --seed-hi 10999 --dump $B/runs/evaldump/cprime_best_s10000.npz"
nohup bash -c "$A && $D" > "$HOME/eval_cprime.log" 2>&1 < /dev/null &
echo "cprime eval pid=$!"
