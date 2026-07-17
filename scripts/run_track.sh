#!/bin/bash
# 单条训练 track:训练(auto-restart 抗OOM/中断)→ 10万局对v4评测 → 聚合。
# 参数: $1=name(如 v5/v11) $2=config路径。setsid 脱离会话运行。
# 幂等:train.py 从 state_file 续训;已 eval 过则跳过。
name=$1; cfg=$2
ROOT=$HOME/Projects/better_mortal
PY=$ROOT/.venv/bin/python
LOG=$ROOT/runs/track_${name}.log
mkdir -p $ROOT/runs/$name
exec >> $LOG 2>&1
source $ROOT/.venv/bin/activate
echo "==== TRACK $name START $(date) ===="

# ---- 训练(auto-restart)----
if [ ! -f $ROOT/runs/${name}_train_done.marker ]; then
  cd $ROOT/Mortal/mortal
  tries=0
  until MORTAL_CFG=$cfg $PY train.py; do
    tries=$((tries+1)); echo "[$name] train crash try $tries $(date)"
    [ $tries -ge 100 ] && { echo "[$name] give up training"; break; }
    sleep 60
  done
  echo "[$name] train DONE $(date)"; touch $ROOT/runs/${name}_train_done.marker
fi

# ---- 评测(10万局对v4)----
if [ ! -f $ROOT/runs/${name}_eval_done.marker ]; then
  cd $ROOT/Mortal/mortal
  elog=$ROOT/runs/$name/eval_100k.log
  etries=0
  until MORTAL_CFG=$cfg $PY one_vs_three.py > $elog 2>&1; do
    etries=$((etries+1)); echo "[$name] eval crash try $etries $(date)"
    [ $etries -ge 30 ] && { echo "[$name] give up eval"; break; }
    sleep 60
  done
  $PY $ROOT/scripts/aggregate_eval.py $elog >> $elog 2>&1
  echo "[$name] eval DONE $(date)"; tail -6 $elog
  touch $ROOT/runs/${name}_eval_done.marker
fi
echo "==== TRACK $name ALL DONE $(date) ===="
