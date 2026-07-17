#!/bin/bash
# 独立评测看门狗:等训练 marker → 自动跑 10万局对 v4 → 聚合。
# 与主队列解耦,setsid 脱离会话,抗 WiFi 断线。
ROOT=$HOME/Projects/better_mortal
PY=$ROOT/.venv/bin/python
LOG=$ROOT/runs/eval_watcher.log
mkdir -p $ROOT/runs
exec >> $LOG 2>&1
source $ROOT/.venv/bin/activate
echo "================ EVAL WATCHER START $(date) ================"

run_eval() {  # $1=name $2=config
  name=$1; cfg=$2
  [ -f $ROOT/runs/${name}_eval_done.marker ] && { echo "[evalw] $name already evaled"; return; }
  echo "[evalw] waiting for ${name}_train_done.marker ..."
  while [ ! -f $ROOT/runs/${name}_train_done.marker ]; do sleep 120; done
  echo "[evalw] $name train done, eval START $(date)"
  cd $ROOT/Mortal/mortal
  mkdir -p $ROOT/runs/$name
  elog=$ROOT/runs/$name/eval_100k.log
  tries=0
  until MORTAL_CFG=$cfg $PY one_vs_three.py > $elog 2>&1; do
    tries=$((tries+1)); echo "[evalw] eval $name crash (try $tries) retry 60s"
    [ $tries -ge 30 ] && break; sleep 60
  done
  $PY $ROOT/scripts/aggregate_eval.py $elog >> $elog 2>&1
  echo "[evalw] $name eval DONE $(date)"; tail -6 $elog
  touch $ROOT/runs/${name}_eval_done.marker
}

run_eval v5 $ROOT/configs/local_train_v5.toml
run_eval v11 $ROOT/configs/local_train_v11.toml
echo "================ EVAL WATCHER ALL DONE $(date) ================"
touch $ROOT/runs/all_eval_done.marker
