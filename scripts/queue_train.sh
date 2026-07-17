#!/bin/bash
# 主控队列(setsid 脱离会话运行,抗 WiFi 断线):
#   下数据 → 编译 v5 libriichi → 训 v5(auto-restart) → 训 v11(auto-restart)
# 状态写 ~/Projects/better_mortal/runs/queue.log 与各 step 的 marker。
ROOT=$HOME/Projects/better_mortal
VENV=$ROOT/.venv/bin/activate
PY=$ROOT/.venv/bin/python
LOG=$ROOT/runs/queue.log
mkdir -p $ROOT/runs
exec >> $LOG 2>&1
echo "================ QUEUE START $(date) ================"
source $VENV

# ---- step 1: 数据 ----
NGAMES=$(find $ROOT/data/houou -name '*.mjson.gz' 2>/dev/null | wc -l)
if [ "$NGAMES" -lt 1000000 ]; then
  echo "[queue] downloading data ($(date))..."
  bash $ROOT/scripts/dl_server_data.sh
fi
echo "[queue] data ready: $(find $ROOT/data/houou -name '*.mjson.gz' | wc -l) games"

# ---- step 2: 编译 v5 libriichi ----
echo "[queue] building libriichi v5 ($(date))..."
cd $ROOT/Mortal
PYO3_PYTHON=$PY $HOME/.cargo/bin/cargo build -p libriichi --lib --release && \
  cp target/release/libriichi.so mortal/libriichi.so
$PY -c "import sys; sys.path.insert(0,'$ROOT/Mortal/mortal'); from libriichi.consts import obs_shape; assert obs_shape(5)==(1022,34), obs_shape(5); print('[queue] libriichi v5 OK', obs_shape(5))" || { echo "[queue] libriichi v5 FAILED, abort"; exit 1; }

run_job() {  # $1=name $2=config
  name=$1; cfg=$2
  echo "[queue] === train $name START $(date) ==="
  cd $ROOT/Mortal/mortal
  tries=0
  until MORTAL_CFG=$cfg $PY train.py; do
    tries=$((tries+1))
    echo "[queue] $name crashed (try $tries), resume in 60s $(date)"
    [ $tries -ge 100 ] && { echo "[queue] $name too many retries, skip"; break; }
    sleep 60
  done
  echo "[queue] === train $name DONE $(date) ==="
  touch $ROOT/runs/${name}_train_done.marker
}

run_eval() {  # $1=name $2=config —— 10万局对 v4,聚合
  name=$1; cfg=$2
  echo "[queue] === eval $name vs v4 START $(date) ==="
  cd $ROOT/Mortal/mortal
  mkdir -p $ROOT/runs/$name
  elog=$ROOT/runs/$name/eval_100k.log
  tries=0
  until MORTAL_CFG=$cfg $PY one_vs_three.py > $elog 2>&1; do
    tries=$((tries+1)); echo "[queue] eval $name crashed (try $tries), retry 60s"
    [ $tries -ge 30 ] && break; sleep 60
  done
  $PY $ROOT/scripts/aggregate_eval.py $elog >> $elog 2>&1
  echo "[queue] === eval $name DONE $(date) ==="; tail -6 $elog
  touch $ROOT/runs/${name}_eval_done.marker
}

# ---- step 3: v5(旗舰:防守特征+LR衰减+256宽)训 + 评 ----
run_job v5 $ROOT/configs/local_train_v5.toml
run_eval v5 $ROOT/configs/local_train_v5.toml

# ---- step 4: v11(LR隔离基线, 192x40)训 + 评 ----
run_job v11 $ROOT/configs/local_train_v11.toml
run_eval v11 $ROOT/configs/local_train_v11.toml

echo "================ QUEUE ALL DONE $(date) ================"
touch $ROOT/runs/queue_all_done.marker
