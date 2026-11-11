#!/bin/bash
# M1 calibrated anchor, online RL on the 5060 Ti (WSL): server + N GPU workers + trainer, all local.
# Same roles as scripts/run_selfplay.sh, with one change of order: the trainer starts only after the workers
# have filled the buffer (>= MINBUF files, or after GATE_MIN minutes), because a trainer that drains a few
# hundred files at a cold start grinds on them until its next test_play (09-12: one snapshot ruined that way).
#   bash run_selfplay_m1.sh --resume --workers 4 --pool base,v4,base,/abs/C392k.pth
# Stop: bash stop_selfplay_m1.sh (by PID / process group only).
set -u
ROOT=$HOME/Projects/better_mortal
MM=$ROOT/Mortal/mortal
PY=$ROOT/.venv/bin/python
SP=${SP:-$ROOT/runs/m1_calanchor}
CFG=${CFG:-$ROOT/configs/online_selfplay_m1.toml}
PIDF=$SP/pids
LOGD=$SP/logs
STATE=$SP/state.pth
mkdir -p $SP $LOGD $SP/buffer $SP/drain

WORKERS=4; POOLSPEC="base"; MINBUF=${MINBUF:-6000}; GATE_MIN=${GATE_MIN:-40}; MODE=""; ONLY_WORKERS=0
while [ $# -gt 0 ]; do case "$1" in
  --resume)       MODE=resume; shift;;
  --workers)      WORKERS="$2"; shift 2;;
  --pool)         POOLSPEC="$2"; shift 2;;
  --only-workers) ONLY_WORKERS=1; shift;;      # smoke: server + workers, no trainer
  *) echo "unknown arg: $1"; exit 1;;
esac; done
[ "$MODE" = resume ] || { echo "use --resume (state.pth = tpt 408k full state)"; exit 1; }
[ -f "$STATE" ] || { echo "!! no $STATE"; exit 1; }
if [ -f $PIDF ]; then
  for p in $(cat $PIDF); do kill -0 "$p" 2>/dev/null && { echo "!! running (PID $p), stop first"; exit 1; }; done
fi

IFS=',' read -ra POOL <<< "$POOLSPEC"
resolve_opp() { case "$1" in base) echo "$SP/BASE.pth";; v4) echo "$ROOT/baseline/mortal_v4.pth";; /*) echo "$1";; *) echo "";; esac; }
for m in "${POOL[@]}"; do p=$(resolve_opp "$m"); [ -n "$p" ] && [ -f "$p" ] || { echo "!! pool member $m -> '$p' missing"; exit 1; }; done

cd $MM
setsid env MORTAL_CFG=$CFG $PY server.py > $LOGD/server.log 2>&1 < /dev/null &
SERVER=$!
echo "[m1] server PID $SERVER"; sleep 5
# workers need a param on the server; the trainer would submit it at startup, here a primer does the same
PYTHONPATH=$MM MORTAL_CFG=$CFG $PY $SP/m1_prime_param.py || { echo "!! param primer failed"; kill -TERM -$SERVER; exit 1; }
WPIDS=()
for i in $(seq 1 $WORKERS); do
  m=${POOL[$(( (i-1) % ${#POOL[@]} ))]}
  OPP=$(resolve_opp "$m")
  W=$SP/worker_${i}.toml
  # per-worker opponent (baseline.train) and per-worker train_play dir (player.py rmtree's it every round)
  sed -E -e "s#^state_file = .*BASE\.pth.*#state_file = '${OPP}'#" \
         -e "s#^log_dir = '.*/train_play'.*#log_dir = '$SP/train_play_w${i}'#" $CFG > $W
  mkdir -p $SP/train_play_w${i}
  setsid env MORTAL_CFG=$W TRAIN_PLAY_PROFILE=default $PY client.py > $LOGD/worker_$i.log 2>&1 < /dev/null &
  WPIDS+=($!)
  echo "[m1] worker $i PID $! opponent $m -> $OPP"
done
{ echo $SERVER; printf "%s\n" "${WPIDS[@]}"; } > $PIDF    # trainer joins as line 2 below
[ $ONLY_WORKERS = 1 ] && { echo "[m1] --only-workers: no trainer"; exit 0; }

echo "[m1] waiting for buffer >= $MINBUF files (max $GATE_MIN min) before starting the trainer"
for t in $(seq 1 $((GATE_MIN * 4))); do
  n=$(ls $SP/buffer | wc -l); [ "$n" -ge "$MINBUF" ] && break; sleep 15
done
echo "[m1] buffer $(ls $SP/buffer | wc -l) files at $(date +%T)"
setsid env MORTAL_CFG=$CFG $PY ${TRAINER:-train.py} > $LOGD/trainer.log 2>&1 < /dev/null &
TRAINER=$!
{ echo $SERVER; echo $TRAINER; printf "%s\n" "${WPIDS[@]}"; } > $PIDF
echo "[m1] trainer PID $TRAINER; pids in $PIDF (line 2 = trainer, for trainer_guard.sh)"
