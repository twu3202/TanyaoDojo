#!/bin/bash
# C' step replay on trx with July's topology, arm ct (patches/cprime_replay/README.md, "Second round"). Chosen by the
# user on 11-07, as the low-priority user of the trx GPU: it runs only while nobody else uses the GPU, marks itself in
# ~/GPU_BUSY_mortal_lowprio as low priority and interruptible, and yields within about a minute to any other GPU
# process or any other ~/GPU_BUSY_* file. It resumes from state.pth once the GPU has been free for 10 minutes.
#   0. set-up of runs/cprime_trx:
#      - state.pth = C392k's full July state (md5 7a679fb4);
#      - BASE.pth = July's BASE.pth, rl1_best (b9c0ff1f), the anchor and the base opponents;
#      - snap_C392k.pth = a fixed copy of C392k for the snap seat;
#      - the config is written from C392k's embedded July config, and ct_setup.py checks it key by key
#   1. server + 7 GPU workers + trainer, as in July:
#      - pool base,base,v4,snap (w1 w2 w5 w6 base, w3 w7 v4, w4 snap);
#      - loader num_workers 0, test_play 2,000 games
#   2. lambda 0.5 up to 240,000. The config is switched to 0.2 just before the trainer's own respawn at 240,000, as
#      July did, without restarting the workers. Then lambda 0.2 up to 278,600.
#   3. archives every 8,000 steps and at 278,600; at the end SP/CT_READY with their md5. A Mac relay copies them to the
#      5060 Ti, which evaluates them on the same device as every other snapshot of this comparison.
# Log SP/ct_chain.log; markers CT_CHAIN_DONE / CT_CHAIN_STOPPED (also after 72 hours without reaching 278,600).
# Stop: kill $(cat SP/ct_chain.pid); bash SP/tools/xfer_helpers.sh stop SP; rm ~/GPU_BUSY_mortal_lowprio
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/cprime_trx/tools/ct_chain.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; SP=$R/cprime_trx; TL=$SP/tools; PY=$B/.venv/bin/python
CFG=$B/configs/online_selfplay_cprime_trx.toml; LOG=$SP/ct_chain.log; MARK=$HOME/GPU_BUSY_mortal_lowprio
SRC=$R/selfplay/state_C392k_snapshot.pth; ANC=$R/selfplay/BASE.pth; CPF=$R/selfplay/cprime_final_0724_0939.pth
PORT=5601; NW=7; POOL="base,base,v4,$SP/snap_C392k.pth"; SWITCH=240000; STOP_AT=278600
CK=$SP/archive/xfer_ct_s$STOP_AT.pth; DMIN=20; DEADLINE=$(( $(date +%s) + 72 * 3600 ))
mkdir -p $SP/logs $SP/buffer $SP/drain $SP/archive
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
dfree() { df -BG --output=avail $HOME | tail -1 | tr -dc 0-9; }
mark_on() {
  printf '%s\n' "低优先级，可随时打断 / LOW PRIORITY, interruptible at any time" \
    "TanyaoDojo mortal: C' step replay on July's topology (server + 7 GPU workers + trainer), on the GPU since $(date '+%F %T')." \
    "要用卡：直接开始你的任务，或者建一个 $HOME/GPU_BUSY_<名字>；mortal 会在 1 分钟内自动停下让出，GPU 空闲 10 分钟后再自动续跑。" \
    "To take the GPU: start your job, or create $HOME/GPU_BUSY_<name>; mortal stops within a minute and resumes after the GPU has been free for 10 minutes." \
    "If your job needs most of the 48 GB as it starts, create the BUSY file first and wait a minute." \
    "This file exists only while mortal is on the GPU. Manual stop: kill \$(cat $SP/ct_chain.pid); bash $TL/xfer_helpers.sh stop $SP; rm $MARK" \
    > $SP/mark.new && mv $SP/mark.new $MARK
}
stopped() { stop_run; say "$1: CT_CHAIN_STOPPED"; exit 1; }
# prints why the GPU is not ours to use, or nothing: another BUSY file, a GPU process that is not this arm's, or no
# nvidia-smi answer (then nothing is known, so it counts as taken)
foreign() {
  local f p out
  for f in $HOME/GPU_BUSY_*; do [ -e "$f" ] && [ "$f" != "$MARK" ] && { echo "booking $(basename $f)"; return; }; done
  out=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) || { echo "nvidia-smi failed"; return; }
  for p in $(echo "$out" | tr -dc '0-9\n'); do
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q '^MORTAL_CFG=.*cprime_trx' \
      || { echo "GPU process $p ($(ps -o user=,args= -p $p 2>/dev/null | cut -c1-80))"; return; }
  done
}
wait_free() {   # wait_free N: N polls of 30 s in a row with the GPU free
  local n=0 r
  while [ $n -lt $1 ]; do
    r=$(foreign); if [ -z "$r" ]; then n=$((n + 1)); else n=0; fi
    [ $(date +%s) -lt $DEADLINE ] || return 1
    sleep 30
  done
}
lam_now() { grep -oE '^weight = [0-9.]+' $CFG | grep -oE '[0-9.]+$'; }
set_lam() { awk -v w="$1" '/^weight = /{print "weight = " w; next} {print}' $CFG > $CFG.new && mv $CFG.new $CFG; }
# the anchor weight the running trainer read, and the latest step it logged after reading it (log lines after L0)
lam_view() {
  tail -n +$((L0 + 1)) $SP/logs/trainer.log 2>/dev/null | tr -d '\r' | awk '
    /anchor enabled: weight=/ { match($0, /weight=[0-9.]+/); w = substr($0, RSTART + 7, RLENGTH - 7); s = 0 }
    /total steps: / { match($0, /total steps: [0-9,]+/); t = substr($0, RSTART + 13, RLENGTH - 13); gsub(",", "", t); s = t + 0 }
    END { print (w == "" ? "-" : w), s + 0 }'
}
state_steps() { $PY -c "import torch,sys; print(torch.load(sys.argv[1],weights_only=True,map_location='cpu')['steps'])" $SP/state.pth; }
last_arch() {
  ls $SP/archive/xfer_ct_s*.pth 2>/dev/null | grep -v mismatch | grep -oE '_s[0-9]+\.pth$' | tr -dc '0-9\n' | sort -n | tail -1
}
stop_run() {
  for p in $(cat $SP/helper_pids 2>/dev/null); do kill -TERM -"$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null; done
  : > $SP/helper_pids
  [ -f $SP/pids ] && bash $TL/xfer_helpers.sh stop $SP >> $LOG 2>&1
  rm -f $MARK
}
start_run() {   # returns 0 started, 1 failed, 3 yielded while starting
  local st la lp r i
  st=$(state_steps) || return 1
  if [ "$st" -ge $SWITCH ]; then set_lam 0.2; else set_lam 0.5; fi
  L0=$(cat $SP/logs/trainer.log 2>/dev/null | wc -l)
  la=$(last_arch); la=${la:-215800}
  say "start at state step $st, lambda $(lam_now), archives after $la"
  mark_on
  SP=$SP CFG=$CFG MINBUF=4000 GATE_MIN=60 bash $TL/run_selfplay_m1.sh --resume --workers $NW --pool $POOL >> $LOG 2>&1 &
  lp=$!
  while kill -0 $lp 2>/dev/null; do   # the launcher waits for the buffer; keep yielding meanwhile
    sleep 30
    r=$(foreign)
    if [ -n "$r" ] && [ -n "$(foreign)" ]; then
      for i in $(seq 1 20); do [ -f $SP/pids ] && break; sleep 3; done   # the launcher writes pids right after the workers
      kill $lp 2>/dev/null; sleep 2; say "yielding while starting: $r"; stop_run; return 3
    fi
  done
  if ! [ "$(sed -n 2p $SP/pids 2>/dev/null)" -gt 0 ] 2>/dev/null; then stop_run; return 1; fi
  for i in $(seq 1 $NW); do
    grep -q "train_play_w$i'" $SP/worker_$i.toml || { say "worker_$i.toml has no train_play dir of its own"; stop_run; return 1; }
  done
  : > $SP/helper_pids
  setsid env EVERY=8000 ALSO=$STOP_AT START_AFTER=$la bash $TL/xfer_helpers.sh archiver $SP xfer_ct < /dev/null > /dev/null 2>&1 & echo $! >> $SP/helper_pids
  setsid env MINBUF=4000 bash $TL/xfer_helpers.sh guard $SP $CFG < /dev/null > /dev/null 2>&1 & echo $! >> $SP/helper_pids
  setsid $PY $TL/xfer_collapse_guard.py $SP < /dev/null >> $SP/collapse_guard.out 2>&1 & echo $! >> $SP/helper_pids
  say "running: pids $(tr '\n' ' ' < $SP/pids), helpers $(tr '\n' ' ' < $SP/helper_pids)"
  return 0
}
echo $$ > $SP/ct_chain.pid
# ---------------- set-up ----------------
if [ ! -e $SP/state.pth ]; then
  [ "$(dfree)" -ge $DMIN ] || stopped "home below $DMIN GB free"
  [ "$(md5sum < $SRC | cut -c1-8)" = 7a679fb4 ] || stopped "C392k md5 mismatch"
  [ "$(md5sum < $ANC | cut -c1-8)" = b9c0ff1f ] || stopped "BASE.pth (rl1_best) md5 mismatch"
  cp $SRC $SP/snap_C392k.pth && cp $ANC $SP/BASE.pth && cp $TL/m1_prime_param.py $SP/ || stopped "copies FAILED"
  $PY $TL/ct_setup.py $SRC $SP $CFG $PORT $CPF >> $LOG 2>&1 || stopped "config set-up FAILED"
  cp $SRC $SP/state.pth.new && mv $SP/state.pth.new $SP/state.pth
  m() { md5sum "$1" | cut -c1-32; }
  MM=$B/Mortal/mortal
  {
    echo "arm ct (trx, July topology), prepared $(date '+%F %T')"
    echo "state.pth $(m $SP/state.pth)  BASE.pth $(m $SP/BASE.pth)  snap_C392k.pth $(m $SP/snap_C392k.pth)  C' final $(m $CPF)"
    echo "config $CFG $(m $CFG)"
    echo "train.py $(m $MM/train.py)  server.py $(m $MM/server.py)  client.py $(m $MM/client.py)  dataloader.py $(m $MM/dataloader.py)"
    echo "model.py $(m $MM/model.py)  player.py $(m $MM/player.py)  libriichi.so $(m $MM/libriichi.so)"
    echo "mortal_v4 $(m $B/baseline/mortal_v4.pth)  grp $(m $R/grp_full/grp.pth)  GPU $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
  } > $SP/manifest.txt
  say "set up: $(tr '\n' ' ' < $SP/manifest.txt)"
fi
$PY - $SP/state.pth >> $LOG 2>&1 <<'PY' || stopped "state check FAILED"
import sys, torch
s = torch.load(sys.argv[1], weights_only=True, map_location="cpu")
sc = s["scheduler"]; lr = [g["lr"] for g in s["optimizer"]["param_groups"]]
print(f"state.pth: steps {s['steps']}, online {s['config']['control']['online']}, scheduler peak/final {sc['peak']}/{sc['final']}, optimizer lr {lr}")
assert s["config"]["control"]["online"] and sc["peak"] == sc["final"] == 5e-6 and all(x == 5e-6 for x in lr)
PY
# ---------------- training, low priority ----------------
L0=0; first=1; seen=0; nyield=0; nfail=0
while [ ! -f $CK ]; do
  [ $(date +%s) -lt $DEADLINE ] || stopped "72 hours without reaching $STOP_AT"
  [ "$(dfree)" -ge $DMIN ] || stopped "home below $DMIN GB free"
  if [ ! -f $SP/pids ]; then
    grep -q ALERT $SP/collapse_guard.log 2>/dev/null && stopped "the collapse guard stopped the run"
    [ $first = 1 ] && say "waiting for a free GPU (3 minutes in a row)" || say "waiting for the GPU to be free for 10 minutes"
    wait_free $([ $first = 1 ] && echo 6 || echo 20) || stopped "72 hours without reaching $STOP_AT"
    start_run; rc=$?
    case $rc in
      0) first=0; seen=0;;
      3) nyield=$((nyield + 1)); first=0;;
      *) nfail=$((nfail + 1)); say "start FAILED ($nfail)"; [ $nfail -ge 3 ] && stopped "three failed starts"; sleep 300;;
    esac
    continue
  fi
  r=$(foreign)
  if [ -n "$r" ]; then seen=$((seen + 1)); else seen=0; fi
  if [ $seen -ge 2 ]; then
    nyield=$((nyield + 1)); say "yielding ($nyield) to $r, last logged step $(lam_view | cut -d' ' -f2)"
    stop_run; seen=0; continue
  fi
  read -r w s <<< "$(lam_view)"
  if [ "$(lam_now)" = 0.5 ] && [ "$w" = 0.5 ] && [ "$s" -ge $((SWITCH - 200)) ]; then
    set_lam 0.2; say "lambda 0.2 written at logged step $s: the trainer reads it at its respawn at $SWITCH"
  fi
  sleep 30
done
sleep 30
stop_run
say "trained to $STOP_AT after $nyield yields"
# which lambda each trainer process read, by the step it started at
tr -d '\r' < $SP/logs/trainer.log | awk -v sw=$SWITCH '
  /anchor enabled: weight=/ { match($0, /weight=[0-9.]+/); w = substr($0, RSTART + 7, RLENGTH - 7); want = 1; next }
  want && /total steps: / { match($0, /total steps: [0-9,]+/); t = substr($0, RSTART + 13, RLENGTH - 13); gsub(",", "", t)
    ok = (t + 0 < sw) ? (w == "0.5") : (w == "0.2"); n++; bad += !ok; if (!ok) list = list " " t ":" w; want = 0 }
  END { printf "lambda check: %d trainer starts, %d with the wrong weight%s\n", n, bad, list }' >> $LOG
$PY $TL/xfer_reuse.py $SP >> $LOG 2>&1
for k in 240000 278600; do echo "xfer_ct_s$k.pth $(md5sum < $SP/archive/xfer_ct_s$k.pth | cut -c1-32)"; done > $SP/CT_READY.new && mv $SP/CT_READY.new $SP/CT_READY
say "archives ready: $(tr '\n' ' ' < $SP/CT_READY)"
say "CT_CHAIN_DONE"
