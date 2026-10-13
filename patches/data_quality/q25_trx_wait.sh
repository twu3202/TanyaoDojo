#!/bin/bash
# 10-13 01:20 user: the trx GPU frees up in about two hours, and this session may use it for a while -> run the q25
# arm (README v7) there instead of queueing it behind v11long on the 5060 Ti (trx is about 1.45x faster).
# 01:30 the Qwen38quant session (whole GPU, Qwen3.6 evaluation until about 04:30, at the latest 05:30) asked us to wait
# for its lock files /mnt/bigdata/q36_v2/GPU_BUSY_q36_* to go away.
# Waits until NOT_BEFORE, until no lock file LOCKS exists, and until the GPU has had no compute process and < 2 GB in use
# for FREE_MIN consecutive minutes, then starts q25 training (symaug_run.sh, from scratch, stop 1,220,000) and its trx
# eval chain. Starts nothing if runs/q25/mortal.pth already exists (a run is already there; resume that by hand).
# The 5060 Ti handover (q25_after_v11long.sh) is held by runs/q25/HOLD_TRX on WSL so the two machines never both
# train q25; migrating = copy mortal.pth to WSL runs/q25 and delete HOLD_TRX there.
# Log: runs/q25/handover.log (GPU status every 10 minutes, Q25_TRX_STARTED once launched)
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/q25/q25_trx_wait.sh > /dev/null 2>&1 &)
B=/home/twu/Projects/better_mortal; R=$B/runs/q25; LOG=$R/handover.log
NOT_BEFORE=${NOT_BEFORE:-2026-10-13 03:00}; FREE_MIN=${FREE_MIN:-15}; LOCKS=${LOCKS:-/mnt/bigdata/q36_v2/GPU_BUSY_q36_*}
say() { echo "$(date +%m-%d_%H:%M:%S) $*" >> $LOG; }
echo $$ > $R/handover.pid
nb=$(date -d "$NOT_BEFORE" +%s)
say "waiting for the GPU (not before $NOT_BEFORE; free = no lock $LOCKS, no process and < 2000 MiB for $FREE_MIN minutes)"
free=0; i=0
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9)
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  locks=$(ls $LOCKS 2>/dev/null | grep -c .)
  if [ -n "$used" ] && [ "$used" -lt 2000 ] && [ "$apps" -eq 0 ] && [ "$locks" -eq 0 ]; then free=$((free + 1)); else free=0; fi
  [ $((i % 10)) -eq 0 ] && say "gpu ${used:-?} MiB, $apps processes, $locks lock files, free for $free min"
  i=$((i + 1))
  [ "$(date +%s)" -ge "$nb" ] && [ $free -ge $FREE_MIN ] && break
  sleep 60
done
if [ -f $R/mortal.pth ]; then say "runs/q25/mortal.pth already exists; not starting (resume by hand)"; exit 1; fi
say "GPU free for $free min (${used} MiB); starting q25"
(NAME=q25 TREE=$B/Mortal_symaug/mortal STOP=1220000 setsid nohup bash $R/symaug_run.sh > /dev/null 2>&1 &)
sleep 5
(setsid nohup bash $R/q25_chain_trx.sh > /dev/null 2>&1 &)
say "Q25_TRX_STARTED (training from scratch, stop 1,220,000; eval chain q25_chain_trx.sh)"
