#!/bin/bash
# 10-12 user: try the data-quality arm q25 (patches/data_quality/README.md v7) on the 5060 Ti. Start it once the
# v11long evaluation chain has finished, so the GPU is never shared; then start its own eval chain.
# Log: runs/q25/handover.log. If the 5060 Ti reboots before the handover, rerun this script.
# 10-13: q25 may run on trx instead (q25_trx_wait.sh). While runs/q25/HOLD_TRX exists this script does not start q25;
# its first line is a deadline (epoch seconds, 0 = none) after which the hold lapses, so the 5060 Ti does not sit idle
# if trx never started. Migrating from trx = copy its mortal.pth into runs/q25 and delete HOLD_TRX (symaug_run.sh
# resumes from mortal.pth).
B=$HOME/Projects/better_mortal; R=$B/runs; LOG=$R/q25/handover.log
say() { echo "$(date +%m-%d_%H:%M:%S) $*" >> $LOG; }
echo $$ > $R/q25/handover.pid
say "waiting for the v11long chain to finish"
while true; do
  grep -q -E " V11LONG_CHAIN_(DONE|STOPPED)$" $R/v11long/chain.log 2>/dev/null && break
  if ! kill -0 $(cat $R/v11long/chain.pid) 2>/dev/null; then say "v11long chain pid gone without a marker; proceeding"; break; fi
  sleep 120
done
held() {
  [ -f $R/q25/HOLD_TRX ] || return 1
  local dl=$(head -n 1 $R/q25/HOLD_TRX | tr -dc 0-9); dl=${dl:-0}
  [ "$dl" -eq 0 ] || [ "$(date +%s)" -lt "$dl" ]
}
if held; then
  say "v11long chain finished; held by HOLD_TRX ($(head -n 1 $R/q25/HOLD_TRX)), waiting"
  while held; do sleep 120; done
  say "hold released ($([ -f $R/q25/HOLD_TRX ] && echo deadline passed || echo HOLD_TRX removed))"
fi
sleep 60
say "v11long chain finished: $(tail -n 2 $R/v11long/chain.log | tr '\n' ' ' | cut -c1-600)"
(NAME=q25 TREE=$B/Mortal_symaug/mortal STOP=1220000 setsid nohup bash $R/q25/symaug_run.sh > /dev/null 2>&1 &)
sleep 5
(setsid nohup bash $R/q25/q25_chain.sh > /dev/null 2>&1 &)
say "q25 training ($([ -f $R/q25/mortal.pth ] && echo resuming from mortal.pth || echo from scratch), stop 1,220,000) and its eval chain started"
