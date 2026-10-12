#!/bin/bash
# 10-12 user: try the data-quality arm q25 (patches/data_quality/README.md v7) on the 5060 Ti. Start it once the
# v11long evaluation chain has finished, so the GPU is never shared; then start its own eval chain.
# Log: runs/q25/handover.log. If the 5060 Ti reboots before the handover, rerun this script.
B=$HOME/Projects/better_mortal; R=$B/runs; LOG=$R/q25/handover.log
say() { echo "$(date +%m-%d_%H:%M:%S) $*" >> $LOG; }
echo $$ > $R/q25/handover.pid
say "waiting for the v11long chain to finish"
while true; do
  grep -q -E " V11LONG_CHAIN_(DONE|STOPPED)$" $R/v11long/chain.log 2>/dev/null && break
  if ! kill -0 $(cat $R/v11long/chain.pid) 2>/dev/null; then say "v11long chain pid gone without a marker; proceeding"; break; fi
  sleep 120
done
sleep 60
say "v11long chain finished: $(tail -n 2 $R/v11long/chain.log | tr '\n' ' ' | cut -c1-600)"
(NAME=q25 TREE=$B/Mortal_symaug/mortal STOP=1220000 setsid nohup bash $R/q25/symaug_run.sh > /dev/null 2>&1 &)
sleep 5
(setsid nohup bash $R/q25/q25_chain.sh > /dev/null 2>&1 &)
say "q25 training (from scratch, stop 1,220,000) and its eval chain started"
