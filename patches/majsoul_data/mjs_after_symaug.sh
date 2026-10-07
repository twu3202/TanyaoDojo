#!/bin/bash
# 10-07 user: run the new-data arm on the 5060 Ti. Start it once the symaug evaluation chain has finished, so the
# GPU is never shared; then start its own eval chain (waits for SYMAUG_TRAIN_DONE in runs/mjs/run.log). Log: runs/mjs/handover.log
B=$HOME/Projects/better_mortal; R=$B/runs; LOG=$R/mjs/handover.log
say() { echo "$(date +%m-%d_%H:%M:%S) $*" >> $LOG; }
echo $$ > $R/mjs/handover.pid
say "waiting for the symaug chain to finish"
while true; do
  grep -q -E " SYMAUG_CHAIN_(DONE|STOPPED)$" $R/symaug/chain.log 2>/dev/null && break
  if ! kill -0 $(cat $R/symaug/chain.pid) 2>/dev/null; then say "symaug chain pid gone without a marker; proceeding"; break; fi
  sleep 120
done
sleep 60
say "symaug chain finished: $(tail -n 2 $R/symaug/chain.log | tr '\n' ' ')"
(NAME=mjs TREE=$B/Mortal_mjs/mortal STOP=2255200 setsid nohup bash $R/mjs/symaug_run.sh > /dev/null 2>&1 &)
sleep 5
(setsid nohup bash $R/mjs/mjs_chain.sh > /dev/null 2>&1 &)
say "mjs training (from scratch, stop 2,255,200) and its eval chain started"
