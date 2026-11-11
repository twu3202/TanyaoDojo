#!/bin/bash
# README addendum (11-10 20:40): if cq_chain.sh ends with "gate not met", relaunch it with the large-gap gate.
# cq_chain.sh resumes: with v11cq_final.pth present it skips the offline step and recomputes the gap readout.
T=$HOME/Projects/better_mortal/runs/xfer_tools; LOG=$T/cq_chain.log
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
while true; do
  grep -q "CQ_CHAIN_DONE (gate not met)" $LOG && break
  grep -qE "gate met \\(ratio|large-gap gate met|CQ_CHAIN_STOPPED|CQ_CHAIN_SKIPPED" $LOG && exit 0
  sleep 60
done
while kill -0 $(cat $T/cq_chain.pid) 2>/dev/null; do sleep 10; done
grep -q "large-gap gate" $LOG && exit 0
say "addendum: relaunching cq_chain.sh with the large-gap gate (CQ_GATE=large)"
CQ_GATE=large setsid nohup bash $T/cq_chain.sh > /dev/null 2>&1 < /dev/null &
