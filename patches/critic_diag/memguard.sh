#!/bin/bash
# memguard.sh PATTERN LOG: kill processes matching PATTERN if WSL's available memory falls under 2 GB
PAT=$1; LOG=$2
sleep 15
while pgrep -f "^/home/r/Projects/better_mortal/.venv/bin/python $PAT" > /dev/null; do
  a=$(free -m | awk 'NR==2{print $7}')
  echo "$(date +%T) avail $a" >> $LOG
  if [ "$a" -lt 2000 ]; then pkill -f "^/home/r/Projects/better_mortal/.venv/bin/python $PAT"; echo KILLED_LOW_MEM >> $LOG; fi
  sleep 5
done
echo "guard ends $(date +%T)" >> $LOG
