#!/bin/bash
LOG=/root/autodl-tmp/data/gzip_fix.log
echo "start $(date)" >> $LOG
for y in 2025 2026; do
  find /root/autodl-tmp/data/houou/$y -name "*.mjson" -print0 | xargs -0 -P 2 -n 200 gzip
  echo "$y done $(date): $(ls /root/autodl-tmp/data/houou/$y | wc -l) files, all gz: $(find /root/autodl-tmp/data/houou/$y -name "*.mjson" | wc -l) remaining plain" >> $LOG
done
echo "all done $(date)" >> $LOG
