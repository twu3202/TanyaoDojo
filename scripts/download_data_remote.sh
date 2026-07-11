#!/bin/bash
# AutoDL 云端全量下载(走学术加速)
set -u
source /etc/network_turbo
BASE=https://github.com/NikkeTryHard/tenhou-to-mjai/releases/download/v2.0.0
DATA=/root/autodl-tmp/data
LOG=$DATA/download.log
mkdir -p $DATA
echo "==== start $(date) ====" >> $LOG
for year in 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026; do
  dest=$DATA/houou/$year
  if [ -d "$dest" ] && [ "$(ls $dest 2>/dev/null | wc -l)" -gt 100 ]; then
    echo "[$year] present, skip" >> $LOG; continue
  fi
  tmp=$DATA/tmp_$year
  rm -rf $tmp && mkdir -p $tmp $dest
  for attempt in 1 2 3 4 5; do
    curl -sSL -C - --retry 5 --retry-all-errors --retry-delay 10 -o $tmp/$year.zip $BASE/$year.zip 2>> $LOG
    if unzip -tq $tmp/$year.zip > /dev/null 2>&1; then break; fi
    echo "[$year] attempt $attempt incomplete, retrying" >> $LOG
    sleep 5
  done
  if ! unzip -q $tmp/$year.zip -d $tmp; then
    echo "[$year] FAILED" >> $LOG; rm -rf $tmp; continue
  fi
  rm $tmp/$year.zip
  n=0
  for f in $tmp/*.mjson; do
    mv "$f" "$dest/$(basename $f).gz" && n=$((n+1))
  done
  rm -rf $tmp
  echo "[$year] done: $n games" >> $LOG
done
echo "==== all done $(date) ====" >> $LOG
echo "total: $(find $DATA/houou -name "*.mjson.gz" | wc -l) games, $(du -sh $DATA/houou | cut -f1)" >> $LOG
