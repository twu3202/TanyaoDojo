#!/bin/bash
# v2: 解压直达目标目录,不逐文件改名(配置 glob 用 *.mjson*)
set -u
source /etc/network_turbo > /dev/null 2>&1
year=$1
BASE=https://github.com/NikkeTryHard/tenhou-to-mjai/releases/download/v2.0.0
DATA=/root/autodl-tmp/data
LOG=$DATA/download.log
dest=$DATA/houou/$year
tmp=$DATA/tmp_$year
mkdir -p $dest $tmp
# 完整判据:目标有文件 且 没有残留 zip(zip 只有解压成功后才会被删)
if [ "$(ls $dest 2>/dev/null | wc -l)" -gt 100 ] && [ ! -f $tmp/$year.zip ]; then
  rm -rf $tmp; echo "[$year] present" >> $LOG; exit 0
fi
ok=0
for attempt in 1 2 3 4 5; do
  curl -sSL -C - --retry 5 --retry-all-errors --retry-delay 10 -o $tmp/$year.zip $BASE/$year.zip 2>> $LOG
  if unzip -tq $tmp/$year.zip > /dev/null 2>&1; then ok=1; break; fi
  echo "[$year] attempt $attempt bad zip, redownload" >> $LOG
  rm -f $tmp/$year.zip
  sleep 5
done
[ $ok -eq 1 ] || { echo "[$year] FAILED" >> $LOG; exit 1; }
unzip -oq $tmp/$year.zip -d $dest || { echo "[$year] extract FAILED" >> $LOG; exit 1; }
rm -rf $tmp
echo "[$year] done: $(ls $dest | wc -l) games" >> $LOG
