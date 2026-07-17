#!/bin/bash
# 本地服务器下载 2019-2026 凤凰卓数据(GitHub 直连,并行),统一规整为 *.mjson.gz
set -u
BASE=https://github.com/NikkeTryHard/tenhou-to-mjai/releases/download/v2.0.0
DATA=$HOME/Projects/better_mortal/data
LOG=$DATA/download.log
mkdir -p $DATA
YEARS="2019 2020 2021 2022 2023 2024 2025 2026"

dl_year() {
  year=$1
  BASE=https://github.com/NikkeTryHard/tenhou-to-mjai/releases/download/v2.0.0
  DATA=$HOME/Projects/better_mortal/data
  LOG=$DATA/download.log
  dest=$DATA/houou/$year
  # 已完整则跳过(有文件且无残留 zip)
  if [ "$(ls $dest 2>/dev/null | wc -l)" -gt 100 ] && [ ! -f $DATA/tmp_$year/$year.zip ]; then
    echo "[$year] present" >> $LOG; return
  fi
  tmp=$DATA/tmp_$year; mkdir -p $dest $tmp
  ok=0
  for a in 1 2 3 4 5; do
    curl -sSL -C - --retry 5 --retry-all-errors --retry-delay 10 -o $tmp/$year.zip $BASE/$year.zip 2>> $LOG
    unzip -tq $tmp/$year.zip > /dev/null 2>&1 && { ok=1; break; }
    echo "[$year] attempt $a bad zip" >> $LOG; rm -f $tmp/$year.zip; sleep 5
  done
  [ $ok -eq 1 ] || { echo "[$year] FAILED" >> $LOG; return; }
  unzip -oq $tmp/$year.zip -d $tmp || { echo "[$year] extract FAILED" >> $LOG; return; }
  rm -f $tmp/$year.zip
  # 统一规整:gzip 内容→改名 .mjson.gz;明文→真 gzip
  n=0
  for f in $tmp/*.mjson; do
    [ -e "$f" ] || continue
    if [ "$(head -c2 "$f" | xxd -p)" = "1f8b" ]; then
      mv "$f" "$dest/$(basename "$f").gz"
    else
      gzip -c "$f" > "$dest/$(basename "$f").gz" && rm -f "$f"
    fi
    n=$((n+1))
  done
  rm -rf $tmp
  echo "[$year] done: $n games" >> $LOG
}
export -f dl_year

echo "==== dl start $(date) ====" >> $LOG
printf "%s\n" $YEARS | xargs -P 4 -I{} bash -c 'dl_year "$@"' _ {}
echo "==== dl done $(date), total $(find $DATA/houou -name '*.mjson.gz' | wc -l) games ====" >> $LOG
