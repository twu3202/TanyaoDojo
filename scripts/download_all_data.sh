#!/bin/zsh
# 全量下载 tenhou-to-mjai v2.0.0(v2:断点续传版,应对间歇性 TLS 掉线)
# zip 半成品保留在 data/zips_pending/,curl -C - 断点续传,外层最多 30 轮
set -u
BASE=https://github.com/NikkeTryHard/tenhou-to-mjai/releases/download/v2.0.0
DATA=/Users/r/HMM/Better_mortal/data
PEND=$DATA/zips_pending
LOG=$DATA/download.log
mkdir -p "$PEND"

YEARS=(2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026)

year_done() {
  [ -d "$DATA/houou/$1" ] && [ "$(ls "$DATA/houou/$1" 2>/dev/null | wc -l)" -gt 100 ]
}

echo "==== v2 start $(date) ====" >> "$LOG"
for round in $(seq 1 30); do
  remaining=0
  for year in $YEARS; do
    year_done "$year" && continue
    remaining=$((remaining+1))
    zipf=$PEND/$year.zip
    echo "[$year] round $round: resume download (have $( [ -f "$zipf" ] && du -h "$zipf" | cut -f1 || echo 0 ))" >> "$LOG"
    curl -sSL --http1.1 -C - --retry 8 --retry-all-errors --retry-delay 15 \
         --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
         -o "$zipf" "$BASE/$year.zip" 2>> "$LOG"
    # 完整性:能通过 unzip 测试才算下完
    if [ -f "$zipf" ] && unzip -tq "$zipf" > /dev/null 2>&1; then
      tmp=$DATA/tmp_$year
      rm -rf "$tmp" && mkdir -p "$tmp" "$DATA/houou/$year"
      if unzip -q "$zipf" -d "$tmp"; then
        n=0
        for f in "$tmp"/*.mjson; do
          mv "$f" "$DATA/houou/$year/$(basename "$f").gz" && n=$((n+1))
        done
        rm -rf "$tmp" "$zipf"
        echo "[$year] DONE: $n games" >> "$LOG"
      else
        echo "[$year] unzip extract failed, will retry" >> "$LOG"
        rm -rf "$tmp"
      fi
    fi
  done
  [ "$remaining" -eq 0 ] && break
  /bin/sleep 20
done
echo "==== v2 end $(date), still missing: $(for y in $YEARS; do year_done $y || echo -n "$y "; done) ====" >> "$LOG"
echo "total games: $(find $DATA/houou -name '*.mjson.gz' | wc -l)" >> "$LOG"
