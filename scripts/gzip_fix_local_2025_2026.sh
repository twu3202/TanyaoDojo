#!/bin/zsh
# 本地 2025/2026:文件名是 .mjson.gz 但内容是明文 JSON —— 去掉假后缀后真 gzip
set -u
LOG=/Users/r/HMM/Better_mortal/data/gzip_fix_local.log
echo "start $(date)" >> $LOG
for y in 2025 2026; do
  d=/Users/r/HMM/Better_mortal/data/houou/$y
  # 先批量去假 .gz 后缀
  find $d -name "*.mjson.gz" -print0 | while IFS= read -r -d '' f; do
    if [ "$(head -c 2 "$f" | xxd -p)" != "1f8b" ]; then mv "$f" "${f%.gz}"; fi
  done
  # 再并行真压缩
  find $d -name "*.mjson" -print0 | xargs -0 -P 8 -n 200 gzip
  echo "$y done $(date): plain remaining $(find $d -name '*.mjson' | wc -l | tr -d ' ')" >> $LOG
done
echo "all done $(date)" >> $LOG
