#!/bin/bash
# 显存闸: 等到空闲显存 >= NEED MiB 再放行, 最多等 WAIT 秒。评测站用它给 trainer 让路
# (trainer 重生时要一次性拿约 4GB, 拿不到就整个进程崩溃 —— 09-14 就是这样丢了两天训练)。
need=${1:-12000}; wait_s=${2:-1800}
for i in $(seq 1 $((wait_s/10))); do
  read used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr ',' ' ')
  [ $((total - used)) -ge "$need" ] && exit 0
  sleep 10
done
exit 1
