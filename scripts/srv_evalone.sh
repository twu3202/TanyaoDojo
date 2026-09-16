#!/bin/bash
# 服务器评测站(2026-09-14 v2): 盯 archiver.log 里 ${PREFIX}_sN.pth(N >= FROM)。
#   每档: 选立直率探针(probe_par.sh, 8 片并行 ~3 分钟, 主要吃 CPU)。
#   full12k.txt 里列出的档: 再跑 s14000 8k(K 进程并行)+ diag_gap / diag_place / diag_bust 转储。
#   s10000 4k 筛选改在本机 5060Ti 跑(v1 在服务器上跑时与 test_play / 自对弈生成撞 GPU, 4k 从 6.6 分钟拖到 18 分钟)。
# 服务器 RTX 6000 Ada 与本机 5060Ti 的 fp32 在 100 副牌里 98 副逐局相同(r=0.988), 两边的 npz 可以配对。
# 产出: dump/<tag>_<name>_s<seg>_gpufp32.npz, gapdump|placedump|bustdump/<tag>_<name>_s<seg>.json, done/<name>.{probe,s14000}
E=/home/twu/evalsta
S=/home/twu/Projects/better_mortal/runs/selfplay
export MORTAL_DIR=/home/twu/Projects/better_mortal/Mortal/mortal
PY=/home/twu/Projects/better_mortal/.venv/bin/python
V4=/home/twu/Projects/better_mortal/baseline/mortal_v4.pth
FROM=${FROM:-0}; PREFIX=${PREFIX:-state}
mkdir -p $E/dump $E/gapdump $E/placedump $E/bustdump $E/done $E/logs
LOG=$E/evalwatch.log
say() { echo "$(date +%m-%d_%H:%M:%S) $*" >> $LOG; }

run_seg() {  # name ck tag seg_start iters
  local name=$1 ck=$2 tag=$3 s=$4 it=$5
  local L=$E/logs/${tag}_${name}_s${s}; rm -rf $L; mkdir -p $L
  local K=$(cat $E/K 2>/dev/null || echo 2)   # 2026-09-16: 3 进程与 trainer 抢显存&CPU, 降到 2
  local t0=$(date +%s)
  say "START $tag $name s$s iters=$it K=$K"
  export GPU_MEM_FRAC=${GPU_MEM_FRAC:-0.15} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 每进程封顶约 7GB
  bash $E/gpugate.sh ${NEED:-12000} 3600 || say "WARN 显存闸等待超时, 仍继续 $tag $name s$s"
  cd /home/twu/Better_mortal/jax_rl/mjai_bot
  for ((c=0; c<it; c+=2)); do echo $((s + c*100)); done | \
    xargs -P $K -I{} $PY run_eval.py $ck --challenger-type mortal --challenger-name "${tag}_${name}" --champion $V4 \
        --device cuda:0 --amp off --games 400 --iters 2 --seed-start {} --log-dir $L > $L.out 2>&1
  local hi=$((s + it*100 - 1)) ng=$((it*400))
  $PY group_ci.py --log-dir $L --seed-lo $s --seed-hi $hi --dump $E/dump/${tag}_${name}_s${s}_gpufp32.npz < /dev/null > $L.ci 2>&1
  $PY diag_gap.py --log-dir $L --seed-lo $s --seed-hi $hi --expect-games $ng --dump $E/gapdump/${tag}_${name}_s${s}.json < /dev/null > $L.gap 2>&1
  $PY diag_place.py --log-dir $L --seed-lo $s --seed-hi $hi --expect-games $ng --dump $E/placedump/${tag}_${name}_s${s}.json < /dev/null > $L.place 2>&1
  $PY diag_bust.py --log-dir $L --seed-lo $s --seed-hi $hi --expect-games $ng --dump $E/bustdump/${tag}_${name}_s${s}.json < /dev/null > $L.bust 2>&1
  local ok=$(grep -a -c "正确口径" $L.ci)
  say "END $tag $name s$s $(( $(date +%s) - t0 ))s ci_ok=$ok $(grep -a '按局重算' $L.ci)"
  [ "$ok" -ge 1 ] && touch $E/done/${name}.s${s}
}

run_seg "$1" "$2" "$3" "$4" "$5"
