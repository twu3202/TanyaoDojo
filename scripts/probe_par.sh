#!/bin/bash
# 选立直率探针(diag_disagree)的服务器并行版: 同一批 v4 轨迹(eval_rl1_best, 4000 局, seeds 10000-10999)按 seed 切 P 片,
# 各片独立 dump, 再把各桶计数相加 —— 与单进程逐局累加的结果相同(计数量只做加法)。
# 用法: probe_par.sh <name> <ckpt> <tag>     产出: probe/<tag>_<name>.json / .txt, done/<name>.probe
E=/home/twu/evalsta
export MORTAL_DIR=/home/twu/Projects/better_mortal/Mortal/mortal
PY=/home/twu/Projects/better_mortal/.venv/bin/python
V4=/home/twu/Projects/better_mortal/baseline/mortal_v4.pth
name=$1; ck=$2; tag=$3; P=${P:-4}   # 8 片曾在 trainer 重生时把显存挤爆(09-14), 降到 4 并加显存闸
mkdir -p $E/probe/shards $E/done
out=$E/probe/${tag}_${name}
t0=$(date +%s)
echo "$(date +%m-%d_%H:%M:%S) PROBE START $tag $name P=$P" >> $E/evalwatch.log
export GPU_MEM_FRAC=${GPU_MEM_FRAC:-0.12} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 每片封顶约 5.7GB: 09-16 不封顶时单片涨到 6-17GB 把 trainer 挤崩
bash $E/gpugate.sh ${NEED:-12000} 3600 || echo "$(date +%m-%d_%H:%M:%S) WARN 显存闸等待超时, 仍继续 $tag $name" >> $E/evalwatch.log
cd /home/twu/Better_mortal/jax_rl/mjai_bot
per=$(( (1000 + P - 1) / P ))
for ((i=0; i<P; i++)); do echo $((10000 + i*per)); done | \
  xargs -P $P -I{} bash -c "$PY diag_disagree.py --log-dir $E/eval_rl1_best --seat-name mortal-v4 --ref $V4 --sub $ck \
      --seed-lo {} --seed-hi \$(( {} + $per - 1 )) --device cuda:0 --dump $E/probe/shards/${tag}_${name}_{}.json > $E/probe/shards/${tag}_${name}_{}.log 2>&1"
$PY - "$out" $E/probe/shards/${tag}_${name}_*.json <<'PY' > $out.txt
import json, sys
out, parts = sys.argv[1], sys.argv[2:]
tot = {"n_games": 0, "bad": 0, "buckets": {}}
for p in parts:
    j = json.load(open(p))
    tot["n_games"] += j["n_games"]; tot["bad"] += j["bad"]
    for b, s in j["buckets"].items():
        t = tot["buckets"].setdefault(b, {k: 0 for k in s})
        for k, v in s.items():
            t[k] += v
json.dump(tot, open(out + ".json", "w"), ensure_ascii=False, indent=1)
B = tot["buckets"]
print(f"  牌谱 {tot['n_games']:,} 局(重放失败 {tot['bad']}, 分片 {len(parts)})")
r = B["riichi"]; c = B["call"]
print(f"  立直桶:ref 选立直 {r['ref_riichi']/max(r['n'],1):.2%},sub 选立直 {r['sub_riichi']/max(r['n'],1):.2%}")
print(f"  鸣牌桶:ref 选过 {c['ref_pass']/max(c['n'],1):.2%},sub 选过 {c['sub_pass']/max(c['n'],1):.2%}")
PY
ng=$(grep -oE "牌谱 [0-9,]+ 局" $out.txt | grep -oE "[0-9,]+" | tr -d ,)
echo "$(date +%m-%d_%H:%M:%S) PROBE END $tag $name $(( $(date +%s) - t0 ))s games=$ng $(grep 立直桶 $out.txt)" >> $E/evalwatch.log
[ "$ng" = 4000 ] && touch $E/done/$name.probe
