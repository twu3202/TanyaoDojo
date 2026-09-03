"""
复核我们自己的评测口径:複式 1v3 的置信区间该按"局"还是按"牌局组"算?

run_eval.py 现在按局算:ci = 1.96 * std(每局顺位点) / sqrt(n),n = 4 × seed 数。
但複式的设计就是同一副牌换四个座位各打一遍——组内四局**不独立**,把它们当
4 个独立样本会把有效样本量当成 4 倍。方向取决于组内相关 ICC:
  ICC < 0(座位优势被抵消,複式的本意) → 真 SE 更小,现行 CI **偏保守**
  ICC > 0(牌型本身对挑战者有利/不利)   → 真 SE 更大,现行 CI **偏窄,显著性被高估**

本脚本从 leanjax_eval/ 的 mjai 日志直接重算两种 CI 并给出 ICC。
自检:按局重算的 avg_pt 必须复现 run_eval 打印的值,否则座位映射错了。

用法: python group_ci.py [--log-dir ...] [--seed-lo 14000] [--seed-hi 15999]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
from collections import defaultdict

import numpy as np

PTS = np.array([90.0, 45.0, 0.0, -135.0])
SUFFIX_SEAT = {"a": 0, "b": 1, "c": 2, "d": 3}   # 旋转后缀 → 挑战者座位
NAME_RE = re.compile(r"(\d+)_(\d+)_([a-d])\.json\.gz$")


def final_scores(path):
    """终局点数。⚠️ 不能只累加 hora/ryukyoku 的 deltas —— mjai 的 reach_accepted
    不带 deltas,立直的 1000 点押金是隐含的;漏掉它会把终局分算高、顺位算错
    (实测整段 avg_pt 偏差 0.47pt)。故以每局 start_kyoku 的权威 scores 为准,
    只对最后一局手工结算:押金 + deltas + 流局时残留供托归第一名(天凤规则)。"""
    sc = None
    kyotaku = 0
    reaches = [0, 0, 0, 0]
    deltas = [0, 0, 0, 0]
    hora = False
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ev = json.loads(line)
            t = ev.get("type")
            if t == "start_kyoku":
                sc = list(ev["scores"])
                kyotaku = int(ev.get("kyotaku", 0))
                reaches = [0, 0, 0, 0]
                deltas = [0, 0, 0, 0]
                hora = False
            elif t == "reach_accepted":
                reaches[int(ev["actor"])] += 1
            elif t in ("hora", "ryukyoku"):
                if t == "hora":
                    hora = True
                for i, v in enumerate(ev.get("deltas", [0] * 4)):
                    deltas[i] += int(v)
    if sc is None:
        return [25000] * 4
    fin = [sc[i] - 1000 * reaches[i] + deltas[i] for i in range(4)]
    if not hora:                       # 流局:台上残留立直棒归终局第一名
        left = kyotaku + sum(reaches)
        if left:
            top = max(range(4), key=lambda i: (fin[i], -i))
            fin[top] += 1000 * left
    return fin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=os.path.expanduser(
        "~/Projects/better_mortal/runs/leanjax_eval"))
    ap.add_argument("--seed-lo", type=int, default=14000)
    ap.add_argument("--seed-hi", type=int, default=15999)
    ap.add_argument("--newer-than", type=float, default=0.0,
                    help="只取 mtime 晚于该 epoch 秒的文件(隔离本次 run 的日志)")
    ap.add_argument("--dump", default=None,
                    help="把逐牌局组的顺位点存成 npz(seed, pt4, mean),供配对比较。"
                         "⚠️ 日志目录按 seed 命名会被后续 run 覆盖,要配对就必须先 dump")
    args = ap.parse_args()

    groups = defaultdict(dict)
    for p in glob.glob(os.path.join(args.log_dir, "*.json.gz")):
        m = NAME_RE.search(os.path.basename(p))
        if not m:
            continue
        seed = int(m.group(1))
        if not (args.seed_lo <= seed <= args.seed_hi):
            continue
        if os.path.getmtime(p) < args.newer_than:
            continue
        groups[seed][m.group(3)] = p

    full = {s: d for s, d in groups.items() if len(d) == 4}
    print(f"完整牌局组 = {len(full):,}  (局数 {4*len(full):,})")
    if not full:
        return

    per_game, per_group, seeds = [], [], []
    rows = []
    for seed in sorted(full):
        pts = []
        for suf, path in sorted(full[seed].items()):
            sc = final_scores(path)
            seat = SUFFIX_SEAT[suf]
            order = sorted(range(4), key=lambda i: (-sc[i], i))
            rank = order.index(seat)
            pts.append(PTS[rank])
        per_game.extend(pts)
        per_group.append(np.mean(pts))
        seeds.append(seed)
        rows.append(pts)
    if args.dump:
        np.savez_compressed(args.dump, seed=np.array(seeds),
                            pt4=np.array(rows), mean=np.array(per_group))
        print(f"dump -> {args.dump}  ({len(seeds):,} 组)")

    g = np.array(per_game)
    gr = np.array(per_group)
    n, k = len(g), len(gr)
    ci_game = 1.96 * g.std(ddof=1) / np.sqrt(n)
    ci_group = 1.96 * gr.std(ddof=1) / np.sqrt(k)
    # ICC:组均值方差 = (σ²/4)(1 + 3·ICC)  →  ICC = (4·Var(组均值)/σ² − 1)/3
    icc = (4 * gr.var(ddof=1) / g.var(ddof=1) - 1) / 3

    print(f"avg_pt(按局重算) = {g.mean():+.3f}   ← 须与 run_eval 打印值一致")
    print(f"  现行口径(按局, n={n:,})   : ±{ci_game:.3f}")
    print(f"  正确口径(按组, k={k:,})   : ±{ci_group:.3f}")
    print(f"  比值 = {ci_group/ci_game:.3f}   组内相关 ICC = {icc:+.4f}")
    if ci_group > ci_game * 1.02:
        print("  → 现行 CI 偏窄,过去所有显著性判读都被高估")
    elif ci_group < ci_game * 0.98:
        print("  → 现行 CI 偏保守(複式确实抵消了座位运气),真实分辨率优于标称")
    else:
        print("  → 两者等价,现行口径无害")


if __name__ == "__main__":
    main()
