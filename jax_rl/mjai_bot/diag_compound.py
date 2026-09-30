"""押过之后的局面:Step 0.3 的「之后」超额是不是我们前面押出来的 —— Step 0.4(2026-09-22)。

**动机**。Step 0.3(diag_pushfold.py --side both)把我方座位多出的点炮牌拆开:tpt 408k 的 +0.042 张/局里,
单步决策 34%、入口局面 7%(不显著)、**同一暴露局后续决策的局面 41%**、暴露局更多 19%。
「后续局面更危险」最自然的解释是累积:前面在 v4 会弃的地方押了,手里留着危险牌、现物被别的牌挤掉,后面更难弃。
但也可能只是轨迹本身不同(挑的非现物不同、做牌不同)。这一步直接检验。

**做法**。同 diag_pushfold 的重放与双模型 argmax。对每个座位的每个暴露局,按时间顺序走它的暴露决策;
在有现物可打的决策上记两类分歧:
    P = 被测会押、v4 会弃;   F = 被测会弃、v4 会押。
每个**后续**决策(非入口)按「此前本局出现过的分歧」分组:P 组 = 出现过 P;F 组 = 只出现过 F;N 组 = 都没有。
**两条轨迹用同一个分组标准** —— 在我方轨迹上 P 意味着我们真的押了;在 v4 轨迹上 P 意味着同样「被测想押」的局面里
v4 实际弃了。于是同组内比较 v4 的点炮牌率(我方轨迹 vs v4 轨迹),局面诱惑程度大体配平,差别就是「押了 vs 没押」的后果。

**读数**(按局 bootstrap 95% 区间):
    组内项_g = 我方该组后续决策数/局 × (v4 在我方轨迹该组的点炮牌率 − v4 在自己轨迹该组的)
    配比项   = 两条轨迹上三组占比不同带来的部分
    三组组内项 + 配比项 = Step 0.3 的「局面项之之后」(逐位相等,自校验)。
另报我方轨迹各组的单步项(同一局面被测 vs v4 的点炮牌率差)—— 看押过之后是不是还接着押(承诺效应)。
**判读**:累积假说成立 ⇔ P 组组内项承担大头、N 组组内项 ≈ 0。

**Step 0.5 追加(2026-09-22)**:0.4 的结果是 N 组(此前押退分类全一致)承担了「之后」项的 80%,累积假说不成立。
于是把每组的组内项再拆成三块 ——
    有现物时的率差 × 有现物占比 + 没现物时的率差 × 没现物占比 + 没现物占比之差 × (v4 没现物 − 有现物时的率)
并报后续决策时的局面构成:没现物占比、手里现物张数、手里危险牌张数(命中立直者真实待牌的)、
立直者待牌种数、听牌、副露。用来区分「现物更早用光」「手里危险牌更多」「对手待牌更宽」。

用法:
  diag_compound.py --log-dir DIR [--log-dir ...] --sub ckpt.pth [--ref mortal_v4.pth] [--max-games N] [--dump x.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_pushfold import replay, cls, NAME_RE, load_model, qvals, cap_gpu_mem  # noqa: E402
import torch  # noqa: E402

GROUPS = ("N", "P", "F")
GLABEL = {"N": "N 此前无分歧", "P": "P 此前有「被测押/v4弃」", "F": "F 此前只有「被测弃/v4押」"}
K = 10          # 每组列: n, v4 点炮, 被测点炮, 没现物决策数, 其中 v4 点炮, 现物张数和, 危险牌张数和, 待牌种数和, 听牌, 副露
W = 3 * K       # 每角色 3 组


def stats(S, G):
    s, r = S[:W].reshape(3, K), S[W:].reshape(3, K)       # 行 N/P/F;列见 K
    Gs, Gr = G, 3 * G
    lr = r[:, 1].sum() / r[:, 0].sum()
    x = {"later_total": s[:, 1].sum() / Gs - s[:, 0].sum() / Gs * lr, "later_rate_v4_on_v4": lr,
         "later_per_game_ours": s[:, 0].sum() / Gs, "later_per_game_v4": r[:, 0].sum() / Gr}
    mix = -s[:, 0].sum() / Gs * lr
    for i, g in enumerate(GROUPS):
        ro, rv = s[i, 1] / s[i, 0], r[i, 1] / r[i, 0]
        x[f"within_{g}"] = s[i, 0] / Gs * (ro - rv)
        x[f"rate_ours_{g}"], x[f"rate_v4_{g}"] = ro, rv
        x[f"rate_diff_{g}"] = ro - rv
        x[f"share_ours_{g}"], x[f"share_v4_{g}"] = s[i, 0] / s[:, 0].sum(), r[i, 0] / r[:, 0].sum()
        x[f"dec_{g}"] = (s[i, 2] - s[i, 1]) / Gs
        x[f"dec_rate_{g}"] = (s[i, 2] - s[i, 1]) / s[i, 0]
        mix += s[i, 0] / Gs * rv
        qo, qv = s[i, 3] / s[i, 0], r[i, 3] / r[i, 0]
        os_, ons = (s[i, 1] - s[i, 4]) / (s[i, 0] - s[i, 3]), s[i, 4] / s[i, 3]
        vs_, vns = (r[i, 1] - r[i, 4]) / (r[i, 0] - r[i, 3]), r[i, 4] / r[i, 3]
        x[f"part_safe_{g}"] = s[i, 0] / Gs * (1 - qo) * (os_ - vs_)
        x[f"part_nosafe_{g}"] = s[i, 0] / Gs * qo * (ons - vns)
        x[f"part_shift_{g}"] = s[i, 0] / Gs * (qo - qv) * (vns - vs_)
        x[f"rate_safe_ours_{g}"], x[f"rate_safe_v4_{g}"] = os_, vs_
        x[f"rate_nosafe_ours_{g}"], x[f"rate_nosafe_v4_{g}"] = ons, vns
        for j, name in ((3, "nosafe"), (5, "nsafe"), (6, "ndanger"), (7, "nwait"), (8, "tenpai"), (9, "open")):
            x[f"c_{name}_ours_{g}"], x[f"c_{name}_v4_{g}"] = s[i, j] / s[i, 0], r[i, j] / r[i, 0]
            x[f"c_{name}_diff_{g}"] = s[i, j] / s[i, 0] - r[i, j] / r[i, 0]
    x["mix"] = mix
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", action="append", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--ref", default=os.environ.get("MORTAL_V4", "/home/r/Projects/better_mortal/baseline/mortal_v4.pth"))
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--dump")
    args = ap.parse_args()

    cap_gpu_mem(args.device)
    dev = torch.device(args.device)
    ref, sub = load_model(args.ref, dev), load_model(args.sub, dev)
    files = sorted(p for d in args.log_dir for p in glob.glob(os.path.join(d, "*.json.gz"))
                   if NAME_RE.search(os.path.basename(p)))
    if args.max_games:
        files = files[:args.max_games]

    rows, bad, pc = [], 0, [0, 0]
    for gi, p in enumerate(files):
        try:
            recs, err = replay(p, "both")
        except Exception as e:                                    # noqa: BLE001
            recs, err = None, f"{type(e).__name__}: {e}"
        if err:
            bad += 1
            if bad <= 3:
                print(f"  ✗ {os.path.basename(p)}: {err}")
            continue
        row = np.zeros(2 * W)
        if recs:
            obs = np.stack([r["obs"] for r in recs])
            mask = np.stack([r["mask"] for r in recs])
            a_ref = qvals(ref, obs, mask, dev).argmax(-1)
            a_sub = qvals(sub, obs, mask, dev).argmax(-1)
            flags = {}
            for k, r in enumerate(recs):
                ar, asb = int(a_ref[k]), int(a_sub[k])
                own = asb if r["role"] == "sub" else ar
                if r["logged"] >= 0:
                    pc[1] += 1
                    pc[0] += int(own == r["logged"])
                f = flags.setdefault((r["seat"], r["kyoku"]), [False, False])
                if not r["first"]:
                    g = 1 if f[0] else (2 if f[1] else 0)
                    o = (0 if r["role"] == "sub" else W) + K * g
                    dv = bool(r["danger"][ar])
                    row[o] += 1
                    row[o + 1] += dv
                    row[o + 2] += bool(r["danger"][asb])
                    if not r["has_safe"]:
                        row[o + 3] += 1
                        row[o + 4] += dv
                    row[o + 5] += r["nsafe"]
                    row[o + 6] += r["ndanger"]
                    row[o + 7] += r["nwait"]
                    row[o + 8] += r["sh"] <= 0
                    row[o + 9] += r["opened"]
                if r["has_safe"]:
                    cr, cs = cls(ar, r), cls(asb, r)
                    if cs == "push" and cr == "fold":
                        f[0] = True
                    elif cs == "fold" and cr == "push":
                        f[1] = True
        rows.append(row)
        if (gi + 1) % 500 == 0:
            print(f"  [{gi + 1}/{len(files)}]", flush=True)

    D = np.array(rows)
    G = len(D)
    print(f"\n牌谱 {G:,} 局(重放失败 {bad})  ref={os.path.basename(args.ref)}  sub={os.path.basename(args.sub)}")
    print(f"  阳性对照:各座位自己的模型重算 = 牌谱实际 {pc[0] / max(pc[1], 1):.2%}({pc[1]:,} 个暴露决策)")
    pt = stats(D.sum(0), G)
    rng = np.random.default_rng(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = [stats(D[rng.integers(0, G, G)].sum(0), G) for _ in range(args.boot)]
    ci = {k: (float(np.nanpercentile([b[k] for b in boots], 2.5)), float(np.nanpercentile([b[k] for b in boots], 97.5)))
          for k in pt}

    def f4(k):
        return f"{pt[k]:+.4f} [{ci[k][0]:+.4f}, {ci[k][1]:+.4f}]"

    print(f"\n==== 后续决策(非入口)的局面项,按此前分歧分组(按局 bootstrap {args.boot} 次) ====")
    print(f"  后续决策/局:我方 {pt['later_per_game_ours']:.2f}  v4 座位 {pt['later_per_game_v4']:.2f};"
          f"v4 在自己轨迹上的后续点炮牌率 {pt['later_rate_v4_on_v4']:.3%}")
    print(f"  局面项之之后(合计) {f4('later_total')} 张/局  ← 应与 diag_pushfold 的同名项相等")
    print(f"  {'组':<26}{'占比 我方':>10}{'占比 v4':>10}{'v4点炮 我方轨迹':>16}{'v4轨迹':>9}{'率差':>9}   组内项 张/局 [95%]")
    for g in GROUPS:
        print(f"  {GLABEL[g]:<24}{pt[f'share_ours_{g}']:>10.2%}{pt[f'share_v4_{g}']:>10.2%}"
              f"{pt[f'rate_ours_{g}']:>16.3%}{pt[f'rate_v4_{g}']:>9.3%}{pt[f'rate_diff_{g}']:>+9.3%}   {f4(f'within_{g}')}")
    print(f"  {'配比项(三组占比不同)':<24}{'':>54}   {f4('mix')}")
    print(f"\n  我方轨迹上各组的单步项(同一局面 被测 − v4 的点炮牌率;承诺效应看 P 组)")
    for g in GROUPS:
        print(f"  {GLABEL[g]:<24}  率差 {pt[f'dec_rate_{g}']:+.3%} [{ci[f'dec_rate_{g}'][0]:+.3%}, {ci[f'dec_rate_{g}'][1]:+.3%}]"
              f"   贡献 {f4(f'dec_{g}')} 张/局")
    print(f"\n  组内项再拆(Step 0.5):有现物时的率差 / 没现物时的率差 / 没现物占比之差,张/局 [95%]")
    for g in GROUPS:
        print(f"  {GLABEL[g]:<24}  有现物 {f4(f'part_safe_{g}')}   没现物 {f4(f'part_nosafe_{g}')}   占比 {f4(f'part_shift_{g}')}")
        print(f"  {'':<24}  v4 点炮率 有现物 {pt[f'rate_safe_ours_{g}']:.3%} vs {pt[f'rate_safe_v4_{g}']:.3%},"
              f"没现物 {pt[f'rate_nosafe_ours_{g}']:.3%} vs {pt[f'rate_nosafe_v4_{g}']:.3%}(我方轨迹 vs v4 轨迹)")
    print(f"\n  后续决策时的局面构成(我方轨迹 vs v4 轨迹)")
    print(f"  {'':<14}" + "".join(f"{GLABEL[g][:1] + ' 组':>34}" for g in GROUPS))
    for name, lab, fmt in (("nosafe", "没现物占比", "{:.2%}"), ("nsafe", "手里现物张数", "{:.3f}"),
                           ("ndanger", "手里危险牌张数", "{:.3f}"), ("nwait", "立直者待牌种数", "{:.3f}"),
                           ("tenpai", "听牌", "{:.2%}"), ("open", "副露", "{:.2%}")):
        cells = []
        for g in GROUPS:
            d = f"c_{name}_diff_{g}"
            fd = fmt.replace(":", ":+")
            cells.append(f"{fmt.format(pt[f'c_{name}_ours_{g}'])}/{fmt.format(pt[f'c_{name}_v4_{g}'])}"
                         f" {fd.format(pt[d])}[{fd.format(ci[d][0])},{fd.format(ci[d][1])}]")
        print(f"  {lab:<12}" + "".join(f"{c:>34}" for c in cells))
    if args.dump:
        with open(args.dump, "w") as fh:
            json.dump({"n_games": G, "bad": bad, "pos_ctrl": pc, "point": pt, "ci": ci,
                       "ref": args.ref, "sub": args.sub}, fh, ensure_ascii=False, indent=1)
        print(f"\n  → {args.dump}")


if __name__ == "__main__":
    main()
