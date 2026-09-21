"""放铳按「和牌方状态 × 我方状态 × 打点 × 巡目」拆解 —— Step 0.1(2026-09-21)。

**动机**。自建血统的放铳率与 v4 持平,但每次放铳贵 90~125 点(RESULTS「缺口定位到放铳更贵」,五个快照同号);
方案 A 换成 v4 基座后这个超额消失 → 它是基座血统的属性,不是在线 RL 造成的。要决定在基座上改什么
(押退判断?读对手打点?),先回答:**多出来的放铳代价来自哪一类局面?**

  · 和牌方状态 W:riichi(立直,明牌威胁)/ open(副露)/ dama(门清未立直,默听)
      —— 立直的威胁看得见,是押退判断问题;默听/副露的打点要靠读,是读手问题
  · 我方状态 D:in_riichi(自己已立直,无法弃牌)/ riichi_decl(立直宣言牌被荣和)/
      free_open(副露中放铳)/ free_closed(门清未立直时放铳)
  · 打点档、和牌方是否庄家、放铳时我方巡目,以及 W × 打点档

**口径**(与 diag_gap 一致):放铳点数 = −deltas[target](含本场,不含供托);荣和才算放铳,自摸不算;
一炮双响按两次放铳计;我方座位取 start_game.names 里唯一不是 mortal-v4 的那个(与文件名后缀交叉核对)。

**差值与 CI**。每行报:次/半庄、均铳、点/半庄(我们 − v4 每席)。差值按「组」(同一 seed 的 4 个轮转)配对,
组层 95% CI,口径同 group_ci。再把差值拆成两项(以 v4 为参照,两项之和恒等于总差):
  频次项 = (我们次数 − v4 次数) × v4 均铳      —— 放铳得更多
  单价项 = 我们次数 × (我们均铳 − v4 均铳)      —— 次数一样但放得更贵

**暴露局**(对立直的押退):一局里有对手立直(reach_accepted)而自己当时未立直,这一局就算该座位「暴露」。
报暴露频率、暴露后打牌数、放铳给立直者/和了/净收支(含自己的立直棒)—— 用来分清「放铳给立直者更多」
是期望值问题(押了但没多和)还是方差问题(押了也多和了,净收支持平,只是尾部更肥 → 被飞更多)。

**自检**:「全部」行的放铳率与均铳必须与同一批牌谱的 diag_gap 输出一致;对 v4 自己打 v4 的零点牌谱,
所有行的差都应与 0 不可分(阴性对照)。

用法:
  diag_dealin.py --log-dir DIR [--log-dir DIR2 ...] [--seed-lo ..] [--seed-hi ..] [--expect-games N] [--dump x.npz] [-j 16]
  diag_dealin.py --merge a.npz b.npz ...          # 多段合并后重算(按 seed 去重)
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

NAME_RE = re.compile(r"(\d+)_(\d+)_([a-d])\.json\.gz$")
SUFFIX_SEAT = {"a": 0, "b": 1, "c": 2, "d": 3}   # 与 group_ci 相同
OPEN_CALLS = ("chi", "pon", "daiminkan", "kakan")  # 暗杠不破门清

W_KINDS = ("riichi", "open", "dama")
D_KINDS = ("in_riichi", "riichi_decl", "free_open", "free_closed")
BINS = ((0, 4000, "<4k"), (4000, 8000, "4-8k"), (8000, 12000, "8-12k"),
        (12000, 18000, "12-18k"), (18000, 10 ** 9, "18k+"))
JUNS = ((0, 6, "early<=6"), (7, 12, "mid7-12"), (13, 999, "late>=13"))

BUCKETS = (["all"]
           + [f"W:{w}" for w in W_KINDS]
           + [f"D:{d}" for d in D_KINDS]
           + [f"WD:{w}|{d}" for w in W_KINDS for d in D_KINDS]
           + [f"bin:{b[2]}" for b in BINS]
           + ["oya:dealer_win", "oya:non_dealer_win"]
           + [f"jun:{j[2]}" for j in JUNS]
           + [f"Wbin:{w}|{b[2]}" for w in W_KINDS for b in BINS])
BI = {b: i for i, b in enumerate(BUCKETS)}
NB = len(BUCKETS)
EXP = ("rounds", "discards", "net", "win_cnt", "win_pts", "dir_cnt", "dir_pts")   # 暴露局累加量
NE = len(EXP)

SECTIONS = (
    ("和牌方状态 W", "W:"),
    ("我方状态 D", "D:"),
    ("W × D", "WD:"),
    ("打点档", "bin:"),
    ("和牌方是否庄家", "oya:"),
    ("放铳时我方巡目", "jun:"),
    ("W × 打点档", "Wbin:"),
)


def _bin(v, table):
    for lo, hi, name in table:
        if lo <= v < hi or (lo <= v <= hi and table is JUNS):
            return name
    return table[-1][2]


def keys_for(w, d, cost, dealer_win, jun):
    cb = _bin(cost, BINS)
    return ("all", f"W:{w}", f"D:{d}", f"WD:{w}|{d}", f"bin:{cb}",
            "oya:dealer_win" if dealer_win else "oya:non_dealer_win",
            f"jun:{_bin(jun, JUNS)}", f"Wbin:{w}|{cb}")


def scan(path):
    """一局牌谱 → (seed, 我方座位, 座位是否与后缀一致, cnt[4,NB], pts[4,NB], kyoku[4], E[4,NE])。"""
    m = NAME_RE.search(os.path.basename(path))
    seed, suffix_seat = int(m.group(1)), SUFFIX_SEAT[m.group(3)]
    cnt = np.zeros((4, NB), dtype=np.int64)
    pts = np.zeros((4, NB), dtype=np.int64)
    kyoku = np.zeros(4, dtype=np.int64)
    E = np.zeros((4, NE), dtype=np.int64)
    riichi = [False] * 4; pend = [False] * 4; opn = [False] * 4; nd = [0] * 4; oya = 0
    exp_ = [False] * 4; ndx = [0] * 4; rd = [0] * 4; wc = [0] * 4; wp = [0] * 4; dc = [0] * 4; dp = [0] * 4
    live = False
    name_seat = None

    def flush():
        for q in range(4):
            if exp_[q]:
                E[q] += (1, ndx[q], rd[q], wc[q], wp[q], dc[q], dp[q])
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ev = json.loads(line)
            t = ev.get("type")
            if t == "start_game":
                cand = [i for i, n in enumerate(ev.get("names", [])) if n != "mortal-v4"]
                if len(cand) == 1:
                    name_seat = cand[0]
            elif t == "start_kyoku":
                if live:
                    flush()
                riichi = [False] * 4; pend = [False] * 4; opn = [False] * 4; nd = [0] * 4
                exp_ = [False] * 4; ndx = [0] * 4; rd = [0] * 4; wc = [0] * 4; wp = [0] * 4; dc = [0] * 4; dp = [0] * 4
                oya = int(ev["oya"])
                kyoku += 1
                live = True
            elif t == "end_kyoku":
                if live:
                    flush()
                live = False
            elif t == "dahai":
                q = int(ev["actor"]); nd[q] += 1
                if exp_[q] and not riichi[q]:
                    ndx[q] += 1
            elif t == "reach":
                pend[int(ev["actor"])] = True
            elif t == "reach_accepted":
                a = int(ev["actor"]); riichi[a] = True; pend[a] = False
                rd[a] -= 1000                                  # 自己的立直棒
                for q in range(4):
                    if q != a and not riichi[q]:
                        exp_[q] = True
            elif t == "ryukyoku":
                d = ev.get("deltas", [0] * 4)
                for q in range(4):
                    rd[q] += int(d[q])
            elif t in OPEN_CALLS:
                opn[int(ev["actor"])] = True
            elif t == "hora":
                a, tg = int(ev["actor"]), int(ev["target"])
                dl = ev.get("deltas", [0] * 4)
                for q in range(4):
                    rd[q] += int(dl[q])
                if exp_[a]:
                    wc[a] += 1; wp[a] += int(dl[a])
                if a == tg:
                    continue
                cost = -int(dl[tg])
                if riichi[a] and not riichi[tg]:
                    dc[tg] += 1; dp[tg] += cost
                w = "riichi" if riichi[a] else ("open" if opn[a] else "dama")
                d = ("in_riichi" if riichi[tg] else "riichi_decl" if pend[tg]
                     else "free_open" if opn[tg] else "free_closed")
                for k in keys_for(w, d, cost, a == oya, nd[tg]):
                    cnt[tg, BI[k]] += 1
                    pts[tg, BI[k]] += cost
    if live:
        flush()
    our = name_seat if name_seat is not None else suffix_seat
    return seed, our, (name_seat is None or name_seat == suffix_seat), cnt, pts, kyoku, E


def collect(files, jobs):
    g = {}
    mismatch = 0
    with Pool(jobs) as pool:
        for seed, our, ok, cnt, pts, ky, E in pool.imap_unordered(scan, files, chunksize=64):
            mismatch += (not ok)
            r = g.setdefault(seed, dict(oc=np.zeros(NB, np.int64), op=np.zeros(NB, np.int64),
                                        vc=np.zeros(NB, np.int64), vp=np.zeros(NB, np.int64),
                                        on=0, vn=0, ok=0, vk=0,
                                        oe=np.zeros(NE, np.int64), ve=np.zeros(NE, np.int64)))
            others = [i for i in range(4) if i != our]
            r["oc"] += cnt[our]; r["op"] += pts[our]
            r["vc"] += cnt[others].sum(0); r["vp"] += pts[others].sum(0)
            r["on"] += 1; r["vn"] += 3
            r["ok"] += int(ky[our]); r["vk"] += int(ky[others].sum())
            r["oe"] += E[our]; r["ve"] += E[others].sum(0)
    seeds = np.array(sorted(g), dtype=np.int64)
    pack = {k: np.array([g[s][k] for s in seeds]) for k in ("oc", "op", "vc", "vp", "on", "vn", "ok", "vk", "oe", "ve")}
    return dict(seeds=seeds, **pack), mismatch


def report(D, title):
    oc, op, vc, vp = D["oc"], D["op"], D["vc"], D["vp"]
    on, vn, ok, vk = D["on"], D["vn"], D["ok"], D["vk"]
    k = len(D["seeds"]); games = int(on.sum())
    print(f"\n==== {title} ====")
    print(f"  {games:,} 局(我方)/ {int(vn.sum()):,} 席局(v4)/ {k:,} 组 seed {D['seeds'].min()}-{D['seeds'].max()}")
    full = int((on == 4).sum())
    if full != k:
        print(f"  ⚠️ 只有 {full:,}/{k:,} 组是完整 4 轮转,不完整组的配对会变弱")
    a = BI["all"]
    print(f"  自检(须与 diag_gap 一致): 放铳率 我们 {oc[:, a].sum() / ok.sum():.2%} / v4 {vc[:, a].sum() / vk.sum():.2%}   "
          f"均铳 我们 {op[:, a].sum() / max(oc[:, a].sum(), 1):,.0f} / v4 {vp[:, a].sum() / max(vc[:, a].sum(), 1):,.0f}")

    per_o = op / on[:, None]; per_v = vp / vn[:, None]
    diff = per_o - per_v                       # 每组: 点/半庄 差(正 = 我们放铳损失更多)
    mean = diff.mean(0)
    ci = 1.96 * diff.std(0, ddof=1) / np.sqrt(k)
    total = mean[a]

    ON, VN = on.sum(), vn.sum()
    ro = oc.sum(0) / ON; rv = vc.sum(0) / VN                     # 次/半庄
    ao = op.sum(0) / np.maximum(oc.sum(0), 1); av = vp.sum(0) / np.maximum(vc.sum(0), 1)
    freq = (ro - rv) * av; sev = ro * (ao - av)

    hdr = (f"  {'桶':<26}{'次/半庄 我们':>12}{'v4':>8}{'均铳 我们':>10}{'v4':>8}"
           f"{'Δ点/半庄':>11}{'± CI':>8}{'z':>7}{'频次项':>9}{'单价项':>9}{'占总差':>8}")
    print(f"\n  总差 = {total:+.1f} ± {ci[a]:.1f} 点/半庄(我们 − v4 每席;正 = 我们放铳损失多)")

    def row(b):
        i = BI[b]
        z = mean[i] / (ci[i] / 1.96) if ci[i] > 0 else 0.0
        share = mean[i] / total if abs(total) > 1e-9 else float("nan")
        print(f"  {b:<26}{ro[i]:>12.3f}{rv[i]:>8.3f}{ao[i]:>10,.0f}{av[i]:>8,.0f}"
              f"{mean[i]:>+11.1f}{ci[i]:>8.1f}{z:>+7.2f}{freq[i]:>+9.1f}{sev[i]:>+9.1f}{share:>8.0%}")

    print(hdr); row("all")
    for name, pre in SECTIONS:
        print(f"  -- {name}")
        for b in BUCKETS:
            if b.startswith(pre):
                row(b)

    if "oe" not in D:
        return
    oe, ve = D["oe"], D["ve"]
    ix = {n: i for i, n in enumerate(EXP)}
    so, sv = oe.sum(0), ve.sum(0)
    print("\n  -- 对立直的押退(暴露局 = 有对手立直、自己当时未立直的局)")
    print(f"  {'':<30}{'我们':>10}{'v4 每席':>10}")
    print(f"  {'暴露局 / 半庄':<30}{so[ix['rounds']] / ON:>10.3f}{sv[ix['rounds']] / VN:>10.3f}")
    print(f"  {'暴露后打牌数 / 暴露局':<28}{so[ix['discards']] / so[ix['rounds']]:>10.2f}{sv[ix['discards']] / sv[ix['rounds']]:>10.2f}")
    print(f"  {'放铳给立直者 / 暴露局':<28}{so[ix['dir_cnt']] / so[ix['rounds']]:>10.2%}{sv[ix['dir_cnt']] / sv[ix['rounds']]:>10.2%}")
    print(f"  {'  其均铳':<30}{so[ix['dir_pts']] / max(so[ix['dir_cnt']], 1):>10,.0f}{sv[ix['dir_pts']] / max(sv[ix['dir_cnt']], 1):>10,.0f}")
    print(f"  {'和了 / 暴露局':<30}{so[ix['win_cnt']] / so[ix['rounds']]:>10.2%}{sv[ix['win_cnt']] / sv[ix['rounds']]:>10.2%}")
    print(f"  {'  其均和':<30}{so[ix['win_pts']] / max(so[ix['win_cnt']], 1):>10,.0f}{sv[ix['win_pts']] / max(sv[ix['win_cnt']], 1):>10,.0f}")
    print(f"  {'净收支 / 暴露局(含立直棒)':<26}{so[ix['net']] / so[ix['rounds']]:>+10,.0f}{sv[ix['net']] / sv[ix['rounds']]:>+10,.0f}")

    def diffrow(label, key):
        x = oe[:, ix[key]] / on - ve[:, ix[key]] / vn
        m, c = x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(k)
        print(f"  Δ {label:<34}{m:>+9.1f} ± {c:>5.1f} 点/半庄   z = {m / (c / 1.96):+.2f}")
    print()
    diffrow("放铳给立直者(点)", "dir_pts")
    diffrow("暴露局里的和了(点)", "win_pts")
    diffrow("暴露局净收支(点)", "net")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", action="append", default=[])
    ap.add_argument("--seed-lo", type=int, default=0)
    ap.add_argument("--seed-hi", type=int, default=10 ** 9)
    ap.add_argument("--expect-games", type=int, default=0)
    ap.add_argument("--dump")
    ap.add_argument("--merge", nargs="*")
    ap.add_argument("--title", default="")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    if args.merge:
        parts = [dict(np.load(p)) for p in args.merge]
        seeds = np.concatenate([p["seeds"] for p in parts])
        _, first = np.unique(seeds, return_index=True)
        if len(first) != len(seeds):
            print(f"  ⚠️ 合并时有 {len(seeds) - len(first)} 个重复 seed,只保留第一次出现")
        D = {key: np.concatenate([p[key] for p in parts])[first] for key in parts[0]}
        report(D, args.title or f"合并 {len(parts)} 段")
        return

    files = []
    for d in args.log_dir:
        for p in glob.glob(os.path.join(d, "*.json.gz")):
            m = NAME_RE.search(os.path.basename(p))
            if m and args.seed_lo <= int(m.group(1)) <= args.seed_hi:
                files.append(p)
    files.sort()
    if not files:
        sys.exit("没有匹配的牌谱")
    if args.expect_games and len(files) != args.expect_games:
        sys.exit(f"局数不符:找到 {len(files)},预期 {args.expect_games}(日志目录可能混了别的 run)")
    D, mismatch = collect(files, args.jobs)
    if mismatch:
        print(f"  ⚠️ {mismatch} 局的 start_game.names 座位与文件名后缀不一致(已按 names 取座位)")
    report(D, args.title or ", ".join(os.path.basename(d.rstrip('/')) for d in args.log_dir))
    if args.dump:
        np.savez(args.dump, **D)
        print(f"\n  → {args.dump}")


if __name__ == "__main__":
    main()
