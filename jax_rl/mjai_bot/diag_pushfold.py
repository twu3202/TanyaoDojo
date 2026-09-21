"""对立直的押退:同一条牌谱轨迹上,在「有对手立直、自己未立直」的每个打牌决策点同时问被测模型与 v4 —— Step 0.2(2026-09-21)。

**动机**。Step 0.1(diag_dealin.py)定位到自建血统与 v4 的差距在「对立直押得太多」:暴露局里放铳给立直者更多,
净收支 −164 ± 55 点/半庄(tpt 408k, 100k),而各类和牌方内部的均铳与 v4 相同(不是读打点的问题)。
离线基座 v11 就有;v5 带手工危险度特征反而最严重 → 不是「看不出危险」,而是「看出来了还押」。
这一步回答:**在哪些局面下我们押、而 v4 弃?**

**做法**。重放牌谱,对选定座位逐事件 PlayerState.update;在该座位的打牌决策点上,若有对手已立直(reach_accepted)
而自己未立直也未宣言,就是一个「暴露决策」。用事件流自己维护每个立直者 R 的安全牌集合:
    R 自己打过的所有牌(振听) ∪ R 立直后任何人打出而 R 没荣和的牌(立直后见逃即永久振听)
多家立直时取交集。动作算「弃」当且仅当它是打牌且牌种在安全集合里;打非现物、宣言立直都算「押」。
**只有手里至少有一张合法现物的决策点**才用来比较弃牌率(没有现物时谈不上弃)。可自摸和的点不算。

同一个观测同时过两个模型(fp32 argmax,不开 autocast):
  · 被测座位:被测模型的重算必须 ≈ 牌谱实际(阳性对照,验证重放 / 编码 / 动作还原);
  · v4 座位:v4 的重算必须 ≈ 实际,再看被测模型在 v4 的轨迹上会怎么选(两条轨迹互为印证)。

**分桶**:我方向听(0 / 1 / 2+)× 门清/副露;放铳时我方巡目;立直家数;手里宝牌数(含赤)。
**读数**:v4 弃牌率、被测弃牌率、差(v4 − 被测,正 = 我们押得多)按局聚类的 95% CI;
「v4 弃而我们押」「v4 押而我们弃」的比例;暴露时可立直的点上两边的立直率(追立)。

**点炮牌率(真值)**:牌谱是全信息的,四个座位的 PlayerState 都在维护 —— 立直者此刻的真实待牌(waits)与
是否振听(at_furiten)是现成的。于是对每个暴露决策可以直接判断:某个打牌动作会不会被立直者荣和(= 点炮牌)。
报 v4 / 被测 / 实际三者选到点炮牌的比率(**全部**暴露决策,含无现物可打的),以及「两边都押」时各自挑到点炮牌的比率
—— 把「愿不愿意弃」和「押的时候挑哪张」拆开。实际打出的点炮牌数应与 diag_dealin 的「放铳给立直者」对得上(第二道阳性对照)。
只算立直者的荣和;默听/副露者的荣和不在此列。

**缺口分解(--side both,Step 0.3)**:把「我方座位实际打出的点炮牌/局 − v4 座位的」拆成三项 ——
    决策项 = 在我方轨迹上,同一局面换成 v4 来选,点炮牌少多少(单步反事实);
    局面项 = v4 在我方轨迹上 vs 在它自己轨迹上的点炮牌率之差 × 我方暴露决策数(我们带着更危险的局面进入/走过暴露);
    次数项 = 暴露决策数之差 × v4 自己的点炮牌率。
局面项再拆成「入口」(每个暴露局我方的第一个暴露决策 —— 这时还没做过任何押退选择,差别只能来自立直前的做牌)
与「之后」(同一暴露局后续决策 —— 含我们前面押过的累积后果)以及两者占比不同的配比项。
入口处另报局面构成:向听、副露、手里有无现物、手里现物张数、宝牌数。全部按局 bootstrap 出 95% 区间。

**已知局限**:安全集合只用现物(含立直后见逃),不含筋/壁/字牌等「半安全」—— 所以这里的「押」包括打筋牌,
这对两个模型是同一把尺子,比较的是差;kakan 见逃(抢杠)不计入安全集合。

用法:
  diag_pushfold.py --log-dir DIR [--log-dir ...] --sub ckpt.pth [--ref mortal_v4.pth] [--side both|sub|ref]
                   [--max-games N] [--device cuda:0] [--dump x.json]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.environ.get("MJAI_BOT_DIR", os.path.dirname(os.path.abspath(__file__))))
from diag_disagree import (NAME_RE, VERSION, ACT_EVENTS, TURN_EVENTS, PlayerState,  # noqa: E402
                           tile_idx, event_to_action, load_model, qvals, cap_gpu_mem)
import torch  # noqa: E402

V4NAME = "mortal-v4"
RED2KIND = {34: 4, 35: 13, 36: 22}


def kind_of_action(a: int) -> int:
    return RED2KIND.get(a, a)


def kind_of_pai(p: str) -> int:
    k = tile_idx(p)
    return RED2KIND.get(k, k)


def dora_of(marker: int) -> int:
    """宝牌指示牌 → 宝牌牌种(数牌 9→1,风 东南西北循环,三元 白发中循环)。"""
    if marker < 27:
        suit, n = divmod(marker, 9)
        return suit * 9 + (n + 1) % 9
    if marker < 31:
        return 27 + (marker - 27 + 1) % 4
    return 31 + (marker - 31 + 1) % 3


def logged_action(events, i, me, can_pass) -> int:
    for e in events[i + 1:]:
        t = e.get("type")
        if t in ACT_EVENTS and int(e.get("actor", -1)) == me:
            return event_to_action(e)
        if t in ACT_EVENTS or t in TURN_EVENTS:
            return 45 if can_pass else -1
    return -1


def replay(path, side):
    events = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    names = events[0].get("names", [])
    subs = [i for i, n in enumerate(names) if n != V4NAME]
    if len(subs) != 1:
        return None, f"names={names}"
    sub_seat = subs[0]
    seats = {"sub": [sub_seat], "ref": [i for i in range(4) if i != sub_seat], "both": [0, 1, 2, 3]}[side]
    ps = {s: PlayerState(s) for s in range(4)}          # 四家都维护:立直者的真实待牌/振听要从它自己的 PlayerState 读
    riichi = [False] * 4
    own = [set() for _ in range(4)]
    after = [set() for _ in range(4)]
    dora, ndisc = [], [0] * 4
    seen = [False] * 4                                   # 本局该座位是否已出现过暴露决策(入口 = 第一个)
    recs = []
    for i, ev in enumerate(events):
        t = ev.get("type")
        # 先更新公共状态:决策点的安全集合要包含到这个事件为止的全部信息
        if t == "start_kyoku":
            riichi = [False] * 4
            own = [set() for _ in range(4)]
            after = [set() for _ in range(4)]
            dora = [dora_of(kind_of_pai(ev["dora_marker"]))]
            ndisc = [0] * 4
            seen = [False] * 4
        elif t == "dora":
            dora.append(dora_of(kind_of_pai(ev["dora_marker"])))
        elif t == "dahai":
            a, k = int(ev["actor"]), kind_of_pai(ev["pai"])
            own[a].add(k)
            ndisc[a] += 1
            for r in range(4):
                if riichi[r] and r != a:
                    after[r].add(k)
        elif t == "reach_accepted":
            riichi[int(ev["actor"])] = True
        line = json.dumps(ev)
        cands = {s: ps[s].update(line) for s in range(4)}
        for s in seats:
            cans = cands[s]
            if not cans.can_act or not cans.can_discard:
                continue
            p = ps[s]
            if p.self_riichi_accepted or p.self_riichi_declared:
                continue
            rs = [r for r in range(4) if r != s and riichi[r]]
            if not rs:
                continue
            obs, mask = p.encode_obs(VERSION, False)
            mask = np.asarray(mask, bool)
            if mask[43]:
                continue
            safe = set.intersection(*[own[r] | after[r] for r in rs])
            danger = set()
            for r in rs:
                if not ps[r].at_furiten:
                    danger |= {k for k, w in enumerate(ps[r].waits) if w}
            safe_mask = np.zeros(46, bool)
            danger_mask = np.zeros(46, bool)
            for a in range(37):
                if mask[a] and kind_of_action(a) in safe:
                    safe_mask[a] = True
                if mask[a] and kind_of_action(a) in danger:
                    danger_mask[a] = True
            if (safe_mask & danger_mask).any():
                raise AssertionError("现物同时是立直者的待牌:安全集合或待牌读取有误")
            tehai = p.tehai
            first, seen[s] = not seen[s], True
            recs.append(dict(
                first=first, nsafe=sum(int(tehai[k]) for k in safe if k < 34),
                role="sub" if s == sub_seat else "ref",
                obs=np.asarray(obs, np.float32), mask=mask, safe=safe_mask, has_safe=bool(safe_mask.any()),
                danger=danger_mask,
                logged=logged_action(events, i, s, cans.can_pass),
                sh=int(p.shanten), opened=(len(p.chis) + len(p.pons) + len(p.minkans)) > 0,
                turn=ndisc[s], nr=len(rs),
                dora=sum(int(tehai[d]) for d in dora) + sum(bool(x) for x in p.akas_in_hand)))
    return recs, None


def cls(a: int, r) -> str:
    if a == 37:
        return "push"
    if a <= 36:
        return "fold" if r["safe"][a] else "push"
    return "other"


def bucket_keys(r):
    sh = "向听0" if r["sh"] <= 0 else ("向听1" if r["sh"] == 1 else "向听2+")
    op = "副露" if r["opened"] else "门清"
    tn = "早≤6" if r["turn"] <= 6 else ("中7-12" if r["turn"] <= 12 else "晚≥13")
    dr = f"宝牌{min(r['dora'], 2)}{'+' if r['dora'] >= 2 else ''}"
    return ["全部", f"向听:{sh}", f"门副:{op}", f"向听×门副:{sh}|{op}", f"巡目:{tn}",
            f"立直家数:{'1' if r['nr'] == 1 else '2+'}", f"宝牌:{dr}"]


ORDER = ["全部"] + [f"向听:{s}" for s in ("向听0", "向听1", "向听2+")] + ["门副:门清", "门副:副露"] + \
        [f"向听×门副:{s}|{o}" for s in ("向听0", "向听1", "向听2+") for o in ("门清", "副露")] + \
        [f"巡目:{t}" for t in ("早≤6", "中7-12", "晚≥13")] + ["立直家数:1", "立直家数:2+"] + \
        [f"宝牌:宝牌{d}" for d in ("0", "1", "2+")]


def cluster_ci(num, den):
    """按局聚类的比率 Σnum/Σden 的 95% 半宽(num 可为差值)。"""
    num, den = np.asarray(num, float), np.asarray(den, float)
    keep = den > 0
    num, den = num[keep], den[keep]
    G, N = len(den), den.sum()
    if G < 2 or N == 0:
        return float("nan"), float("nan")
    r = num.sum() / N
    se = np.sqrt(G / (G - 1) * ((num - r * den) ** 2).sum()) / N
    return r, 1.96 * se


def decomp_stats(S, G, K):
    """S = 按局求和后的一行(sub 在前 K 列,ref 在后 K 列);G = 局数。ref 有三个座位,每座位按 3G 归一。"""
    s, r = S[:K], S[K:]
    Gs, Gr = G, 3 * G
    rr = r[1] / r[0]                                     # v4 在自己轨迹上的点炮牌率
    x = {}
    x["gap"] = s[3] / Gs - r[3] / Gr
    x["decision"] = (s[2] - s[1]) / Gs
    x["state"] = (s[1] / s[0] - rr) * s[0] / Gs
    x["count"] = (s[0] / Gs - r[0] / Gr) * rr
    es, er = s[5] / s[4], r[5] / r[4]                    # 入口:v4 的点炮牌率
    ls, lr = (s[1] - s[5]) / (s[0] - s[4]), (r[1] - r[5]) / (r[0] - r[4])
    x["state_entry"] = (es - er) * s[4] / Gs
    x["state_later"] = (ls - lr) * (s[0] - s[4]) / Gs
    x["state_mix"] = (s[4] * er + (s[0] - s[4]) * lr) / Gs - s[0] / Gs * rr
    x["entry_rate_v4_on_ours"], x["entry_rate_v4_on_v4"] = es, er
    x["later_rate_v4_on_ours"], x["later_rate_v4_on_v4"] = ls, lr
    x["entry_rate_sub_on_ours"] = s[6] / s[4]
    x["rounds_per_game_ours"], x["rounds_per_game_v4"] = s[4] / Gs, r[4] / Gr
    x["dec_per_round_ours"], x["dec_per_round_v4"] = s[0] / s[4], r[0] / r[4]
    for j, name in ((7, "entry_tenpai"), (8, "entry_1shanten"), (9, "entry_open"), (10, "entry_has_safe"),
                    (11, "entry_nsafe"), (12, "entry_dora")):
        x[name + "_ours"], x[name + "_v4"] = s[j] / s[4], r[j] / r[4]
    return x


def decomp(D, K, B=1000, seed=0):
    G = len(D)
    pt = decomp_stats(D.sum(0), G, K)
    rng = np.random.default_rng(seed)
    boots = [decomp_stats(D[rng.integers(0, G, G)].sum(0), G, K) for _ in range(B)]
    ci = {k: (float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5)))
          for k in pt}
    dci = {}
    for k in pt:
        if k.endswith("_ours") and k[:-5] + "_v4" in pt:
            base = k[:-5]
            diffs = [b[base + "_ours"] - b[base + "_v4"] for b in boots]
            dci[base] = (pt[base + "_ours"] - pt[base + "_v4"], float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
    for a, b_ in (("entry_rate_v4_on_ours", "entry_rate_v4_on_v4"), ("later_rate_v4_on_ours", "later_rate_v4_on_v4")):
        diffs = [b[a] - b[b_] for b in boots]
        dci[a[:-8]] = (pt[a] - pt[b_], float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))

    g = pt["gap"]
    print(f"\n==== 缺口分解:我方座位实际点炮牌/局 − v4 座位(按局 bootstrap {B} 次,95% 区间) ====")
    print(f"  缺口 {g:+.4f} 张/局  [{ci['gap'][0]:+.4f}, {ci['gap'][1]:+.4f}]")
    for k, lab in (("decision", "决策项(同一局面换 v4 来选)"), ("state", "局面项(v4 在我方轨迹上更易点炮)"),
                   ("count", "次数项(暴露决策更多)")):
        print(f"    {lab:<30}{pt[k]:+.4f}  [{ci[k][0]:+.4f}, {ci[k][1]:+.4f}]  占 {pt[k] / g:.0%}")
    for k, lab in (("state_entry", "入口(立直前做牌带进来的)"), ("state_later", "之后(含前面押过的累积)"),
                   ("state_mix", "配比(入口/之后占比不同)")):
        print(f"      局面项之{lab:<22}{pt[k]:+.4f}  [{ci[k][0]:+.4f}, {ci[k][1]:+.4f}]")
    print(f"  v4 的点炮牌率 —— 入口:我方轨迹 {pt['entry_rate_v4_on_ours']:.3%} vs v4 轨迹 {pt['entry_rate_v4_on_v4']:.3%}"
          f"(差 {dci['entry_rate_v4'][0]:+.3%} [{dci['entry_rate_v4'][1]:+.3%}, {dci['entry_rate_v4'][2]:+.3%}]);"
          f"被测在入口 {pt['entry_rate_sub_on_ours']:.3%}")
    print(f"                 之后:我方轨迹 {pt['later_rate_v4_on_ours']:.3%} vs v4 轨迹 {pt['later_rate_v4_on_v4']:.3%}"
          f"(差 {dci['later_rate_v4'][0]:+.3%} [{dci['later_rate_v4'][1]:+.3%}, {dci['later_rate_v4'][2]:+.3%}])")
    print(f"  {'入口局面构成':<16}{'我方':>10}{'v4':>10}{'差':>10}   95% 区间")
    for base, lab, fmt in (("rounds_per_game", "暴露局/局", "{:.3f}"), ("dec_per_round", "每暴露局决策数", "{:.2f}"),
                           ("entry_tenpai", "入口听牌", "{:.2%}"), ("entry_1shanten", "入口一向听", "{:.2%}"),
                           ("entry_open", "入口副露", "{:.2%}"), ("entry_has_safe", "入口有现物", "{:.2%}"),
                           ("entry_nsafe", "入口现物张数", "{:.3f}"), ("entry_dora", "入口宝牌数", "{:.3f}")):
        d, lo, hi = dci[base]
        f = fmt.replace(":", ":+")
        print(f"  {lab:<16}{fmt.format(pt[base + '_ours']):>10}{fmt.format(pt[base + '_v4']):>10}{f.format(d):>10}"
              f"   [{f.format(lo)}, {f.format(hi)}]")
    return dict(point=pt, ci=ci, diff_ci=dci, n_games=G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", action="append", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--ref", default=os.environ.get("MORTAL_V4", "/home/r/Projects/better_mortal/baseline/mortal_v4.pth"))
    ap.add_argument("--side", default="both", choices=["both", "sub", "ref"])
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dump")
    args = ap.parse_args()

    cap_gpu_mem(args.device)
    dev = torch.device(args.device)
    ref, sub = load_model(args.ref, dev), load_model(args.sub, dev)

    files = sorted(p for d in args.log_dir for p in glob.glob(os.path.join(d, "*.json.gz"))
                   if NAME_RE.search(os.path.basename(p)))
    if args.max_games:
        files = files[:args.max_games]

    # G[role][bucket] = 每局一行 [n, ref_fold, sub_fold, log_fold, ref_fold&sub_push, ref_push&sub_fold]
    G = {role: defaultdict(list) for role in ("sub", "ref")}
    RI = {role: defaultdict(list) for role in ("sub", "ref")}    # 可立直的暴露点: [n, ref_riichi, sub_riichi]
    PC = {role: [0, 0] for role in ("sub", "ref")}                # 阳性对照: 轨迹所属模型重算 == 实际
    NS = {role: [0, 0] for role in ("sub", "ref")}                # [暴露决策总数, 其中无现物可打]
    # DI[role][bucket] = 每局一行 [n, ref点炮牌, sub点炮牌, 实际点炮牌];BP = 两边都押时 [n, ref点炮, sub点炮]
    DI = {role: defaultdict(list) for role in ("sub", "ref")}
    BP = {role: [] for role in ("sub", "ref")}
    # DEC 每局一行,sub 与 ref 各 12 列: [N, Dref, Dsub, Dlog, F, Fref, Fsub, 入口向听0, 入口向听1, 入口副露, 入口有现物, 入口现物张数]
    # 外加入口宝牌数一列 → 每角色 13 列
    NDEC = 13
    DEC = []
    ng, bad = 0, 0
    for gi, p in enumerate(files):
        try:
            recs, err = replay(p, args.side)
        except Exception as e:                                    # noqa: BLE001
            recs, err = None, f"{type(e).__name__}: {e}"
        if err:
            bad += 1
            if bad <= 3:
                print(f"  ✗ {os.path.basename(p)}: {err}")
            continue
        ng += 1
        if not recs:
            for role in G:
                for b in ORDER:
                    G[role][b].append([0] * 6)
                    DI[role][b].append([0] * 4)
            DEC.append([0] * (2 * NDEC))
            continue
        obs = np.stack([r["obs"] for r in recs])
        mask = np.stack([r["mask"] for r in recs])
        a_ref = qvals(ref, obs, mask, dev).argmax(-1)
        a_sub = qvals(sub, obs, mask, dev).argmax(-1)
        per = {role: defaultdict(lambda: [0] * 6) for role in G}
        dper = {role: defaultdict(lambda: [0] * 4) for role in G}
        bp = {role: [0, 0, 0] for role in G}
        rip = {role: [0, 0, 0] for role in G}
        dec = [0] * (2 * NDEC)
        for k, r in enumerate(recs):
            role = r["role"]
            own_a = a_sub[k] if role == "sub" else a_ref[k]
            if r["logged"] >= 0:
                PC[role][1] += 1
                PC[role][0] += int(own_a == r["logged"])
            NS[role][0] += 1
            dz = r["danger"]
            dr_, ds_ = bool(dz[a_ref[k]]), bool(dz[a_sub[k]])
            dl_ = bool(dz[r["logged"]]) if 0 <= r["logged"] < 46 else False
            o = 0 if role == "sub" else NDEC
            dec[o + 0] += 1; dec[o + 1] += dr_; dec[o + 2] += ds_; dec[o + 3] += dl_
            if r["first"]:
                dec[o + 4] += 1; dec[o + 5] += dr_; dec[o + 6] += ds_
                dec[o + 7] += r["sh"] <= 0; dec[o + 8] += r["sh"] == 1; dec[o + 9] += r["opened"]
                dec[o + 10] += r["has_safe"]; dec[o + 11] += r["nsafe"]; dec[o + 12] += r["dora"]
            for b in bucket_keys(r):
                c = dper[role][b]
                c[0] += 1; c[1] += dr_; c[2] += ds_; c[3] += dl_
            if cls(int(a_ref[k]), r) == "push" and cls(int(a_sub[k]), r) == "push" and a_ref[k] <= 36 and a_sub[k] <= 36:
                bp[role][0] += 1; bp[role][1] += dr_; bp[role][2] += ds_
            if r["mask"][37]:
                rip[role][0] += 1
                rip[role][1] += int(a_ref[k] == 37)
                rip[role][2] += int(a_sub[k] == 37)
            if not r["has_safe"]:
                NS[role][1] += 1
                continue
            cr, cs = cls(int(a_ref[k]), r), cls(int(a_sub[k]), r)
            cl = cls(int(r["logged"]), r) if r["logged"] >= 0 else "other"
            for b in bucket_keys(r):
                c = per[role][b]
                c[0] += 1
                c[1] += cr == "fold"
                c[2] += cs == "fold"
                c[3] += cl == "fold"
                c[4] += cr == "fold" and cs == "push"
                c[5] += cr == "push" and cs == "fold"
        for role in G:
            for b in ORDER:
                G[role][b].append(per[role][b])
                DI[role][b].append(dper[role][b])
            RI[role]["all"].append(rip[role])
            BP[role].append(bp[role])
        DEC.append(dec)
        if (gi + 1) % 500 == 0:
            print(f"  [{gi + 1}/{len(files)}]", flush=True)

    print(f"\n牌谱 {ng:,} 局(重放失败 {bad})  ref={os.path.basename(args.ref)}  sub={os.path.basename(args.sub)}")
    out = {"n_games": ng, "bad": bad, "ref": args.ref, "sub": args.sub, "roles": {}}
    for role, title in (("sub", "被测模型自己的轨迹"), ("ref", "v4 座位的轨迹")):
        if not PC[role][1]:
            continue
        pc = PC[role][0] / PC[role][1]
        owner = "被测" if role == "sub" else "v4"
        print(f"\n==== {title} ====")
        print(f"  阳性对照:{owner}模型重算 = 牌谱实际 {pc:.2%}({PC[role][1]:,} 个可还原的暴露决策)")
        print(f"  暴露决策 {NS[role][0]:,} 个,其中手里没有现物可打 {NS[role][1] / max(NS[role][0], 1):.1%}(不参与弃牌率比较)")
        print(f"  {'桶':<22}{'决策数':>9}{'v4 弃':>8}{'被测弃':>8}{'实际弃':>8}{'差 v4−被测':>12}{'± CI':>8}{'z':>7}"
              f"{'v4弃我押':>9}{'v4押我弃':>9}")
        rows = {}
        for b in ORDER:
            arr = np.array(G[role][b], float)
            if arr.size == 0 or arr[:, 0].sum() == 0:
                continue
            n = arr[:, 0]
            fr, fs, fl = arr[:, 1].sum() / n.sum(), arr[:, 2].sum() / n.sum(), arr[:, 3].sum() / n.sum()
            d, ci = cluster_ci(arr[:, 1] - arr[:, 2], n)
            z = d / (ci / 1.96) if ci > 0 else float("nan")
            fp, pf = arr[:, 4].sum() / n.sum(), arr[:, 5].sum() / n.sum()
            rows[b] = dict(n=int(n.sum()), ref_fold=fr, sub_fold=fs, log_fold=fl, diff=d, ci=ci, v4fold_subpush=fp, v4push_subfold=pf)
            label = b.split(":", 1)[-1] if b != "全部" else "全部"
            indent = "" if b == "全部" else "  "
            print(f"  {indent + label:<22}{int(n.sum()):>9,}{fr:>8.2%}{fs:>8.2%}{fl:>8.2%}{d:>+12.2%}{ci:>8.2%}{z:>+7.2f}"
                  f"{fp:>9.2%}{pf:>9.2%}")
        print(f"\n  点炮牌率(所有暴露决策,含无现物的;真值 = 立直者此刻未振听的待牌)")
        print(f"  {'桶':<22}{'决策数':>9}{'v4':>8}{'被测':>8}{'实际':>8}{'差 被测−v4':>12}{'± CI':>8}{'z':>7}")
        drows = {}
        for b in ORDER:
            arr = np.array(DI[role][b], float)
            if arr.size == 0 or arr[:, 0].sum() == 0:
                continue
            n = arr[:, 0]
            pr, ps_, pl = arr[:, 1].sum() / n.sum(), arr[:, 2].sum() / n.sum(), arr[:, 3].sum() / n.sum()
            d, ci = cluster_ci(arr[:, 2] - arr[:, 1], n)
            z = d / (ci / 1.96) if ci > 0 else float("nan")
            drows[b] = dict(n=int(n.sum()), ref=pr, sub=ps_, logged=pl, diff=d, ci=ci)
            label = b.split(":", 1)[-1] if b != "全部" else "全部"
            indent = "" if b == "全部" else "  "
            print(f"  {indent + label:<22}{int(n.sum()):>9,}{pr:>8.2%}{ps_:>8.2%}{pl:>8.2%}{d:>+12.2%}{ci:>8.2%}{z:>+7.2f}")
        bpa = np.array(BP[role], float)
        if bpa.size and bpa[:, 0].sum():
            n = bpa[:, 0]
            d, ci = cluster_ci(bpa[:, 2] - bpa[:, 1], n)
            print(f"  两边都押(都打非现物)的 {int(n.sum()):,} 个点上各自挑到点炮牌:v4 {bpa[:, 1].sum() / n.sum():.2%},"
                  f"被测 {bpa[:, 2].sum() / n.sum():.2%},差 {d:+.2%} ± {ci:.2%}  ← 押的时候挑牌的水平")
        print(f"  实际打出的点炮牌 {int(np.array(DI[role]['全部'], float)[:, 3].sum()):,} 张(应与该座位「放铳给立直者」次数相近)")
        ri = np.array(RI[role]["all"], float)
        if ri.size and ri[:, 0].sum():
            n = ri[:, 0]
            d, ci = cluster_ci(ri[:, 2] - ri[:, 1], n)
            print(f"  暴露时可立直的点 {int(n.sum()):,} 个:v4 选立直 {ri[:, 1].sum() / n.sum():.2%},"
                  f"被测选立直 {ri[:, 2].sum() / n.sum():.2%},差(被测 − v4) {d:+.2%} ± {ci:.2%}")
        out["roles"][role] = dict(pos_ctrl=pc, n_exposed=NS[role][0], no_safe=NS[role][1], buckets=rows, dealin_tile=drows)
    if args.side == "both" and DEC:
        out["decomp"] = decomp(np.array(DEC, float), NDEC)
    if args.dump:
        with open(args.dump, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"\n  → {args.dump}")


if __name__ == "__main__":
    main()
