"""差距定位:我们的 agent 与 mortal_v4 在同一批牌局上的分项对比。

**动机**。到 2026-09-11 为止,立直线上花掉的算力几乎全用在没量过头寸的方向上:
加步数(5 亿之后为零)、两个 bug 修正(同等算力下 +0.09 ± 1.91)。而弃张效率那条
轴早就用 `ukeire_headroom.py` 量过——向听最优 96.4%,没有头寸。所以在训练之前,
先回答"那 4.66 分到底在哪":是攻(和不了)、是守(放铳)、还是判断(立直/副露时机)。

**为什么这个对比是公平的**。复式协议让挑战者轮坐四席、每副牌打四遍,所以把
"我们的座位"与"三个 mortal 座位"在**同一批牌局**上汇总,座位运气和牌运都被抵消。
不需要任何前向推理,纯解析已经落盘的牌谱。

**口径与已知局限**:
  · 座位从 start_game 的 names 定位,不信文件名后缀(可自校验);
  · 和了打点取 deltas[和牌者],**含供托**,故略高于纯打点 —— 两边同口径,可比;
  · 流局听牌从 deltas 符号反推,**deltas 全零时不可分辨**(全听或全不听),
    这类局单独计数,不计入听牌率分母;
  · 一局可能有多个 hora(双响),逐个计入。

用法: python diag_gap.py [--log-dir ...] [--seed-lo 10000] [--seed-hi 15999]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
from collections import defaultdict

NAME_RE = re.compile(r"(\d+)_(\d+)_([a-d])\.json\.gz$")
NAKI = ("chi", "pon", "ankan", "kakan", "daiminkan")


def blank():
    return dict(kyoku=0, win=0, tsumo=0, win_pts=0, deal_in=0, deal_in_pts=0,
                riichi=0, naki=0, ryu=0, ryu_tenpai=0, ryu_known=0)


def scan_game(path):
    """返回 (our_seat, per_seat_stats)。our_seat=None 表示这局不是我们打的。"""
    stats = [blank() for _ in range(4)]
    our = None
    cur = None          # 本局的临时累加
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            t = ev.get("type")
            if t == "start_game":
                names = ev.get("names", [])
                for i, n in enumerate(names):
                    if n != "mortal-v4":
                        our = i
            elif t == "start_kyoku":
                cur = [dict(riichi=0, naki=0) for _ in range(4)]
                for i in range(4):
                    stats[i]["kyoku"] += 1
            elif cur is None:
                continue
            elif t == "reach":
                cur[int(ev["actor"])]["riichi"] = 1
            elif t in NAKI:
                cur[int(ev["actor"])]["naki"] = 1
            elif t == "hora":
                a, tg = int(ev["actor"]), int(ev["target"])
                d = ev.get("deltas", [0] * 4)
                stats[a]["win"] += 1
                stats[a]["win_pts"] += int(d[a])
                if a == tg:
                    stats[a]["tsumo"] += 1
                else:
                    stats[tg]["deal_in"] += 1
                    stats[tg]["deal_in_pts"] += -int(d[tg])
            elif t == "ryukyoku":
                d = ev.get("deltas", [0] * 4)
                known = any(x != 0 for x in d)
                for i in range(4):
                    stats[i]["ryu"] += 1
                    if known:
                        stats[i]["ryu_known"] += 1
                        if d[i] > 0:
                            stats[i]["ryu_tenpai"] += 1
            elif t == "end_kyoku":
                for i in range(4):
                    stats[i]["riichi"] += cur[i]["riichi"]
                    stats[i]["naki"] += cur[i]["naki"]
                cur = None
    return our, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=os.path.expanduser(
        "~/Projects/better_mortal/runs/leanjax_eval"))
    ap.add_argument("--seed-lo", type=int, default=10000)
    ap.add_argument("--seed-hi", type=int, default=15999)
    ap.add_argument("--newer-than", type=float, default=0.0,
                    help="只取 mtime 晚于该 epoch 秒的文件。**几乎总是需要**:见下方注记")
    ap.add_argument("--expect-games", type=int, default=0,
                    help="预期局数;不符则报错。防的是下面这个静默污染")
    ap.add_argument("--dump", default=None, help="把累加量存 JSON,供跨快照对比")
    args = ap.parse_args()
    # ⚠️ 日志目录按 seed 命名,跨 run 复用。被后续 run **覆盖**的那些会丢(见 eval_dump.sh),
    # 而**没被覆盖**的那些会静默混进来 —— 2026-09-11 实测:seeds 10000-15999 取到 24,000 局,
    # 因为中间的 11000-13999 是 08-20 老 run 的残留,从没被任何后续评测覆盖过。
    # 座位分布检查抓不到它(每个 run 都是均匀轮转)。所以必须用 --newer-than 卡时间戳,
    # 并用 --expect-games 把预期局数写死。

    ours, theirs = blank(), blank()
    n_games = 0
    seat_hist = defaultdict(int)
    for p in sorted(glob.glob(os.path.join(args.log_dir, "*.json.gz"))):
        m = NAME_RE.search(os.path.basename(p))
        if not m:
            continue
        seed = int(m.group(1))
        if not (args.seed_lo <= seed <= args.seed_hi):
            continue
        if os.path.getmtime(p) < args.newer_than:
            continue
        our, st = scan_game(p)
        if our is None:
            continue
        n_games += 1
        seat_hist[our] += 1
        for k in ours:
            ours[k] += st[our][k]
            theirs[k] += sum(st[i][k] for i in range(4) if i != our)

    if not n_games:
        print("没有匹配的牌谱")
        return
    print(f"牌谱 {n_games:,} 局   我们的座位分布 {dict(sorted(seat_hist.items()))}")
    print(f"  (复式应四席均匀;不均匀说明取到的日志不是完整轮转)\n")
    if args.expect_games and n_games != args.expect_games:
        raise SystemExit(f"局数 {n_games:,} ≠ 预期 {args.expect_games:,} —— "
                         f"多半混进了未被覆盖的旧 run 日志,收紧 --newer-than")
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump({"n_games": n_games, "ours": ours, "theirs": theirs}, f,
                      ensure_ascii=False, indent=1)
        print(f"  累加量 → {args.dump}\n")

    def row(label, f_o, f_t, fmt="{:.2%}", better="low"):
        o, t = f_o(ours), f_t(theirs)
        d = o - t
        mark = ""
        if better != "-":
            worse = (d > 0) if better == "low" else (d < 0)
            mark = "  ← 差" if worse else "  ← 好"
        print(f"  {label:<16}{fmt.format(o):>10}{fmt.format(t):>12}"
              f"{fmt.format(d):>12}{mark}")

    print(f"  {'指标':<14}{'我们':>12}{'mortal_v4':>12}{'差':>12}")
    print("  " + "-" * 50)
    row("和了率", lambda s: s["win"] / s["kyoku"], lambda s: s["win"] / s["kyoku"],
        better="high")
    row("放铳率", lambda s: s["deal_in"] / s["kyoku"], lambda s: s["deal_in"] / s["kyoku"],
        better="low")
    row("立直率", lambda s: s["riichi"] / s["kyoku"], lambda s: s["riichi"] / s["kyoku"],
        better="-")
    row("副露率", lambda s: s["naki"] / s["kyoku"], lambda s: s["naki"] / s["kyoku"],
        better="-")
    row("自摸比例", lambda s: s["tsumo"] / max(s["win"], 1),
        lambda s: s["tsumo"] / max(s["win"], 1), better="-")
    print()
    row("平均和了", lambda s: s["win_pts"] / max(s["win"], 1),
        lambda s: s["win_pts"] / max(s["win"], 1), fmt="{:+,.0f}", better="high")
    row("平均放铳", lambda s: s["deal_in_pts"] / max(s["deal_in"], 1),
        lambda s: s["deal_in_pts"] / max(s["deal_in"], 1), fmt="{:,.0f}", better="low")
    print()
    if ours["ryu_known"]:
        row("流局听牌率", lambda s: s["ryu_tenpai"] / max(s["ryu_known"], 1),
            lambda s: s["ryu_tenpai"] / max(s["ryu_known"], 1), better="high")
    print(f"  流局 {ours['ryu']:,} 局,其中 deltas 全零(全听/全不听,不可分辨) "
          f"{ours['ryu'] - ours['ryu_known']:,}")
    print()
    # 每局期望收支:把攻守两侧折成点数,便于看差距主要由哪一侧贡献
    def per_kyoku(s):
        return (s["win_pts"] - s["deal_in_pts"]) / s["kyoku"]
    o, t = per_kyoku(ours), per_kyoku(theirs)
    print(f"  每局净收支(和了得点 − 放铳失点,不含立直棒/流局):")
    print(f"    我们 {o:+,.0f}   mortal_v4 {t:+,.0f}   差 {o - t:+,.0f} 点/局")
    wo = ours["win_pts"] / ours["kyoku"] - theirs["win_pts"] / theirs["kyoku"]
    do = -(ours["deal_in_pts"] / ours["kyoku"] - theirs["deal_in_pts"] / theirs["kyoku"])
    print(f"    其中 攻(和了侧) {wo:+,.0f} 点/局   守(放铳侧) {do:+,.0f} 点/局")


if __name__ == "__main__":
    main()
