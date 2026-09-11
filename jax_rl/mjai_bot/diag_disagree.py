"""逐决策分歧图:在同一条牌谱轨迹上,同时问参考模型(mortal_v4)与被测模型,按决策类型分桶。

**动机**(2026-09-11)。立直线的攻守诊断(diag_gap.py)只看得到聚合结果:JAX RL agent
的缺口几乎全在进攻侧。要决定下一步是"换 RL 方法"还是"向 v4 学判断",需要知道被测模型
**在哪一类决策上**与 v4 分道,以及 v4 自己认为这些分歧值多少。

**为什么只能在 Mortal 栈上做**。JAX 线的 agent 用 87 维动作空间和 36 平面观测,与 v4
对不齐;价值线的 v11/rl1_best 与 v4 用**同一个 libriichi v4 编码 (1012,34) 与同一个
46 维动作空间**,所以可以把同一个 PlayerState 的编码同时喂给两边,argmax 直接可比。

**做法**。重放牌谱:对指定座位逐事件 `PlayerState.update`;凡 `can_act` 的时点就是一个
决策点,当场 `encode_obs(4, False)` 取观测与掩码;再从后续事件还原该座位**实际打出**
的动作。批量过两个网络,记录:
  · 参考 vs 实际、被测 vs 实际 —— **阳性对照**:模型在自己的轨迹上重算 argmax,
    必须几乎全等于牌谱里它实际打的,否则重放/编码有错,后面的数不可信;
  · 参考 vs 被测 的分歧率;
  · 分歧处参考模型的 Q 差 `Q_ref[a_ref] − Q_ref[a_sub]`(参考模型眼中被测选择的损失,
    单位是 Mortal 的训练奖励尺度,只用于分桶排序,**不是**顺位点)。

**决策类型**(按优先级归桶,一个决策点只进一个桶):
  agari(可和) > riichi(可立直) > call(可吃碰明杠,响应他家打牌) > self_kan(可暗杠/加杠)
  > ryukyoku(九种九牌) > discard(纯打牌,≥2 张可选) > forced(只有 1 个合法动作,不计分歧)

**已知局限**(均在输出中单独可见,不混进主结论):
  · v4 评测时开 `enable_rule_based_agari_guard`,而 `rule_based_agari` 未导出到 Python,
    这里无法复刻 → agari 桶的阳性对照可能略低于其它桶,这是预期的;
  · 他家抢先荣和/碰时,本座"想碰"的意图不在牌谱里,只能记为"过" → call 桶有小的向下偏差;
  · 杠的第二段选牌(at_kan_select)不单独计。

用法:
  python diag_disagree.py --log-dir ~/Projects/better_mortal/runs/eval_rl1_best \\
      --seat-name rl1_best --ref .../mortal_v4.pth --sub .../rl1_best_155k.pth \\
      --expect-games 4000 --device cuda:0
阳性对照专用(不需要被测模型的牌谱):--seat-name mortal-v4 --sub 同 --ref
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

MORTAL_DIR = "/home/r/Projects/better_mortal/Mortal/mortal"
sys.path.insert(0, MORTAL_DIR)
import prelude  # noqa: F401,E402
import torch  # noqa: E402
from model import Brain, DQN  # noqa: E402
from libriichi.state import PlayerState  # noqa: E402

VERSION = 4
NAME_RE = re.compile(r"(\d+)_(\d+)_([a-d])\.json\.gz$")
HONORS = ["E", "S", "W", "N", "P", "F", "C"]
RED = {"5mr": 34, "5pr": 35, "5sr": 36}
ACT_EVENTS = {"dahai", "reach", "chi", "pon", "daiminkan", "ankan", "kakan", "hora", "ryukyoku"}
TURN_EVENTS = {"tsumo", "end_kyoku", "end_game", "start_kyoku"}
BUCKETS = ["agari", "riichi", "call", "self_kan", "ryukyoku", "discard", "forced"]


def tile_idx(pai: str) -> int:
    if pai in RED:
        return RED[pai]
    if pai in HONORS:
        return 27 + HONORS.index(pai)
    return int(pai[0]) - 1 + {"m": 0, "p": 9, "s": 18}[pai[1]]


def rank(pai: str) -> int:
    return int(pai[0]) if pai[0].isdigit() else -1


def event_to_action(ev: dict) -> int:
    t = ev["type"]
    if t == "dahai":
        return tile_idx(ev["pai"])
    if t == "reach":
        return 37
    if t == "chi":
        called = rank(ev["pai"])
        others = sorted(rank(x) for x in ev["consumed"])
        if called < others[0]:
            return 38
        if called < others[1]:
            return 39
        return 40
    if t == "pon":
        return 41
    if t in ("daiminkan", "ankan", "kakan"):
        return 42
    if t == "hora":
        return 43
    if t == "ryukyoku":
        return 44
    raise ValueError(t)


def bucket_of(mask: np.ndarray, cans) -> str:
    if mask.sum() <= 1:
        return "forced"
    if mask[43]:
        return "agari"
    if mask[37]:
        return "riichi"
    if cans.can_chi or cans.can_pon or cans.can_daiminkan:
        return "call"
    if cans.can_ankan or cans.can_kakan:
        return "self_kan"
    if mask[44]:
        return "ryukyoku"
    return "discard"


def load_model(path: str, device: torch.device):
    st = torch.load(path, weights_only=True, map_location="cpu")
    cfg = st["config"]
    ver = cfg["control"].get("version", 1)
    assert ver == VERSION, f"{path}: version={ver},本工具只支持 v4 编码"
    brain = Brain(version=ver, conv_channels=cfg["resnet"]["conv_channels"],
                  num_blocks=cfg["resnet"]["num_blocks"]).eval()
    dqn = DQN(version=ver).eval()
    brain.load_state_dict(st["mortal"])
    dqn.load_state_dict(st["current_dqn"])
    return brain.to(device), dqn.to(device)


@torch.inference_mode()
def qvals(model, obs: np.ndarray, mask: np.ndarray, device) -> np.ndarray:
    brain, dqn = model
    o = torch.as_tensor(obs, device=device)
    m = torch.as_tensor(mask, device=device)
    return dqn(brain(o), m).float().cpu().numpy()      # fp32:不开 autocast,贴近 CPU 评测的 argmax


def replay_game(path: str, seat_name: str):
    """返回 [(obs, mask, bucket, logged_action)],以及 None 或错误说明。"""
    events = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    names = events[0].get("names", [])
    if seat_name not in names:
        return [], f"座位名 {seat_name} 不在 {names}"
    me = names.index(seat_name)
    ps = PlayerState(me)
    out = []
    for i, ev in enumerate(events):
        cans = ps.update(json.dumps(ev))
        if not cans.can_act:
            continue
        obs, mask = ps.encode_obs(VERSION, False)
        mask = np.asarray(mask, bool)
        # 还原实际动作:往后找第一个"动作事件或轮转事件"
        logged = None
        j = i + 1
        while j < len(events):
            e2 = events[j]
            t2 = e2.get("type")
            if t2 == "hora":                         # 可能一炮多响:连续 hora 里有我就算我和了
                k = j
                while k < len(events) and events[k].get("type") == "hora":
                    if int(events[k]["actor"]) == me:
                        logged = 43
                    k += 1
                if logged is None:
                    logged = 45 if cans.can_pass else None
                break
            if t2 in ACT_EVENTS:
                if int(e2.get("actor", -1)) == me:
                    logged = event_to_action(e2)
                elif t2 == "ryukyoku" and "actor" not in e2 and mask[44]:
                    # 九种九牌:mjai 的 ryukyoku 事件不带 actor。只在本座此刻有 44 可选时才归给它;
                    # 否则这是荒牌流局(本座刚放弃了对最后一张的响应),仍记为"过"
                    logged = 44
                elif cans.can_pass:
                    logged = 45
                break
            if t2 in TURN_EVENTS:
                logged = 45 if cans.can_pass else None
                break
            j += 1
        if logged is None or not mask[logged]:
            logged = -1                               # 还原失败,只计入"未还原"
        out.append((np.asarray(obs, np.float32), mask, bucket_of(mask, cans), logged))
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--seat-name", required=True)
    ap.add_argument("--ref", default="/home/r/Projects/better_mortal/baseline/mortal_v4.pth")
    ap.add_argument("--sub", required=True)
    ap.add_argument("--seed-lo", type=int, default=0)
    ap.add_argument("--seed-hi", type=int, default=10**9)
    ap.add_argument("--newer-than", type=float, default=0.0)
    ap.add_argument("--expect-games", type=int, default=0)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    dev = torch.device(args.device)
    ref = load_model(args.ref, dev)
    same = os.path.realpath(args.ref) == os.path.realpath(args.sub)
    sub = ref if same else load_model(args.sub, dev)

    files = []
    for p in sorted(glob.glob(os.path.join(args.log_dir, "*.json.gz"))):
        m = NAME_RE.search(os.path.basename(p))
        if not m or not (args.seed_lo <= int(m.group(1)) <= args.seed_hi):
            continue
        if os.path.getmtime(p) < args.newer_than:
            continue
        files.append(p)
    if args.expect_games and len(files) != args.expect_games:
        raise SystemExit(f"牌谱 {len(files):,} 局 ≠ 预期 {args.expect_games:,}")
    if args.max_games:
        files = files[:args.max_games]

    S = defaultdict(lambda: dict(n=0, unmapped=0, ref_ok=0, sub_ok=0, dis=0, qgap=0.0,
                                 ref_riichi=0, sub_riichi=0, ref_pass=0, sub_pass=0))
    n_games, bad = 0, 0
    for gi, p in enumerate(files):
        try:
            recs, err = replay_game(p, args.seat_name)
        except Exception as e:                        # noqa: BLE001
            recs, err = [], f"{type(e).__name__}: {e}"
        if err:
            bad += 1
            if bad <= 3:
                print(f"  ✗ {os.path.basename(p)}: {err}")
            continue
        n_games += 1
        if not recs:
            continue
        obs = np.stack([r[0] for r in recs])
        mask = np.stack([r[1] for r in recs])
        q_ref = qvals(ref, obs, mask, dev)
        q_sub = q_ref if same else qvals(sub, obs, mask, dev)
        a_ref = q_ref.argmax(-1)
        a_sub = q_sub.argmax(-1)
        for k, (_, _, b, logged) in enumerate(recs):
            s = S[b]
            s["n"] += 1
            if logged < 0:
                s["unmapped"] += 1
            else:
                s["ref_ok"] += int(a_ref[k] == logged)
                s["sub_ok"] += int(a_sub[k] == logged)
            if a_ref[k] != a_sub[k]:
                s["dis"] += 1
                s["qgap"] += float(q_ref[k, a_ref[k]] - q_ref[k, a_sub[k]])
            s["ref_riichi"] += int(a_ref[k] == 37)
            s["sub_riichi"] += int(a_sub[k] == 37)
            s["ref_pass"] += int(a_ref[k] == 45)
            s["sub_pass"] += int(a_sub[k] == 45)
        if (gi + 1) % 500 == 0:
            print(f"  [{gi + 1}/{len(files)}]", flush=True)

    print(f"\n牌谱 {n_games:,} 局(重放失败 {bad})  座位={args.seat_name}")
    print(f"ref={os.path.basename(args.ref)}  sub={os.path.basename(args.sub)}"
          f"{'  (同一模型:阳性对照模式)' if same else ''}\n")
    print(f"  {'桶':<9}{'决策点':>9}{'未还原':>7}{'ref=实际':>9}{'sub=实际':>9}"
          f"{'分歧率':>8}{'分歧均Q差':>10}{'Q差/局':>9}")
    print("  " + "-" * 70)
    tot_gap = 0.0
    for b in BUCKETS:
        if b not in S:
            continue
        s = S[b]
        mapped = max(s["n"] - s["unmapped"], 1)
        gap_per_game = s["qgap"] / max(n_games, 1)
        tot_gap += gap_per_game
        print(f"  {b:<9}{s['n']:>9,}{s['unmapped']:>7,}{s['ref_ok'] / mapped:>9.2%}"
              f"{s['sub_ok'] / mapped:>9.2%}{s['dis'] / max(s['n'], 1):>8.2%}"
              f"{s['qgap'] / max(s['dis'], 1):>10.4f}{gap_per_game:>9.4f}")
    print(f"\n  合计 Q差/局 = {tot_gap:.4f}(参考模型眼中被测模型每局损失,仅作分桶排序)")
    if "riichi" in S:
        s = S["riichi"]
        print(f"  立直桶:ref 选立直 {s['ref_riichi'] / max(s['n'], 1):.2%},"
              f"sub 选立直 {s['sub_riichi'] / max(s['n'], 1):.2%}")
    if "call" in S:
        s = S["call"]
        print(f"  鸣牌桶:ref 选过 {s['ref_pass'] / max(s['n'], 1):.2%},"
              f"sub 选过 {s['sub_pass'] / max(s['n'], 1):.2%}")
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump({"n_games": n_games, "bad": bad, "seat": args.seat_name,
                       "ref": args.ref, "sub": args.sub, "buckets": dict(S)}, f,
                      ensure_ascii=False, indent=1)
        print(f"  → {args.dump}")


if __name__ == "__main__":
    main()
