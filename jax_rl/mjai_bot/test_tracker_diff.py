"""
R4 差分验证:tracker 从事件前缀重建的 obs vs replay_env 的 env 真值,逐决策点比对。

判据:planes/scalars 全等(shanten 用真值重算对齐 make_bc_dataset 的修复;
furiten 近似允许统计级小错配,单列报告)。
用法:PYTHONPATH=~/mahjax:../data_bridge python test_tracker_diff.py "<glob>" <n_games>
"""
from __future__ import annotations
import sys
import glob as globmod
import random
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_bridge"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mjai_parser import parse_game
from replay_env import replay_kyoku, ReplayReject, DECISION_TYPES
from obs_lean import observe_lean
from tracker import KyokuTracker, make_jax_helpers

import jax
from mahjax.red_mahjong.shanten import Shanten

obs_j = jax.jit(observe_lean)
shan_j = jax.jit(Shanten.number)
shan_fn, waits_fn = make_jax_helpers()

PLANE_NAMES = (
    ["hand>=1", "hand>=2", "hand>=3", "hand>=4", "red_in_hand", "drawn"]
    + [f"river_{i}" for i in range(4)]
    + [f"last_{i}" for i in range(4)]
    + [f"meld_{i}" for i in range(4)]
    + ["visible", "dora_ind"]
)


def synth_start(k) -> dict:
    return {
        "type": "start_kyoku", "bakaze": k.bakaze, "kyoku": k.kyoku_num,
        "honba": k.honba, "kyotaku": k.kyotaku, "oya": k.oya,
        "scores": k.scores or [25000] * 4, "dora_marker": k.dora_markers[0],
        "tehais": k.tehais,
    }


def main():
    pat = sys.argv[1]
    n_games = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    files = sorted(globmod.glob(pat))
    random.Random(7).shuffle(files)
    files = files[:n_games]

    stats = {
        "decisions": 0,
        "plane_mismatch": np.zeros(20, np.int64),
        "scalar_mismatch": np.zeros(26, np.int64),
    }
    first = {}

    for fp in files:
        g = parse_game(fp)
        for ki, k in enumerate(g.kyokus):
            raw_ix = [i for i, ev in enumerate(k.events) if ev["type"] in DECISION_TYPES]
            start_ev = synth_start(k)
            trackers = []
            for s in range(4):
                tr = KyokuTracker(s, shan_fn, waits_fn)
                tr.feed(start_ev)
                trackers.append(tr)
            fed = [0, 0, 0, 0]

            def probe(state, a, from_log, ptr, _k=k, _raw=raw_ix, _fp=fp, _ki=ki,
                      _trs=trackers, _fed=fed):
                cp = int(state.current_player)
                # 前缀时点规则:
                #  本回合型决策(dahai/reach/ankan/kakan/自摸和)= 该事件前的全部;
                #  响应型(pon/chi/daiminkan/荣和/隐式 PASS)= env 在下一摸牌前服务,
                #  前缀须切到触发牌(最近一张 dahai 或 kakan)为止。
                evs = _k.events
                if a == 85:  # KYUUSHU:合成动作不在决策表内,前缀=全部事件
                    from_log = False
                    prefix_end = len(evs)
                elif from_log:
                    dix = _raw[ptr - 1]
                    dev = evs[dix]
                    is_resp = dev["type"] in ("pon", "chi", "daiminkan") or (
                        dev["type"] == "hora" and dev["actor"] != dev["target"]
                    )
                    bound = dix
                    if not is_resp:
                        prefix_end = dix
                    else:
                        prefix_end = None
                else:
                    bound = _raw[ptr] if ptr < len(_raw) else len(evs)
                    prefix_end = None
                if prefix_end is None:
                    prefix_end = bound
                    for i in range(bound - 1, -1, -1):
                        if evs[i]["type"] in ("dahai", "kakan"):
                            prefix_end = i + 1
                            break
                tr = _trs[cp]
                for ev in evs[_fed[cp]:prefix_end]:
                    tr.feed(ev)
                _fed[cp] = max(_fed[cp], prefix_end)
                mine = tr.build_obs()
                envo = obs_j(state)
                env_p = np.asarray(envo["planes"])
                env_s = np.array(envo["scalars"])
                env_s[8] = float(shan_j(state.players.hand[cp])) / 6.0
                stats["decisions"] += 1
                dp = np.abs(mine["planes"] - env_p).max(axis=0) > 1e-5
                ds = np.abs(mine["scalars"] - env_s) > 1e-5
                stats["plane_mismatch"] += dp
                stats["scalar_mismatch"] += ds
                for j in np.flatnonzero(dp):
                    first.setdefault(f"P{j}:{PLANE_NAMES[j]}",
                                     f"{_fp.split('/')[-1]}#k{_ki} ptr={ptr} cp={cp}")
                for j in np.flatnonzero(ds):
                    first.setdefault(f"S{j}",
                                     f"{_fp.split('/')[-1]}#k{_ki} ptr={ptr} cp={cp} "
                                     f"mine={mine['scalars'][j]:.4f} env={env_s[j]:.4f}")
                if dp[18] and "P18_DETAIL" not in first:
                    d = mine["planes"][:, 18] - env_p[:, 18]
                    first["P18_DETAIL"] = (
                        f"types={np.flatnonzero(np.abs(d) > 1e-5).tolist()} "
                        f"delta={d[np.abs(d) > 1e-5].tolist()} "
                        f"recent={[e['type'] + str(e.get('actor', '')) for e in evs[max(0, prefix_end-6):prefix_end]]}"
                    )
                # 见逃真值信号:PASS 且 env 荣合法 → 与评测侧 cans.can_ron_agari 同语义
                if a == 84 and bool(np.asarray(state.legal_action_mask)[74]):
                    tr.note_ron_passed()

            try:
                replay_kyoku(k, on_decision=probe)
            except ReplayReject:
                pass

    n = stats["decisions"]
    print(f"decisions={n}")
    print("plane mismatch rates:")
    for j in range(20):
        c = stats["plane_mismatch"][j]
        if c:
            print(f"  P{j:2d} {PLANE_NAMES[j]:12s} {c:6d} ({c/n:.4%})  e.g. {first.get(f'P{j}:{PLANE_NAMES[j]}')}")
    print("scalar mismatch rates:")
    for j in range(26):
        c = stats["scalar_mismatch"][j]
        if c:
            print(f"  S{j:2d} {c:6d} ({c/n:.4%})  e.g. {first.get(f'S{j}')}")
    if "P18_DETAIL" in first:
        print("P18_DETAIL:", first["P18_DETAIL"])
    total_bad = stats["plane_mismatch"].sum() + stats["scalar_mismatch"].sum()
    print("ALL_MATCH" if total_bad == 0 else f"TOTAL_MISMATCH_CELLS={total_bad}")


if __name__ == "__main__":
    main()
