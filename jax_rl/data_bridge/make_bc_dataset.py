"""
R2 BC 数据集落盘:重放回路 × 双观测(lean 平面 + 上游 dict),同一决策点采集。

每个决策点存:
  planes  (34,20) uint8(×4 无损量化,值域恰为 0/0.25/0.5/0.75/1)
  scalars (26,)   float32
  hand(14) last_draw(1) action_history(3,200) shanten furiten scores(4)
  round honba kyotaku prevalent seat dora_indicators(5)   —— 上游 dict obs 原料
  action  uint8   legal_mask (87,) bool   from_log bool   game_id uint32
用法:PYTHONPATH=~/mahjax python make_bc_dataset.py "<glob>" <n_games> <out_dir>
"""
from __future__ import annotations
import sys
import glob as globmod
import random
import time
from pathlib import Path

import numpy as np
import jax

from mjai_parser import parse_game
from replay_env import replay_kyoku, ReplayReject

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from obs_lean import observe_lean

from mahjax.red_mahjong.observation import _observe_dict

SHARD = 100_000


LEAN_KEYS = ("planes", "scalars", "action", "legal_mask", "from_log", "game_id")
DICT_KEYS = ("hand", "last_draw", "action_history", "shanten", "furiten", "scores",
             "round", "honba", "kyotaku", "prevalent", "seat", "dora_indicators")


class Buf:
    def __init__(self, lean_only=False):
        self.lean_only = lean_only
        keys = LEAN_KEYS if lean_only else LEAN_KEYS + DICT_KEYS
        self.rows = {k: [] for k in keys}

    def __len__(self):
        return len(self.rows["action"])

    def add(self, ol, od, mask, a, from_log, gid):
        r = self.rows
        r["planes"].append(np.round(np.asarray(ol["planes"]) * 4).astype(np.uint8))
        r["scalars"].append(np.asarray(ol["scalars"], np.float32))
        r["action"].append(np.uint8(a))
        r["legal_mask"].append(mask)
        r["from_log"].append(from_log)
        r["game_id"].append(np.uint32(gid))
        if self.lean_only:
            return
        r["hand"].append(np.asarray(od["hand"], np.int8))
        r["last_draw"].append(np.int8(od["last_draw"]))
        r["action_history"].append(np.asarray(od["action_history"], np.int8))
        r["shanten"].append(np.int8(od["shanten_count"]))
        r["furiten"].append(bool(od["furiten"]))
        r["scores"].append(np.asarray(od["scores"], np.int16))
        r["round"].append(np.int8(od["round"]))
        r["honba"].append(np.int8(od["honba"]))
        r["kyotaku"].append(np.int8(od["kyotaku"]))
        r["prevalent"].append(np.int8(od["prevalent_wind"]))
        r["seat"].append(np.int8(od["seat_wind"]))
        r["dora_indicators"].append(np.asarray(od["dora_indicators"], np.int8))

    def flush(self, out_dir: Path, shard_ix: int):
        arrs = {k: np.stack(v) for k, v in self.rows.items()}
        path = out_dir / f"shard_{shard_ix:04d}.npz"
        np.savez_compressed(path, **arrs)
        n = len(self)
        self.__init__(self.lean_only)
        return path, n


def main():
    pat, n_games, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    lean_only = len(sys.argv) > 4 and sys.argv[4] == "lean"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(globmod.glob(pat))
    random.Random(1).shuffle(files)
    files = files[:n_games]

    obs_l = jax.jit(observe_lean)
    obs_d = jax.jit(_observe_dict)
    # ⚠️ verify_step 走 update_shanten=FALSE,state.shanten_current_player 在重放中
    # 恒为初始值 → obs 标量[8] 必须在采集点用当前玩家真实手牌重算(PPO/评测侧是真值,
    # 不修会造成 BC 与在线的特征分布错位)。
    from mahjax.red_mahjong.shanten import Shanten
    shan = jax.jit(Shanten.number)
    buf = Buf(lean_only)
    shard_ix = n_k = ok = total = 0
    rejects = 0
    t0 = time.time()
    for gid, fp in enumerate(files):
        g = parse_game(fp)
        for k in g.kyokus:
            n_k += 1

            def collect(state, a, from_log, _ptr=None, _gid=gid):
                mask = np.asarray(state.legal_action_mask)
                od = None if lean_only else obs_d(state)
                ol = obs_l(state)
                scal = np.array(ol["scalars"])
                scal[8] = float(shan(state.players.hand[int(state.current_player)])) / 6.0
                buf.add({"planes": ol["planes"], "scalars": scal}, od, mask, a, from_log, _gid)

            try:
                replay_kyoku(k, on_decision=collect)
                ok += 1
            except ReplayReject:
                rejects += 1
            if len(buf) >= SHARD:
                path, n = buf.flush(out, shard_ix)
                total += n
                shard_ix += 1
                print(f"[{time.time()-t0:6.0f}s] {path.name} +{n} (games {gid+1}/{len(files)})",
                      flush=True)
    if len(buf):
        path, n = buf.flush(out, shard_ix)
        total += n
        print(f"[{time.time()-t0:6.0f}s] {path.name} +{n}", flush=True)
    print(f"DONE games={len(files)} kyokus={n_k} ok={ok} rejects={rejects} "
          f"samples={total} {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
