"""
R4 评测入口:LeanJax(挑战者,mjai-log 引擎)vs Mortal v4(冠军)1v3 复式对局。

神圣协议参数(与价值线 -2.50/-1.96 系列直接可比):
  seed_key=20260711, seed_start=10000 起连续铺,pt=[90,45,0,-135]。
用法(WSL,Mortal .venv):
  JAX_PLATFORMS=cpu python run_eval.py <params.pkl> [--games 100] [--seed-start 10000]
    [--iters 1]  # 每 iter 跑 games 局,seed 自动续铺
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

# 服务器(/home/twu)与本机(/home/r)路径不同:用环境变量覆盖,默认保持本机不变
MORTAL_DIR = os.environ.get("MORTAL_DIR", "/home/r/Projects/better_mortal/Mortal/mortal")
sys.path.insert(0, MORTAL_DIR)
import prelude  # noqa: F401  (logging/torch 前置)
import numpy as np
import torch
from model import Brain, DQN
from engine import MortalEngine  # Mortal 侧引擎(champion)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
# LeanJaxEngine 延迟到真用时才导入:它拖进 jax/mahjax,而 Mortal 格式挑战者只需要 torch+libriichi,
# 服务器上的 Mortal venv 没装 jax。

from libriichi.arena import OneVsThree

PTS = np.array([90.0, 45.0, 0.0, -135.0])


def load_champion(state_file: str, device: str, amp: str = "auto") -> MortalEngine:
    state = torch.load(state_file, weights_only=True, map_location="cpu")
    cfg = state["config"]
    version = cfg["control"].get("version", 1)
    brain = Brain(version=version, conv_channels=cfg["resnet"]["conv_channels"],
                  num_blocks=cfg["resnet"]["num_blocks"]).eval()
    dqn = DQN(version=version).eval()
    brain.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    return MortalEngine(
        brain, dqn, is_oracle=False, version=version,
        device=torch.device(device),
        # auto = 历史行为(GPU 上开 AMP)。off 用于验证/加速:GPU fp32 应与 CPU fp32 逐局一致
        enable_amp=(device != "cpu") if amp == "auto" else (amp == "on"),
        enable_rule_based_agari_guard=True, name="mortal-v4",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params")
    ap.add_argument("--games", type=int, default=100, help="每 iter 局数(须为 4 的倍数)")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=10000)
    ap.add_argument("--seed-key", type=int, default=20260711)
    ap.add_argument("--champion",
                    default=os.environ.get("MORTAL_V4", "/home/r/Projects/better_mortal/baseline/mortal_v4.pth"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--amp", default="auto", choices=("auto", "on", "off"),
                    help="Mortal 引擎的 AMP。auto=GPU 开/CPU 关(历史口径);off=GPU 上也走 fp32")
    ap.add_argument("--log-dir", default="/home/r/Projects/better_mortal/runs/leanjax_eval")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--obs", default="lean", choices=("lean", "v2"))
    ap.add_argument("--challenger-type", default="leanjax", choices=("leanjax", "mortal"),
                    help="mortal = 价值线 Mortal 格式权重(v11/rl1_best 等),与冠军同引擎同动作空间")
    ap.add_argument("--challenger-name", default=None,
                    help="写进牌谱 names 的名字;必须 != mortal-v4,否则 diag_gap 找不到挑战者座位")
    args = ap.parse_args()

    cham = load_champion(args.champion, args.device, args.amp)
    if args.challenger_type == "mortal":
        chal = load_champion(args.params, args.device, args.amp)
        chal.name = args.challenger_name or "challenger"
    else:
        from jax_engine import LeanJaxEngine
        chal = LeanJaxEngine(args.params, name=args.challenger_name or "leanjax",
                             channels=args.channels, blocks=args.blocks, obs=args.obs)
    assert chal.name != cham.name, "挑战者与冠军同名 -> 牌谱里分不出座位"
    seeds_per_iter = args.games // 4
    total = np.zeros(4, np.int64)
    t0 = time.time()
    for i in range(args.iters):
        seed = args.seed_start + i * seeds_per_iter
        env = OneVsThree(disable_progress_bar=True, log_dir=args.log_dir)
        rankings = np.array(env.py_vs_py(
            challenger=chal, champion=cham,
            seed_start=(seed, args.seed_key), seed_count=seeds_per_iter,
        ))
        total += rankings
        n = total.sum()
        avg_rank = total @ np.arange(1, 5) / n
        avg_pt = total @ PTS / n
        # 每局 pt 方差 → 95% CI
        per_game = np.repeat(PTS, total)
        ci = 1.96 * per_game.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        print(f"[iter {i}] n={n} rankings={total.tolist()} avg_rank={avg_rank:.4f} "
              f"avg_pt={avg_pt:+.3f}±{ci:.3f} "
              f"fallback={getattr(chal, 'fallback_count', 0)}/{getattr(chal, 'decision_count', 0)} "
              f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
