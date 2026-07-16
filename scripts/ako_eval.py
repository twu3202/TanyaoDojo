#!/usr/bin/env python3
"""Cross-lineage sentinel eval: our checkpoint (challenger) vs 3x akochan, duplicate arena.
akochan is an independent (pre-NN, EV-simulation) lineage — gains vs v4 that don't
transfer here indicate opponent-specific overfitting ("Mortal killer" alarm).

Usage (server):
  AKOCHAN_DIR=$HOME/Projects/better_mortal/vendor/akochan \
  AKOCHAN_TACTICS=$HOME/Projects/better_mortal/vendor/akochan/tactics_ours.json \
  LD_LIBRARY_PATH=$AKOCHAN_DIR OMP_NUM_THREADS=2 \
  MORTAL_DIR=$HOME/Projects/better_mortal/Mortal/mortal \
  python ako_eval.py --ckpt <path.pth> --device cpu --iters 5 --seeds-per-iter 10
"""
import sys, os, argparse
MORTAL_DIR = os.environ.get("MORTAL_DIR", "/Users/r/HMM/Better_mortal/Mortal/mortal")
sys.path.insert(0, MORTAL_DIR)
os.chdir(MORTAL_DIR)  # so libriichi.so is found

import numpy as np
import torch
from model import Brain, DQN
from engine import MortalEngine
from libriichi.arena import OneVsThree

JUN_PT = np.array([90, 45, 0, -135])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--seeds-per-iter", type=int, default=10)  # games = 4x
    ap.add_argument("--seed-key", type=int, default=20260716)
    args = ap.parse_args()

    state = torch.load(args.ckpt, weights_only=True, map_location="cpu")
    cfg = state["config"]
    version = cfg["control"].get("version", 1)
    cc, nb = cfg["resnet"]["conv_channels"], cfg["resnet"]["num_blocks"]
    brain = Brain(version=version, conv_channels=cc, num_blocks=nb).eval()
    dqn = DQN(version=version).eval()
    brain.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    chal = MortalEngine(brain, dqn, is_oracle=False, version=version,
                        device=torch.device(args.device),
                        enable_rule_based_agari_guard=True, name="challenger")
    total_games = args.iters * args.seeds_per_iter * 4
    print(f"challenger: {args.ckpt} (v{version} {cc}ch x {nb}blk) vs 3x akochan, "
          f"{total_games} games, seed_key={args.seed_key}", flush=True)
    print(f"AKOCHAN_DIR={os.environ.get('AKOCHAN_DIR')} TACTICS={os.environ.get('AKOCHAN_TACTICS')}", flush=True)

    per_iter_pt = []
    tot = np.zeros(4)
    seed_start = 10000
    for i in range(args.iters):
        seed = seed_start + i * args.seeds_per_iter
        env = OneVsThree(disable_progress_bar=True, log_dir=None)
        rk = np.array(env.py_vs_ako(engine=chal, seed_start=(seed, args.seed_key),
                                    seed_count=args.seeds_per_iter))
        tot += rk
        pt = rk @ JUN_PT / rk.sum()
        per_iter_pt.append(pt)
        print(f"iter {i}: {rk.tolist()} rank={rk @ np.arange(1,5)/rk.sum():.4f} pt={pt:+.3f}", flush=True)
    n = tot.sum()
    avg_rank = tot @ np.arange(1, 5) / n
    avg_pt = tot @ JUN_PT / n
    ci = 1.96*np.std(per_iter_pt, ddof=1)/np.sqrt(len(per_iter_pt)) if len(per_iter_pt) > 1 else float("nan")
    print(f"=== AKO SENTINEL: dist={tot.astype(int).tolist()} avg_rank={avg_rank:.4f} "
          f"avg_pt={avg_pt:+.3f} ± {ci:.3f} (n={int(n)}) ===", flush=True)
    print("pt>0 => our model stronger than akochan", flush=True)

if __name__ == "__main__":
    main()
