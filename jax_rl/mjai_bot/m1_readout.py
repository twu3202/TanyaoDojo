"""M1 mechanism readout (10-03): did the calibrated anchor unlock the policy, and are its new choices better?

For each snapshot, against tpt 408k (the start, whose actions were logged in these games):
  1. on the step20 low-noise test pairs (1000 A3 label games, push/fold and riichi/dama pairs, target = label
     averaged over 8 walls): correlation and calibration slope of the predicted margin, flips per game (the
     snapshot now prefers the alternative to the logged 408k action) and their gain per game, seed-group CI;
  2. on every non-forced decision of the first N of those games: how often the greedy action differs from 408k's,
     and the median top-2 Q margin in sacred pt (the compression the anchor is meant to produce).
Pre-registered reading: patches/calibrated_anchor/README.md.

    python m1_readout.py --labels L --log-dir DIR --test-pairs P --games 200 --model 408k=A.pth --model m1_448k=B.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a3_finetune import PT_SCALE, build_test, index_games, iter_decisions, load_labels, read_game, test_eval  # noqa: E402
from a3_confirm import load_model  # noqa: E402
from riichi_eval import env  # noqa: E402


def all_decisions(games, keys, ref_name):
    obs, mask = [], []
    for key in keys:
        g = read_game(games[key])
        seat = next(i for i, n in enumerate(g.names) if n != ref_name)
        for d in iter_decisions(g, [seat]):
            if d.bucket != "forced" and d.mask.sum() >= 2:
                obs.append(d.obs.astype(np.float16))
                mask.append(d.mask)
    return obs, np.stack(mask)


def qvals(model, obs, mask, device, bs=512):
    b, q = model
    out = []
    with torch.no_grad():
        for i in range(0, len(obs), bs):
            o = torch.from_numpy(np.stack(obs[i:i + bs])).to(device).float()
            m = torch.from_numpy(mask[i:i + bs]).to(device)
            out.append(q(b(o), m).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--test-pairs", required=True)
    ap.add_argument("--games", type=int, default=200, help="games for the policy-change / margin readout")
    ap.add_argument("--model", action="append", required=True, help="NAME=path; the first one is the reference (408k)")
    ap.add_argument("--ref-name", default="mortal-v4")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    env.setup()
    device = torch.device(args.device)
    labels, _ = load_labels([args.labels])
    games = index_games([args.log_dir])
    rows = [json.loads(ln) for ln in open(args.test_pairs)]
    T, miss = build_test(rows, labels, games, args.ref_name)
    keys = sorted({(r["seed"], r["rot"]) for r in rows})[: args.games]
    obs, mask = all_decisions(games, [k for k in keys if k in games], args.ref_name)
    print(f"test pairs {len(T['tgt']):,} ({miss} unmatched); policy readout on {len(obs):,} decisions of {len(keys)} games")
    ref_arg = None
    print(f"\n{'model':<12}{'corr':>7}{'slope':>8}{'flips/g':>9}{'gain/g':>16}{'  greedy != ref':>16}{'median top-2 (pt)':>19}")
    for spec in args.model:
        name, path = spec.split("=", 1)
        model = load_model(path, device)
        m, _ = test_eval(*model, T, device)
        model[0].eval()                     # test_eval leaves the brain in train mode (BN frozen)
        q = np.where(mask, qvals(model, obs, mask, device), -np.inf)
        arg = q.argmax(1)
        top2 = np.sort(q, 1)[:, -2:]
        marg = np.median(top2[:, 1] - top2[:, 0]) / PT_SCALE
        if ref_arg is None:
            ref_arg = arg
        a = m["all"]
        print(f"{name:<12}{a['corr']:>7.3f}{a['slope']:>8.3f}{a['flips_per_game']:>9.2f}"
              f"{a['gain_per_game']:>+9.3f} ± {a['gain_ci']:.3f}{(arg != ref_arg).mean():>15.2%}{marg:>19.2f}")
        print(f"{'':<12}by kind corr: " + " ".join(f"{k} {v:.3f}" for k, v in m["corr_by_kind"].items()))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
