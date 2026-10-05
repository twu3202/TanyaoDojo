"""A3 confirmation (10-02): is the first-order gain of the tuned model's flips real on games nobody has looked at?

On the step20 low-noise test (1000 games), the learning-curve runs (`a3_finetune.py --train-frac 0.25/0.5/1`) all
flip about 0.9 of the labelled pairs per game and gain about +0.74 pt/game there (95% CI ±0.63), the same for every
data fraction; push→fold flips carry most of it. That test set was used to choose the reported step, so it is
checked here on fresh games: the held-out seed groups (seed % 10 == 0) that `a3_finetune.py` never trains on,
minus the step20 test seeds and the first 400 held-out games its training-time check prints (the same ordering as
its `hold_keys`). None of these games was used for training, for choosing a step or for any reading so far.

    pick:  forward every model on the labelled pairs of those games; write the pairs any model flips
           (`flips.jsonl`, the input of riichi-eval `tools/label_noise.py --only`) and every pair's prediction
           (`pred.npz`, with the single-wall label from the label file)
    score: price each model's flips with the wall-averaged label from label_noise.py (logged wall + K re-dealt walls)

Pre-registered (written before the rollouts): the A3 first-order gain is confirmed  <=>  the 100% model's flips gain
> 0 pt/game on these games with the 95% seed-group CI above 0. Then the next test is the tuned model's strength in
full play; otherwise the +0.74 was noise and A3 stays closed. Per-kind numbers are descriptive only.

    python a3_confirm.py pick --labels L.jsonl --log-dir DIR --test-pairs P.jsonl --out DIR \
        --model base=B.pth --model f100=M.pth ...
    python a3_confirm.py score --out DIR --pairs DIR/noise/pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a3_finetune import PT_SCALE, index_games, load_labels, read_game, iter_decisions, env  # noqa: E402


def load_model(path, device):
    from model import Brain, DQN  # Mortal/mortal/model.py
    state = torch.load(path, weights_only=True, map_location="cpu")
    cfg = state["config"]
    version = cfg["control"].get("version", 1)
    b = Brain(version=version, conv_channels=cfg["resnet"]["conv_channels"], num_blocks=cfg["resnet"]["num_blocks"])
    q = DQN(version=version)
    b.load_state_dict(state["mortal"])
    q.load_state_dict(state["current_dqn"])
    return b.to(device).eval(), q.to(device).eval()


def confirm_keys(labels, games, test_pairs, holdout_games):
    test_seeds = {json.loads(ln)["seed"] for ln in open(test_pairs)}
    keys = sorted(k for k in games if k in labels and k[0] not in test_seeds)
    hold = [k for k in keys if k[0] % 10 == 0]
    return hold[holdout_games:]


def pick(args):
    env.setup()
    device = torch.device(args.device)
    labels, _ = load_labels(args.labels)
    games = index_games(args.log_dir)
    keys = confirm_keys(labels, games, args.test_pairs, args.holdout_games)
    seeds = sorted({k[0] for k in keys})
    full = {s for s in seeds if sum((s, r) in games for r in "abcd") == 4}
    if args.max_groups:
        full = set(sorted(full)[: args.max_groups])
    keys = [k for k in keys if k[0] in full]
    models = {}
    for a in args.models:
        name, path = a.split("=", 1)
        models[name] = load_model(path, device)
    print(f"{len(keys):,} confirmation games ({len(full):,} seed groups); models {list(models)}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    meta, preds = defaultdict(list), defaultdict(list)

    def flush(obs, mask, ours, alt, bs=512):
        for s in range(0, len(obs), bs):
            o = torch.from_numpy(np.stack(obs[s:s + bs])).to(device).float()
            m = torch.from_numpy(np.stack(mask[s:s + bs])).to(device)
            i = np.arange(len(o))
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                for name, (b, q) in models.items():
                    qv = q(b(o), m).float().cpu().numpy()
                    preds[name].append((qv[i, alt[s:s + bs]] - qv[i, ours[s:s + bs]]) / PT_SCALE)

    for gi in range(0, len(keys), args.chunk):
        obs, mask, ours, alt = [], [], [], []
        for key in keys[gi:gi + args.chunk]:
            g = read_game(games[key])
            seat = next(i for i, n in enumerate(g.names) if n != args.ref_name)
            lab = labels[key]
            for d in iter_decisions(g, [seat], select=lambda r, pos, st: (r, pos) in lab):
                if d.bucket == "forced":
                    continue
                for o, a, dg, kd in lab.get((d.r, d.pos), []):
                    if d.logged != o or not d.mask[a]:
                        continue
                    obs.append(d.obs.astype(np.float16))
                    mask.append(d.mask)
                    ours.append(o)
                    alt.append(a)
                    for n, v in (("seed", key[0]), ("rot", key[1]), ("r", d.r), ("pos", d.pos), ("kind", kd),
                                 ("dlab", dg)):
                        meta[n].append(v)
        flush(obs, mask, np.array(ours), np.array(alt))
        print(f"  [{min(gi + args.chunk, len(keys)):,}/{len(keys):,}] {len(meta['seed']):,} pairs", flush=True)

    P = {n: np.concatenate(v).astype(np.float32) for n, v in preds.items()}
    M = {n: np.array(v) for n, v in meta.items()}
    np.savez(os.path.join(args.out, "pred.npz"), groups=np.array(sorted(full)), names=np.array(list(P)), **M,
             **{f"pred_{n}": v for n, v in P.items()})
    any_flip = np.zeros(len(M["seed"]), bool)
    for n, p in P.items():
        print(f"  {n:<8} flips/game {(p > 0).sum() / len(keys):.3f}", flush=True)
        any_flip |= p > 0
    with open(os.path.join(args.out, "flips.jsonl"), "w") as f:
        for i in np.flatnonzero(any_flip):
            f.write(json.dumps(dict(seed=int(M["seed"][i]), rot=str(M["rot"][i]), r=int(M["r"][i]),
                                    pos=int(M["pos"][i]), kind=str(M["kind"][i]))) + "\n")
    print(f"-> {args.out}/flips.jsonl: {any_flip.sum():,} pairs flipped by any model", flush=True)


def score(args):
    z = np.load(os.path.join(args.out, "pred.npz"), allow_pickle=True)
    groups = z["groups"]
    G = 4 * len(groups)
    seed, kind, dlab = z["seed"], z["kind"], z["dlab"]
    key = list(zip(seed.tolist(), z["rot"].tolist(), z["r"].tolist(), z["pos"].tolist(), kind.tolist()))
    tgt = {}
    for ln in open(args.pairs):
        x = json.loads(ln)
        v = np.asarray(x["v"], np.float64)
        tgt[(x["seed"], x["rot"], x["r"], x["pos"], x["kind"])] = (v[:, 1] - v[:, 0]).mean()
    t = np.array([tgt.get(k, np.nan) for k in key])
    idx = [np.flatnonzero(seed == s) for s in groups]     # seed groups without labelled pairs count as 0

    def gain(sel, val):
        y = np.array([np.nansum(val[i][sel[i]]) / 4 for i in idx])
        return y.mean(), 1.96 * y.std(ddof=1) / np.sqrt(len(y))

    print(f"{G:,} confirmation games, {len(groups):,} seed groups, {len(key):,} labelled pairs, "
          f"{len(tgt):,} priced with re-dealt walls")
    for n in z["names"]:
        p = z[f"pred_{n}"]
        fl = p > 0
        miss = int(np.isnan(t[fl]).sum())
        g, ci = gain(fl, t)
        g1, ci1 = gain(fl, dlab)
        print(f"\n{n}: flips/game {fl.sum() / G:.3f}  gain/game {g:+.3f} ± {ci:.3f} (wall-averaged)   "
              f"{g1:+.3f} ± {ci1:.3f} (single wall, label file)   unpriced flips {miss}")
        for kd in sorted(set(kind.tolist())):
            sel = fl & (kind == kd)
            if sel.any():
                gk, cik = gain(sel, t)
                print(f"   {kd:<12} flips/game {sel.sum() / G:.3f}  gain/game {gk:+.3f} ± {cik:.3f}  "
                      f"per flip {np.nanmean(t[sel]):+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("pick", "score"))
    ap.add_argument("--model", action="append", dest="models", default=[], help="NAME=path.pth (pick; repeat)")
    ap.add_argument("--labels", action="append")
    ap.add_argument("--log-dir", action="append")
    ap.add_argument("--test-pairs")
    ap.add_argument("--holdout-games", type=int, default=400)
    ap.add_argument("--ref-name", default="mortal-v4")
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--max-groups", type=int, default=0, help="smoke test: first N seed groups only")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", help="label_noise.py pairs.jsonl (score)")
    args = ap.parse_args()
    pick(args) if args.mode == "pick" else score(args)


if __name__ == "__main__":
    main()
