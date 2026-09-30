"""A3: fine-tune a Mortal-format model on rollout-priced action pairs, anchored to itself.

Labels come from riichi-eval's policy probe (`tools/policy_probe.py --kinds push2fold,riichi2dama,fold2push,
dama2riichi`): at a decision where the model took action `ours`, a one-step rollout (the alternative `alt`
played once, then the same model continuing in its seat, opponents unchanged, true wall) prices

    d_grp = value(alt) − value(ours)       (kyoku horizon, GRP expected placement pt, sacred pt units)

which is a paired sample of Q^π(s, alt) − Q^π(s, ours) for the model's own policy π. Training pulls the
model's own margin towards it:

    pair   = ((Q(s,alt) − Q(s,ours)) − scale·d_grp)²          on labelled states
    anchor = mean over legal a of (Q(s,a) − Q_base(s,a))²      on labelled + random unlabelled states
    loss   = pair + λ·anchor

`scale` converts sacred pt to the Q units the model was trained in (online pts [2.390, 1.195, 0, −3.585]
= 0.02656 × [90, 45, 0, −135]).

The label games must NOT be sacred-protocol eval games (seed_key 20260711): generate them on another key.
Held-out seeds (seed % 10 == 0) give an honest first-order check before any game is played: the summed
d_grp of the held-out pairs the tuned model now flips to `alt`, per game.

    python a3_finetune.py --labels L.jsonl --log-dir DIR --base tpt_s408000.pth --out a3.pth \
        [--steps 6000 --batch 256 --lr 1e-5 --anchor 1.0]

Needs riichi-eval on the path (RIICHI_EVAL_SRC, default ~/riichi-eval/src) plus its env vars
(MORTAL_DIR, RIICHI_EVAL_LIBRIICHI).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

sys.path.insert(0, os.environ.get("RIICHI_EVAL_SRC", os.path.expanduser("~/riichi-eval/src")))
from riichi_eval import env  # noqa: E402
from riichi_eval.logs import NAME_RE, read_game  # noqa: E402
from riichi_eval.replay import iter_decisions  # noqa: E402

PT_SCALE = 2.390 / 90


def load_labels(paths):
    """(seed, rot) -> {(r, pos): [(ours, alt, d_grp, kind)]}, deduplicated per (state, alt)."""
    out = defaultdict(lambda: defaultdict(list))
    seen, n = set(), 0
    for p in paths:
        with open(p) as f:
            for ln in f:
                x = json.loads(ln)
                if x.get("d_grp") is None:
                    continue
                k = (x["seed"], x["rot"], x["r"], x["pos"], x["alt"])
                if k in seen:
                    continue
                seen.add(k)
                out[(x["seed"], x["rot"])][(x["r"], x["pos"])].append((x["ours"], x["alt"], x["d_grp"], x["kind"]))
                n += 1
    return out, n


def index_games(dirs):
    games = {}
    for d in dirs:
        for p in glob.glob(os.path.join(d, "*.json.gz")):
            m = NAME_RE.search(os.path.basename(p))
            if m:
                games[(int(m.group(1)), m.group(3))] = p
    return games


def game_samples(path, lab, ref_name, unl_prob, rng):
    """Yield (obs f16, mask, ours, alt, d_grp, w, kind) for the labelled states of one game plus random others."""
    g = read_game(path)
    seat = next(i for i, n in enumerate(g.names) if n != ref_name)

    def sel(r, pos, s):
        return (r, pos) in lab or rng.random() < unl_prob

    for d in iter_decisions(g, [seat], select=sel):
        if d.bucket == "forced":
            continue
        obs = d.obs.astype(np.float16)
        pairs = lab.get((d.r, d.pos))
        if pairs:
            for ours, alt, dg, kd in pairs:
                if d.logged == ours and d.mask[alt]:
                    yield obs, d.mask, ours, alt, np.float32(dg), np.float32(1), kd
        else:
            yield obs, d.mask, 0, 0, np.float32(0), np.float32(0), ""


class A3Data(IterableDataset):
    def __init__(self, games, labels, ref_name, unl_prob, seed, buffer=4096):
        self.games, self.labels, self.ref_name = games, labels, ref_name
        self.unl_prob, self.seed, self.buffer = unl_prob, seed, buffer

    def __iter__(self):
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = random.Random(self.seed * 1000 + wid)
        mine = self.games[wid::nw]
        buf = []
        epoch = 0
        while True:
            order = mine[:]
            rng.shuffle(order)
            for key, path in order:
                for *s, _ in game_samples(path, self.labels.get(key, {}), self.ref_name, self.unl_prob, rng):
                    if len(buf) < self.buffer:
                        buf.append(s)
                    else:
                        i = rng.randrange(len(buf))
                        yield buf[i]
                        buf[i] = s
            epoch += 1


def load_models(path, device):
    env.setup()
    from model import Brain, DQN  # Mortal/mortal/model.py
    state = torch.load(path, weights_only=True, map_location="cpu")
    cfg = state["config"]
    version = cfg["control"].get("version", 1)

    def mk():
        b = Brain(version=version, conv_channels=cfg["resnet"]["conv_channels"], num_blocks=cfg["resnet"]["num_blocks"])
        q = DQN(version=version)
        b.load_state_dict(state["mortal"])
        q.load_state_dict(state["current_dqn"])
        return b.to(device), q.to(device)

    return state, mk(), mk()


def holdout_eval(brain, dqn, hold, device, n_games, seeds_of, bs=512):
    """First-order held-out check. Returns dict of metrics."""
    brain.eval()
    qs = []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for i in range(0, len(hold["obs"]), bs):
            obs = torch.from_numpy(hold["obs"][i:i + bs]).to(device).float()
            mask = torch.from_numpy(hold["mask"][i:i + bs]).to(device)
            qs.append(dqn(brain(obs), mask).float().cpu().numpy())
    brain.train()
    brain.freeze_bn(True)
    q = np.concatenate(qs)
    lab = hold["w"] > 0
    ours, alt, dg = hold["ours"], hold["alt"], hold["dg"]
    idx = np.arange(len(q))
    diff = q[idx, alt] - q[idx, ours]
    out = {"pair_mse": float(np.mean((diff[lab] - PT_SCALE * dg[lab]) ** 2))}
    flip = lab & (diff > 0)
    per_seed = defaultdict(float)
    for i in np.nonzero(flip)[0]:
        per_seed[seeds_of[i]] += dg[i]
    seeds = sorted(set(seeds_of[lab]))
    y = np.array([per_seed.get(s, 0.0) for s in seeds])
    out["flips_per_game"] = float(flip.sum() / n_games)
    out["gain_per_game"] = float(y.sum() / n_games)
    out["gain_ci"] = float(1.96 * (y / 4).std(ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else float("nan")
    arg = np.where(hold["mask"], q, -np.inf).argmax(1)
    out["third_action"] = float(np.mean((arg[flip] != alt[flip]) & (arg[flip] != ours[flip]))) if flip.any() else 0.0
    out["by_kind"] = {}
    for k in sorted(set(hold["kind"][lab])):
        sel = lab & (hold["kind"] == k)
        f = sel & (diff > 0)
        out["by_kind"][k] = (float(f.sum() / n_games), float(dg[f].sum() / n_games))
    unl = ~lab
    if unl.any():
        out["unlabelled_argmax_changed"] = float(np.mean(arg[unl] != hold["base_arg"][unl]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", action="append", required=True)
    ap.add_argument("--log-dir", action="append", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-name", default="mortal-v4")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--anchor", type=float, default=1.0)
    ap.add_argument("--unl-per-game", type=float, default=16, help="random unlabelled anchor states per game")
    ap.add_argument("--holdout-games", type=int, default=1200)
    ap.add_argument("--holdout-offset", type=int, default=0,
                    help="skip this many held-out games first (a fresh, independent check set for a re-run)")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="pipeline test: allows sacred-key games but never writes a checkpoint")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    labels, n_lab = load_labels(args.labels)
    games = index_games(args.log_dir)
    sacred = [p for p in games.values() if "_20260711_" in os.path.basename(p)]
    if sacred and not args.smoke:
        sys.exit(f"{sacred[0]}: sacred-protocol eval game in the training set; generate label games on another key")
    keys = sorted(k for k in games if k in labels)
    train = [(k, games[k]) for k in keys if k[0] % 10 != 0]
    hold_keys = [k for k in keys if k[0] % 10 == 0][args.holdout_offset: args.holdout_offset + args.holdout_games]
    print(f"{n_lab:,} labelled pairs in {len(keys):,} games; train {len(train):,} games, "
          f"held-out check on {len(hold_keys):,} games", flush=True)

    # the select hook sees every can-act point of our seat (~150 per game, call passes included)
    unl_prob = min(1.0, args.unl_per_game / 150)
    rng = random.Random(12345)
    H = defaultdict(list)
    seeds_of = []
    for k in hold_keys:
        for sample in game_samples(games[k], labels[k], args.ref_name, unl_prob, rng):
            for name, v in zip(("obs", "mask", "ours", "alt", "dg", "w", "kind"), sample):
                H[name].append(v)
            seeds_of.append(k[0])
    hold = {n: np.stack(v) if n in ("obs", "mask") else np.array(v) for n, v in H.items()}
    seeds_of = np.array(seeds_of)
    print(f"held-out: {int((hold['w'] > 0).sum()):,} labelled, {int((hold['w'] == 0).sum()):,} unlabelled states",
          flush=True)

    state, (brain, dqn), (abrain, adqn) = load_models(args.base, device)
    for m in (abrain, adqn):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    brain.train()
    brain.freeze_bn(True)
    dqn.train()

    with torch.no_grad():
        qb = []
        for i in range(0, len(hold["obs"]), 512):
            obs = torch.from_numpy(hold["obs"][i:i + 512]).to(device).float()
            mask = torch.from_numpy(hold["mask"][i:i + 512]).to(device)
            qb.append(adqn(abrain(obs), mask).float().cpu().numpy())
        hold["base_arg"] = np.concatenate(qb).argmax(1)
    n_hold_games = len(hold_keys)
    m0 = holdout_eval(brain, dqn, hold, device, n_hold_games, seeds_of)
    print(f"step 0: {json.dumps(m0)}", flush=True)

    params = [p for p in list(brain.parameters()) + list(dqn.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    loader = DataLoader(A3Data(train, labels, args.ref_name, unl_prob, args.seed), batch_size=args.batch,
                        num_workers=args.workers, persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)
    it = iter(loader)
    t0 = time.time()
    agg = defaultdict(float)
    hist = [{"step": 0, **m0}]
    for step in range(1, args.steps + 1):
        for _ in range(args.accum):
            obs, mask, ours, alt, dg, w = next(it)
            obs, mask = obs.to(device, non_blocking=True).float(), mask.to(device, non_blocking=True)
            ours, alt = ours.to(device).long()[:, None], alt.to(device).long()[:, None]
            dg, w = dg.to(device), w.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                q = dqn(brain(obs), mask)
                with torch.no_grad():
                    qa = adqn(abrain(obs), mask)
            q, qa = q.float(), qa.float()
            qm, qam = torch.where(mask, q, 0.0), torch.where(mask, qa, 0.0)
            anchor = ((qm - qam) ** 2).sum() / mask.sum()
            diff = (qm.gather(1, alt) - qm.gather(1, ours)).squeeze(1)
            pair = (w * (diff - PT_SCALE * dg) ** 2).sum() / w.sum().clamp(min=1)
            loss = pair + args.anchor * anchor
            scaler.scale(loss / args.accum).backward()
            agg["pair"] += float(pair) / args.accum
            agg["anchor"] += float(anchor) / args.accum
            agg["labelled"] += float(w.mean()) / args.accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if step % 100 == 0:
            el = time.time() - t0
            print(f"step {step}: pair {agg['pair'] / 100:.4f} anchor {agg['anchor'] / 100:.5f} "
                  f"labelled {agg['labelled'] / 100:.2f}  {el / step:.2f}s/step", flush=True)
            agg.clear()
        if step % args.eval_every == 0 or step == args.steps:
            m = holdout_eval(brain, dqn, hold, device, n_hold_games, seeds_of)
            hist.append({"step": step, **m})
            print(f"step {step} held-out: {json.dumps(m)}", flush=True)

    if args.smoke:
        print("smoke run: no checkpoint written")
        return
    state["mortal"] = brain.state_dict()
    state["current_dqn"] = dqn.state_dict()
    state["a3"] = json.dumps({"args": vars(args), "history": hist})
    torch.save(state, args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
