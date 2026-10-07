"""Source-quality probe for the new-data arm (2026-10-06): how often does the human action agree with mortal_v4's
greedy action, per source (Tenhou houou / Mahjong Soul throne / Mahjong Soul jade)?

v4 is stronger than all three populations, so a lower agreement means play further from strong play. Compared on
the same kind of kyoku for every source: kyoku with a kan and the all-last kyoku are excluded everywhere, because
repaired jade logs lose those (see mjs_convert.py). Non-forced decisions only (>= 2 legal actions). Reported overall
and by decision kind, with game-clustered 95% CIs. The jade inclusion rule is pre-registered in
patches/majsoul_data/README.md.

    python mjs_quality_probe.py --source houou=LIST1 --source throne=LIST2 --source jade=LIST3 --model V4.pth
Run from a Mortal tree (libriichi.so and model.py importable).
"""
from __future__ import annotations

import argparse
import gzip
import json

import numpy as np
import torch
from libriichi.dataset import GameplayLoader
from model import Brain, DQN

KINDS = ("discard", "riichi", "call", "agari", "other")


def kind_of(mask, act):
    if mask[43]:
        return "agari"
    if mask[37]:
        return "riichi"
    if mask[38:43].any():
        return "call"
    if act <= 36:
        return "discard"
    return "other"


def excluded_kyoku(raw):
    """kyoku indices (0-based log order) with a kan, plus the all-last kyoku."""
    k, out = -1, set()
    for ln in raw.splitlines():
        if not ln.strip():
            continue
        e = json.loads(ln)
        if e["type"] == "start_kyoku":
            k += 1
        elif e["type"] in ("ankan", "kakan", "daiminkan"):
            out.add(k)
    out.add(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="NAME=file list")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    st = torch.load(args.model, weights_only=True, map_location="cpu")
    cfg = st["config"]
    v = cfg["control"].get("version", 1)
    brain = Brain(version=v, conv_channels=cfg["resnet"]["conv_channels"], num_blocks=cfg["resnet"]["num_blocks"])
    dqn = DQN(version=v)
    brain.load_state_dict(st["mortal"])
    dqn.load_state_dict(st["current_dqn"])
    brain, dqn = brain.to(dev).eval(), dqn.to(dev).eval()
    loader = GameplayLoader(version=v, oracle=False, player_names=[], excludes=[])
    print(f"{'source':<8}{'games':>7}{'decisions':>11}{'agree':>16}" + "".join(f"{k:>16}" for k in KINDS))
    for spec in args.source:
        name, lst = spec.split("=", 1)
        paths = [ln.strip() for ln in open(lst) if ln.strip()]
        per_game = []          # (n, agree) per game, overall and per kind
        for p in paths:
            raw = gzip.open(p, "rt", encoding="utf-8").read()
            excl = excluded_kyoku(raw)
            try:
                games = loader.load_log(raw)
            except Exception:  # noqa: BLE001
                continue
            obs, act, msk = [], [], []
            for g in games:
                o, a, m = g.take_obs(), g.take_actions(), g.take_masks()
                ak = bytes(g.take_at_kyoku())
                for i in range(len(a)):
                    if ak[i] in excl or m[i].sum() < 2:
                        continue
                    obs.append(o[i]); act.append(a[i]); msk.append(m[i])
            if not act:
                continue
            with torch.no_grad():
                q = []
                for i in range(0, len(obs), 1024):
                    ob = torch.from_numpy(np.stack(obs[i:i + 1024])).to(dev)
                    mk = torch.from_numpy(np.stack(msk[i:i + 1024])).to(dev)
                    q.append(dqn(brain(ob), mk).float().cpu().numpy())
            q = np.concatenate(q)
            mk = np.stack(msk)
            best = np.where(mk, q, -np.inf).argmax(1)
            a = np.asarray(act)
            hit = best == a
            rec = {"all": (len(a), int(hit.sum()))}
            kinds = np.array([kind_of(mk[i], a[i]) for i in range(len(a))])
            for k in KINDS:
                sel = kinds == k
                rec[k] = (int(sel.sum()), int(hit[sel].sum()))
            per_game.append(rec)

        def ci(key):
            n = np.array([r[key][0] for r in per_game], float)
            h = np.array([r[key][1] for r in per_game], float)
            if n.sum() == 0:
                return "      -"
            p = h.sum() / n.sum()
            # ratio estimator, game-clustered variance
            se = np.sqrt(((h - p * n) ** 2).sum() / (len(n) * (len(n) - 1))) / n.mean()
            return f"{100 * p:6.2f} ± {196 * se:4.2f}"
        nd = sum(r["all"][0] for r in per_game)
        print(f"{name:<8}{len(per_game):>7}{nd:>11,}{ci('all'):>16}" + "".join(f"{ci(k):>16}" for k in KINDS), flush=True)


if __name__ == "__main__":
    main()
