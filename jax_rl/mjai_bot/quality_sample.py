"""Decision-quality yardstick for the data-quality study (2026-10-11, patches/data_quality/README.md).

For each game in a list: how often the human action equals mortal_v4's greedy action, on non-forced decisions
(>= 2 legal actions), for all four seats. Counted separately for:
  e1   decisions in the first kyoku (East 1, honba 0): every game starts from the same scores here, so differences
       between groups of games are differences in the players, not in the situation;
  bal  decisions in kyoku that start with a score spread < 10,000 and are not the last kyoku;
  all  every decision.
One row per game, so any per-game metric can be checked against the same inference later. v4 is only a yardstick for
choosing a filter; it is never a training signal.

    python quality_sample.py --list PATHS.txt --out OUT.tsv --model V4.pth [--device cuda:0]
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

KINDS = ("discard", "riichi", "call")


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


def kyoku_flags(raw):
    """per kyoku (0-based log order): (is East 1 honba 0, starts balanced and is not the last kyoku)."""
    flags = []
    for ln in raw.splitlines():
        if '"start_kyoku"' not in ln:
            continue
        e = json.loads(ln)
        s = e["scores"]
        flags.append([e["bakaze"] == "E" and e["kyoku"] == 1 and e["honba"] == 0, max(s) - min(s) < 10000])
    if flags:
        flags[-1][1] = False
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    st = torch.load(a.model, weights_only=True, map_location="cpu")
    cfg = st["config"]
    v = cfg["control"].get("version", 1)
    brain = Brain(version=v, conv_channels=cfg["resnet"]["conv_channels"], num_blocks=cfg["resnet"]["num_blocks"])
    dqn = DQN(version=v)
    brain.load_state_dict(st["mortal"])
    dqn.load_state_dict(st["current_dqn"])
    brain, dqn = brain.to(dev).eval(), dqn.to(dev).eval()
    loader = GameplayLoader(version=v, oracle=False, player_names=[], excludes=[])
    paths = [ln.strip() for ln in open(a.list) if ln.strip()]
    cols = ["path", "n_e1", "h_e1", "n_bal", "h_bal", "n_all", "h_all"] + \
           [f"{p}_{k}" for k in KINDS for p in ("n_e1", "h_e1")]
    with open(a.out, "w", encoding="utf-8") as out:
        out.write("\t".join(cols) + "\n")
        for i, p in enumerate(paths):
            try:
                op = gzip.open if p.endswith(".gz") else open
                raw = op(p, "rt", encoding="utf-8").read()
                flags = kyoku_flags(raw)
                games = loader.load_log(raw)
            except Exception:  # noqa: BLE001 - a log that does not load is skipped and counted by its absence
                continue
            obs, act, msk, kk = [], [], [], []
            for g in games:
                o, ac, m = g.take_obs(), g.take_actions(), g.take_masks()
                ak = bytes(g.take_at_kyoku())
                for j in range(len(ac)):
                    if m[j].sum() < 2:
                        continue
                    obs.append(o[j]); act.append(ac[j]); msk.append(m[j]); kk.append(ak[j])
            if not act:
                continue
            with torch.no_grad():
                q = []
                for j in range(0, len(obs), 1024):
                    ob = torch.from_numpy(np.stack(obs[j:j + 1024])).to(dev)
                    mk = torch.from_numpy(np.stack(msk[j:j + 1024])).to(dev)
                    q.append(dqn(brain(ob), mk).float().cpu().numpy())
            q = np.concatenate(q)
            mk = np.stack(msk)
            hit = np.where(mk, q, -np.inf).argmax(1) == np.asarray(act)
            kk = np.asarray(kk)
            e1 = np.array([flags[k][0] if k < len(flags) else False for k in kk])
            bal = np.array([flags[k][1] if k < len(flags) else False for k in kk])
            kinds = np.array([kind_of(mk[j], act[j]) for j in range(len(act))])
            row = [p, int(e1.sum()), int(hit[e1].sum()), int(bal.sum()), int(hit[bal].sum()), len(hit), int(hit.sum())]
            for k in KINDS:
                sel = e1 & (kinds == k)
                row += [int(sel.sum()), int(hit[sel].sum())]
            out.write("\t".join(map(str, row)) + "\n")
            if (i + 1) % 500 == 0:
                out.flush()
                print(f"{i + 1} / {len(paths)}", flush=True)


if __name__ == "__main__":
    main()
