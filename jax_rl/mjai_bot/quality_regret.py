"""Stakes-weighted decision quality for the data-quality study (2026-10-11, patches/data_quality/README.md, v2).

For every non-forced decision (>= 2 legal actions, all four seats) of every game in a list, and for each judge
model: did the human pick the judge's greedy action, and the judge's regret for the human action,
    regret = Q_judge(best legal) - Q_judge(human action)   (>= 0, in the judge's own value units).
Regret is near 0 when the choice hardly matters (a near-tie in tile efficiency) and large when it does (push / fold,
placement fights late in the game), so it weights decisions by their stakes without hand-picking kyoku.

Each decision also gets the situation it was made in, so the analysis can compare like with like:
phase (E1..S4, West as 8), all-last flag, kan-in-kyoku flag, the player's score and rank at the start of the kyoku and
the point gaps to the players directly above and below, decision kind, number of legal actions.

v3 (2026-10-11, the pro's remark "what the AI recommends is not always best, but what it says to avoid must not be
played"): per judge it also keeps its own greedy action and, for every other judge k, x_k = Q_j(own best) -
Q_j(judge k's best). The spread of x over decisions where judges disagree is how far apart strong choices can be;
the analysis uses it as a tolerance band. v4 (same day): any number of judges, including a "human school" judge
(the offline base trained only on houou games), and the human action itself is kept for typing disagreements.

    python quality_regret.py --list PATHS.txt --out OUT.npz --judge v4=V4.pth --judge cp=CP.pth [--device mps]
Resumable: every --shard games (default 250) are saved as OUT.partNNNN.npz; a rerun skips finished shards and the
final OUT.npz is merged from them. Sleep only pauses the process; a restart or kill loses at most one shard.
Run with a Mortal tree on PYTHONPATH (libriichi + model.py). The judges are never a training signal.
"""
from __future__ import annotations

import argparse
import gzip
import json

import numpy as np
import torch
from libriichi.dataset import GameplayLoader
from model import Brain, DQN

KIND = {"discard": 0, "riichi": 1, "call": 2, "agari": 3, "other": 4}


def kind_of(mask, act):
    if mask[43]:
        return 3
    if mask[37]:
        return 1
    if mask[38:43].any():
        return 2
    if act <= 36:
        return 0
    return 4


def kyoku_info(raw):
    """per kyoku in log order: phase, scores at start, has a kan; the last kyoku is flagged all-last."""
    out = []
    for ln in raw.splitlines():
        if '"start_kyoku"' in ln:
            e = json.loads(ln)
            ph = {"E": 0, "S": 4, "W": 8, "N": 8}[e["bakaze"]] + (e["kyoku"] - 1 if e["bakaze"] in "ES" else 0)
            out.append({"phase": min(ph, 8), "scores": e["scores"], "kan": False})
        elif out and any(t in ln for t in ('"ankan"', '"kakan"', '"daiminkan"')):
            out[-1]["kan"] = True
    return out


def rank_and_gaps(scores, seat):
    order = sorted(range(4), key=lambda i: (-scores[i], i))
    r = order.index(seat)
    above = scores[order[r - 1]] - scores[seat] if r > 0 else 0
    below = scores[seat] - scores[order[r + 1]] if r < 3 else 0
    return r + 1, above, below


def load_judge(path, dev):
    st = torch.load(path, weights_only=True, map_location="cpu")
    cfg = st["config"]
    v = cfg["control"].get("version", 1)
    brain = Brain(version=v, conv_channels=cfg["resnet"]["conv_channels"], num_blocks=cfg["resnet"]["num_blocks"])
    dqn = DQN(version=v)
    brain.load_state_dict(st["mortal"])
    dqn.load_state_dict(st["current_dqn"])
    return v, brain.to(dev).eval(), dqn.to(dev).eval()


def shard_path(out: str, k: int) -> str:
    return f"{out}.part{k:04d}.npz"


def merge_shards(out: str, n_games: int, size: int, paths) -> None:
    """Concatenate the finished shards (games [k*size, (k+1)*size)) into OUT in the original single-file format."""
    parts = [np.load(shard_path(out, k)) for k in range((n_games + size - 1) // size)]
    keys = [k for k in parts[0].files if k not in ("paths", "ok")]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    merged["ok"] = np.concatenate([p["ok"] for p in parts])
    merged["paths"] = np.asarray(paths)
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, **merged)
    import os
    os.replace(tmp, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", action="append", required=True, help="NAME=checkpoint")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--shard", type=int, default=250, help="games per checkpoint shard; finished shards are skipped")
    ap.add_argument("--first", default="", help="comma-separated shard indices to run before the rest")
    ap.add_argument("--only", default="", help="run only these shards (e.g. 10-23) and do not merge")
    a = ap.parse_args()
    dev = torch.device(a.device)
    import os
    if dev.type == "cuda" and os.environ.get("GPU_MEM_FRAC"):
        # several workers share one GPU: cap each process so the caching allocator cannot crowd the others out
        torch.cuda.set_per_process_memory_fraction(float(os.environ["GPU_MEM_FRAC"]), dev)
    judges = {}
    version = None
    for spec in a.judge:
        name, p = spec.split("=", 1)
        v, brain, dqn = load_judge(p, dev)
        assert version in (None, v), "judges must share an observation version"
        version = v
        judges[name] = (brain, dqn)
    loader = GameplayLoader(version=version, oracle=False, player_names=[], excludes=[])
    paths = [ln.strip() for ln in open(a.list) if ln.strip()]
    import os
    nshard = (len(paths) + a.shard - 1) // a.shard
    first = [int(x) for x in a.first.split(",") if x.strip()]
    todo = first + [k for k in range(nshard) if k not in first]
    if a.only:
        lo_, hi_ = (int(x) for x in a.only.split("-"))
        todo = [k for k in todo if lo_ <= k <= hi_]
    for sk in todo:
        if os.path.exists(shard_path(a.out, sk)):
            continue
        lo, hi = sk * a.shard, min(len(paths), (sk + 1) * a.shard)
        out = score(paths, range(lo, hi), judges, loader, dev)
        tmp = shard_path(a.out, sk) + ".tmp.npz"
        np.savez_compressed(tmp, **out)
        os.replace(tmp, shard_path(a.out, sk))
        print(f"{hi} / {len(paths)} (shard {sk + 1}/{nshard} saved)", flush=True)
    if a.only:
        print(f"shards {a.only} done (not merged)")
        return
    merge_shards(a.out, len(paths), a.shard, paths)
    z = np.load(a.out)
    print(f"done: {int(z['ok'].sum())} / {len(paths)} games, {len(z['game']):,} decisions")


def score(paths, gis, judges, loader, dev):
    """score games paths[gi] for gi in gis; game ids are global so shards concatenate."""
    cols = {k: [] for k in ("game", "seat", "kyoku", "phase", "last", "kan", "score", "rank", "above", "below", "kind",
                            "nlegal")}
    res = {n: {"hit": [], "regret": [], "top2": [], "best": [], **{f"x_{k}": [] for k in judges if k != n}}
           for n in judges}
    acts = []
    ok = []
    for gi in gis:
        p = paths[gi]
        try:
            op = gzip.open if p.endswith(".gz") else open
            raw = op(p, "rt", encoding="utf-8").read()
            info = kyoku_info(raw)
            games = loader.load_log(raw)
        except Exception:  # noqa: BLE001 - unreadable logs are skipped; ok[] records which loaded
            ok.append(False)
            continue
        ok.append(True)
        obs, act, msk, feat = [], [], [], []
        for seat, g in enumerate(games):
            o, ac, m = g.take_obs(), g.take_actions(), g.take_masks()
            ak = bytes(g.take_at_kyoku())
            for j in range(len(ac)):
                if m[j].sum() < 2 or ak[j] >= len(info):
                    continue
                ki = info[ak[j]]
                r, up, dn = rank_and_gaps(ki["scores"], seat)
                obs.append(o[j]); act.append(ac[j]); msk.append(m[j])
                feat.append((gi, seat, ak[j], ki["phase"], ak[j] == len(info) - 1, ki["kan"], ki["scores"][seat], r, up,
                             dn, kind_of(m[j], ac[j]), int(m[j].sum())))
        if not act:
            continue
        mk = np.stack(msk)
        at = np.asarray(act)
        qd = {}
        with torch.no_grad():
            for name, (brain, dqn) in judges.items():
                q = []
                for j in range(0, len(obs), 1024):
                    ob = torch.from_numpy(np.stack(obs[j:j + 1024])).to(dev)
                    mm = torch.from_numpy(mk[j:j + 1024]).to(dev)
                    q.append(dqn(brain(ob), mm).float().cpu().numpy())
                qd[name] = np.where(mk, np.concatenate(q), -np.inf)
        idx = np.arange(len(at))
        bests = {n: q.argmax(1) for n, q in qd.items()}
        for name, q in qd.items():
            best = bests[name]
            qs = np.sort(q, 1)
            res[name]["hit"].append(best == at)
            res[name]["regret"].append((q[idx, best] - q[idx, at]).astype(np.float32))
            res[name]["top2"].append((qs[:, -1] - qs[:, -2]).astype(np.float32))
            res[name]["best"].append(best.astype(np.int8))
            for k in qd:
                if k != name:
                    res[name][f"x_{k}"].append((q[idx, best] - q[idx, bests[k]]).astype(np.float32))
        acts.append(at.astype(np.int8))
        for k, col in zip(cols, zip(*feat)):
            cols[k].extend(col)
    out = {k: np.asarray(v, dtype=np.int32) for k, v in cols.items()}
    for n, d in res.items():
        out[f"{n}_hit"] = np.concatenate(d["hit"]) if d["hit"] else np.zeros(0, bool)
        out[f"{n}_regret"] = np.concatenate(d["regret"]) if d["regret"] else np.zeros(0, np.float32)
        out[f"{n}_top2"] = np.concatenate(d["top2"]) if d["top2"] else np.zeros(0, np.float32)
        for key, v in d.items():
            if key not in ("hit", "regret", "top2"):
                out[f"{n}_{key}"] = np.concatenate(v) if v else np.zeros(0, np.float32)
    out["act"] = np.concatenate(acts) if acts else np.zeros(0, np.int8)
    out["ok"] = np.asarray(ok)
    return out

if __name__ == "__main__":
    main()
