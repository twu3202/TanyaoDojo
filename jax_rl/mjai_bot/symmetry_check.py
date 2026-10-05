"""D1 (10-03): how equivariant are our models under the symmetries of riichi mahjong?

Permuting the three suits (6 ways) or the three dragons (6 ways) maps any game to an equivalent game: same rules,
same values, with one exception (ryuuiisou, a yakuman that needs souzu and hatsu). A model trained without that
augmentation (all our configs: `enable_augmentation = false`, and Mortal's own augmentation only swaps man/pin)
need not respect it. Here every decision of the challenger seat is re-encoded from the permuted log (libriichi
replay of the relabelled mjai events), and the model's Q is mapped back to the original action labels.

Reported per model: how often the greedy action changes under a symmetry (pure noise in the policy), how much Q
moves (relative to the spread of its own action margins), and how often the symmetrized policy (Q averaged over
the 11 non-trivial symmetries plus the identity) differs from the plain one, i.e. how much test-time
symmetrization would change play. Also checks whether the permuted observation is just a column permutation of
the original (rows that are not would need care in a test-time engine).

    python symmetry_check.py --log-dir DIR --games 300 --model tpt408=PATH --model v4=PATH
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get("RIICHI_EVAL_SRC", os.path.expanduser("~/riichi-eval/src")))
from riichi_eval import env  # noqa: E402
from riichi_eval.logs import NAME_RE, parse_game  # noqa: E402
from riichi_eval.replay import iter_decisions  # noqa: E402

SUITS = "mps"
DRAGONS = "PFC"


def make_perms(dragons="all"):
    """(suit permutation, dragon permutation) pairs: 6 suit perms with dragons fixed, then the non-trivial dragon perms
    with suits fixed: all 5 ('all', as in D1) or only the 2 cyclic rotations ('rot'). The dragon transpositions are not
    exact symmetries (10-05): the dora indicator relation P -> F -> C -> P is directed, so a swap gives dora to the wrong
    tile in games with a dragon indicator; 'rot' is the group the symmetry augmentation trains on."""
    sp = [dict(zip(SUITS, p)) for p in itertools.permutations(SUITS)]
    if dragons == "rot":
        dp = [dict(zip(DRAGONS, DRAGONS[k:] + DRAGONS[:k])) for k in (1, 2)]
    else:
        dp = [dict(zip(DRAGONS, p)) for p in itertools.permutations(DRAGONS)][1:]
    ident_d = dict(zip(DRAGONS, DRAGONS))
    ident_s = dict(zip(SUITS, SUITS))
    return [(s, ident_d) for s in sp] + [(ident_s, d) for d in dp]


def tile_map(s_map, d_map):
    """Tile string -> tile string, and action index -> action index (0..36 discards; others unchanged)."""
    def f(t):
        if len(t) >= 2 and t[0] in "123456789" and t[1] in SUITS:
            return t[0] + s_map[t[1]] + t[2:]
        return d_map.get(t, t)

    a = np.arange(46)
    base = {"m": 0, "p": 9, "s": 18}
    for su, off in base.items():
        a[off:off + 9] = base[s_map[su]] + np.arange(9)
        a[34 + "mps".index(su)] = 34 + "mps".index(s_map[su])
    for i, dr in enumerate(DRAGONS):
        a[31 + i] = 31 + DRAGONS.index(d_map[dr])
    return f, a


def relabel(obj, f):
    if isinstance(obj, str):
        return f(obj)
    if isinstance(obj, list):
        return [relabel(x, f) for x in obj]
    if isinstance(obj, dict):
        return {k: (v if k in ("type", "names", "bakaze") else relabel(v, f)) for k, v in obj.items()}
    return obj


def load_model(path, device):
    env.setup()
    from model import Brain, DQN
    st = torch.load(path, weights_only=True, map_location="cpu")
    cfg = st["config"]
    v = cfg["control"].get("version", 1)
    b = Brain(version=v, conv_channels=cfg["resnet"]["conv_channels"], num_blocks=cfg["resnet"]["num_blocks"])
    q = DQN(version=v)
    b.load_state_dict(st["mortal"])
    q.load_state_dict(st["current_dqn"])
    return b.to(device).eval(), q.to(device).eval(), v


def qvals(model, obs, mask, device, bs=512):
    b, q, _ = model
    out = []
    with torch.no_grad():
        for i in range(0, len(obs), bs):
            o = torch.from_numpy(np.stack(obs[i:i + bs])).to(device).float()
            m = torch.from_numpy(np.stack(mask[i:i + bs])).to(device)
            out.append(q(b(o), m).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--model", action="append", required=True, help="NAME=path.pth")
    ap.add_argument("--ref-name", default="mortal-v4")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=10, help="games per chunk (memory)")
    ap.add_argument("--dragon-perms", choices=("all", "rot"), default="all",
                    help="all 5 non-trivial dragon perms (D1) or the 2 rotations only (exact symmetries)")
    args = ap.parse_args()
    device = torch.device(args.device)
    models = {}
    for a in args.model:
        n, p = a.split("=", 1)
        models[n] = load_model(p, device)
    version = next(iter(models.values()))[2]
    assert all(m[2] == version for m in models.values()), "models must share the observation version"
    perms = make_perms(args.dragon_perms)
    maps = [tile_map(s, d) for s, d in perms]

    paths = sorted(p for p in glob.glob(os.path.join(args.log_dir, "*.json.gz"))
                   if NAME_RE.search(os.path.basename(p)))[: args.games]
    import gzip
    rows_bad = None
    n_dec = 0
    QS = {n: [] for n in models}
    MASKS = []
    for c0 in range(0, len(paths), args.chunk):
        OBS = [[] for _ in perms]
        MASK = [[] for _ in perms]
        for path in paths[c0:c0 + args.chunk]:
            with gzip.open(path, "rt") as fh:
                evs = [json.loads(ln) for ln in fh if ln.strip()]
            for e in evs:
                e.pop("meta", None)
            seat = next(i for i, n in enumerate(evs[0]["names"]) if n != args.ref_name)
            per = []
            for f, _ in maps:
                g = parse_game([json.dumps(relabel(e, f)) for e in evs], path)
                per.append([(d.obs, d.mask) for d in iter_decisions(g, [seat], version) if d.bucket != "forced"])
            k = len(per[0])
            assert all(len(p) == k for p in per), f"{path}: decision count differs across symmetries"
            n_dec += k
            for pi, ((_, amap), p) in enumerate(zip(maps, per)):
                # tile columns: the permuted obs should equal the original with columns relabelled
                cmap = amap[:34]
                for (o0, _), (o1, _) in zip(per[0], p):
                    bad = ~np.all(o1[:, cmap] == o0, axis=1)
                    rows_bad = bad.astype(np.int64) if rows_bad is None else rows_bad + bad
                OBS[pi].extend(o.astype(np.float16) for o, _ in p)
                MASK[pi].extend(m for _, m in p)
        MASKS.append(np.stack(MASK[0]))
        for name, model in models.items():
            QS[name].append(np.stack([qvals(model, OBS[pi], MASK[pi], device)[:, amap]   # back to original labels
                                      for pi, (_, amap) in enumerate(maps)]))
        print(f"  [{min(c0 + args.chunk, len(paths))}/{len(paths)}] {n_dec:,} decisions", flush=True)
    print(f"{len(paths)} games, {n_dec:,} non-forced decisions of the challenger, {len(perms)} symmetries "
          f"(identity + 5 suit + {len(perms) - 6} dragon permutations)")
    nb = np.flatnonzero(rows_bad)
    print(f"obs rows that are not a pure column relabelling under some symmetry: {len(nb)} of {len(rows_bad)} "
          f"(row ids {nb[:40].tolist()}{' ...' if len(nb) > 40 else ''})")
    mask = np.concatenate(MASKS)

    for name in models:
        Q = np.concatenate(QS[name], axis=1)        # (perm, decision, action)
        Qm = np.where(mask, Q, -np.inf)
        a0 = Qm[0].argmax(1)
        nleg = mask.sum(1)
        multi = nleg >= 2
        top2 = np.sort(np.where(mask, Q[0], -np.inf), 1)[:, -2:]
        margin = (top2[:, 1] - top2[:, 0])[multi]
        print(f"\n{name}: {multi.sum():,} decisions with >= 2 legal actions, median top-2 Q margin {np.median(margin):.4f}")
        for lab, rng in (("suit perms", range(1, 6)), ("dragon perms", range(6, len(perms)))):
            ch = np.mean([(Qm[p].argmax(1) != a0)[multi].mean() for p in rng])
            dq = np.mean([np.abs(Q[p] - Q[0])[mask].mean() for p in rng])
            print(f"  {lab:<13} greedy action changes on {ch:.2%} of decisions; mean |dQ| {dq:.4f} "
                  f"({dq / np.median(margin):.2f}x the median top-2 margin)")
        Qs = np.where(mask, Q.mean(0), -np.inf)
        tta = (Qs.argmax(1) != a0)[multi]
        print(f"  symmetrized (mean over all {len(perms)}) differs from plain greedy on {tta.mean():.2%} of decisions")
        disc = multi & (a0 <= 36)
        print(f"    of which discards: {((Qs.argmax(1) != a0) & disc).sum() / disc.sum():.2%} of discard decisions; "
              f"non-discard: {((Qs.argmax(1) != a0) & multi & ~disc).sum() / max((multi & ~disc).sum(), 1):.2%}")


if __name__ == "__main__":
    main()
