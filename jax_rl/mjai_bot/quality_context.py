"""Decision context for the data-quality study (patches/data_quality/README.md, v5).

No model. For every decision that quality_regret.py scores (same games, same order, same filter: >= 2 legal actions),
add what kind of moment it was:
  at_turn    the player's turn number in the kyoku (from the loader);
  shanten    the player's shanten at the decision (from the loader);
  opp_riichi how many opponents have a riichi accepted at that moment;
  melds      how many melds the player already has (0 on a call decision = the first call of the hand).
opp_riichi and melds come from replaying the log with libriichi's PlayerState for each seat. The replay's decision
points are aligned to the loader's by their legal-action signature (difflib), because the two can differ by one point
in a game; decisions that do not align get -1.

    python quality_context.py --list PATHS.txt --out OUT.npz
Resumable in the same way as quality_regret.py (OUT.partNNNN.npz every --shard games; reruns skip finished shards).
"""
from __future__ import annotations

import argparse
import difflib
import gzip
import json

import numpy as np
from libriichi.dataset import GameplayLoader
from libriichi.state import PlayerState


def sig_mask(m):
    return (bool(m[:37].any()), bool(m[37]), bool(m[38]), bool(m[39]), bool(m[40]), bool(m[41]), bool(m[42]),
            bool(m[43]), bool(m[44]), bool(m[45]))


def sig_cans(c):
    return (c.can_discard, c.can_riichi, c.can_chi_low, c.can_chi_mid, c.can_chi_high, c.can_pon,
            c.can_daiminkan or c.can_ankan or c.can_kakan, c.can_agari, c.can_ryukyoku, c.can_pass)


def replay(lines, seat):
    st = PlayerState(seat)
    sigs, ctx = [], []
    riichi = set()
    for ln in lines:
        e = json.loads(ln)
        t = e["type"]
        if t == "start_kyoku":
            riichi = set()
        elif t == "reach_accepted":
            riichi.add(e["actor"])
        cans = st.update(ln)
        if cans is not None and cans.can_act:
            melds = len(st.chis) + len(st.pons) + len(st.minkans) + len(st.ankans)
            sigs.append(sig_cans(cans))
            ctx.append((len(riichi - {seat}), melds))
    return sigs, ctx


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
    ap.add_argument("--shard", type=int, default=250, help="games per checkpoint shard; finished shards are skipped")
    a = ap.parse_args()
    import os
    loader = GameplayLoader(version=4, oracle=False, player_names=[], excludes=[])
    paths = [ln.strip() for ln in open(a.list) if ln.strip()]
    nshard = (len(paths) + a.shard - 1) // a.shard
    for sk in range(nshard):
        if os.path.exists(shard_path(a.out, sk)):
            continue
        lo, hi = sk * a.shard, min(len(paths), (sk + 1) * a.shard)
        out, unaligned, total = context(paths, range(lo, hi), loader)
        tmp = shard_path(a.out, sk) + ".tmp.npz"
        np.savez_compressed(tmp, **out)
        os.replace(tmp, shard_path(a.out, sk))
        print(f"{hi} / {len(paths)} (shard {sk + 1}/{nshard}; unaligned {unaligned} / {total})", flush=True)
    merge_shards(a.out, len(paths), a.shard, paths)
    z = np.load(a.out)
    print(f"done: {int(z['ok'].sum())} / {len(paths)} games, {len(z['game']):,} decisions, "
          f"unaligned {int((z['opp_riichi'] < 0).sum())}")


def context(paths, gis, loader):
    cols = {k: [] for k in ("game", "seat", "kind_check", "nlegal", "at_turn", "shanten", "opp_riichi", "melds")}
    ok = []
    unaligned = total = 0
    for gi in gis:
        p = paths[gi]
        try:
            op = gzip.open if p.endswith(".gz") else open
            raw = op(p, "rt", encoding="utf-8").read()
            nkyoku = raw.count('"start_kyoku"')
            games = loader.load_log(raw)
        except Exception:  # noqa: BLE001 - same skips as quality_regret.py
            ok.append(False)
            continue
        ok.append(True)
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        for seat, g in enumerate(games):
            m, ac = g.take_masks(), g.take_actions()
            ak, turns, sh = bytes(g.take_at_kyoku()), bytes(g.take_at_turns()), bytes(g.take_shantens())
            lsig = [sig_mask(m[j]) for j in range(len(ac))]
            rsig, rctx = replay(lines, seat)
            match = {}
            for blk in difflib.SequenceMatcher(None, lsig, rsig, autojunk=False).get_matching_blocks():
                for k in range(blk.size):
                    match[blk.a + k] = blk.b + k
            for j in range(len(ac)):
                if m[j].sum() < 2 or ak[j] >= nkyoku:
                    continue
                total += 1
                if j in match:
                    orr, mel = rctx[match[j]]
                else:
                    orr, mel = -1, -1
                    unaligned += 1
                cols["game"].append(gi); cols["seat"].append(seat); cols["kind_check"].append(int(ac[j]))
                cols["nlegal"].append(int(m[j].sum())); cols["at_turn"].append(turns[j]); cols["shanten"].append(sh[j])
                cols["opp_riichi"].append(orr); cols["melds"].append(mel)
    out = {k: np.asarray(v, dtype=np.int32) for k, v in cols.items()}
    out["ok"] = np.asarray(ok)
    return out, unaligned, total

if __name__ == "__main__":
    main()
