"""Rollout check of the human-vs-AI disagreement types (data-quality study, patches/data_quality/README.md v8).

The type costs in quality_regret_analyze.py come from a regression of what the human's seat went on to score in
that kyoku on how often the human deviated from the AI, so they can carry whatever made the human choose
differently. Here each disagreement is priced directly: from the decision in a real Tenhou game, the human's
action and the AI's are both rolled out with mortal_v4 playing all four seats afterwards (one-step deviation),
to the end of the kyoku, valued for the deciding seat by GRP expected placement pt (raw points kept as well).

Walls. Tenhou logs carry no libriichi seed, so each kyoku's wall is rebuilt from the log (riichi_eval
`log_wall`): dealt hands, logged draws and revealed indicators in place, never-seen tiles at random (k = 0), and
on K re-deals of every tile still hidden at the branch (k = 1..K, one seed per k shared by all branches = common
random numbers). Given the four hands the hidden order is uniform, so the re-deals average over the wall exactly.

Kinds (AI = mortal_v4 and C' lambda0.2 greedy choose the same class; no ron available at call decisions):
  pass2call    human passed a chi / pon,     AI calls          alt = v4's call
  call2pass    human called chi / pon,       AI passes         alt = pass
  dama2riichi  riichi legal, human discarded, AI declares riichi  alt = riichi (v4 picks the tile);
               branch 2 = v4's best discard without riichi, so d splits into riichi-vs-dama and tile choice
  riichi2dama  human declared riichi,        AI discards       alt = v4's best discard
  disc2disc    control: both AIs pick the same tile kind, the human another; sampled at --disc-rate
d = V(alt) - V(human), per wall; branch 0 = the human's logged action, branch 1 = alt.

    python quality_rollout.py --list PATHS --out DIR --v4 V4.pth --cp CP.pth --grp GRP.pth [--walls 8]
    python quality_rollout.py --fidelity 50 --list PATHS --v4 V4.pth      (positive control: forced replay of
                                                   every kyoku on the rebuilt wall reproduces the logged result)
    python quality_rollout.py --report DIR
Needs RIICHI_EVAL_SRC (riichi-eval/src), RIICHI_EVAL_LIBRIICHI (a libriichi build with Rollout.SUPPORTS_WALL)
and MORTAL_DIR (Mortal/mortal). Resumable: DIR/shard_NNNN.jsonl per --shard games, finished shards are skipped.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.environ.get("RIICHI_EVAL_SRC", os.path.expanduser("~/Projects/riichi-eval/src")))
from riichi_eval.logs import parse_game  # noqa: E402
from riichi_eval.actions import action_to_event  # noqa: E402
from riichi_eval.protocol import SACRED, placement_pt  # noqa: E402
from riichi_eval.replay import iter_decisions, kind_of_action  # noqa: E402
from riichi_eval.rollout import driver  # noqa: E402
from riichi_eval.rollout.ev_map import _logged_round_end  # noqa: E402
from riichi_eval.rollout.spec import NEVER, kyoku_spec, log_wall  # noqa: E402

KINDS = ("pass2call", "call2pass", "dama2riichi", "riichi2dama", "disc2disc")
CALLS = (38, 39, 40, 41)
ACT = ("dahai", "reach", "chi", "pon", "daiminkan", "ankan", "kakan", "hora")


def seed_of(*key) -> int:
    return int.from_bytes(hashlib.blake2b(repr(key).encode(), digest_size=8).digest(), "little")


def phase_of(index: int) -> str:
    return "E" if index < 4 else "S12" if index < 6 else "S34" if index < 8 else "W"


def read(path: str):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        g = parse_game([ln.rstrip("\n") for ln in f if ln.strip()], path)
    # Tenhou logs carry no libriichi seed; this one only feeds start_game and the walls of later kyoku (horizon
    # "game"), never the branch kyoku, whose wall is rebuilt from the log
    g.seed = (0, seed_of(path))
    return g


def logged_event(rd, pos: int, actor: int) -> dict:
    """The human's reaction at live log position pos (ForcedPrefixAgent's reference rule), as mjai."""
    for ev in rd.events[pos:]:
        t = ev["type"]
        if t in ("reach_accepted", "dora") or t == "hora" and int(ev["actor"]) != actor:
            continue
        if t in ACT and int(ev["actor"]) == actor:
            return {k: v for k, v in ev.items() if k not in ("meta", "deltas", "ura_markers")}
        return {"type": "none"}
    return {"type": "none"}


def value_rows(games, specs, meta, res, grp, pts=SACRED.pts):
    """V (GRP pt for the deciding seat) and raw kyoku points per rollout; NaN where the rollout failed."""
    V = np.full(len(specs), np.nan)
    RAW = np.full(len(specs), np.nan)
    q, seat_, idx = [], [], []
    err = []
    for i, (m, x) in enumerate(zip(meta, res)):
        g, rd, actor = games[m["game"]], games[m["game"]].rounds[m["r"]], m["actor"]
        if x.error:
            err.append(x.error)
            continue
        RAW[i] = x.kyoku_scores[actor] - rd.scores[actor]
        if x.next_kyoku is None:
            V[i] = placement_pt(x.final_scores, actor, pts)
        else:
            pre = [grp.feature(r.index, r.honba, r.kyotaku, r.scores) for r in g.rounds[:m["r"] + 1]]
            q.append(pre + [grp.feature(x.next_kyoku[0], x.next_kyoku[1], x.kyoku_kyotaku, x.kyoku_scores)])
            seat_.append(actor)
            idx.append(i)
    for s in range(0, len(q), 8192):
        V[idx[s:s + 8192]] = grp.expected_pt(q[s:s + 8192], seat_[s:s + 8192])
    return V, RAW, err


def find(games, engines, rng, disc_rate, batch=2048):
    """Candidates over all four seats of every game: dicts with game, r, pos, actor, kind, events per branch."""
    v4, cp = engines
    pool = []
    for gi, g in enumerate(games):
        for d in iter_decisions(g, [0, 1, 2, 3], 4):
            if d.bucket == "forced" or d.logged < 0:
                continue
            m, a = d.mask, d.logged
            want = (a == 45 and m[38:42].any() and not m[43]) or (a in CALLS and not m[43]) \
                or (m[37] and 0 <= a <= 36) or a == 37 \
                or (0 <= a <= 36 and not m[37] and rng.random() < disc_rate)
            if want:
                pool.append((gi, d))
    if not pool:
        return []
    qs = []
    for eng in (v4, cp):
        q = []
        for s in range(0, len(pool), batch):
            ch = pool[s:s + batch]
            q.extend(eng.react_batch([d.obs for _, d in ch], [d.mask for _, d in ch], None)[1])
        q = np.asarray(q, np.float64)
        q[~np.stack([d.mask for _, d in pool])] = -np.inf
        qs.append(q)
    out = []
    for (gi, d), q4, qc in zip(pool, *qs):
        b4, bc, a, m = int(q4.argmax()), int(qc.argmax()), d.logged, d.mask
        kind, alt = None, None
        if a == 45 and b4 in CALLS and bc in CALLS:
            kind, alt = "pass2call", b4
        elif a in CALLS and b4 == 45 and bc == 45:
            kind, alt = "call2pass", 45
        elif m[37] and 0 <= a <= 36 and b4 == 37 and bc == 37:
            kind, alt = "dama2riichi", 37
        elif a == 37 and b4 <= 36 and bc <= 36:
            kind, alt = "riichi2dama", b4
        elif 0 <= a <= 36 and not m[37] and b4 <= 36 and bc <= 36 \
                and kind_of_action(b4) == kind_of_action(bc) != kind_of_action(a):
            kind, alt = "disc2disc", b4
        if kind is None:
            continue
        rd = games[gi].rounds[d.r]
        evs = [logged_event(rd, d.pos, d.actor), action_to_event(alt, d.actor, d.ctx)]
        if kind == "dama2riichi":
            disc = [x for x in range(37) if m[x]]
            evs.append(action_to_event(max(disc, key=lambda x: q4[x]), d.actor, d.ctx))
        if any(e is None for e in evs) or (evs[0]["type"] == "none") != (a == 45):
            continue
        out.append(dict(game=gi, r=d.r, pos=d.pos, actor=d.actor, kind=kind, phase=phase_of(rd.index),
                        turn=d.turn, shanten=d.shanten, n_riichi=d.n_riichi, opened=d.opened, exposed=d.exposed,
                        m4=float(q4[b4] - q4[a]), mc=float(qc[bc] - qc[a]), events=evs))
    return out


def run_shard(paths, engines, grp, walls, disc_rate, rng, batch):
    games, keep = [], []
    for p in paths:
        try:
            games.append(read(p))
            keep.append(p)
        except Exception:  # noqa: BLE001
            pass
    cands = find(games, engines, rng, disc_rate)
    specs, meta = [], []
    for ci, c in enumerate(cands):
        g = games[c["game"]]
        rd = g.rounds[c["r"]]
        try:
            wall = log_wall(rd, seed_of(g.path, c["r"]))
        except ValueError:
            c["skip"] = "wall"
            continue
        for k in range(walls + 1):
            rs = None if k == 0 else seed_of(g.path, c["r"], c["pos"], k)
            for b, ev in enumerate(c["events"]):
                specs.append(kyoku_spec(g, c["r"], seat_engines=[0, 0, 0, 0], branch_pos=c["pos"],
                                        branch_actor=c["actor"], replacement=json.dumps(ev), horizon="kyoku",
                                        reshuffle=rs, wall=wall))
                meta.append(dict(ci=ci, k=k, b=b, game=c["game"], r=c["r"], actor=c["actor"]))
    res = driver.run([engines[0]], specs, batch=batch) if specs else []
    V, RAW, err = value_rows(games, specs, meta, res, grp)
    nb = {ci: len(c["events"]) for ci, c in enumerate(cands)}
    vv = {ci: np.full((walls + 1, nb[ci]), np.nan) for ci in nb}
    rr = {ci: np.full((walls + 1, nb[ci]), np.nan) for ci in nb}
    for m, v, r in zip(meta, V, RAW):
        vv[m["ci"]][m["k"], m["b"]] = v
        rr[m["ci"]][m["k"], m["b"]] = r
    rows = []
    for ci, c in enumerate(cands):
        if c.get("skip") or np.isnan(vv[ci]).any():
            continue
        row = {k: v for k, v in c.items() if k not in ("game", "events")}
        row.update(path=os.path.relpath(games[c["game"]].path, os.path.dirname(os.path.dirname(games[c["game"]].path))),
                   alt=c["events"][1], v=np.round(vv[ci], 4).tolist(), raw=rr[ci].astype(int).tolist())
        rows.append(row)
    return rows, dict(games=len(games), cands=len(cands), rollouts=len(specs), errors=len(err),
                      dropped=len(cands) - len(rows), err_example=err[:2])


def fidelity(paths, engine, batch):
    """Every kyoku replayed in full from the log on its rebuilt wall must end exactly as logged."""
    games = [read(p) for p in paths]
    specs, meta = [], []
    walls_bad = 0
    for gi, g in enumerate(games):
        for r, rd in enumerate(g.rounds):
            try:
                w = log_wall(rd, seed_of(g.path, r))
            except ValueError:
                walls_bad += 1
                continue
            for rs in (None, seed_of(g.path, r, "fid")):
                specs.append(kyoku_spec(g, r, seat_engines=[0, 0, 0, 0], branch_pos=NEVER, horizon="kyoku",
                                        reshuffle=rs, wall=w))
                meta.append((gi, r, rs is not None))
    res = driver.run([engine], specs, batch=batch)
    ok, bad, err = [0, 0], [], []
    for (gi, r, re_), x in zip(meta, res):
        g = games[gi]
        if x.error:
            err.append((g.path, r, x.error[:300]))
            continue
        want = _logged_round_end(g.rounds[r], g.rounds[r + 1] if r + 1 < len(g.rounds) else None)
        if list(x.kyoku_scores) == want:
            ok[re_] += 1
        else:
            bad.append((g.path, r, re_, list(x.kyoku_scores), want))
    n = len(specs) // 2
    print(f"fidelity: {len(games)} games, {n} kyoku; rebuilt wall unusable for {walls_bad}")
    print(f"  logged wall: {ok[0]} / {n} reproduce the logged result; re-dealt hidden tiles: {ok[1]} / {n}")
    print(f"  errors {len(err)}, mismatches {len(bad)}")
    for e in err[:5]:
        print("  ERR", e)
    for b in bad[:5]:
        print("  MISMATCH", b)


def report(out_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, "shard_*.jsonl"))):
        with open(p) as f:
            rows.extend(json.loads(ln) for ln in f)
    print(f"{len(rows):,} disagreements priced, {len({r['path'] for r in rows}):,} games")
    for unit, key in (("GRP pt", "v"), ("raw points", "raw")):
        print(f"\n== d = V(AI) - V(human) in {unit}; mean over re-dealt walls k=1..K (k=0 logged wall in brackets);"
              " 95% CI over decisions")
        by = defaultdict(list)
        for r in rows:
            for ph in (r["phase"], "all"):
                by[(r["kind"], ph)].append(r)
        print(f"  {'kind':<12}{'phase':<6}{'n':>7}{'d (re-dealt)':>22}{'z':>7}{'d (k=0)':>10}"
              f"{'riichi vs dama':>22}{'tile choice':>22}")
        for kind in KINDS:
            for ph in ("E", "S12", "S34", "W", "all"):
                rs = by.get((kind, ph))
                if not rs:
                    continue
                x = np.asarray([r[key] for r in rs], np.float64)   # (n, K+1, branches)
                d = x[:, 1:, 1].mean(1) - x[:, 1:, 0].mean(1)
                d0 = x[:, 0, 1] - x[:, 0, 0]
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
                line = f"  {kind:<12}{ph:<6}{len(d):>7}{d.mean():>+12.3f} ± {1.96 * se:<7.3f}{d.mean() / se:>+7.1f}{d0.mean():>+10.3f}"
                if x.shape[2] == 3:      # dama2riichi: riichi (1) vs v4's dama (2), v4's dama vs the human's tile (0)
                    for a, b in ((1, 2), (2, 0)):
                        e = x[:, 1:, a].mean(1) - x[:, 1:, b].mean(1)
                        line += f"{e.mean():>+12.3f} ± {1.96 * e.std(ddof=1) / np.sqrt(len(e)):<7.3f}"
                print(line)
    nk = sum(1 for r in rows if np.asarray(r["v"]).shape[0] > 1)
    print(f"\nwalls per decision: {np.asarray(rows[0]['v']).shape[0] - 1 if rows else 0} re-dealt + logged; {nk:,} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list")
    ap.add_argument("--out")
    ap.add_argument("--v4")
    ap.add_argument("--cp")
    ap.add_argument("--grp")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--walls", type=int, default=8)
    ap.add_argument("--disc-rate", type=float, default=0.01)
    ap.add_argument("--shard", type=int, default=100)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--only", default="", help="run only shards LO-HI")
    ap.add_argument("--fidelity", type=int, default=0)
    ap.add_argument("--report")
    a = ap.parse_args()
    if a.report:
        report(a.report)
        return
    from riichi_eval.engines.mortal import cap_gpu_mem, load_mortal
    from riichi_eval.rollout.value import GrpValuer
    cap_gpu_mem(a.device)
    paths = [ln.strip() for ln in open(a.list) if ln.strip()]
    v4 = load_mortal(a.v4, a.device, "off", "mortal_v4")
    if a.fidelity:
        fidelity(paths[:a.fidelity], v4, a.batch)
        return
    cp = load_mortal(a.cp, a.device, "off", "cprime")
    grp = GrpValuer(a.grp)
    os.makedirs(a.out, exist_ok=True)
    nshard = (len(paths) + a.shard - 1) // a.shard
    todo = range(nshard)
    if a.only:
        lo, hi = (int(x) for x in a.only.split("-"))
        todo = [k for k in todo if lo <= k <= hi]
    for k in todo:
        out = os.path.join(a.out, f"shard_{k:04d}.jsonl")
        if os.path.exists(out):
            continue
        t0 = time.time()
        rows, st = run_shard(paths[k * a.shard:(k + 1) * a.shard], (v4, cp), grp, a.walls, a.disc_rate,
                             np.random.default_rng(seed_of(20261013, k)), a.batch)
        with open(out + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(out + ".tmp", out)
        kinds = defaultdict(int)
        for r in rows:
            kinds[r["kind"]] += 1
        print(f"shard {k + 1}/{nshard}: {st} {dict(kinds)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
