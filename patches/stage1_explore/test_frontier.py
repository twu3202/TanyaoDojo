"""Consistency check of explore_engine.frontier against a plain-Python reference built from PlayerState replay.

For every can-act decision of every seat in some logs: v5 obs + mask from PlayerState, Q from a real model, then
  engine:    frontier(q, obs, mask) on the whole batch (vectorised, reads obs rows)
  reference: riichi flags and riichi players' own discards from the event stream, self-riichi from PlayerState,
             the same pair rules written with Python sets.
Kind, alt and margin must agree exactly (margin to fp tolerance). Also checks that row 870 is the kan-select flag.

    MORTAL_CFG=... python test_frontier.py LOG_DIR N_GAMES MODEL.pth
"""
import glob
import gzip
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/twu/Projects/better_mortal/Mortal/mortal")
from libriichi.state import PlayerState  # noqa: E402
from model import Brain, DQN  # noqa: E402
from explore_engine import frontier, KINDS, V4_ROWS  # noqa: E402

TILES = [f"{n}{s}" for s in "mps" for n in range(1, 10)] + ["E", "S", "W", "N", "P", "F", "C"]
KIND_OF = list(range(34)) + [4, 13, 22]


def main():
    log_dir, n_games, model_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    st = torch.load(model_path, weights_only=True, map_location="cpu")
    cfg = st["config"]
    brain = Brain(version=4, conv_channels=cfg["resnet"]["conv_channels"], num_blocks=cfg["resnet"]["num_blocks"]).to(dev).eval()
    dqn = DQN(version=4).to(dev).eval()
    brain.load_state_dict(st["mortal"])
    dqn.load_state_dict(st["current_dqn"])

    obs_l, mask_l, ref_l = [], [], []
    ks_checked = ks_ok = 0
    for p in sorted(glob.glob(log_dir + "/*.json.gz"))[:n_games]:
        lines = gzip.open(p, "rt").read().splitlines()
        ps = [PlayerState(s) for s in range(4)]
        riichi, own = [False] * 4, [set() for _ in range(4)]
        for ln in lines:
            ev = json.loads(ln)
            t = ev["type"]
            if t == "start_kyoku":
                riichi, own = [False] * 4, [set() for _ in range(4)]
            cans = [s.update(ln) for s in ps]
            if t == "dahai":
                own[ev["actor"]].add({"5mr": "5m", "5pr": "5p", "5sr": "5s"}.get(ev["pai"], ev["pai"]))
            elif t == "reach_accepted":
                riichi[ev["actor"]] = True
            for s in range(4):
                if not cans[s].can_act:
                    continue
                o5, m = ps[s].encode_obs(5, False)
                o5, m = np.asarray(o5), np.asarray(m, bool)
                if cans[s].can_ankan or cans[s].can_kakan:
                    ok_, _ = ps[s].encode_obs(4, True)
                    ks_checked += 1
                    ks_ok += int((np.asarray(ok_)[870] == 1).all() and (o5[870] == 0).all())
                rs = [o for o in range(4) if o != s and riichi[o]]
                in_riichi = ps[s].self_riichi_declared or ps[s].self_riichi_accepted
                safe = set.intersection(*[own[o] for o in rs]) if rs else None
                obs_l.append(o5)
                mask_l.append(m)
                ref_l.append((bool(rs) and not in_riichi, safe))

    obs = torch.as_tensor(np.stack(obs_l), device=dev)
    masks = torch.as_tensor(np.stack(mask_l), device=dev)
    qs = []
    with torch.no_grad():
        for i in range(0, len(obs), 2048):
            qs.append(dqn(brain(obs[i:i + 2048, :V4_ROWS]), masks[i:i + 2048]).float())
    q = torch.cat(qs)
    greedy, alt, margin, kind = frontier(q, obs, masks)
    greedy, alt, margin, kind = (x.cpu().numpy() for x in (greedy, alt, margin, kind))
    qn, mn = q.cpu().numpy(), masks.cpu().numpy()

    bad, counts = 0, np.zeros(len(KINDS), int)
    for i, (exposed, safe) in enumerate(ref_l):
        m = mn[i]
        qq = np.where(m, qn[i], -np.inf)
        g = int(np.argmax(qq))
        disc = [a for a in range(37) if m[a]]
        pairs = []
        if exposed:
            is_safe = {a: TILES[KIND_OF[a]] in safe for a in disc}
            safe_a = [a for a in disc if is_safe[a]]
            unsafe_a = [a for a in disc if not is_safe[a]]
            if (g == 37 or (g < 37 and not is_safe[g])) and safe_a:
                a = max(safe_a, key=lambda x: qq[x])
                pairs.append((qq[g] - qq[a], 0, a))
            if g < 37 and is_safe[g] and unsafe_a:
                a = max(unsafe_a, key=lambda x: qq[x])
                pairs.append((qq[g] - qq[a], 1, a))
        if m[37]:
            if g == 37 and disc:
                a = max(disc, key=lambda x: qq[x])
                pairs.append((qq[g] - qq[a], 2, a))
            if g < 37:
                pairs.append((qq[g] - qq[37], 3, 37))
        want = min(pairs) if pairs else None
        got_k = int(kind[i])
        if want is None:
            ok = got_k == -1
        else:
            ok = got_k == want[1] and int(alt[i]) == want[2] and abs(float(margin[i]) - float(want[0])) < 1e-4
            counts[want[1]] += 1
        if int(greedy[i]) != g:
            ok = False
        if not ok:
            bad += 1
            if bad <= 5:
                print("MISMATCH", i, "exposed", exposed, "greedy", g, int(greedy[i]), "want", want,
                      "got", got_k, int(alt[i]), float(margin[i]))
    print(f"{len(ref_l):,} decisions from {n_games} games: {bad} mismatches; pairs by kind "
          + ", ".join(f"{k} {c}" for k, c in zip(KINDS, counts)))
    print(f"kan-select row 870: {ks_ok}/{ks_checked} decisions with a kan option behave as expected")
    mq = margin[kind >= 0] / (2.390 / 90)
    for lo, hi in ((0, 5), (5, 10), (10, 20)):
        print(f"  pairs with margin in [{lo},{hi}) pt: {((mq >= lo) & (mq < hi)).sum():,}")


if __name__ == "__main__":
    main()
