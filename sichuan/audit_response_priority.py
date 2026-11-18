"""Count legal ron opportunities suppressed by a supplied pre-fix engine.

Measures seeded random win-first play, NOT trained-policy score bias.
"""
import argparse
import importlib.util
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor


def audit_chunk(args):
    legacy, start, count = args
    spec = importlib.util.spec_from_file_location("legacy_reference", legacy)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    affected = events = suppressed = decisions = 0
    examples = []
    for seed in range(start, start + count):
        g = mod.SichuanGame(seed)
        rng = random.Random(seed ^ 0xABCDEF)
        hit = False
        while g.phase != "over":
            i, actions = g.legal_actions()
            wins = [a for a in actions if a[0] in ("zimo", "ron")]
            a = rng.choice(wins or actions)
            decisions += 1
            if g.phase == "response" and a[0] in ("peng", "zhigang") and not g.ron_winners:
                tile = g.pending_discard[1]
                later = []
                for j in g.response_queue[1:]:
                    p = g.players[j]
                    h = p.hand[:]
                    h[tile] += 1
                    if mod.is_hu(h, len(p.melds), p.void):
                        later.append(j)
                if later:
                    hit = True
                    events += 1
                    suppressed += len(later)
                    if len(examples) < 5:
                        examples.append(dict(seed=seed, turn=g.turn_guard, claimant=i,
                                             claim=a[0], later_winners=later))
            g.step(a)
            assert g.tile_conservation() == 108 and sum(g.scores()) == 0
        affected += hit
    return dict(games=count, affected_games=affected, claim_events=events,
                suppressed_winners=suppressed, decisions=decisions, examples=examples)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--legacy", required=True)
    p.add_argument("--games", type=int, default=100000)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    t = time.monotonic()
    chunks = [(a.legacy, start, min(1000, a.games-start)) for start in range(0, a.games, 1000)]
    totals = dict(games=0, affected_games=0, claim_events=0, suppressed_winners=0, decisions=0)
    examples = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(audit_chunk, chunks):
            for k in totals:
                totals[k] += row[k]
            examples.extend(row['examples'])
    totals.update(affected_game_fraction=totals['affected_games']/totals['games'],
                  elapsed_seconds=round(time.monotonic()-t, 2), examples=examples[:10],
                  policy="uniform legal actions, win first; seeds 0..games-1; RNG seed xor 0xABCDEF",
                  limitation="Opportunity count under legacy random trajectories, not trained-policy score bias.")
    print(json.dumps(totals, ensure_ascii=False, indent=2))
