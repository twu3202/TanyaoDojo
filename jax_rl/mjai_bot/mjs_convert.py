"""Mahjong Soul archive games (nikketryhard/tenhou-houou-mjai, majsoul-{jade,throne}-*.tar) -> Mortal-loadable mjai
logs (one gzip per game), with per-game structural checks and a metadata line per game (2026-10-06).

Two dialects in the source (checked on 1000-2000-game samples per room):
  - throne: standard mjai (13-tile haipai, the dealer's first draw as a `tsumo`, call targets, kan dora events,
    `deltas` and `ura_markers` on hora / ryukyoku). Passed through unchanged.
  - jade: a different converter with systematic defects, repaired where the log determines the answer:
      * the dealer's 14 tiles are all in `tehais`: the 14th becomes the first `tsumo` (which tile was really drawn
        is unknowable; the dealer's first discard is always flagged tedashi, elsewhere the flag is intact);
      * every chi / pon / daiminkan has target = actor: set to the actor of the claimed (last) discard;
      * added kans are labelled `ankan` and concealed kans `kakan` (without `consumed`): swapped back, the kakan's
        `consumed` taken from the actor's pon;
      * `reach_accepted` is written before a ron on the riichi tile: dropped (the riichi was not accepted);
      * no kan dora is ever revealed (0/2020 kans): the indicators are lost, so every kyoku with a kan is dropped;
      * no `deltas` / `ura_markers`: the final scores, hence the final ranking and the all-last kyoku's reward, are
        unknown, so the all-last kyoku is dropped too. Placeholder deltas [0, 0, 0, 0] let libriichi load the game;
        earlier kyoku rewards only need the scores at each kyoku start, which the logs carry.
    Dropped kyoku are listed in the start_game event (`drop_kyoku`, 0-based log order); the training loader skips
    every sample of those kyoku.
Mode (hanchan vs tonpuu) has no per-game field. Throne games ending in the East round all end with a bust (76/76
in a 1000-game sample), and jade's ending positions have the same distribution, so both are taken as hanchan;
the per-archive ending distribution is written to the summary for the record.

    python mjs_convert.py --root ROOT --archive majsoul-throne-2026.tar --out OUT.tar --meta OUT.meta.jsonl.gz
ROOT is the handoff data root (majsoul_recent_manifest.jsonl.gz under training_handoff_20261006/).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import io
import json
import multiprocessing as mp
import os
import tarfile
import time

import zstandard

TILES = {f"{n}{s}" for n in "123456789" for s in "mps"} | {f"5{s}r" for s in "mps"} | set("ESWNPFC")
KNOWN = {"start_game", "start_kyoku", "tsumo", "dahai", "chi", "pon", "daiminkan", "kakan", "ankan", "dora",
         "reach", "reach_accepted", "hora", "ryukyoku", "end_kyoku", "end_game"}


def base(t):
    return t[:2] if t.endswith("r") else t


def convert(args):
    row, path = args
    meta = {"game_id": row["game_id"], "room": row["room"], "day": row["day"], "archive": row["archive"],
            "member": row["member"]}

    def fail(reason):
        meta.update(status="fail", reason=reason)
        return row, None, meta

    with open(path, "rb") as f:
        f.seek(row["offset"])
        blob = f.read(row["size"])
    try:
        text = zstandard.ZstdDecompressor().decompressobj().decompress(blob).decode("utf-8")
        evs = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    except Exception as e:  # noqa: BLE001
        return fail(f"decode: {type(e).__name__}")

    if not evs or evs[0].get("type") != "start_game" or len(evs[0].get("names", [])) != 4:
        return fail("start_game")
    if evs[-1].get("type") != "end_game":
        return fail("no end_game")
    unknown = {e.get("type") for e in evs} - KNOWN
    if unknown:
        return fail(f"event types {sorted(unknown)}")
    sk = [i for i, e in enumerate(evs) if e["type"] == "start_kyoku"]
    if not sk:
        return fail("no kyoku")
    # dialect: the jade converter deals the dealer 14 tiles; every other defect below comes with it
    jade = any(len(e["tehais"][e["oya"]]) == 14 for e in evs if e["type"] == "start_kyoku")
    out = []
    drop = set()        # kyoku indices (log order, 0-based) whose samples training must drop
    fixes = collections.Counter()
    k = -1
    pons = {}           # (actor, base tile) -> the pon's three tiles, current kyoku
    last_dahai = None
    for i, e in enumerate(evs):
        t = e["type"]
        if t == "start_kyoku":
            k += 1
            pons, last_dahai = {}, None
            th, oya = e["tehais"], e["oya"]
            lens = [len(x) for x in th]
            if not jade and lens == [13, 13, 13, 13]:
                pass
            elif jade and lens[oya] == 14 and all(lens[s] == 13 for s in range(4) if s != oya):
                th = [list(x) for x in th]
                draw = th[oya].pop(13)          # which of the 14 was really drawn is unknowable
                out.append(dict(e, tehais=th))
                out.append({"type": "tsumo", "actor": oya, "pai": draw})
                fixes["haipai14"] += 1
                continue
            else:
                return fail(f"haipai lengths {lens}")
        elif jade and t in ("chi", "pon", "daiminkan"):
            # jade writes target = actor on every call; the claimed tile is the last discard
            if last_dahai is None or last_dahai["actor"] == e["actor"] or base(last_dahai["pai"]) != base(e["pai"]):
                return fail("call without a matching discard")
            e = dict(e, target=last_dahai["actor"])
            fixes["call_target"] += 1
            if t == "pon":
                pons[(e["actor"], base(e["pai"]))] = list(e["consumed"]) + [e["pai"]]
            if t == "daiminkan":
                drop.add(k)
        elif jade and t == "ankan":
            # jade labels an added kan as ankan (always on the actor's own pon; 758/758 in a 2000-game sample)
            key = (e["actor"], base(e["consumed"][0]))
            if key not in pons:
                return fail("jade ankan without own pon")
            rest = list(e["consumed"])
            for x in pons[key]:
                if x not in rest:
                    return fail("kakan tiles do not match the pon")
                rest.remove(x)
            if len(rest) != 1:
                return fail("kakan tiles do not match the pon")
            e = {"type": "kakan", "actor": e["actor"], "pai": rest[0], "consumed": pons[key]}
            fixes["ankan_to_kakan"] += 1
            drop.add(k)
        elif jade and t == "kakan":
            # ... and a concealed kan as kakan without `consumed` (never on a pon: 0/826)
            if (e["actor"], base(e["pai"])) in pons:
                return fail("jade kakan on own pon")
            tile = base(e["pai"])
            consumed = [tile + "r", tile, tile, tile] if tile[0] == "5" and tile[1] in "mps" else [tile] * 4
            e = {"type": "ankan", "actor": e["actor"], "consumed": consumed}
            fixes["kakan_to_ankan"] += 1
            drop.add(k)
        elif jade and t == "reach_accepted":
            nxt = evs[i + 1] if i + 1 < len(evs) else {}
            if nxt.get("type") == "hora" and nxt.get("target") == e["actor"]:
                fixes["reach_accepted_before_ron"] += 1   # the riichi tile was ronned: riichi not accepted
                continue
        elif jade and t == "hora" and e["target"] == e["actor"]:
            prev = out[-1]
            if not (prev["type"] == "tsumo" and prev["actor"] == e["actor"]):
                j = len(out) - 1
                while out[j]["type"] in ("hora", "reach_accepted"):
                    j -= 1
                if out[j]["type"] not in ("dahai", "kakan") or base(out[j]["pai"]) != base(e["pai"]):
                    return fail("ron without a matching discard")
                e = dict(e, target=out[j]["actor"])
                fixes["ron_target"] += 1
        if t == "dahai":
            last_dahai = e
        out.append(e)
    # jade logs never reveal a kan dora (0/2020 kans), so every kyoku with a kan is dropped (set above)
    # tile multiset sanity per kyoku: no tile kind more than 4 times among haipai + draws + dora indicators, at most
    # one of each red five
    kyoku_starts = [i for i, e in enumerate(out) if e["type"] == "start_kyoku"] + [len(out)]
    for a, b in zip(kyoku_starts, kyoku_starts[1:]):
        cnt = collections.Counter()
        for t in (t for hand in out[a]["tehais"] for t in hand):
            cnt[t] += 1
        cnt[out[a]["dora_marker"]] += 1
        for e in out[a + 1:b]:
            if e["type"] == "tsumo":
                cnt[e["pai"]] += 1
            elif e["type"] == "dora":
                cnt[e["dora_marker"]] += 1
        if any(t not in TILES for t in cnt):
            return fail("unknown tile")
        kinds = collections.Counter()
        for t, c in cnt.items():
            kinds[base(t)] += c
        if max(kinds.values()) > 4 or any(cnt[f"5{s}r"] > 1 for s in "mps"):
            return fail("tile count")
    last = out[kyoku_starts[-2]]
    meta["n_kyoku"] = len(kyoku_starts) - 1
    meta["end"] = f"{last['bakaze']}{last['kyoku']}"
    # all-last kyoku: deltas present (throne) or placeholder (jade; final ranking unknown -> its samples dropped)
    al_end = [e for e in out[kyoku_starts[-2]:] if e["type"] in ("hora", "ryukyoku")]
    if not al_end:
        return fail("all-last kyoku without hora/ryukyoku")
    if all("deltas" in e for e in al_end):
        if jade:
            return fail("jade game with deltas")
    elif jade and not any("deltas" in e for e in al_end):
        for e in al_end:
            e["deltas"] = [0, 0, 0, 0]
        drop.add(meta["n_kyoku"] - 1)
    else:
        return fail("missing deltas in all-last")
    if jade:
        out[0] = dict(out[0], source="majsoul-jade (repaired, mjs_convert.py)", drop_kyoku=sorted(drop))
    meta.update(status="ok", jade_repaired=jade, drop_kyoku=sorted(drop), fixes=dict(fixes))
    meta["path"] = f"{row['room']}/{row['day'][:7].replace('-', '')}/{row['game_id']}.mjson.gz"
    payload = gzip.compress(("\n".join(json.dumps(e, ensure_ascii=False, separators=(",", ":")) for e in out)
                             + "\n").encode("utf-8"), compresslevel=6)
    return row, payload, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--archive", required=True, help="archive file name, e.g. majsoul-jade-2025.tar")
    ap.add_argument("--out", required=True, help="output .tar of converted .mjson.gz members")
    ap.add_argument("--meta", required=True, help="per-game metadata .jsonl.gz")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    man = os.path.join(args.root, "training_handoff_20261006", "majsoul_recent_manifest.jsonl.gz")
    rows = [r for r in (json.loads(ln) for ln in gzip.open(man, "rt")) if r["archive"] == args.archive]
    rows.sort(key=lambda r: r["offset"])
    if args.limit:
        rows = rows[:args.limit]
    path = os.path.join(args.root, rows[0]["archive_path"])
    t0 = time.time()
    stats = collections.Counter()
    ends = collections.Counter()
    with mp.Pool(args.procs) as pool, tarfile.open(args.out, "w") as tar, gzip.open(args.meta, "wt") as mf:
        for i, (row, payload, meta) in enumerate(pool.imap(convert, ((r, path) for r in rows), chunksize=256)):
            stats[meta["status"] if meta["status"] == "ok" else "fail: " + meta["reason"]] += 1
            if payload is not None:
                stats["jade_repaired"] += meta["jade_repaired"]
                stats["kyoku"] += meta["n_kyoku"]
                stats["kyoku_dropped"] += len(meta["drop_kyoku"])
                for f_, n_ in meta["fixes"].items():
                    stats["fix_" + f_] += n_
                ends[meta["end"]] += 1
                ti = tarfile.TarInfo(meta["path"])
                ti.size = len(payload)
                ti.mtime = 0
                tar.addfile(ti, io.BytesIO(payload))
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            if (i + 1) % 100000 == 0:
                print(f"  {i + 1:,}/{len(rows):,}  {time.time() - t0:.0f}s", flush=True)
    summary = {"archive": args.archive, "rows": len(rows), "stats": dict(stats),
               "ends": dict(sorted(ends.items())), "seconds": round(time.time() - t0, 1)}
    print(json.dumps(summary, ensure_ascii=False))
    with open(args.out + ".summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
