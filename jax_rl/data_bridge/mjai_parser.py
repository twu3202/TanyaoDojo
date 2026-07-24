"""
R1 数据桥·第一件:mjai 牌谱解析器。
输入:天凤转档的 *.mjson / *.mjson.gz(每文件一场,JSON lines,mjai 事件流)。
输出:结构化 Game/Kyoku(初始手牌、宝牌、庄家、逐步事件),供 replay.py 重放进 Mahjax。
纯 Python、零依赖,可独立跑在全量 251 万局上做统计与格式校验。
"""
from __future__ import annotations
import gzip
import json
from dataclasses import dataclass, field
from typing import Iterator, Optional

MJAI_TILES = [f"{r}{s}" for s in "mps" for r in range(1, 10)] + list("ESWNPFC")
RED_TILES = {"5mr", "5pr", "5sr"}
KNOWN_EVENTS = {
    "start_game", "start_kyoku", "tsumo", "dahai", "pon", "chi",
    "daiminkan", "ankan", "kakan", "reach", "reach_accepted",
    "dora", "hora", "ryukyoku", "end_kyoku", "end_game", "none",
}


def tile_ok(t: str) -> bool:
    return t in MJAI_TILES or t in RED_TILES or t == "?"


@dataclass
class Kyoku:
    bakaze: str = "E"
    kyoku_num: int = 1
    honba: int = 0
    kyotaku: int = 0
    oya: int = 0
    scores: list = field(default_factory=list)
    dora_markers: list = field(default_factory=list)
    tehais: list = field(default_factory=list)          # 4 × 13 张
    events: list = field(default_factory=list)          # start_kyoku 之后的原始事件 dict
    end_type: Optional[str] = None                       # hora / ryukyoku


@dataclass
class Game:
    names: list = field(default_factory=list)
    kyokus: list = field(default_factory=list)


class ParseError(Exception):
    pass


def parse_game(path) -> Game:
    op = gzip.open if str(path).endswith(".gz") else open
    game = Game()
    cur: Optional[Kyoku] = None
    with op(path, "rt", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError as e:
                raise ParseError(f"{path}:{ln} 非法 JSON: {e}")
            t = ev.get("type")
            if t not in KNOWN_EVENTS:
                raise ParseError(f"{path}:{ln} 未知事件类型 {t!r}")
            if t == "start_game":
                game.names = ev.get("names", [])
            elif t == "start_kyoku":
                cur = Kyoku(
                    bakaze=ev["bakaze"], kyoku_num=ev["kyoku"], honba=ev["honba"],
                    kyotaku=ev.get("kyotaku", 0), oya=ev["oya"],
                    scores=ev.get("scores", []), dora_markers=[ev["dora_marker"]],
                    tehais=ev["tehais"],
                )
                for hand in cur.tehais:
                    if len(hand) != 13:
                        raise ParseError(f"{path}:{ln} 初始手牌非 13 张: {len(hand)}")
                    for x in hand:
                        if not tile_ok(x):
                            raise ParseError(f"{path}:{ln} 非法牌面 {x!r}")
            elif t in ("end_kyoku",):
                if cur is not None:
                    game.kyokus.append(cur)
                    cur = None
            elif t == "end_game":
                break
            else:
                if cur is None:
                    raise ParseError(f"{path}:{ln} 局外事件 {t}")
                if t == "dora":
                    cur.dora_markers.append(ev["dora_marker"])
                if t in ("hora", "ryukyoku"):
                    cur.end_type = t
                for k in ("pai",):
                    if k in ev and not tile_ok(ev[k]):
                        raise ParseError(f"{path}:{ln} 非法牌面 {ev[k]!r}")
                for x in ev.get("consumed", []):
                    if not tile_ok(x):
                        raise ParseError(f"{path}:{ln} 非法牌面 {x!r}")
                cur.events.append(ev)
    if not game.kyokus:
        raise ParseError(f"{path} 无有效对局")
    return game


def iter_draws(kyoku: Kyoku) -> Iterator[tuple]:
    """按发生顺序产出 (actor, tile) 的摸牌序列(重构牌山用)。"""
    for ev in kyoku.events:
        if ev["type"] == "tsumo":
            yield ev["actor"], ev["pai"]


def kyoku_stats(kyoku: Kyoku) -> dict:
    n_draws = sum(1 for _ in iter_draws(kyoku))
    n_dahai = sum(1 for e in kyoku.events if e["type"] == "dahai")
    n_meld = sum(1 for e in kyoku.events if e["type"] in ("pon", "chi", "daiminkan", "ankan", "kakan"))
    return {"draws": n_draws, "dahai": n_dahai, "melds": n_meld, "end": kyoku.end_type}


if __name__ == "__main__":
    import sys, glob, random
    pat = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(pat))
    random.Random(0).shuffle(files)
    files = files[: int(sys.argv[2]) if len(sys.argv) > 2 else 100]
    ok = bad = kyokus = horas = 0
    for fp in files:
        try:
            g = parse_game(fp)
            ok += 1
            kyokus += len(g.kyokus)
            horas += sum(1 for k in g.kyokus if k.end_type == "hora")
        except ParseError as e:
            bad += 1
            print("PARSE_FAIL:", e)
    print(f"files ok={ok} bad={bad}  kyokus={kyokus}  hora_rate={horas/max(kyokus,1):.3f}")
