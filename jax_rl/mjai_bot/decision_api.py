"""A decision-model interface for Mortal-format riichi models: game state in, a probability over the legal options
out, in one forward pass. The request and response follow the shape of the `/v1/decisions` APIs (a list of `choice`
questions, answered with the chosen value, a probability per option and a confidence). The state is the mjai event
log up to the decision, seen from the deciding seat.

    python decision_api.py serve --weights MODEL.pth [--port 8000] [--name tanyao-408k]
    curl -s localhost:8000/v1/decisions -d @request.json

Request (fields beyond `input` and `questions` are optional):
    {"model": "tanyao-408k",
     "input": "<mjai events, one JSON object per line>",  # or a list of event objects, or a list of messages whose
                                                         # content holds that text as an input_text part
     "player_id": 0,                                     # the deciding seat; inferred when exactly one seat can act
     "temperature": 1.0,
     "questions": [{"type": "choice", "name": "action",
                    "choices": [{"value": "dahai:5m"}, {"value": "reach"}]}]}   # omit "choices" for every legal option
Response:
    {"model": ..., "answers": [{"type": "choice", "name": "action", "choice": "dahai:C", "confidence": 0.62,
       "probabilities": [{"value": "dahai:C", "probability": 0.62, "q": 0.31, "description": "discard C"}, ...]}],
     "state": {"player_id": 0, "hand": "123m0p...", "shanten": 1, ...}, "usage": {"decisions": 1, ...}}

What the numbers are: the model is a Q network. `q` is its value estimate of each option (placement points on the
training scale, higher is better), and `probability` is softmax(q / temperature) over the asked options: a
Boltzmann policy, not a calibrated probability that the option is best. The chosen value is the argmax of q.
Option values: `dahai:<tile>` (mjai tile names, red fives 5mr/5pr/5sr), `reach`, `chi:low|mid|high`, `pon`,
`daiminkan`, `ankan:<tile>`, `kakan:<tile>`, `hora`, `ryukyoku`, `none` (pass).

Also `nanikiru_events(...)`: builds the events of a "what to discard" position from a hand in Tenhou notation, for
demos without a game log. AGPL-3.0 (links libriichi, imports Mortal's model code), like the rest of this directory.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.environ.get("MORTAL_DIR", os.path.expanduser("~/Projects/better_mortal/Mortal/mortal")))
from libriichi.state import PlayerState  # noqa: E402
from model import Brain, DQN  # noqa: E402

TILES = [f"{n}{s}" for s in "mps" for n in range(1, 10)] + ["E", "S", "W", "N", "P", "F", "C", "5mr", "5pr", "5sr"]
VALUES = [f"dahai:{t}" for t in TILES] + ["reach", "chi:low", "chi:mid", "chi:high", "pon", "kan", "hora",
                                          "ryukyoku", "none"]
assert len(VALUES) == 46
KAN = 42


def _desc(value: str, ps: PlayerState) -> str:
    kind, _, arg = value.partition(":")
    last = ps.last_kawa_tile()
    return {"dahai": f"discard {arg}", "reach": "declare riichi", "chi": f"chi {last} ({arg})", "pon": f"pon {last}",
            "daiminkan": f"open kan on {last}", "ankan": f"closed kan of {arg}", "kakan": f"added kan of {arg}",
            "hora": "win (tsumo)" if ps.last_cans.can_tsumo_agari else "win (ron)", "ryukyoku": "abortive draw (nine terminals)",
            "none": "pass"}.get(kind, value)


class DecisionModel:
    def __init__(self, weights: str, device: str = "cpu", name: str | None = None):
        st = torch.load(weights, weights_only=True, map_location="cpu")
        cfg = st["config"]
        self.version = cfg["control"].get("version", 1)
        self.brain = Brain(version=self.version, conv_channels=cfg["resnet"]["conv_channels"],
                           num_blocks=cfg["resnet"]["num_blocks"]).eval()
        self.dqn = DQN(version=self.version).eval()
        self.brain.load_state_dict(st["mortal"])
        self.dqn.load_state_dict(st["current_dqn"])
        self.device = torch.device(device)
        self.brain.to(self.device)
        self.dqn.to(self.device)
        self.name = name or os.path.basename(weights).rsplit(".", 1)[0]

    @torch.inference_mode()
    def _q(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(obs[None], device=self.device)
        m = torch.as_tensor(mask[None], device=self.device)
        return self.dqn(self.brain(o), m)[0].float().cpu().numpy()

    def options(self, events: list[dict], player_id: int) -> tuple[dict[str, float], PlayerState]:
        """Legal options of `player_id` after `events`, with their q. A kan in one's own turn is split into one option
        per tile (the model's kan-select head picks among them), carrying the q of the kan decision."""
        ps = PlayerState(player_id)
        for ev in events:
            cans = ps.update(json.dumps(ev))
        if not cans.can_act:
            raise ValueError(f"seat {player_id} has no decision to make after the last event")
        obs, mask = ps.encode_obs(self.version, False)
        q = self._q(obs, mask)
        out = {VALUES[i]: float(q[i]) for i in np.flatnonzero(mask) if i != KAN}
        if mask[KAN]:
            if cans.can_daiminkan:
                out["daiminkan"] = float(q[KAN])
            else:
                kobs, kmask = ps.encode_obs(self.version, True)
                kq = self._q(kobs, kmask)
                ank, kak = set(ps.ankan_candidates()), set(ps.kakan_candidates())
                sel = [i for i in np.flatnonzero(kmask) if i < 37]
                w = np.exp(kq[sel] - kq[sel].max())
                w /= w.sum()
                for i, p in zip(sel, w):
                    t = TILES[i]
                    kind = "ankan" if t in ank or t.rstrip("r") in ank else "kakan" if t in kak or t.rstrip("r") in kak else "kan"
                    # the kan decision's q is shared; log(p) orders the tiles the way the kan-select head ranks them
                    out[f"{kind}:{t}"] = float(q[KAN]) + float(np.log(max(p, 1e-12)))
        return out, ps

    def decide(self, events: list[dict], player_id: int, choices: list[str] | None = None,
               temperature: float = 1.0) -> dict[str, Any]:
        legal, ps = self.options(events, player_id)
        asked = list(legal) if not choices else choices
        bad = [c for c in asked if c not in legal]
        if bad:
            raise ValueError(f"not legal here: {bad}; legal options: {sorted(legal)}")
        q = np.array([legal[c] for c in asked])
        z = (q - q.max()) / max(temperature, 1e-6)
        p = np.exp(z) / np.exp(z).sum()
        order = np.argsort(-q)
        probs = [{"value": asked[i], "probability": round(float(p[i]), 6), "q": round(float(q[i]), 5),
                  "description": _desc(asked[i], ps)} for i in order]
        return {"choice": asked[order[0]], "confidence": round(float(p[order[0]]), 6), "probabilities": probs,
                "state": {"player_id": player_id, "hand": _hand(ps), "shanten": int(ps.shanten), "turn": float(ps.at_turn),
                          "riichi_declared": bool(ps.self_riichi_declared)}}


def _hand(ps: PlayerState) -> str:
    counts, akas = list(ps.tehai), list(ps.akas_in_hand)
    out = []
    for si, s in enumerate("mps"):
        digits = ""
        for n in range(9):
            c = counts[si * 9 + n]
            if n == 4 and c and akas[si]:
                digits += "0" + "5" * (c - 1)
            else:
                digits += str(n + 1) * c
        if digits:
            out.append(digits + s)
    hz = "".join(str(i + 1) * counts[27 + i] for i in range(7))
    return "".join(out) + (hz + "z" if hz else "")


# ---- request handling (OpenAI-style /v1/decisions) ----

def _events_from_input(inp: Any) -> list[dict]:
    if isinstance(inp, list) and inp and isinstance(inp[0], dict) and inp[0].get("type") == "message":
        texts = []
        for msg in inp:
            content = msg.get("content")
            if isinstance(content, str):
                texts.append(content)
            else:
                texts += [c.get("text", "") for c in content if c.get("type") in ("input_text", "text")]
        inp = "\n".join(texts)
    if isinstance(inp, str):
        inp = inp.strip()
        if inp.startswith("["):
            return json.loads(inp)
        return [json.loads(line) for line in inp.splitlines() if line.strip()]
    if isinstance(inp, list):
        return inp
    raise ValueError("input must be mjai events: a JSON-lines string, a list of events, or messages holding them")


def _infer_player(events: list[dict]) -> int:
    able = []
    for pid in range(4):
        ps = PlayerState(pid)
        try:
            for ev in events:
                cans = ps.update(json.dumps(ev))
        except Exception:  # noqa: BLE001
            continue
        if cans.can_act:
            able.append(pid)
    if len(able) != 1:
        raise ValueError(f"cannot infer the deciding seat (seats able to act: {able}); pass player_id")
    return able[0]


def handle(model: DecisionModel, body: dict) -> tuple[int, dict]:
    try:
        events = _events_from_input(body.get("input"))
        pid = body.get("player_id")
        pid = _infer_player(events) if pid is None else int(pid)
        temp = float(body.get("temperature", 1.0))
        answers, state = [], None
        for qi, qs in enumerate(body.get("questions") or [{"type": "choice", "name": "action"}]):
            name = qs.get("name", f"q{qi}")
            if qs.get("type", "choice") != "choice":
                answers.append({"type": "refusal", "name": name, "reason": "only choice questions are supported"})
                continue
            choices = [c["value"] if isinstance(c, dict) else c for c in qs.get("choices") or []]
            r = model.decide(events, pid, choices or None, temp)
            state = r.pop("state")
            answers.append({"type": "choice", "name": name, **r})
        return 200, {"object": "decision", "model": model.name, "answers": answers, "state": state,
                     "usage": {"decisions": len(answers), "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
    except (ValueError, KeyError, json.JSONDecodeError, RuntimeError) as e:
        return 400, {"error": {"type": "invalid_request_error", "message": str(e)}}


# ---- "what to discard" positions from a hand ----

def parse_tiles(s: str) -> list[str]:
    """Tenhou notation, e.g. '123m406p789s1122z' (0 = red five; z: 1-7 = E S W N P F C)."""
    out, digits = [], ""
    for ch in s.replace(" ", ""):
        if ch.isdigit():
            digits += ch
        elif ch in "mpsz":
            for d in digits:
                if ch == "z":
                    if not "1" <= d <= "7":
                        raise ValueError(f"honor tiles are 1z-7z, got {d}z")
                    out.append("ESWNPFC"[int(d) - 1])
                else:
                    out.append(f"5{ch}r" if d == "0" else f"{d}{ch}")
            digits = ""
        else:
            raise ValueError(f"unexpected character {ch!r} in {s!r}")
    if digits:
        raise ValueError(f"digits without a suit at the end of {s!r}")
    return out


def nanikiru_events(hand: str, dora: str = "1z", bakaze: str = "E", kyoku: int = 1, honba: int = 0,
                    seat_wind: str = "E", scores: list[int] | None = None, rivers: list[str] | None = None,
                    riichi: list[bool] | None = None) -> tuple[list[dict], int]:
    """Events for a closed-hand discard decision. `hand` holds 14 tiles, the last one being the tile just drawn.
    `rivers` lists the discards so far for the seats in the order self, shimocha, toimen, kamicha (Tenhou notation);
    `riichi` marks seats (same order) whose last listed discard was their riichi declaration."""
    tiles = parse_tiles(hand)
    if len(tiles) != 14:
        raise ValueError(f"the hand needs 14 tiles (13 + the drawn tile), got {len(tiles)}")
    oya = kyoku - 1
    pid = (oya + "ESWN".index(seat_wind)) % 4
    rel = [parse_tiles(r) if r else [] for r in (rivers or ["", "", "", ""])]
    riichi = riichi or [False] * 4
    if riichi[0]:
        raise ValueError("this demo decides for a seat that has not declared riichi")
    seat_river = {(pid + k) % 4: rel[k] for k in range(4)}
    seat_riichi = {(pid + k) % 4: riichi[k] for k in range(4)}
    tehais = [["?"] * 13 for _ in range(4)]
    tehais[pid] = tiles[:13]
    sc = scores or [25000] * 4
    evs = [{"type": "start_game", "names": ["0", "1", "2", "3"]},
           {"type": "start_kyoku", "bakaze": bakaze, "dora_marker": parse_tiles(dora)[0], "kyoku": kyoku,
            "honba": honba, "kyotaku": 0, "oya": oya, "scores": sc, "tehais": tehais}]
    # replay the rivers turn by turn from the dealer; our own past discards are drawn and discarded at once
    n_self = len(seat_river[pid])
    order = [(oya + k) % 4 for k in range(4)]
    for t in range(n_self + 1):
        for s in order:
            if s == pid and t == n_self:
                evs.append({"type": "tsumo", "actor": pid, "pai": tiles[13]})
                for s2 in order[order.index(pid) + 1:]:
                    if len(seat_river[s2]) != n_self:
                        raise ValueError("river lengths do not fit the turn order: seats after you need as many discards as you")
                for s2 in order[:order.index(pid)]:
                    if len(seat_river[s2]) != n_self + 1:
                        raise ValueError("river lengths do not fit the turn order: seats before you (from the dealer) need one more discard than you")
                return evs, pid
            if t >= len(seat_river[s]):
                raise ValueError("river lengths do not fit the turn order")
            tile = seat_river[s][t]
            evs.append({"type": "tsumo", "actor": s, "pai": tile if s == pid else "?"})
            last = t == len(seat_river[s]) - 1
            if last and seat_riichi[s]:
                evs.append({"type": "reach", "actor": s})
            evs.append({"type": "dahai", "actor": s, "pai": tile, "tsumogiri": s != pid})
            if last and seat_riichi[s]:
                sc = list(sc)
                sc[s] -= 1000
                evs.append({"type": "reach_accepted", "actor": s, "deltas": [-1000 if i == s else 0 for i in range(4)],
                            "scores": sc})
    raise AssertionError("unreachable")


def serve(model: DecisionModel, port: int):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path.rstrip("/") != "/v1/decisions":
                self.send_error(404)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                code, out = handle(model, body)
            except json.JSONDecodeError as e:
                code, out = 400, {"error": {"type": "invalid_request_error", "message": f"bad JSON: {e}"}}
            data = json.dumps(out).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    print(f"serving {model.name} on http://127.0.0.1:{port}/v1/decisions")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--weights", required=True)
    s.add_argument("--name")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--device", default="cpu")
    n = sub.add_parser("nanikiru", help="print the decision for a hand, e.g. nanikiru --weights W 123m406p789s11223z")
    n.add_argument("hand")
    n.add_argument("--weights", required=True)
    n.add_argument("--dora", default="1z")
    n.add_argument("--seat-wind", default="E")
    args = ap.parse_args()
    model = DecisionModel(args.weights, getattr(args, "device", "cpu"), getattr(args, "name", None))
    if args.cmd == "serve":
        serve(model, args.port)
    else:
        events, pid = nanikiru_events(args.hand, dora=args.dora, seat_wind=args.seat_wind)
        print(json.dumps(model.decide(events, pid), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
