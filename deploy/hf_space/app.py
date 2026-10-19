"""TanyaoDojo decision demo (Hugging Face Space): a "what to discard" (nanikiru) board and a /v1/decisions endpoint,
both answered by one forward pass of a Mortal-format Q network trained by TanyaoDojo.

Local run (from the repository):
    TANYAO_WEIGHTS=/path/model.pth MORTAL_DIR=/path/Mortal/mortal PYTHONPATH=/dir/with/libriichi python deploy/hf_space/app.py
In the Space, build_space.sh bundles decision_api.py, Mortal's model.py and a Linux libriichi next to this file.
AGPL-3.0: the Space links libriichi and Mortal's model code; source at https://github.com/twu3202/TanyaoDojo."""
import json
import os
import random
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "..", "jax_rl", "mjai_bot")):
    sys.path.insert(0, os.path.abspath(p))
os.environ.setdefault("MORTAL_DIR", HERE)

import gradio as gr  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

import tiles as T  # noqa: E402
from decision_api import DecisionModel, handle, nanikiru_events, parse_tiles  # noqa: E402

WEIGHTS = os.environ.get("TANYAO_WEIGHTS", os.path.join(HERE, "model.pth"))
NAME = os.environ.get("TANYAO_MODEL_NAME", "tanyao-408k")
API_ON = os.environ.get("TANYAO_DECISIONS_API", "1") == "1"
model = DecisionModel(WEIGHTS, name=NAME)
TILE_DIR = os.path.join(tempfile.gettempdir(), "tanyao_tiles")
ICONS = T.write_all(TILE_DIR)
gr.set_static_paths(paths=[TILE_DIR])

WINDS = ["E", "S", "W", "N"]
WIND_LABEL = {"E": "East 東", "S": "South 南", "W": "West 西", "N": "North 北"}
TILE_LABEL = {"E": "East 東", "S": "South 南", "W": "West 西", "N": "North 北", "P": "White 白", "F": "Green 發",
              "C": "Red 中", "5mr": "red 5m", "5pr": "red 5p", "5sr": "red 5s"}
TARGETS = ["Hand", "Dora indicator", "Your river", "Shimocha river", "Toimen river", "Kamicha river"]
SEATS = ["You", "Shimocha (right)", "Toimen (across)", "Kamicha (left)"]


def label(t):
    return TILE_LABEL.get(t, t)


def encode(names):
    """mjai tile names -> Tenhou notation, keeping the order."""
    out, suit = "", None
    for t in names:
        d, s = (("0", t[1]) if t.endswith("r") else (t[0], t[1])) if t[0].isdigit() else (str("ESWNPFC".index(t) + 1), "z")
        if suit is not None and s != suit:
            out += suit
        out += d
        suit = s
    return out + (suit or "")


def sort_key(t):
    return (3, "ESWNPFC".index(t), 0) if not t[0].isdigit() else ("mps".index(t[1]), int(t[0]), t.endswith("r"))


def safe_parse(s):
    try:
        return parse_tiles(s or ""), None
    except ValueError as e:
        return [], str(e)


CSS = """
.tile svg{display:inline-block;vertical-align:middle;margin:1px;filter:drop-shadow(0 1px 1px rgba(0,0,0,.25))}
.board{background:#1e5a3c;border-radius:14px;padding:14px 16px;color:#f4f1e8}
.board .lbl{font-size:12px;opacity:.85;margin:6px 0 2px}
.board .hand{margin-top:10px;padding-top:8px;border-top:1px solid rgba(255,255,255,.2)}
.riichi{background:#c8102e;color:#fff;border-radius:6px;padding:0 6px;font-size:11px;margin-left:6px}
.tilebtn{min-width:34px!important;max-width:46px!important;padding:2px!important;background:transparent!important;border:none!important;box-shadow:none!important}
.tilebtn img{width:30px!important;height:40px!important;max-height:40px!important;margin:0!important}
.opt{display:grid;grid-template-columns:minmax(150px,1.3fr) 1fr 56px 64px;align-items:center;gap:8px;padding:3px 6px;border-radius:8px}
.opt.best{background:rgba(46,160,67,.18)}
.bar{height:12px;border-radius:6px;background:linear-gradient(90deg,#2ea043,#7ee787)}
.verdict{font-size:20px;font-weight:700;display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:4px 0 8px}
.small{font-size:12px;opacity:.8}
"""


def board(hand, dora, bakaze, kyoku, honba, seat, r_self, r_shimo, r_toi, r_kami, rr_shimo, rr_toi, rr_kami):
    h, eh = safe_parse(hand)
    d, _ = safe_parse(dora)
    rivers = [safe_parse(x)[0] for x in (r_self, r_shimo, r_toi, r_kami)]
    rr = [False, rr_shimo, rr_toi, rr_kami]
    head = (f'<b>{WIND_LABEL.get(bakaze, bakaze)} {int(kyoku or 1)}</b> · honba {int(honba or 0)} · '
            f'you sit {WIND_LABEL.get(seat, seat)} · dora indicator {T.row(d[:1], 34) if d else "—"}')
    rows = "".join(f'<div class="lbl">{SEATS[k]}{"<span class=riichi>riichi</span>" if rr[k] else ""}</div>'
                   f'{T.row(rivers[k], 30) if rivers[k] else "<span class=small>no discards</span>"}'
                   for k in (2, 3, 1, 0))
    n = len(h)
    hint = "" if n == 14 else f'<div class="small">{n} tiles: the hand needs 14 (13 + the tile just drawn, last)</div>'
    shown = sorted(h[:13], key=sort_key) + h[13:] if n == 14 else sorted(h, key=sort_key)
    hand_html = T.row(shown, 58, gap_before_last=(n == 14)) if h else '<span class="small">empty: click tiles below</span>'
    err = f'<div class="small">⚠️ {eh}</div>' if eh else ""
    return f'<div class="board">{head}{rows}<div class="hand"><div class="lbl">Your hand</div>{hand_html}{hint}{err}</div></div>'


def add_tile(code, target, *fields):
    fields = list(fields)
    i = TARGETS.index(target)
    cur, _ = safe_parse(fields[i])
    if i == 1:
        cur = [code]
    elif i == 0 and len(cur) >= 14:
        cur = cur[:13] + [code]
    else:
        cur = cur + [code]
    fields[i] = encode(cur)
    return fields


def undo(target, *fields):
    fields = list(fields)
    i = TARGETS.index(target)
    cur, _ = safe_parse(fields[i])
    fields[i] = encode(cur[:-1])
    return fields


def clear(target, *fields):
    fields = list(fields)
    fields[TARGETS.index(target)] = ""
    return fields


def random_hand():
    wall = [t for t in T.ALL[:34] for _ in range(4)]
    for r in ("5m", "5p", "5s"):
        wall.remove(r)
        wall.append(r + "r")
    random.shuffle(wall)
    return encode(sorted(wall[:13], key=sort_key) + wall[13:14]), encode(wall[14:15]), "", "", "", "", "Hand"


def _opt_label(p):
    kind, _, arg = p["value"].partition(":")
    if kind == "dahai":
        return f'{T.img(arg, 36)} <span>discard {label(arg)}</span>'
    if kind in ("ankan", "kakan"):
        return f'{T.img(arg, 36)} <span>{"closed" if kind == "ankan" else "added"} kan</span>'
    return {"reach": "🀄 <b>Riichi</b> 立直", "none": "Pass", "hora": "🎉 <b>Win</b> 和了", "pon": "Pon",
            "daiminkan": "Open kan", "ryukyoku": "Abort (nine terminals)"}.get(kind, p["description"])


def _options_html(res, title):
    rows = []
    for i, p in enumerate(res["probabilities"]):
        rows.append(f'<div class="opt{" best" if i == 0 else ""}"><div>{_opt_label(p)}</div>'
                    f'<div><div class="bar" style="width:{max(p["probability"] * 100, 0.5):.1f}%"></div></div>'
                    f'<div>{p["probability"]:.1%}</div><div class="small">Q {p["q"]:+.2f}</div></div>')
    return f'<div class="small" style="margin-top:6px"><b>{title}</b></div>' + "".join(rows)


def analyse(hand, dora, bakaze, kyoku, honba, seat, r_self, r_shimo, r_toi, r_kami, rr_shimo, rr_toi, rr_kami,
            scores, temperature):
    try:
        sc_rel = [int(x) for x in scores.replace(" ", "").split(",")] if scores.strip() else [25000] * 4
        if len(sc_rel) != 4:
            raise ValueError("scores: four numbers, in the order you, shimocha, toimen, kamicha")
        oya = int(kyoku) - 1
        pid = (oya + WINDS.index(seat)) % 4
        sc = [0] * 4
        for k in range(4):
            sc[(pid + k) % 4] = sc_rel[k]
        events, pid = nanikiru_events(hand, dora=dora or "1z", bakaze=bakaze, kyoku=int(kyoku), honba=int(honba or 0),
                                      seat_wind=seat, scores=sc, rivers=[r_self, r_shimo, r_toi, r_kami],
                                      riichi=[False, rr_shimo, rr_toi, rr_kami])
        res = model.decide(events, pid, temperature=float(temperature))
    except Exception as e:  # noqa: BLE001
        return f'<div class="verdict">⚠️ {e}</div>', ""
    best = res["probabilities"][0]
    st = res["state"]
    out = (f'<div class="verdict">{_opt_label(best)} <span class="small">{best["probability"]:.0%}</span></div>'
           f'<div class="small">shanten {st["shanten"]} · {NAME} · temperature {float(temperature):g}</div>'
           + _options_html(res, "All legal options"))
    if any(p["value"] == "reach" for p in res["probabilities"]):
        r2 = model.decide(events + [{"type": "reach", "actor": pid}], pid, temperature=float(temperature))
        out += "<br>" + _options_html(r2, "If you declare riichi, discard")
    req = {"model": NAME, "input": "\n".join(json.dumps(e) for e in events), "player_id": pid,
           "questions": [{"type": "choice", "name": "action"}]}
    return out, json.dumps(req, ensure_ascii=False, indent=1)


def api_try(body):
    try:
        code, out = handle(model, json.loads(body))
    except json.JSONDecodeError as e:
        code, out = 400, {"error": {"type": "invalid_request_error", "message": f"bad JSON: {e}"}}
    return json.dumps(out, ensure_ascii=False, indent=1)


EXAMPLES = [  # hand, dora, round wind, kyoku, honba, seat, rivers (you, shimocha, toimen, kamicha), riichi x3, scores
    ["123m406p789s11223z", "1p", "E", 1, 0, "E", "", "", "", "", False, False, False, ""],
    ["23455m3467p24688s", "7z", "E", 2, 0, "S", "19m", "1z9p", "4z2z", "1m9s7z", False, False, False, ""],
    ["1145m2367p456s556z", "3s", "S", 3, 1, "W", "9p1z", "7z2s", "6p1s8m", "9s3z2p", False, True, False,
     "18000,32000,24000,26000"],
]

ABOUT = f"""
### The model: **{NAME}** (TanyaoDojo "tpt 408k")

- **Strength.** −0.64 ± 0.46 placement points per hanchan against Mortal v4, measured over 100k duplicate games.
  The challenger takes every seat on the same walls, and the other three seats are v4; pt [90, 45, 0, −135].
  That puts it a little below v4, which is one of the strongest open-source riichi AIs.
- **How it was trained.** It is TanyaoDojo's own pipeline, end to end:
  - an offline base learned only from Tenhou Phoenix-room games (2.5M games, Monte-Carlo targets + CQL);
  - then online self-play RL whose targets are the game's real placement points, anchored to the project's own
    earlier checkpoints.
- **Where v4 comes in.** v4's weights and v4's choices are never trained on, so this is not a v4 imitation. v4 did
  play as one of the self-play opponents (about 2 of every 7 generated games). It is also the yardstick: every number
  is measured against it, and this snapshot was picked by those measurements.
- **The numbers on the board.** Q is the model's value estimate of each option, in placement points on its training
  scale. Probability is softmax(Q / temperature) over the options, a Boltzmann policy, not a calibrated chance of being
  right.
- **Limits of the board.**
  - Calls (chi / pon / kan already made) are not entered here.
  - Other players' rivers are replayed as discards straight after their draws.
  - Use the Decision API with a full mjai log for any position.

For study and post-game review; please do not use it in rated games.
Source and license: [TanyaoDojo](https://github.com/twu3202/TanyaoDojo) (AGPL-3.0; it links
[Mortal](https://github.com/Equim-chan/Mortal)'s libriichi).
"""

with gr.Blocks(title="TanyaoDojo: what to discard", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🀄 What would the model discard?\nBuild a position by clicking tiles, then ask a riichi AI trained by "
                "[TanyaoDojo](https://github.com/twu3202/TanyaoDojo) (about 0.6 points per hanchan below Mortal v4). "
                "Every legal option is ranked in one forward pass.")
    with gr.Tab("What to discard"):
        with gr.Row():
            with gr.Column(scale=3):
                preview = gr.HTML()
                target = gr.Radio(TARGETS, value="Hand", label="Clicking a tile adds it to")
                pal = []
                for suit_tiles in ([f"{n}m" for n in range(1, 10)] + ["5mr"], [f"{n}p" for n in range(1, 10)] + ["5pr"],
                                   [f"{n}s" for n in range(1, 10)] + ["5sr"], list("ESWNPFC")):
                    with gr.Row(equal_height=True):
                        for t in suit_tiles:
                            pal.append((t, gr.Button("", icon=ICONS[t], elem_classes="tilebtn", min_width=34)))
                with gr.Row():
                    b_undo = gr.Button("⌫ Undo", size="sm")
                    b_clear = gr.Button("Clear field", size="sm")
                    b_rand = gr.Button("🎲 Random hand", size="sm")
                with gr.Accordion("Position details (or type tiles: 123m406p789s1234567z, 0 = red five)", open=False):
                    hand = gr.Textbox(label="Hand (14 tiles, the drawn tile last)", value=EXAMPLES[0][0], max_lines=1)
                    with gr.Row():
                        dora = gr.Textbox(label="Dora indicator", value="1p", max_lines=1)
                        bakaze = gr.Dropdown(WINDS, value="E", label="Round wind")
                        kyoku = gr.Dropdown([1, 2, 3, 4], value=1, label="Hand no.")
                        honba = gr.Number(value=0, label="Honba", precision=0)
                        seat = gr.Dropdown(WINDS, value="E", label="Your seat wind")
                    gr.Markdown("Rivers in turn order. From the dealer, seats before you need one more discard than "
                                "you, seats after you the same number.")
                    with gr.Row():
                        r_self = gr.Textbox(label="Your river", max_lines=1)
                        r_shimo = gr.Textbox(label="Shimocha (right)", max_lines=1)
                        r_toi = gr.Textbox(label="Toimen (across)", max_lines=1)
                        r_kami = gr.Textbox(label="Kamicha (left)", max_lines=1)
                    with gr.Row():
                        rr_shimo = gr.Checkbox(label="shimocha riichi (on the last listed discard)")
                        rr_toi = gr.Checkbox(label="toimen riichi")
                        rr_kami = gr.Checkbox(label="kamicha riichi")
                    with gr.Row():
                        scores = gr.Textbox(label="Scores: you, shimocha, toimen, kamicha (blank = 25000 each)", max_lines=1)
                        temp = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Temperature")
                go = gr.Button("Ask the model", variant="primary", size="lg")
            with gr.Column(scale=2):
                result = gr.HTML('<div class="small">The answer appears here.</div>')
                with gr.Accordion("This position as a /v1/decisions request", open=False):
                    req_box = gr.Code(language="json")
        fields = [hand, dora, r_self, r_shimo, r_toi, r_kami]
        bd_inputs = [hand, dora, bakaze, kyoku, honba, seat, r_self, r_shimo, r_toi, r_kami, rr_shimo, rr_toi, rr_kami]
        for t, b in pal:
            b.click(lambda tg, *f, _t=t: add_tile(_t, tg, *f), [target] + fields, fields)
        b_undo.click(undo, [target] + fields, fields)
        b_clear.click(clear, [target] + fields, fields)
        b_rand.click(random_hand, None, fields + [target])
        for c in bd_inputs:
            c.change(board, bd_inputs, preview)
        go.click(analyse, bd_inputs + [scores, temp], [result, req_box])
        gr.Examples(EXAMPLES, bd_inputs + [scores], label="Example positions")
        demo.load(board, bd_inputs, preview)
    with gr.Tab("Decision API"):
        gr.Markdown(
            ("`POST /v1/decisions` on this Space takes " if API_ON else "Run `decision_api.py serve` locally for ")
            + "a request in the shape of the `/v1/decisions` decision-model APIs:\n"
            "- `input`: the mjai event log up to the decision, as JSON lines, a list of events, or messages holding them.\n"
            "- `player_id`: optional; inferred when exactly one seat can act.\n"
            "- `choice` questions whose `choices` are option values: `dahai:<tile>`, `reach`, `chi:low|mid|high`, "
            "`pon`, `daiminkan`, `ankan:<tile>`, `kakan:<tile>`, `hora`, `ryukyoku`, `none`. Leave `choices` out "
            "for every legal option.\n\n"
            "The board tab shows its position as a request.")
        body = gr.Code(language="json", label="Request", value=json.dumps(
            {"model": NAME, "input": [{"type": "start_game", "names": ["0", "1", "2", "3"]},
                                      {"type": "start_kyoku", "bakaze": "E", "dora_marker": "1p", "kyoku": 1, "honba": 0,
                                       "kyotaku": 0, "oya": 0, "scores": [25000] * 4,
                                       "tehais": [["1m", "2m", "3m", "4p", "5pr", "6p", "7s", "8s", "9s", "E", "E", "S", "S"],
                                                  ["?"] * 13, ["?"] * 13, ["?"] * 13]},
                                      {"type": "tsumo", "actor": 0, "pai": "W"}],
             "questions": [{"type": "choice", "name": "riichi_or_dama",
                            "choices": [{"value": "reach"}, {"value": "dahai:W"}]}]}, indent=1))
        out = gr.Code(language="json", label="Response")
        gr.Button("Send").click(api_try, body, out)
    with gr.Tab("About the model"):
        gr.Markdown(ABOUT)

app = FastAPI()
if API_ON:
    @app.post("/v1/decisions")
    async def decisions(request: Request):
        try:
            body = await request.json()
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": {"type": "invalid_request_error", "message": f"bad JSON: {e}"}}, 400)
        code, out = handle(model, body)
        return JSONResponse(out, status_code=code)
# No server-side rendering under a plain FastAPI mount (there is no Node server to serve its chunks), and behind the
# Spaces proxy the public origin is SPACE_HOST, not the bind address.
ROOT = f"https://{os.environ['SPACE_HOST']}" if os.environ.get("SPACE_HOST") else ""
app = gr.mount_gradio_app(app, demo, path="/", allowed_paths=[TILE_DIR], ssr_mode=False, root_path=ROOT)

if __name__ == "__main__":
    import uvicorn
    # Gradio Spaces route traffic to GRADIO_SERVER_PORT (7860); PORT can belong to something else there
    uvicorn.run(app, host=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"), port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
                proxy_headers=True, forwarded_allow_ips="*")
