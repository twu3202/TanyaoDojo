"""Riichi tiles drawn as SVG (original drawings for this demo; no external image assets), plus small HTML helpers.
Tile names follow mjai: 1m..9m, 1p..9p, 1s..9s, E S W N P F C, red fives 5mr 5pr 5sr."""
from __future__ import annotations

import html
import os

W, H = 60, 80
INK, RED, GREEN, BLUE = "#1d2b53", "#c8102e", "#1a7f37", "#1f5fbf"
NUM = "一二三四五六七八九"
FONT = "'Noto Serif CJK JP','Noto Serif CJK SC','Hiragino Mincho ProN','Songti SC','SimSun',serif"
HONOR = {"E": ("東", INK), "S": ("南", INK), "W": ("西", INK), "N": ("北", INK), "F": ("發", GREEN), "C": ("中", RED)}

PIN = {1: [(30, 40)], 2: [(30, 22), (30, 58)], 3: [(16, 18), (30, 40), (44, 62)],
       4: [(18, 24), (42, 24), (18, 56), (42, 56)], 5: [(18, 22), (42, 22), (30, 40), (18, 58), (42, 58)],
       6: [(19, 18), (41, 18), (19, 40), (41, 40), (19, 62), (41, 62)],
       7: [(14, 14), (30, 21), (46, 28), (19, 48), (41, 48), (19, 66), (41, 66)],
       8: [(19, 14), (41, 14), (19, 32), (41, 32), (19, 50), (41, 50), (19, 68), (41, 68)],
       9: [(15, 17), (30, 17), (45, 17), (15, 40), (30, 40), (45, 40), (15, 63), (30, 63), (45, 63)]}
PIN_R = {1: 15, 2: 10, 3: 9, 4: 9, 5: 8.5, 6: 8, 7: 7, 8: 7, 9: 6.8}
SOU = {2: [(30, 24), (30, 56)], 3: [(30, 24), (19, 56), (41, 56)], 4: [(19, 24), (41, 24), (19, 56), (41, 56)],
       5: [(17, 24), (43, 24), (30, 40), (17, 56), (43, 56)], 6: [(15, 24), (30, 24), (45, 24), (15, 56), (30, 56), (45, 56)],
       7: [(30, 16), (15, 40), (30, 40), (45, 40), (15, 64), (30, 64), (45, 64)],
       8: [(12, 24), (24, 24), (36, 24), (48, 24), (12, 56), (24, 56), (36, 56), (48, 56)],
       9: [(15, 16), (30, 16), (45, 16), (15, 40), (30, 40), (45, 40), (15, 64), (30, 64), (45, 64)]}


def _frame(inner: str, red: bool = False) -> str:
    edge = RED if red else "#b9b2a3"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
            f'<rect x="1.5" y="3.5" width="{W - 3}" height="{H - 5}" rx="7" fill="#8a7f6b"/>'
            f'<rect x="1.5" y="1.5" width="{W - 3}" height="{H - 5}" rx="7" fill="#fbf7ee" stroke="{edge}" stroke-width="1.5"/>'
            f'{inner}</svg>')


def _text(x, y, s, size, color, weight="700"):
    return (f'<text x="{x}" y="{y}" text-anchor="middle" dominant-baseline="central" font-family="{FONT}" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}">{s}</text>')


def _pin(n, red):
    r = PIN_R[n]
    out = []
    for i, (x, y) in enumerate(PIN[n]):
        c = RED if red or n == 1 or (n in (3, 5, 9) and (x, y) == (30, 40)) or (n == 7 and i < 3) else (
            GREEN if n in (6, 8, 9) and i % 2 else BLUE)
        out.append(f'<circle cx="{x}" cy="{y - 2}" r="{r}" fill="{c}"/><circle cx="{x}" cy="{y - 2}" r="{r * 0.62}" '
                   f'fill="#fbf7ee"/><circle cx="{x}" cy="{y - 2}" r="{r * 0.3}" fill="{c}"/>')
    return "".join(out)


def _sou(n, red):
    if n == 1:
        return (f'<rect x="25" y="12" width="10" height="46" rx="5" fill="{RED if red else GREEN}"/>'
                f'<circle cx="30" cy="14" r="7" fill="{RED}"/><rect x="21" y="56" width="18" height="6" rx="3" fill="{GREEN}"/>')
    h = 14 if n in (7, 9) else 18
    out = []
    for i, (x, y) in enumerate(SOU[n]):
        c = RED if red or (n in (5, 7) and i == (2 if n == 5 else 0)) or (n == 9 and x == 30) else GREEN
        out.append(f'<rect x="{x - 3.2}" y="{y - h / 2 - 2}" width="6.4" height="{h}" rx="3" fill="{c}"/>'
                   f'<line x1="{x - 3.2}" y1="{y - 2}" x2="{x + 3.2}" y2="{y - 2}" stroke="#fbf7ee" stroke-width="1.2"/>')
    return "".join(out)


def tile_svg(name: str) -> str:
    red = name.endswith("r")
    base = name.rstrip("r")
    if base in HONOR:
        ch, col = HONOR[base]
        return _frame(_text(30, 38, ch, 36, col))
    if base == "P":
        return _frame('<rect x="13" y="13" width="34" height="50" rx="4" fill="none" stroke="#1f5fbf" stroke-width="3"/>')
    n, suit = int(base[0]), base[1]
    if suit == "m":
        return _frame(_text(30, 24, NUM[n - 1], 26, RED if red else INK) + _text(30, 55, "萬", 28, RED), red)
    return _frame(_pin(n, red) if suit == "p" else _sou(n, red), red)


ALL = [f"{n}{s}" for s in "mps" for n in range(1, 10)] + ["E", "S", "W", "N", "P", "F", "C", "5mr", "5pr", "5sr"]


def write_all(directory: str) -> dict[str, str]:
    os.makedirs(directory, exist_ok=True)
    paths = {}
    for t in ALL:
        p = os.path.join(directory, f"{t}.svg")
        with open(p, "w", encoding="utf-8") as f:
            f.write(tile_svg(t))
        paths[t] = p
    return paths


def img(name: str, h: int = 44, extra: str = "") -> str:
    """Inline tile for HTML output."""
    svg = tile_svg(name).replace(f'width="{W}" height="{H}"', f'width="{h * W / H:.0f}" height="{h}"', 1)
    return f'<span class="tile" title="{html.escape(name)}" {extra}>{svg}</span>'


def row(names: list[str], h: int = 44, gap_before_last: bool = False) -> str:
    parts = [img(t, h) for t in names]
    if gap_before_last and len(parts) > 1:
        parts[-1] = '<span style="display:inline-block;width:10px"></span>' + parts[-1]
    return '<span class="tilerow">' + "".join(parts) + "</span>"
