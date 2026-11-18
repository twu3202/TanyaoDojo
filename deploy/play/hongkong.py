"""No-flower Hong Kong Old Style, based on the CUHK alumni tournament table."""

from dataclasses import dataclass

from .honors import DRAGONS, ORPHANS, TERMINALS, WINDS, standard_shapes, thirteen_orphans

VARIANT = 'hong-kong-old-style'
MIN_FAAN = 3
CAP = 8
SELF_DRAW_SHARE = (0, 0, 0, 16, 32, 48, 64, 96, 128)


@dataclass(frozen=True)
class Faan:
    raw: int
    patterns: tuple[str, ...]

    @property
    def fan(self):
        return min(self.raw, CAP)


def faan(hand, melds, ctx):
    """Best legal interpretation, or None if the shape or its faan is insufficient."""
    if thirteen_orphans(hand, melds):
        return Faan(CAP, (ctx.opening or '十三幺',))
    if ctx.kind == 'rob_concealed':
        return None
    interpretations = [
        _score_shape(hand, melds, pair, melds + sets, ctx)
        for pair, sets in standard_shapes(hand, len(melds))
    ]
    best = max(interpretations, key=lambda score: score.raw, default=None)
    return best if best is not None and best.raw >= MIN_FAAN else None


def _score_shape(hand, melds, pair, sets, ctx):
    if ctx.opening:
        return Faan(CAP, (ctx.opening,))
    if ctx.kind == 'zimo' and ctx.kong_chain >= 2:
        return Faan(CAP, ('杠上杠',))
    if ctx.kind == 'rob_kong' and ctx.kong_chain:
        return Faan(CAP, ('抢杠上杠',))

    tiles = {tile for tile, count in enumerate(hand) if count}
    tiles.update(tile for meld in melds for tile in meld.tiles)
    pungs = {meld.tile for meld in sets if meld.kind != 'chi'}
    dragon_pungs = pungs & DRAGONS
    wind_pungs = pungs & WINDS
    all_pungs = all(meld.kind != 'chi' for meld in sets)
    fixed = (
        (len(wind_pungs) == 4, '大四喜'),
        (len(wind_pungs) == 3 and pair in WINDS, '小四喜'),
        (len(dragon_pungs) == 3, '大三元'),
        (len(melds) == 4 and all(meld.kind.startswith('gang') for meld in melds), '十八罗汉'),
        (tiles <= TERMINALS, '清幺九'),
        (tiles <= WINDS | DRAGONS, '字一色'),
        (all_pungs and not melds and (ctx.kind == 'zimo' or pair == ctx.tile), '坎坎胡'),
    )
    for qualifies, label in fixed:
        if qualifies:
            return Faan(CAP, (label,))

    points = []
    suited = {tile // 9 for tile in tiles if tile < 27}
    if len(suited) == 1:
        points.append((3, '混一色') if tiles & (WINDS | DRAGONS) else (7, '清一色'))
    if all_pungs:
        points.append((3, '对对胡'))
    elif all(meld.kind == 'chi' for meld in sets):
        points.append((1, '平胡'))
    if tiles <= ORPHANS:
        points.append((3, '花幺'))
    if len(dragon_pungs) == 2 and pair in DRAGONS:
        points.append((4, '小三元'))
    else:
        points.extend((1, label) for tile, label in ((31, '白板刻'), (32, '发财刻'), (33, '红中刻'))
                      if tile in dragon_pungs)
    if ctx.seat_wind in wind_pungs:
        points.append((1, '门风刻'))
    if ctx.round_wind in wind_pungs:
        points.append((1, '圈风刻'))
    if all(meld.kind == 'gang_an' for meld in melds):
        points.append((1, '门清'))
    if ctx.kind == 'zimo':
        points.append((1, '自摸'))
        if ctx.kong_chain:
            points.append((1, '杠上花'))
    if ctx.kind == 'rob_kong':
        points.append((1, '抢杠'))
    return Faan(sum(value for value, _ in points), tuple(label for _, label in points))


def payments(fan, kind, winner, provider=None, liable=None):
    """Positive amounts owed by the four seats for one winning hand."""
    share = SELF_DRAW_SHARE[fan]
    if kind == 'ron':
        return tuple(2 * share if seat == provider else 0 for seat in range(4))
    if kind in ('rob_kong', 'rob_concealed'):
        liable = provider
    if liable is not None:
        return tuple(3 * share if seat == liable else 0 for seat in range(4))
    return tuple(0 if seat == winner else share for seat in range(4))


def share(fan):
    return SELF_DRAW_SHARE[fan]


def liability(melds, last):
    if len(melds) == 4:
        return last.provider
    if last.kind != 'chi' and last.tile in DRAGONS:
        exposed = {meld.tile for meld in melds if meld.kind in ('peng', 'gang_ming', 'gang_bu')}
        if DRAGONS <= exposed:
            return last.provider
    return None


def next_dealer(dealer, winners):
    return dealer if dealer in winners else (dealer + 1) % 4
