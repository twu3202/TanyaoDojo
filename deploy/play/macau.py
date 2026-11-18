"""Macau University alumni tournament rules for the 136-tile table."""

from dataclasses import dataclass

from .honors import DRAGONS, ORPHANS, TERMINALS, WINDS, standard_shapes, thirteen_orphans

VARIANT = 'macau-tournament'
WALL_RESERVE = 14
MATCH_HANDS = 16
MIN_FAAN = 0
SWAP_LOCK = False
PASS_LOCK = False


@dataclass(frozen=True)
class Score:
    raw: int
    patterns: tuple[str, ...]
    tier: str | None = None
    liable_patterns: frozenset[str] = frozenset()

    @property
    def fan(self):
        return None if self.tier else min(6, self.raw)

    @property
    def share(self):
        return {'A': 250, 'B': 500}[self.tier] if self.tier else 10 * self.fan


def _nine_gates(hand, melds, tile):
    if melds or tile >= 27:
        return False
    base = tile // 9 * 9
    waiting = [hand[i] - (i == tile) for i in range(base, base + 9)]
    return waiting == [3, 1, 1, 1, 1, 1, 1, 1, 3]


def faan(hand, melds, ctx):
    opening = {'天胡': '天胡', '地胡': '人和'}.get(ctx.opening)
    if ctx.first_nondealer_draw:
        opening = '地胡'
    if thirteen_orphans(hand, melds):
        return Score(0, (opening,), 'B') if opening else Score(0, ('十三幺',), 'A')
    if ctx.kind == 'rob_concealed':
        return None
    options = (_score_shape(hand, melds, pair, melds + sets, ctx, opening)
               for pair, sets in standard_shapes(hand, len(melds)))
    return max(options, key=lambda score: (score.share, score.raw), default=None)


def _score_shape(hand, melds, pair, sets, ctx, opening):
    chows = {meld.tile for meld in sets if meld.kind == 'chi'}
    pungs = {meld.tile for meld in sets if meld.kind != 'chi'}
    dragon_pungs, wind_pungs = pungs & DRAGONS, pungs & WINDS
    tiles = {tile for tile, count in enumerate(hand) if count}
    tiles.update(tile for meld in melds for tile in meld.tiles)
    kongs = sum(meld.kind.startswith('gang') for meld in melds)
    suits = {tile // 9 for tile in tiles if tile < 27}
    honors = bool(tiles & (WINDS | DRAGONS))
    full_flush = len(suits) == 1 and not honors
    liable_patterns = frozenset(label for qualifies, label in (
        (full_flush, '清一色'), (len(dragon_pungs) == 3, '大三元'),
        (len(wind_pungs) == 4, '大四喜'), (kongs == 4, '四杠'),
    ) if qualifies)

    category_b = (
        (bool(opening), opening), (tiles <= TERMINALS, '清幺九'),
        (_nine_gates(hand, melds, ctx.tile), '九莲宝灯'),
        ('大四喜' in liable_patterns, '大四喜'), ('四杠' in liable_patterns, '四杠'),
    )
    category_a = (
        ('大三元' in liable_patterns, '大三元'),
        (len(wind_pungs) == 3 and pair in WINDS, '小四喜'),
        (not chows and all(meld.kind == 'gang_an' for meld in melds)
         and (ctx.kind == 'zimo' or pair == ctx.tile), '坎坎胡'),
        (not suits, '字一色'),
    )
    for tier, candidates in (('B', category_b), ('A', category_a)):
        labels = tuple(label for qualifies, label in candidates if qualifies)
        if labels:
            return Score(0, labels, tier, liable_patterns)

    points = []
    if len(suits) == 1:
        points.append((6, '清一色') if full_flush else (3, '混一色'))
    if not chows:
        points.append((3, '对对胡'))
    elif not pungs:
        points.append((1, '平胡'))
    if tiles <= ORPHANS:
        points.append((3, '花幺'))
    if len(dragon_pungs) == 2 and pair in DRAGONS:
        points.append((6, '小三元'))
    else:
        points.extend((1, label) for tile, label in ((31, '白板刻'), (32, '发财刻'), (33, '红中刻'))
                      if tile in dragon_pungs)
    if ctx.seat_wind in wind_pungs:
        points.append((1, '门风刻'))
    if ctx.round_wind in wind_pungs:
        points.append((1, '圈风刻'))
    if all(meld.kind == 'gang_an' for meld in melds):
        points.append((1, '门清'))
    if any({suit, suit + 3, suit + 6} <= chows for suit in (0, 9, 18)):
        points.append((1, '一条龙'))
    if kongs == 3:
        points.append((6, '三杠'))
    if ctx.kind == 'zimo':
        points.append((1, '自摸'))
        if ctx.kong_chain:
            points.append((1, '杠上花'))
    elif ctx.kind == 'rob_kong':
        points.append((1, '抢杠'))
    return Score(sum(value for value, _ in points), tuple(label for _, label in points) or ('鸡胡',),
                 liable_patterns=liable_patterns)


def settlement(score, kind, winner, provider, liabilities):
    if kind != 'zimo':
        amount = 5 if score.fan == 0 else 2 * score.share
        return tuple(amount if seat == provider else 0 for seat in range(4))
    liable = next((seat for label, seat in reversed(liabilities) if label in score.liable_patterns), None)
    if liable is not None:
        return tuple(3 * score.share if seat == liable else 0 for seat in range(4))
    return tuple(0 if seat == winner else score.share for seat in range(4))


def liability(melds, last):
    result = []
    if len(melds) == 4 and all(meld.provider is not None and meld.tile < 27
                               and meld.tile // 9 == last.tile // 9 for meld in melds):
        result.append('清一色')
    if last.kind != 'chi':
        exposed = {meld.tile for meld in melds if meld.kind != 'gang_an' and meld.kind != 'chi'}
        if last.tile in DRAGONS and DRAGONS <= exposed:
            result.append('大三元')
        if last.tile in WINDS and WINDS <= exposed:
            result.append('大四喜')
    if last.kind == 'gang_ming' and all(meld.kind.startswith('gang') for meld in melds) and len(melds) == 4:
        result.append('四杠')
    return tuple(result)


def next_dealer(dealer, winners, provider=None):
    return (dealer + 1) % 4
