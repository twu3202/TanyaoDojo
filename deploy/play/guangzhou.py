"""Guangzhou three-faan-cap room, with separate uncapped special hands."""

from dataclasses import dataclass
from typing import ClassVar

from .honors import DRAGONS, WINDS, TableOptions, standard_shapes, thirteen_orphans

VARIANT = 'guangzhou-three-faan'
MIN_FAAN = 0
WALL_RESERVE = 0
MATCH_HANDS = None
SWAP_LOCK = False
PASS_LOCK = False
OPTIONS = TableOptions(pass_win_same=True, pass_win_clear_by_call=True,
                       kong_requires_live_tile=True, first_liability_only=frozenset({'twelve_open'}))


@dataclass(frozen=True)
class Score:
    fan: int
    patterns: tuple[str, ...]
    liable_patterns: frozenset[str] = frozenset()
    kong_provider: int | None = None
    tier: ClassVar[None] = None

    @property
    def share(self):
        return 1 << self.fan


def _opening(ctx):
    if ctx.opening == '天胡':
        return '天胡'
    if ctx.opening == '地胡':
        return '人和'
    if ctx.first_nondealer_draw:
        return '地胡'
    return None


def _shape(hand, melds, sets):
    groups = melds + sets
    pungs = {meld.tile for meld in groups if meld.kind != 'chi'}
    big = tuple(name for found, name in (
        (DRAGONS <= pungs, '大三元'),
        (WINDS <= pungs, '大四喜'),
    ) if found)
    if big:
        fan = 8 if '大四喜' in big else 6
        return fan, big, frozenset(big)
    tiles = {tile for tile, count in enumerate(hand) if count} | {tile for meld in melds for tile in meld.tiles}
    suits = {tile // 9 for tile in tiles if tile < 27}
    honors = any(tile >= 27 for tile in tiles)
    fan, patterns = 0, []
    if not pungs:
        fan += 1
        patterns.append('平胡')
    elif all(meld.kind != 'chi' for meld in groups):
        fan += 2
        patterns.append('对对胡')
    if len(suits) == 1:
        fan += 2 if honors else 3
        patterns.append('混一色' if honors else '清一色')
    if not suits:
        fan += 3
        patterns.append('字一色')
    return min(fan, 3), tuple(patterns) or ('鸡胡',), frozenset()


def faan(hand, melds, ctx):
    if thirteen_orphans(hand, melds):
        fan, patterns, liable = 6, ('十三幺',), frozenset()
    elif ctx.kind == 'rob_concealed':
        return None
    else:
        best = max((_shape(hand, melds, sets) for _, sets in standard_shapes(hand, len(melds))),
                   default=None, key=lambda row: row[0])
        if best is None:
            return None
        fan, patterns, liable = best
    opening = _opening(ctx)
    if opening is not None:
        return Score(6, (opening,))
    event = None
    if ctx.kind in ('rob_kong', 'rob_concealed'):
        event = '抢杠'
    elif ctx.kind == 'zimo' and ctx.kong_chain:
        event = '杠上花'
    elif ctx.kind == 'zimo' and ctx.last_live_draw:
        event = '海底'
    elif ctx.kind == 'ron' and ctx.winner_count == 3:
        event = '一炮三响'
    if event is not None:
        fan = max(fan, 3)
        patterns += (event,)
    if len(melds) == 4 and all(meld.provider is not None for meld in melds):
        liable = liable | {'twelve_open'}
    return Score(fan, patterns, liable, ctx.kong_provider)


def liability(melds, last):
    exposed = [meld for meld in melds if meld.provider is not None]
    pungs = {meld.tile for meld in exposed if meld.kind != 'chi'}
    return tuple(label for qualifies, label in (
        (len(exposed) == 4, 'twelve_open'),
        (last.tile in DRAGONS and DRAGONS <= pungs, '大三元'),
        (last.tile in WINDS and WINDS <= pungs, '大四喜'),
    ) if qualifies)


def settlement(score, kind, winner, provider, liabilities):
    if kind == 'ron':
        return tuple(score.share if seat == provider else 0 for seat in range(4))
    if kind in ('rob_kong', 'rob_concealed'):
        liable = provider
    else:
        liable = score.kong_provider
        if liable is None:
            liable = next((seat for pattern, seat in liabilities if pattern in score.liable_patterns), None)
    if liable is not None:
        return tuple(3 * score.share if seat == liable else 0 for seat in range(4))
    return tuple(score.share if seat != winner else 0 for seat in range(4))


def next_dealer(dealer, winners, provider=None):
    return dealer if dealer in winners else (dealer + 1) % 4
