"""Golden Island's no-chow Guangdong rules and bilateral horse settlement."""

from dataclasses import dataclass
from typing import ClassVar

from .honors import (DRAGONS, ORPHANS, TERMINALS, WINDS, KongPoints, TableOptions,
                     seven_pairs, standard_shapes, thirteen_orphans)

VARIANT = 'golden-island-guangdong'
MIN_FAAN = 0
WALL_RESERVE = 4
MATCH_HANDS = None
SWAP_LOCK = False
PASS_LOCK = False
BONUS_MODE = 'bilateral_horses'
OPTIONS = TableOptions(
    chow=False, seven_pairs=True, pass_win_higher_all=True, pass_win_clear_by_call=True,
    kong_points=KongPoints(3, 1, 2, single_direct_payer=True, refund_on_draw=True),
    horse_counts=(4, 4, 4), dealer_echo=True, exchange_count=3, closed_live_tail=3,
)


@dataclass(frozen=True)
class Score:
    share: int
    patterns: tuple[str, ...]
    kong_provider: int | None = None
    four_open: bool = False
    fan: ClassVar[None] = None
    tier: ClassVar[None] = None


def _tile_patterns(tiles):
    suits = {tile // 9 for tile in tiles if tile < 27}
    if tiles <= WINDS | DRAGONS:
        return [(20, '字一色')]
    patterns = []
    if len(suits) == 1:
        patterns.append((6, '混一色') if any(tile >= 27 for tile in tiles) else (12, '清一色'))
    if tiles <= TERMINALS:
        patterns.append((16, '清幺九'))
    elif tiles <= ORPHANS:
        patterns.append((12, '花幺'))
    return patterns


def _standard_patterns(melds, pair, sets):
    groups = melds + sets
    pungs = {meld.tile for meld in groups if meld.kind != 'chi'}
    if len(melds) == 4 and all(meld.kind.startswith('gang_') for meld in melds):
        patterns = [(36, '十八罗汉')]
    elif len(pungs) == 4:
        patterns = [(6, '对对胡')]
    else:
        patterns = []
    for tiles, small, big in ((DRAGONS, '小三元', '大三元'), (WINDS, '小四喜', '大四喜')):
        if tiles <= pungs:
            patterns.append((20, big))
        elif len(tiles & pungs) == len(tiles) - 1 and pair in tiles:
            patterns.append((12, small))
    return patterns


def faan(hand, melds, ctx):
    orphans = thirteen_orphans(hand, melds)
    if ctx.kind == 'rob_concealed' and not orphans:
        return None
    tiles = {tile for tile, count in enumerate(hand) if count} | {tile for meld in melds for tile in meld.tiles}
    common = _tile_patterns(tiles)
    if ctx.opening or ctx.first_nondealer_draw:
        common.append((20, ctx.opening or '地胡'))
    candidates = []
    for pair, sets in standard_shapes(hand, len(melds)):
        patterns = common + _standard_patterns(melds, pair, sets)
        if not patterns and ctx.kind != 'ron':
            patterns = [(3, '平胡')] if all(group.kind == 'chi' for group in sets) and not melds else [(2, '鸡胡')]
        if patterns:
            candidates.append(patterns)
    if seven_pairs(hand, melds):
        quads = hand.count(4)
        candidates.append(common + [((8, 16, 24, 36)[quads], ('七对', '龙七对', '双豪七', '三豪七')[quads])])
    if orphans:
        candidates.append(common + [(20, '十三幺')])
    if not candidates:
        return None
    best = max(candidates, key=lambda patterns: sum(points for points, _ in patterns))
    if ctx.kind == 'zimo' and ctx.late_wall:
        best.append((1, '海底'))
    if ctx.kind == 'zimo' and ctx.kong_chain:
        best.append((1, '杠上花'))
    if ctx.kind in ('rob_kong', 'rob_concealed'):
        best.append((1, '抢杠'))
    return Score(sum(points for points, _ in best), tuple(pattern for _, pattern in best),
                 ctx.kong_provider, len(melds) == 4 and all(meld.provider is not None for meld in melds))


def liability(melds, last):
    return ('twelve_open',) if len(melds) == 4 and all(meld.provider is not None for meld in melds) else ()


def horse_seat(tile, winner):
    offset = (tile % 9) % 4 if tile < 27 else (0, 1, 2, 3, 2, 1, 0)[tile - 27]
    return (winner + offset) % 4


def settle_hand(game, winners, offer):
    game.horse_tiles = game.wall[:4]
    del game.wall[:4]
    outcomes = []
    for winner in winners:
        actor, score = winner.actor, winner.score
        hits = [0] * 4
        for tile in game.horse_tiles:
            hits[horse_seat(tile, actor)] += 1
        game.bilateral_horses.append({'actor': actor, 'hits': hits})
        shares = [score.share * (1 + hits[actor] + hits[seat]) if actor != seat else 0 for seat in range(4)]
        if offer is not None:
            payments = tuple(shares[seat] if offer.actor == seat else 0 for seat in range(4))
        else:
            cover = score.kong_provider
            if cover is None and score.four_open:
                cover = next((seat for label, seat in reversed(game.players[actor].liabilities)
                              if label == 'twelve_open'), None)
            payments = tuple(sum(shares) if seat == cover else 0 for seat in range(4)) if cover is not None else shares
        outcomes.append((score, tuple(payments)))
    return outcomes


def next_dealer(dealer, winners, provider=None):
    if not winners:
        return dealer
    return winners[0] if len(winners) == 1 else provider
