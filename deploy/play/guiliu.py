"""Guiliu wake draws, special shapes, and reciprocal payment table."""

from dataclasses import dataclass, replace
from typing import ClassVar

from .honors import KongPoints, TableOptions, seven_pairs, standard_shapes

VARIANT = 'guiliu-wake'
MIN_FAAN = 0
WALL_RESERVE = 0
MATCH_HANDS = None
SWAP_LOCK = False
PASS_LOCK = False
OPTIONS = TableOptions(
    chow=False, seven_pairs=True, orphans=False, host_deals_first=True, show_winds=False,
    drawn_added_kong_only=True, kong_requires_live_tile=True,
    pass_win_higher_all=True, pass_win_clear_by_call=True, reciprocal_liability=True,
    kong_points=KongPoints(3, 3, 2, single_direct_payer=True, added_payer_from_pong=True),
    winner_echo=True, echo_share=2,
)


@dataclass(frozen=True)
class Score:
    share: int
    dealer_share: int
    factor: int
    patterns: tuple[str, ...]
    fan: ClassVar[None] = None
    tier: ClassVar[None] = None


def _ragged(hand, melds):
    honors = sum(hand[27:])
    if melds or sum(hand) != 14 or max(hand) > 1 or honors < 5:
        return 0
    for suit in (0, 9, 18):
        ranks = [rank for rank in range(9) if hand[suit + rank]]
        if any(high - low < 3 for low, high in zip(ranks, ranks[1:])):
            return 0
    return honors


def faan(hand, melds, ctx):
    if ctx.kind == 'rob_concealed':
        return None
    ragged = _ragged(hand, melds)
    pairs = seven_pairs(hand, melds)
    shapes = list(standard_shapes(hand, len(melds)))
    if not (ragged or pairs or shapes):
        return None
    candidates = []
    if ragged:
        candidates.append((1, ('七星烂',)) if ragged == 7 else (0, ('烂牌',)))
    if pairs or shapes:
        tiles = {tile for tile, n in enumerate(hand) if n} | {tile for meld in melds for tile in meld.tiles}
        common = []
        if min(tiles) >= 27:
            common.append('字一色')
        elif max(tiles) < 27 and len({tile // 9 for tile in tiles}) == 1:
            common.append('清一色')
        if pairs:
            quad_count = hand.count(4)
            label = ('七对', '龙七对', '双豪七', '三豪七')[quad_count]
            candidates.append((len(common) + 1 + quad_count, (*common, label)))
        if shapes:
            ordinary = common.copy()
            four_open_ron = len(melds) == 4 and all(meld.provider is not None for meld in melds) and ctx.kind != 'zimo'
            if four_open_ron:
                ordinary.append('全求人')
            elif all(meld.kind != 'chi' for meld in melds) and any(
                all(group.kind == 'triplet' for group in sets) for _, sets in shapes
            ):
                ordinary.append('对对胡')
            candidates.append((len(ordinary), tuple(ordinary) or ('平胡',)))
    fan, patterns = max(candidates, key=lambda row: row[0])
    dealer = ctx.seat_wind == 27
    if fan:
        share = (15 if ctx.kind == 'zimo' else 35) * (1 << (fan - 1))
        dealer_share = share
    elif ctx.kind == 'zimo':
        share, dealer_share = ((15, 15) if dealer else (3, 5)) if ragged else ((4, 4) if dealer else (2, 4))
    else:
        share = (10 if dealer else 8) if ragged else (7 if dealer else 5)
        dealer_share = share
    factor = 1
    if all(meld.kind == 'gang_an' for meld in melds):
        patterns += ('门清',)
        factor *= 2
    if ctx.kong_chain and ctx.kind in ('zimo', 'ron'):
        patterns += ('杠上花' if ctx.kind == 'zimo' else '杠上炮',)
        factor *= 2
    if ctx.kind == 'rob_kong':
        patterns += ('抢杠',)
    return Score(share * factor, dealer_share * factor, factor, patterns)


def liability(melds, last):
    return ('three_feeds',) if sum(meld.provider == last.provider for meld in melds) == 3 else ()


def adjacent(tile):
    if tile < 27:
        return tuple(other for other in (tile - 1, tile + 1) if other // 9 == tile // 9)
    group = (27, 28, 29, 30) if tile < 31 else (33, 32, 31)
    index = group.index(tile)
    return group[index - 1], group[(index + 1) % len(group)]


def _wake(game, winner, last_tile):
    hand = winner.hand.copy()
    for meld in game.players[winner.actor].melds:
        for tile in meld.tiles:
            hand[tile] += 1
    if last_tile is not None:
        game.wake_tiles = [last_tile]
        return 1, sum(hand[tile] for tile in adjacent(last_tile))
    hits = count = 0
    while count < len(game.wake_tiles) or game.wall:
        if count == len(game.wake_tiles):
            tile = game.wall.pop()
            game.wake_draws.append(tile)
            game.wake_tiles.append(tile)
        hits_now = sum(hand[tile] for tile in adjacent(game.wake_tiles[count]))
        hits += hits_now
        count += 1
        if hits_now < 3:
            break
    return count, hits


def settle_hand(game, winners, offer):
    provider = None if offer is None else offer.actor
    last_tile = (game.drawn_tile if offer is None else offer.tile) if not game.remaining else None
    scores = {}
    for winner in winners:
        count, hits = _wake(game, winner, last_tile)
        score = winner.score
        extra = hits * score.factor * (1 if winner.kind == 'zimo' else 3)
        scores[winner.actor] = replace(score, share=score.share + extra, dealer_share=score.dealer_share + extra)
        game.wake_details.append({'actor': winner.actor, 'count': count, 'hits': hits})
    outcomes = []
    for winner in winners:
        actor, score = winner.actor, scores[winner.actor]
        partners = {seat for _, seat in game._liabilities_for(actor)}
        if winner.kind == 'zimo':
            ordinary = [0 if seat == actor else score.dealer_share if seat == game.dealer else score.share
                        for seat in range(4)]
            payments = [2 * sum(ordinary) if seat in partners else 0 for seat in range(4)] if partners else ordinary
        else:
            payments = [score.share if seat == provider else 0 for seat in range(4)]
            for partner in partners:
                if partner not in scores or scores[partner].share != score.share:
                    payments[partner] += score.share
        outcomes.append((score, tuple(payments)))
    return outcomes


def next_dealer(dealer, winners, provider=None):
    return winners[0] if winners else dealer
