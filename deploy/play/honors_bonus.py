"""Four-seat regional tables with reserved, post-win horse tiles."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import ClassVar

from .honors import (DRAGONS, ORPHANS, WINDS, KongPoints, Meld, TableOptions, WinContext,
                     seven_pairs, standard_shapes, thirteen_orphans)


@dataclass(frozen=True)
class Score:
    share: int
    patterns: tuple[str, ...]
    kong_provider: int | None = None
    fan: ClassVar[None] = None
    tier: ClassVar[None] = None


def _tiles(hand, melds):
    return {tile for tile, count in enumerate(hand) if count} | {tile for meld in melds for tile in meld.tiles}


def _all_pungs(melds, shapes):
    return all(meld.kind != 'chi' for meld in melds) and any(
        all(group.kind == 'triplet' for group in sets) for _, sets in shapes)


def _shangsi(hand, melds, ctx):
    if thirteen_orphans(hand, melds):
        name = '十三幺'
    elif seven_pairs(hand, melds):
        name = '七对'
    elif next(standard_shapes(hand, len(melds)), None) is not None:
        name = '平胡'
    else:
        return None
    return Score(2 if ctx.kind == 'zimo' else 1, (name,))


def _hundred(hand, melds, ctx):
    if ctx.kind == 'ron':
        return None
    shapes = list(standard_shapes(hand, len(melds)))
    pairs = seven_pairs(hand, melds)
    if not shapes and not pairs:
        return None
    tiles = _tiles(hand, melds)
    if tiles <= WINDS | DRAGONS:
        return Score(16, ('字一色',))
    if tiles <= ORPHANS:
        return Score(12, ('花幺',))

    suits = {tile // 9 for tile in tiles if tile < 27}
    flush = len(suits) == 1
    partial = flush and bool(tiles & (WINDS | DRAGONS))
    full = flush and not partial
    # Select the best interpretation, adding only the combinations in the published table.
    options = [Score(8 if full else 4 if partial else 2,
                     ('清一色',) if full else ('混一色',) if partial else ('平胡',))]
    if pairs:
        options.append(Score(16 if full else 8, ('七对', '清一色') if full else ('七对',)))
    if _all_pungs(melds, shapes):
        labels = ('对对胡',) + (('清一色',) if full else ('混一色',) if partial else ())
        options.append(Score(4 + (8 if full else 4 if partial else 0), labels))
    return max(options, key=lambda score: score.share)


def _hezhou(hand, melds, ctx):
    if thirteen_orphans(hand, melds):
        share, patterns = 10, ['十三幺']
    else:
        shapes = list(standard_shapes(hand, len(melds)))
        pairs = seven_pairs(hand, melds)
        if not shapes and not pairs:
            return None
        share, patterns = 1, []
        if pairs:
            dragon = 4 in hand
            share *= 4 if dragon else 2
            patterns.append('龙七对' if dragon else '七对')
        elif _all_pungs(melds, shapes):
            share *= 2
            patterns.append('对对胡')
        tiles = _tiles(hand, melds)
        if len({tile // 9 for tile in tiles}) == 1 and max(tiles) < 27:
            share *= 2
            patterns.append('清一色')
        if ctx.kind == 'zimo' and all(meld.kind == 'gang_an' for meld in melds):
            share *= 2
            patterns.append('门清')
        if ctx.kind == 'zimo' and ctx.kong_chain:
            share *= 2
            patterns.append('杠上花')
    if ctx.kind == 'rob_kong' or (ctx.kind == 'ron' and ctx.kong_chain):
        share += 1
        patterns.append('抢杠' if ctx.kind == 'rob_kong' else '杠上炮')
    return Score(share, tuple(patterns) or ('平胡',))


def _liuzhou(hand, melds, ctx):
    shapes = list(standard_shapes(hand, len(melds)))
    pairs = seven_pairs(hand, melds)
    if not shapes and not pairs:
        return None
    tiles = _tiles(hand, melds)
    suited = {tile // 9 for tile in tiles}
    big = tuple(label for found, label in (
        (len(suited) == 1 and max(tiles) < 27, '清一色'),
        (_all_pungs(melds, shapes), '对对胡'),
        (pairs, '七对'),
    ) if found)
    everyone_open = len(melds) == 4 and all(meld.provider is not None for meld in melds)
    patterns = list(big)
    if ctx.kind == 'zimo':
        if ctx.kong_chain:
            share = 2 * (1 << len(big))
            patterns.append('杠上花')
        elif everyone_open:
            share = 2 * (1 << max(0, len(big) - 1))
            patterns.append('全求人')
        else:
            share = 1 << len(big)
        if all(meld.kind == 'gang_an' for meld in melds):
            share *= 2
            patterns.append('门清')
    else:
        event = '抢杠' if ctx.kind == 'rob_kong' else '杠上炮' if ctx.kong_chain else (
            '全求人' if everyone_open else None)
        share = 3 * (1 << len(big)) if event else (3 * (1 << (len(big) - 1)) if big else 1)
        if event:
            patterns.append(event)
    return Score(share, tuple(patterns) or ('平胡',), ctx.kong_provider)


@dataclass(frozen=True)
class HorseRoom:
    VARIANT: str
    OPTIONS: TableOptions
    shape_score: Callable[[list[int], tuple[Meld, ...], WinContext], Score | None]
    fixed_horses: bool = False
    cover_robbery: bool = False
    MATCH_HANDS: ClassVar[None] = None
    MIN_FAAN: ClassVar[int] = 0
    SWAP_LOCK: ClassVar[bool] = False
    PASS_LOCK: ClassVar[bool] = False

    @property
    def WALL_RESERVE(self):
        return max(self.OPTIONS.horse_counts)

    def faan(self, hand, melds, ctx):
        return None if ctx.kind == 'rob_concealed' else self.shape_score(hand, melds, ctx)

    def horse_hit(self, tile, winner, dealer):
        if self.fixed_horses:
            return tile in (27, 33) or (tile < 27 and tile % 9 in (0, 4, 8))
        position = (winner - dealer) % 4
        if tile < 27:
            return (tile % 9) % 4 == position
        return tile in ((27, 33), (28, 32), (29, 31), (30,))[position]

    @staticmethod
    def with_horses(score, hits):
        return replace(score, share=score.share * (hits + 1))

    def liability(self, melds, last):
        if self.OPTIONS.reciprocal_liability and sum(meld.provider == last.provider for meld in melds) == 3:
            return ('three_feeds',)
        return ()

    def settlement(self, score, kind, winner, provider, liabilities):
        partners = [partner for _, partner in liabilities]
        if kind == 'zimo':
            if partners:
                return tuple(3 * score.share if seat == partners[-1] else 0 for seat in range(4))
            return tuple(score.share if seat != winner else 0 for seat in range(4))
        covers = (provider in partners) or (kind == 'rob_kong' and self.cover_robbery)
        return tuple(score.share * (3 if covers else 1) if seat == provider else 0 for seat in range(4))

    @staticmethod
    def next_dealer(dealer, winners, provider=None):
        if not winners:
            return dealer
        return winners[0] if len(winners) == 1 else provider


class LiuzhouRoom(HorseRoom):
    SWAP_LOCK = True
    BONUS_MODE = 'exploding_fish'

    def horse_hit(self, tile, winner, dealer):
        return tile % 9 + 1 if tile < 27 else 5

    def settlement(self, score, kind, winner, provider, liabilities):
        if kind == 'zimo' and score.kong_provider is not None:
            return tuple(3 * score.share if seat == score.kong_provider else 0 for seat in range(4))
        partners = {seat for _, seat in liabilities}
        if kind == 'zimo':
            return tuple((3 if seat in partners else 1) * score.share if seat != winner else 0
                         for seat in range(4))
        return tuple(((seat == provider) + (seat in partners)) * score.share for seat in range(4))

    @staticmethod
    def next_dealer(dealer, winners, provider=None):
        if not winners:
            return dealer
        return provider if len(winners) == 3 else winners[0]


def _options(**changes):
    return TableOptions(seven_pairs=True, drawn_added_kong_only=True, host_deals_first=True,
                        show_winds=False, **changes)


SHANGSI = HorseRoom(
    'shangsi', _options(chow=False, pass_pon_same=True, pass_win_higher_same=True,
                        kong_points=KongPoints(1, 1, 2, refund_on_draw=True), horse_counts=(4, 10, 4),
                        dealer_echo=True), _shangsi,
)
HUNDRED = HorseRoom(
    'hundred-tiles', _options(tile_kinds=tuple(range(9, 34)), chow=False, orphans=False,
                              kong_points=KongPoints(3, 1, 2, single_direct_payer=True), horse_counts=(0, 4, 4),
                              concealed_kong_faces=1), _hundred, cover_robbery=True,
)
HEZHOU = HorseRoom(
    'hezhou', _options(pass_pon_all=True, pass_win_all=True,
                       kong_points=KongPoints(1, 1, 1, single_direct_payer=True), horse_counts=(4, 4, 4),
                       reciprocal_liability=True), _hezhou, fixed_horses=True,
)
LIUZHOU = LiuzhouRoom(
    'liuzhou-exploding-fish',
    _options(orphans=False, pass_pon_same=True, pass_win_all=True, chow_lock_extensions=False,
             kong_points=KongPoints(1, 1, 1, single_direct_payer=True, added_payer_from_pong=True),
             horse_counts=(1, 1, 1), reciprocal_liability=True),
    _liuzhou,
)
