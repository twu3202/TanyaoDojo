"""Zhejiang Taizhou points, reciprocal bills and liability rules."""

from dataclasses import replace
from functools import lru_cache
from itertools import combinations

from .hangzhou import FOUR_HANGZHOU, WHITE, HangzhouGame, HangzhouRules, split_wild
from .honors import DRAGONS, TILE_COUNT, WINDS, Meld
from .honors_bonus import Score
from .tiantai import _set_hu, _winning_placements
from .wildcard_shapes import can_form_standard, wildcard_decompositions, without_tile


LIMIT = 100
FRESH_STAGE_WALL = 30
_RESERVE = 16
_TILES = tuple(range(TILE_COUNT))


def _meld_hu(meld, seat_wind, open_triplet=False):
    if meld.tile in WINDS and (meld.kind == 'peng' or open_triplet):
        return 4
    return _set_hu(meld, seat_wind, open_triplet)


def _value_set(meld, seat_wind):
    return meld.kind != 'chi' and (meld.tile == seat_wind or meld.tile in DRAGONS - {WHITE})


class TaizhouRules(HangzhouRules):
    WALL_RESERVE = _RESERVE

    def next_dealer(self, dealer, winners, provider):
        return dealer if winners and winners[0] == dealer else (dealer + 1) % self.OPTIONS.seat_count

    @staticmethod
    def can_win(hand, melds, ctx):
        if ctx.kind not in ('zimo', 'ron', 'rob_kong') or ctx.tile is None or not hand[ctx.tile]:
            return False
        if (ctx.kind != 'zimo' and ctx.tile == WHITE) or sum(hand) != 14 - 3 * len(melds):
            return False
        return can_form_standard(*split_wild(hand))

    def faan(self, hand, melds, ctx, *, indicator=WHITE):
        if not self.can_win(hand, melds, ctx):
            return None
        natural, wild = split_wild(hand)
        self_draw = ctx.kind == 'zimo'
        fixed_hu = 10 + 2 * self_draw + sum(_meld_hu(meld, ctx.seat_wind) for meld in melds)
        open_faces = {tile for meld in melds for tile in meld.tiles}
        valued_pairs = DRAGONS | {ctx.seat_wind}
        best = None
        for filled, assignments, shapes in wildcard_decompositions(natural, wild, len(melds), _TILES):
            used = open_faces | {tile for tile, count in enumerate(filled) if count}
            suits = {tile // 9 for tile in used if tile < 27}
            bonus = []
            if len(suits) == 1:
                bonus.extend(('清一色',) * 3 if max(used) < 27 else ('混一色',))
            if not wild:
                bonus.append('无得')
            elif assignments == (indicator,):  # Every wildcard uses the same printed face.
                bonus.append('得还原')
            winning_tiles = assignments if ctx.tile == WHITE else (ctx.tile,)
            for pair, concealed in shapes:
                all_sets = (*melds, *concealed)
                fan = [*bonus]
                fan.extend('门风刻杠' if m.tile == ctx.seat_wind else '字牌刻杠'
                           for m in all_sets if _value_set(m, ctx.seat_wind))
                all_pungs = all(meld.kind != 'chi' for meld in all_sets)
                for tile in winning_tiles:
                    for wait, opened in _winning_placements(pair, concealed, tile):
                        inside = wait == '嵌档'
                        hu = fixed_hu + 2 * ((pair in valued_pairs) + all_pungs + inside)
                        hu += sum(_meld_hu(m, ctx.seat_wind, not self_draw and i == opened)
                                  for i, m in enumerate(concealed))
                        raw = hu << len(fan)
                        if best is None or raw > best[0]:
                            names = (*dict.fromkeys(fan), *(('大对子',) if all_pungs else ()),
                                     *(('嵌档',) if inside else ()), f'底胡x{hu}', f'番数x{len(fan)}')
                            best = raw, Score(min(raw, LIMIT), names)
        return best[1] if best is not None else None

    @staticmethod
    def settle_hand(game, winners, offer):
        winner, = winners
        liable, reason = game.liable_seat(winner, offer)
        values = [game.loser_points(seat, offer) if seat != winner.actor else winner.score.share
                  for seat in range(game.seat_count)]
        if liable is not None:
            values[liable] = 0
        bills = pair_bills(values, game.dealer, liable)
        game.taizhou_settlement = {'values': values, 'liable': liable, 'liability_reason': reason, 'bills': bills}
        for recipient, payments in enumerate(zip(*bills, strict=True)):
            if recipient != winner.actor and any(payments):
                game._transfer(recipient, payments)
        return [(winner.score, tuple(row[winner.actor] for row in bills))]


def pair_bills(values, dealer, liable=None):
    bills = [[0] * len(values) for _ in values]
    for a, b in combinations(range(len(values)), 2):
        low, high = sorted((a, b), key=values.__getitem__)
        if values[low] == values[high]:
            continue
        amount = LIMIT if values[high] == LIMIT else values[high] - values[low]
        if values[high] != LIMIT and dealer not in (low, high):
            amount /= 2
        payer = low if liable is None else liable
        if payer != high:
            bills[payer][high] += amount
    return bills


@lru_cache(maxsize=8192)
def _ready(hand, capacity):
    for tile, count in enumerate(hand):
        if count < capacity[tile]:
            augmented = without_tile(hand, tile, -1)
            if can_form_standard(*split_wild(augmented)):
                return True
    return False


def _one_from_ready(hand, melds):
    capacity = [4] * TILE_COUNT
    capacity[WHITE] -= 1  # The flipped tile stays out of play.
    for meld in melds:
        for tile in meld.tiles:
            capacity[tile] -= 1
    capacity = tuple(capacity)
    if _ready(hand, capacity):
        return False
    for draw, count in enumerate(hand):
        if count >= capacity[draw]:
            continue
        augmented = without_tile(hand, draw, -1)
        if any(_ready(without_tile(augmented, discard), capacity)
               for discard, n in enumerate(hand) if n and discard != draw):
            return True
    return False


TAIZHOU = TaizhouRules('taizhou-four-hu', replace(FOUR_HANGZHOU.OPTIONS, host_deals_first=False,
                                               pass_win_same=True, pass_pon_same=True, pass_win_clear_by_call=True,
                                               seven_pairs=False, show_winds=True), fixed_dealer=1, flip_wildcard=True)


class TaizhouGame(HangzhouGame):
    def __init__(self, seed, dealer, round_wind, *, rules, dealer_streak=0):
        self.played_faces = set()
        self.fresh_discard = False
        self.taizhou_settlement = None
        super().__init__(seed, dealer, round_wind, rules=rules, dealer_streak=dealer_streak)

    def _replacement_index(self):
        return self.rules.WALL_RESERVE

    @property
    def fresh_stage(self):
        return len(self.wall) <= FRESH_STAGE_WALL

    def _has_float_wait(self, hand, meld_count):
        return False

    def _throw_tile(self, actor, tile):
        self.fresh_discard = tile not in self.played_faces
        self.played_faces.add(tile)
        super()._throw_tile(actor, tile)

    def _call_discard(self, offer):
        actor = self.claim[0]
        super()._call_discard(offer)
        self.players[actor].declined_calls.clear()

    def _can_take_win(self, actor, offer):
        return offer.kind in ('discard', 'gang_bu')

    def _score(self, actor, hand, melds, ctx):
        return self.rules.faan(hand, melds, ctx, indicator=self.wildcard_indicator)

    def loser_points(self, actor, offer):
        player = self.players[actor]
        hand = player.hand.copy()
        if offer is not None and offer.kind == 'gang_bu' and offer.actor == actor:
            hand[offer.tile] -= 1
        printed = self.wildcard_indicator
        if printed != WHITE:
            hand[printed] += hand[WHITE]
            hand[WHITE] = 0
        wind = 27 + (actor - self.dealer) % self.seat_count
        hu = sum(2 for tile in DRAGONS | {wind} if hand[tile] % 3 == 2)
        fan = 0
        for meld in player.melds:
            hu += _meld_hu(meld, wind)
            fan += _value_set(meld, wind)
        for tile, count in enumerate(hand):
            if count >= 3:
                pung = Meld('triplet', tile)
                hu += (count // 3) * _meld_hu(pung, wind)
                fan += (count // 3) * _value_set(pung, wind)
        return min(LIMIT, hu << fan)

    def liable_seat(self, winner, offer):
        if offer is not None and offer.kind == 'discard' and self.fresh_discard and self.fresh_stage:
            return offer.actor, 'late_fresh_tile'
        chows = [meld for meld in self.players[winner.actor].melds if meld.kind == 'chi']
        if '清一色' in winner.score.patterns and len(chows) >= 3:
            return (chows[0].provider if offer is None else offer.actor), 'pure_suit_feeds'
        if offer is not None and offer.kind == 'discard' and self.fresh_discard:
            printed = self.physical_tile(offer.tile)
            player = self.players[offer.actor]
            hand = tuple(player.hand)
            natural_dragon = printed in DRAGONS and offer.tile in DRAGONS
            white_case = (printed == WHITE and self.wildcard_indicator < 27
                          and any(hand[t] == 1 for t in DRAGONS - {WHITE}))
            if (natural_dragon or white_case) and _one_from_ready(hand, player.melds):
                return offer.actor, 'fresh_dragon'
        return None, None
