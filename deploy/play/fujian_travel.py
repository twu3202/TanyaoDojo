"""Quanzhou weighted sets and Ningde flower awards over five-set gold travel."""

from dataclasses import replace
from functools import lru_cache
from itertools import product

from .fujian_sixteen import WHITE, FujianSixteenGame, FujianSixteenRules, kong_units
from .honors import PASS, WinContext
from .honors_bonus import Score
from .longyan import TRAVEL
from .sixteen import OPTIONS, SET_COUNT
from .wildcard_shapes import can_form_sets, without_tile


QUANZHOU_FACTORS = {'平胡': 1, '抢杠胡': 1, '自摸': 2, '三金倒': 3, '天胡': 4,
                    '八花': 4, '游金': 3, '双游': 6, '三游': 9}
TRAVEL_PATTERNS = ('游金', '双游', '三游')
KONG_FEES = {'gang_bu': 1, 'gang_ming': 2, 'gang_an': 3}
EXPOSED_WEIGHTS = {'peng': 1, 'gang_bu': 2, 'gang_ming': 2, 'gang_an': 3}


@lru_cache(maxsize=16384)
def _quanzhou_sets(natural, wild):
    """Maximum set points on an exact split, or None for an incomplete split."""
    first = next((tile for tile, count in enumerate(natural) if count), None)
    if first is None:
        return 2 * (wild // 3) if wild % 3 == 0 else None
    scores = []
    for used in range(max(1, 3 - wild), min(natural[first], 3) + 1):
        tail = _quanzhou_sets(without_tile(natural, first, used), wild - (3 - used))
        if tail is not None:
            scores.append(tail + 1 + (first >= 27))
    if first < 27:
        suit = first // 9 * 9
        for start in range(max(suit, first - 2), min(first, suit + 6) + 1):
            other = tuple(tile for tile in range(start, start + 3) if tile != first)
            for use in product(*(range(1 + bool(natural[tile])) for tile in other)):
                missing = 2 - sum(use)
                if missing > wild:
                    continue
                rest = list(natural)
                rest[first] -= 1
                for tile, amount in zip(other, use):
                    rest[tile] -= amount
                tail = _quanzhou_sets(tuple(rest), wild - missing)
                if tail is not None:
                    scores.append(tail)
    return max(scores, default=None)


def quanzhou_closed_points(natural, wild, *, waiting=False):
    if waiting:
        # One actual gold is already committed to the eyes for the instant third travel.
        return _quanzhou_sets(natural, wild - 1) if wild else None
    best = _quanzhou_sets(natural, wild - 2) if wild >= 2 else None
    for tile, count in enumerate(natural):
        for used in range(max(1, 2 - wild), min(count, 2) + 1):
            tail = _quanzhou_sets(without_tile(natural, tile, used), wild - (2 - used))
            if tail is not None and (best is None or tail > best):
                best = tail
    return best


class TravelingSixteenRules(FujianSixteenRules):
    MAX_TRAVEL = 1
    EIGHT_FLOWERS = False

    def travel_wait(self, hand, melds, gold):
        if sum(hand) != 3 * (SET_COUNT - len(melds)) + 1 or not hand[gold]:
            return False
        natural, wild = self.split(hand, gold)
        return can_form_sets(natural, wild - 1)


class QuanzhouRules(TravelingSixteenRules):
    OPTIONS = replace(OPTIONS, pass_win_same=True, pass_pon_same=True)
    MAX_TRAVEL = 3
    EIGHT_FLOWERS = True

    @staticmethod
    def next_dealer(dealer, winners, provider):
        return winners[0] if winners else dealer

    def score(self, game, hand, melds, ctx, flowers, *, travel_level=0):
        natural, wild = self.split(hand, game.wildcard_tile)
        set_points = quanzhou_closed_points(natural, wild, waiting=ctx.kind == 'third_travel')
        opening = {'opening_heaven': '天胡', 'three_gold': '三金倒', 'flower_hu': '八花'}
        pattern = opening.get(ctx.kind)
        if travel_level:
            pattern = TRAVEL_PATTERNS[travel_level - 1]
        elif ctx.kind == 'zimo':
            if ctx.opening == '天胡' and set_points is not None:
                pattern = '天胡'
            else:
                pattern = '三金倒' if wild == 3 else '自摸'
        elif pattern is None:
            pattern = '抢杠胡' if ctx.kind == 'rob_kong' else '平胡'
        if set_points is None:
            # For a special win without a complete shape, only intact natural pungs count.
            set_points = sum(1 + (tile >= 27) for tile, count in enumerate(natural) if count >= 3)
        set_points += sum(EXPOSED_WEIGHTS[meld.kind] + (meld.tile >= 27) for meld in melds if meld.kind != 'chi')
        kong_points = sum(KONG_FEES.get(meld.kind, 0) for meld in melds)
        basic = 1 + len(flowers) + wild + set_points + kong_points
        return Score(basic * QUANZHOU_FACTORS[pattern],
                     (pattern, f'刻分x{set_points}', f'杠分x{kong_points}', f'金牌x{wild}', f'花数x{len(flowers)}'))

    @staticmethod
    def payments(game, actor, provider, score):
        dealer_extra = game.dealer_streak * QUANZHOU_FACTORS[score.patterns[0]]
        return tuple(score.share + dealer_extra * (game.dealer in (actor, seat))
                     if seat != actor and provider in (None, seat) else 0 for seat in range(game.seat_count))


class NingdeRandomGoldRules(TravelingSixteenRules):
    OPTIONS = replace(OPTIONS, pass_win_same=True)
    WALL_RESERVE = 12
    WHITE_PROXY = WHITE
    DISCARD_AWARDS = {'天胡': 30, '抢金': 30, '三金倒': 40, '金雀': 60, '金龙': 120, '混一色': 120, '清一色': 240}
    DRAW_AWARDS = DISCARD_AWARDS

    def score(self, game, hand, melds, ctx, flowers, *, travel_level=0):
        gold = game.wildcard_tile
        water = len(flowers) + hand[gold] + kong_units(melds)
        awards = self.awards(hand, melds, ctx, gold, flowers)
        if water <= 1:
            awards.append(('一水胡', 15) if water else ('无水胡', 30))
        pattern, bonus = max(awards, key=lambda award: award[1])
        factor = 1 if ctx.kind in ('ron', 'rob_kong') else 2
        tags = (pattern, '游金') if travel_level else (pattern,)
        return Score((water + game.dealer_streak) * factor + bonus,
                     (*tags, f'水数x{water}', f'金牌x{hand[gold]}'))


QUANZHOU = QuanzhouRules('quanzhou-triplet-water')
NINGDE_RANDOM_GOLD = NingdeRandomGoldRules('ningde-random-gold')
FUJIAN_TRAVEL_ROOMS = (QUANZHOU, NINGDE_RANDOM_GOLD)


class FujianTravelGame(FujianSixteenGame):
    RECORD_FOLLOW = False

    def __init__(self, seed, dealer, round_wind, *, rules, dealer_streak=0):
        self.travel_level = 0
        self.travel_owner = None
        super().__init__(seed, dealer, round_wind, rules=rules, dealer_streak=dealer_streak)

    def _open_flipped_gold(self, seats):
        if not self.rules.EIGHT_FLOWERS:
            super()._open_flipped_gold(seats)
            return
        self.wildcard_tile = gold = self._flip_gold_indicator(self._replacement_index())
        entries = [(actor, 'flower_hu') for actor in seats if self._eight_flowers(actor)]
        if self._qualifies_for_gold(self.players[self.dealer].hand):
            entries.append((self.dealer, 'opening_heaven'))
        entries.extend((actor, 'three_gold') for actor in seats if self.players[actor].hand[gold] == 3)
        self._queue_gold_choices(entries)

    def _eight_flowers(self, actor):
        return self.rules.EIGHT_FLOWERS and len(self.flowers[actor]) == 8

    def _finish_gold_opening(self, actor, kind, hand):
        if kind != 'flower_hu':
            super()._finish_gold_opening(actor, kind, hand)
            return
        self._settle_travel_special(actor, kind, kind, hand)

    def _settle_travel_special(self, actor, kind, reason, hand):
        ctx = WinContext(kind, self.wildcard_tile, 27, self.round_wind)
        level = 3 if kind == 'third_travel' else 0
        score = self.rules.score(self, hand, self.players[actor].melds, ctx, self.flowers[actor], travel_level=level)
        self._settle_special(actor, None, reason, score, hand)

    def _draw(self, actor, kong_chain=0, provider=None):
        already_had_eight = self._eight_flowers(actor)
        super()._draw(actor, kong_chain, provider)
        if self.phase == 'over' and self.reason_code == 'draw' and not already_had_eight and self._eight_flowers(actor):
            self._finish_gold_opening(actor, 'flower_hu', self.players[actor].hand.copy())

    def _response_actions(self):
        actor, actions = super()._response_actions()
        return (actor, [PASS]) if self.travel_level >= 2 or actor == self.travel_owner else (actor, actions)

    def legal_actions(self):
        actor, actions = super().legal_actions()
        if self.phase != 'action' or self.offer is not None or self.drawn_tile is None:
            return actor, actions
        if self._eight_flowers(actor):
            actions.append(('flower_hu', None))
        if self.travel_owner not in (None, actor) or self.travel_level >= self.rules.MAX_TRAVEL:
            return actor, actions
        if self.travel_level < 2 and self.remaining < self.seat_count:
            return actor, actions
        gold = self.wildcard_tile
        if self.travel_level and self.drawn_tile != gold:
            return actor, actions
        player = self.players[actor]
        hand = tuple(player.hand)
        if hand[gold]:
            actions += [(TRAVEL, tile) for kind, tile in actions if kind == 'discard' and (tile == gold or not self.travel_level)
                        and self.rules.travel_wait(without_tile(hand, tile), player.melds, gold)]
        return actor, actions

    def step(self, action):
        kind, tile = action
        if self.phase == 'gold_opening' or kind not in (TRAVEL, 'flower_hu'):
            super().step(action)
            return
        actor, actions = self.legal_actions()
        if action not in actions:
            raise ValueError('Illegal Fujian travel action')
        self.last_call = None
        hand = self.players[actor].hand
        if kind == 'flower_hu':
            self._finish_gold_opening(actor, kind, hand.copy())
            return
        self.travel_level += 1
        self.travel_owner = actor
        super()._discard(actor, tile)
        if self.travel_level == 3:
            self._settle_travel_special(actor, 'third_travel', 'third_travel_win', hand.copy())

    def _discard(self, actor, tile):
        if actor == self.travel_owner:
            self.travel_level = 0
            self.travel_owner = None
        super()._discard(actor, tile)

    def _score(self, actor, hand, melds, ctx):
        if ctx.opening == '天胡' and self.passed_heaven:
            return None
        if self.rules.can_win(hand, melds, ctx, gold=self.wildcard_tile):
            level = self.travel_level if ctx.kind == 'zimo' and self.travel_owner == actor else 0
            return self.rules.score(self, hand, melds, ctx, self.flowers[actor], travel_level=level)
        return None
