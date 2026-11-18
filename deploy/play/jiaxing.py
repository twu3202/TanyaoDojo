"""Jiaxing's qualifying tai, carried bonuses and continue-or-fold settlement."""

from collections import deque
from dataclasses import replace
import random

from .hangzhou import FOUR_HANGZHOU, HangzhouGame, HangzhouRules, split_wild
from .honors import DRAGONS, PASS, TILE_COUNT
from .honors_bonus import Score
from .wildcard_shapes import can_form_all_pungs, can_form_standard, wildcard_decompositions


_TILES = tuple(range(TILE_COUNT))


class PrintedWildcardGame(HangzhouGame):
    """A flipped face is wild while every other tile keeps its printed identity."""

    def _flip_wildcard(self):
        index = next(i for i, tile in enumerate(self.wall) if tile < TILE_COUNT)
        self.wildcard_indicator = self.indicator_tile = self.wildcard_tile = self.wall.pop(index)

    def physical_tile(self, tile):
        return tile

    def _chows(self, player, tile, provider):
        return [choice for choice in super()._chows(player, tile, provider) if self.wildcard_tile not in choice[1]]

    def _replacement_index(self):
        return self.rules.WALL_RESERVE

    def _has_float_wait(self, hand, meld_count):
        return False


class JiaxingRules(HangzhouRules):
    @staticmethod
    def qualifying(natural, wild, melds):
        suited = {tile // 9 for tile, count in enumerate(natural[:27]) if count}
        suited.update(meld.tile // 9 for meld in melds if meld.tile < 27)
        if len(suited) <= 1 and can_form_standard(natural, wild):
            return True
        return all(meld.kind != 'chi' for meld in melds) and can_form_all_pungs(natural, wild)

    def can_win(self, hand, melds, ctx, *, wildcard):
        if ctx.kind not in ('zimo', 'rob_kong') or ctx.tile is None or sum(hand) != 14 - 3 * len(melds):
            return False
        if not hand[ctx.tile] or (ctx.kind == 'rob_kong' and ctx.tile == wildcard):
            return False
        natural, wild = split_wild(hand, wildcard)
        if ctx.kind == 'rob_kong' and wild:
            capacity = [4] * TILE_COUNT
            capacity[wildcard] -= 1
            for meld in melds:
                for tile in meld.tiles:
                    capacity[tile] -= 1
            before = list(hand)
            before[ctx.tile] -= 1
            for tile, n in enumerate(before):
                if n >= capacity[tile]:
                    continue
                before[tile] += 1
                any_draw = self.qualifying(*split_wild(before, wildcard), melds)
                before[tile] -= 1
                if not any_draw:
                    break
            else:
                return False
        return self.qualifying(natural, wild, melds)

    def faan(self, hand, melds, ctx, *, wildcard, yellow=None):
        if not self.can_win(hand, melds, ctx, wildcard=wildcard):
            return None
        natural, wild = split_wild(hand, wildcard)
        opened = {tile for meld in melds for tile in meld.tiles}
        base_fan = []
        if wild == 3:
            base_fan.append('三百搭')
        if ctx.kong_chain and ctx.kind == 'zimo':
            base_fan.append('杠上开花')
        if ctx.last_live_draw and ctx.kind == 'zimo':
            base_fan.append('海底捞月')
        if ctx.opening == '天胡' and ctx.kind == 'zimo':
            base_fan.extend(('天胡',) * 2)
        if ctx.first_nondealer_draw and ctx.kind == 'zimo':
            base_fan.extend(('地胡',) * 2)
        if yellow:
            base_fan.append(yellow)
        if len(melds) == 4 and (ctx.tile == wildcard or natural[ctx.tile] == 2):
            base_fan.append('单钓')

        best = None
        for filled, _, shapes in wildcard_decompositions(natural, wild, len(melds), _TILES):
            faces = opened | {tile for tile, count in enumerate(filled) if count}
            suits = {tile // 9 for tile in faces if tile < 27}
            for _, concealed in shapes:
                sets = (*melds, *concealed)
                all_pungs = all(meld.kind != 'chi' for meld in sets)
                if len(suits) > 1 and not all_pungs:
                    continue
                labels = []
                tai = int(not wild)
                if not wild:
                    labels.append('无百搭')
                if not suits:
                    tai += 10
                    labels.append('全老头')
                elif len(suits) == 1:
                    pure = max(faces) < 27
                    tai += 4 if pure else 2
                    labels.append('清一色' if pure else '混一色')
                if all_pungs:
                    tai += 2
                    labels.append('大对子')
                value_sets = {meld.tile for meld in sets if meld.kind != 'chi'}
                tai += len(value_sets & DRAGONS) + (ctx.seat_wind in value_sets) + (ctx.round_wind in value_sets)
                labels.extend((*dict.fromkeys(base_fan), f'台数x{tai}', f'番数x{len(base_fan)}'))
                score = Score(tai << len(base_fan), tuple(labels))
                if best is None or score.share > best.share:
                    best = score
        return best

    def next_dealer(self, dealer, winners, provider):
        return dealer if not winners or dealer in winners else (dealer + 1) % self.OPTIONS.seat_count

    @staticmethod
    def liability(melds, meld):
        if sum(m.provider == meld.provider for m in melds) == 3:
            return ('three_feeds',)
        return ()

    @staticmethod
    def settle_hand(game, winners, offer):
        winner, = winners
        actor = winner.actor
        links = {frozenset(pair) for pair in game.related_seats}
        participants = {seat for pair in links for seat in pair}
        factors = [0] * game.seat_count
        for seat in range(game.seat_count):
            if seat == actor or seat in game.folded:
                continue
            if actor in participants:
                factors[seat] = 5 if frozenset((actor, seat)) in links else 2
            elif links:
                factors[seat] = 2 if seat in participants else 1
            elif offer is None:
                factors[seat] = 1
        if offer is not None:
            factors[offer.actor] += 3
        return [(winner.score, tuple(winner.score.share * factor for factor in factors))]


JIAXING = JiaxingRules('jiaxing-raise-head', replace(FOUR_HANGZHOU.OPTIONS, show_winds=True, seven_pairs=False,
                                                      reciprocal_liability=True, drawn_added_kong_only=True),
                       fixed_dealer=1, flip_wildcard=True)


class JiaxingGame(PrintedWildcardGame):
    def __init__(self, seed, dealer, round_wind, *, rules, dealer_streak=0):
        self.folded = set()
        self.liability_choices = deque()
        self.yellow = deque()
        self.dice = random.Random(f'jiaxing-dice:{seed}')
        super().__init__(seed, dealer, round_wind, rules=rules, dealer_streak=dealer_streak)

    def carry_yellow(self, previous):
        if previous is not None:
            self.yellow.extend(previous.yellow)
        if self.wildcard_tile >= 27:
            self.yellow.append('老头黄')
        if self.dice.randint(1, 6) == self.dice.randint(1, 6):
            self.yellow.append('豹子黄')
        if previous is not None and self.wildcard_tile == previous.wildcard_tile:
            self.yellow.append('百搭连庄')

    def _score(self, actor, hand, melds, ctx):
        return self.rules.faan(hand, melds, ctx, wildcard=self.wildcard_tile, yellow=self.yellow[0] if self.yellow else None)

    def _can_take_win(self, actor, offer):
        return offer.kind == 'gang_bu'

    def legal_actions(self):
        if self.phase == 'liability_choice':
            return self.liability_choices[0], [('liability_choice', 0), ('liability_choice', 1)]
        if self.phase == 'action' and self.cur in self.folded:
            return self.cur, [('discard', self.drawn_tile)]
        return super().legal_actions()

    def _response_actions(self):
        if self.responses[0] in self.folded:
            return self.responses[0], [PASS]
        return super()._response_actions()

    def step(self, action):
        if self.phase != 'liability_choice':
            return super().step(action)
        if action not in self.legal_actions()[1]:
            raise ValueError('Illegal Jiaxing liability choice')
        self.last_call = None
        seat = self.liability_choices.popleft()
        if action[1] == 0:
            self.folded.add(seat)
        if not self.liability_choices:
            self.phase = 'action'

    def _call_discard(self, offer):
        previous = len(self.related_seats)
        super()._call_discard(offer)
        if len(self.related_seats) == previous:
            return
        if offer.actor in self.folded:
            self.related_seats.pop()
            return
        excluded = self.folded | {seat for pair in self.related_seats for seat in pair}
        self.liability_choices.extend(seat for seat in range(self.seat_count) if seat not in excluded)
        if self.liability_choices:
            self.phase = 'liability_choice'

    def _end(self, reason):
        super()._end(reason)
        if reason == 'draw':
            self.yellow.append('黄番')
        elif self.yellow:
            self.yellow.popleft()
