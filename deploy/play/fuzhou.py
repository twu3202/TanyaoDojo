"""Fuzhou 36-flower Mahjong with an opening gold claim."""

from collections import deque
from dataclasses import replace

from .honors import DRAGONS, PASS, WINDS
from .honors_bonus import Score
from .sixteen import OPTIONS, SET_COUNT, SixteenGame, SixteenRules
from .wildcard_shapes import can_form_sets, can_form_standard, without_tile


HONOR_FLOWERS = WINDS | DRAGONS
GOLD_CLAIMS = ('claim_gold', 'three_gold')


def split_gold(hand, gold):
    return without_tile(tuple(hand), gold, hand[gold]), hand[gold]


class FuzhouRules(SixteenRules):
    VARIANT = 'fuzhou-36-flowers'
    WALL_RESERVE = 18
    OPTIONS = replace(OPTIONS, pass_win_same=True, show_winds=False)

    @staticmethod
    def can_win(hand, melds, ctx, *, gold):
        if ctx.kind not in ('zimo', 'ron') or sum(hand) != 3 * (SET_COUNT - len(melds)) + 2:
            return False
        natural, wild = split_gold(hand, gold)
        if ctx.kind == 'ron':
            if ctx.tile == gold:
                return False
            # A spare gold can pair with any drawn face once all other sets are ready.
            if wild and can_form_sets(without_tile(natural, ctx.tile), wild - 1, HONOR_FLOWERS):
                return False
        return (ctx.kind == 'zimo' and wild == 3) or can_form_standard(natural, wild, HONOR_FLOWERS)

    @staticmethod
    def score(hand, melds, *, gold, flowers, dealer_streak, kind, opening=False):
        natural, wild = split_gold(hand, gold)
        exposed = sum(m.kind in ('gang_ming', 'gang_bu') for m in melds)
        concealed = sum(m.kind == 'gang_an' for m in melds)
        drawn = kind != 'ron'
        awards = [(0, '平胡')]
        if opening:
            awards.append((30, '天胡'))
        if kind == 'gold_claim':
            awards.append((30, '抢金'))
        if kind in ('zimo', 'three_gold_win') and wild == 3:
            awards.append((40, '三金倒'))
            if can_form_standard(natural, 0, HONOR_FLOWERS):
                awards.append((120, '金龙'))
        elif kind == 'zimo' and wild == 2 and can_form_sets(natural, 0, HONOR_FLOWERS):
            awards.append((60, '金雀'))
        suits = {tile // 9 for tile, count in enumerate(natural) if count} | {m.tile // 9 for m in melds}
        if kind != 'gold_claim' and len(suits) == 1 and can_form_standard(natural, wild, HONOR_FLOWERS):
            awards.append((240, '清一色'))
        if not flowers and not (exposed or concealed):
            awards.append((30, '无花无杠'))
        elif flowers == 1:
            awards.append((15, '一张花'))
        special, pattern = max(awards, key=lambda award: award[0])
        points = (flowers + wild + exposed + 2 * concealed + dealer_streak) * (2 if drawn else 1) + special
        return Score(points, (pattern, f'花牌x{flowers}', f'金牌x{wild}'))

    @staticmethod
    def payments(game, actor, provider, score):
        return tuple(score.share if seat != actor and provider in (None, seat) else 0
                     for seat in range(game.seat_count))


FUZHOU = FuzhouRules()


class FuzhouGame(SixteenGame):
    excluded_tiles = HONOR_FLOWERS

    def _prepare_wall(self):
        super()._prepare_wall()
        self.flower_kinds = self.flower_kinds | HONOR_FLOWERS
        self.indicator_tile = None
        self.wildcard_tile = None
        self.gold_discards = set()

    def _deal_opening(self):
        super()._deal_opening()
        while (tile := self.wall.pop(self._replacement_index())) in self.flower_kinds:
            self.flowers[self.dealer].append(tile)
        self.indicator_tile = self.wildcard_tile = tile

        order = [(self.dealer + step) % self.seat_count for step in range(self.seat_count)]
        order.sort(key=lambda actor: self.players[actor].hand[tile] != 3)
        self.gold_queue = deque(order)
        self.phase = 'gold_opening'
        self.cur = self.gold_queue[0]

    def physical_counts(self):
        counts = super().physical_counts()
        counts[self.indicator_tile] += 1
        return counts

    def _gold_actions(self):
        hand = tuple(self.players[self.cur].hand)
        gold = self.wildcard_tile
        if hand[gold] == 3:
            return [PASS, ('three_gold', None)]
        if self.cur != self.dealer:
            natural, wild = split_gold(hand, gold)
            return [PASS, ('claim_gold', None)] if can_form_standard(natural, wild + 1, HONOR_FLOWERS) else [PASS]
        actions = [PASS]
        for tile, count in enumerate(hand):
            if count and tile != gold:
                natural, wild = split_gold(without_tile(hand, tile), gold)
                if can_form_standard(natural, wild + 1, HONOR_FLOWERS):
                    actions.append(('claim_gold', tile))
        return actions

    def legal_actions(self):
        if self.phase == 'gold_opening':
            return self.cur, self._gold_actions()
        return super().legal_actions()

    def step(self, action):
        if self.phase != 'gold_opening':
            super().step(action)
            return
        if action not in self._gold_actions():
            raise ValueError('Illegal opening gold action')
        if action == PASS:
            self.gold_queue.popleft()
            self.phase = 'gold_opening' if self.gold_queue else 'action'
            self.cur = self.gold_queue[0] if self.gold_queue else self.dealer
            return
        actor = self.cur
        kind, discarded = action
        hand = self.players[actor].hand.copy()
        if kind == 'claim_gold':
            if discarded is not None:
                hand[discarded] -= 1
                self.players[actor].hand[discarded] -= 1
                self.discards[actor].append(discarded)
            hand[self.wildcard_tile] += 1
        reason = 'gold_claim' if kind == 'claim_gold' else 'three_gold_win'
        score = self.rules.score(hand, (), gold=self.wildcard_tile, flowers=len(self.flowers[actor]),
                                 dealer_streak=self.dealer_streak, kind=reason)
        self._settle_special(actor, None, reason, score, hand)

    def _draw(self, actor, kong_chain=0, provider=None):
        if not kong_chain:
            self.gold_discards.discard(actor)
        super()._draw(actor, kong_chain, provider)

    def _discard(self, actor, tile):
        if tile == self.wildcard_tile:
            self.gold_discards.add(actor)
        super()._discard(actor, tile)

    def _chows(self, player, tile, provider):
        return (choice for choice in super()._chows(player, tile, provider) if self.wildcard_tile not in choice[1])

    def _response_actions(self):
        if self.offer.tile == self.wildcard_tile:
            return self.responses[0], [PASS]
        return super()._response_actions()

    def _score(self, actor, hand, melds, ctx):
        if ctx.kind == 'ron' and actor in self.gold_discards:
            return None
        if not self.rules.can_win(hand, melds, ctx, gold=self.wildcard_tile):
            return None
        return self.rules.score(hand, melds, gold=self.wildcard_tile, flowers=len(self.flowers[actor]),
                                dealer_streak=self.dealer_streak, kind=ctx.kind, opening=ctx.opening == '天胡')
