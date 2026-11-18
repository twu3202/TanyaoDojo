"""Shared public gold indicator and one opening claim window per seat."""

from collections import deque

from .honors import PASS


GOLD_CLAIMS = ('claim_gold', 'three_gold', 'four_gold', 'opening_heaven')


def split_gold(hand, gold, *, substitute=None):
    natural = list(hand)
    wild = natural[gold]
    natural[gold] = 0
    if substitute is not None and substitute != gold:
        natural[gold], natural[substitute] = natural[substitute], 0
    return tuple(natural), wild


def record_opening_follow(game, actor, tile):
    if not game.following or not game.uninterrupted:
        return
    number = len(game.discards[actor]) - 1
    if actor != game.dealer and tile != game.discards[game.dealer][number]:
        game.following = False
    elif actor == (game.dealer - 1) % game.seat_count:
        share = (2 if tile < 27 else 1) * 2 ** game.follow_rounds
        game.follow_rounds += 1
        for seat, player in enumerate(game.players):
            amount = share if seat != game.dealer else (1 - game.seat_count) * share
            player.score += amount
            game.follow_deltas[seat] += amount


class GoldOpening:
    def _flip_gold_indicator(self, index):
        while (tile := self.wall.pop(index)) in self.flower_kinds:
            self._reveal_flower(self.dealer, tile)
        self.indicator_tile = tile
        return tile

    def _start_gold_opening(self, index, seats, *, heaven=False):
        self.wildcard_tile = tile = self._flip_gold_indicator(index)
        order = sorted(seats, key=lambda actor: self.players[actor].hand[tile] != 3)
        entries = [(actor, None) for actor in order]
        if heaven and self._qualifies_for_gold(self.players[self.dealer].hand):
            entries.insert(0, (self.dealer, 'opening_heaven'))
        self._queue_gold_choices(entries)

    def _queue_gold_choices(self, entries):
        self.gold_queue = deque(entries)
        if entries:
            self.phase = 'gold_opening'
            self.cur = entries[0][0]

    def physical_counts(self):
        counts = super().physical_counts()
        if self.indicator_tile is not None:
            counts[self.indicator_tile] += 1
        return counts

    def _gold_actions(self):
        if (choice := self.gold_queue[0][1]) is not None:
            return [PASS, (choice, None)]
        hand = self.players[self.cur].hand
        gold = self.wildcard_tile
        if hand[gold] == 3:
            return [PASS, ('three_gold', None)]
        augmented = hand.copy()
        augmented[gold] += 1
        if self.cur != self.dealer:
            return [PASS, ('claim_gold', None)] if self._qualifies_for_gold(augmented) else [PASS]
        actions = [PASS]
        for tile, count in enumerate(hand):
            if count and tile != gold:
                augmented[tile] -= 1
                if self._qualifies_for_gold(augmented):
                    actions.append(('claim_gold', tile))
                augmented[tile] += 1
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
            self.cur = self.gold_queue[0][0] if self.gold_queue else self.dealer
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
        self._finish_gold_opening(actor, kind, hand)
