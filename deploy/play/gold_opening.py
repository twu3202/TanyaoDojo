"""Shared public gold indicator and one opening claim window per seat."""

from collections import deque

from .honors import PASS


GOLD_CLAIMS = ('claim_gold', 'three_gold', 'opening_heaven')


class GoldOpening:
    def _start_gold_opening(self, index, seats, *, heaven=False):
        while (tile := self.wall.pop(index)) in self.flower_kinds:
            self._reveal_flower(self.dealer, tile)
        self.indicator_tile = self.wildcard_tile = tile
        order = sorted(seats, key=lambda actor: self.players[actor].hand[tile] != 3)
        self.gold_queue = deque((actor, False) for actor in order)
        if heaven and self._qualifies_for_gold(self.players[self.dealer].hand):
            self.gold_queue.appendleft((self.dealer, True))
        self.phase = 'gold_opening'
        self.cur = self.gold_queue[0][0]

    def physical_counts(self):
        counts = super().physical_counts()
        counts[self.indicator_tile] += 1
        return counts

    def _gold_actions(self):
        if self.gold_queue[0][1]:
            return [PASS, ('opening_heaven', None)]
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
