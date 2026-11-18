"""Shared opening, replacement wall and settlement for 16-tile tables."""

from .flower_wall import FlowerWall
from .honors import HonorsGame, PASS, TableOptions, Win

SET_COUNT = 5
WAITING_HAND = SET_COUNT * 3 + 1
OPTIONS = TableOptions(orphans=False, concealed_kong_faces=0, chow_lock_extensions=False,
                       kong_requires_live_tile=True)


class SixteenRules:
    OPTIONS = OPTIONS
    MATCH_HANDS = 8
    MIN_FAAN = 0
    PASS_LOCK = False
    SWAP_LOCK = True
    WALL_RESERVE = 16

    @staticmethod
    def liability(melds, last):
        return ()

    @staticmethod
    def next_dealer(dealer, winners, provider):
        return (dealer + 1) % 4 if winners and dealer not in winners else dealer

    @classmethod
    def settle_hand(cls, game, winners, offer):
        return [(winner.score, cls.payments(game, winner.actor, offer.actor if offer else None, winner.score))
                for winner in winners]


class SixteenGame(FlowerWall, HonorsGame):
    def __init__(self, seed, dealer, round_wind, *, rules, dealer_streak=0):
        self.dealer_streak = dealer_streak
        super().__init__(seed, dealer, round_wind, rules=rules)

    def _replacement_index(self):
        return self.rules.WALL_RESERVE

    def _deal_opening(self):
        order = tuple((self.dealer + i) % self.seat_count for i in range(self.seat_count))
        racks = [[] for _ in self.players]
        for _ in range(4):
            for actor in order:
                racks[actor].extend(self.wall.pop() for _ in range(4))
        racks[self.dealer].append(self.wall.pop())
        while any(tile in self.flower_kinds for rack in racks for tile in rack):
            for actor in order:
                for index, tile in enumerate(racks[actor]):
                    if tile in self.flower_kinds:
                        self.flowers[actor].append(tile)
                        racks[actor][index] = self.wall.pop(self._replacement_index())
        for actor, rack in enumerate(racks):
            for tile in rack:
                self.players[actor].hand[tile] += 1
        self.players[self.dealer].draws = 1
        self.drawn_tile = racks[self.dealer][-1]

    def _response_actions(self):
        if self.winners:
            return self.responses[0], [PASS]
        return super()._response_actions()

    def _settle_special(self, actor, provider, kind, score, hand):
        payments = self.rules.payments(self, actor, provider, score)
        self.offer = None
        self.winners = [actor]
        self.revealed[actor] = hand
        self._transfer(actor, payments)
        self.win_details.append(Win(actor, provider, kind, score.fan, score.share, score.patterns, payments))
        self._end(kind)
