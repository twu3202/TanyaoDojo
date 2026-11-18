"""Shared opening, replacement wall and settlement for 16-tile tables."""

from .flower_wall import FlowerWall
from .honors import HonorsGame, PASS, TableOptions

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
        self._deal_flower_racks(WAITING_HAND)

    def _response_actions(self):
        if self.winners:
            return self.responses[0], [PASS]
        return super()._response_actions()
