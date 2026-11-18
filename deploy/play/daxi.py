"""Daxi mahjong with an opening buy-in and supplier-specific kong payments."""

from .honors import HonorsGame, KongPoints, TableOptions
from .honors_bonus import Score
from .wildcard_shapes import can_form_standard


class DaxiRules:
    VARIANT = 'daxi-buy-base'
    OPTIONS = TableOptions(orphans=False, concealed_kong_faces=0, pass_win_same=True, pass_pon_same=True,
                           kong_points=KongPoints(1, 1, 1, single_direct_payer=True, added_payer_from_pong=True))
    MATCH_HANDS = 8
    MIN_FAAN = 0
    WALL_RESERVE = 16
    PASS_LOCK = False
    SWAP_LOCK = False

    @staticmethod
    def faan(hand, melds, ctx):
        if ctx.kind not in ('zimo', 'ron', 'rob_kong') or sum(hand) != 14 - 3 * len(melds):
            return None
        if can_form_standard(tuple(hand), 0):
            return Score(1, ('平胡',))
        return None

    @staticmethod
    def next_dealer(dealer, winners, provider):
        return dealer if dealer in winners else (dealer + 1) % DAXI.OPTIONS.seat_count

    @staticmethod
    def liability(melds, meld):
        return ()

    @staticmethod
    def settle_hand(game, winners, offer):
        outcomes = []
        for winner in winners:
            actor = winner.actor
            payers = [seat != actor and (offer is None or seat == offer.actor) for seat in range(game.seat_count)]
            buy_ins = tuple(int(game.buy_base[actor]) + int(game.buy_base[seat]) if pays else 0
                            for seat, pays in enumerate(payers))
            game._transfer(actor, buy_ins, game.buy_base_deltas)
            outcomes.append((winner.score, tuple(map(int, payers))))
        return outcomes


DAXI = DaxiRules()


class DaxiGame(HonorsGame):
    def __init__(self, seed, dealer, round_wind, *, rules):
        super().__init__(seed, dealer, round_wind, rules=rules)
        self.buy_base = [None] * self.seat_count
        self.buy_base[dealer] = True
        self.buy_base_deltas = [0] * self.seat_count
        self.opening_draw = self.drawn_tile
        self.drawn_tile = None
        self.phase = 'buy_base'
        self.cur = (dealer + 1) % self.seat_count

    def legal_actions(self):
        if self.phase == 'buy_base':
            return self.cur, [('buy_base', 0), ('buy_base', 1)]
        return super().legal_actions()

    def step(self, action):
        if self.phase != 'buy_base':
            return super().step(action)
        actor, legal = self.legal_actions()
        if action not in legal:
            raise ValueError('Illegal Daxi buy-in choice')
        self.buy_base[actor] = bool(action[1])
        self.cur = (actor + 1) % self.seat_count
        if self.cur == self.dealer:
            self.drawn_tile = self.opening_draw
            self.phase = 'action'

    def _replacement_index(self):
        return self.rules.WALL_RESERVE
