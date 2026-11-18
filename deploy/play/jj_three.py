"""JJ's three-seat call-points room, using the 2017 settlement table."""

from dataclasses import dataclass
from .honors import (DRAGONS, PASS, RON, ZIMO, HonorsGame, KongPoints, TableOptions,
                     ready_discards, seven_pairs, standard_shapes)


VARIANT = 'jj-three-player'
MIN_FAAN = 0
WALL_RESERVE = 0
MATCH_HANDS = None
SWAP_LOCK = False
PASS_LOCK = False
OPTIONS = TableOptions(tile_kinds=(*range(18), *sorted(DRAGONS)), seat_count=3, seven_pairs=True,
                       orphans=False, show_winds=False, drawn_added_kong_only=True,
                       kong_requires_live_tile=True, kong_points=KongPoints(5, 1, 5, single_direct_payer=True))


@dataclass(frozen=True)
class Score:
    share: int = 5
    fan: None = None
    tier: None = None
    patterns: tuple[str, ...] = ('底分',)


SCORE = Score()


def faan(hand, melds, ctx):
    if ctx.kind == 'rob_concealed':
        return None
    if seven_pairs(hand, melds) or next(standard_shapes(hand, len(melds)), None) is not None:
        return SCORE
    return None


def liability(melds, last):
    return ()


def settle_hand(game, winners, offer):
    outcomes = []
    for winner in winners:
        payers = (set(range(game.seat_count)) - {winner.actor}) if offer is None else {offer.actor}
        payments = tuple(winner.score.share if seat in payers else 0 for seat in range(game.seat_count))
        outcomes.append((winner.score, payments))
    return outcomes


def next_dealer(dealer, winners, provider):
    if not winners:
        return dealer
    return winners[0] if len(winners) == 1 else provider


class JJThreeGame(HonorsGame):
    def __init__(self, seed, dealer, round_wind, *, rules):
        super().__init__(seed, dealer, round_wind, rules=rules)
        self.side_deltas = {key: [0] * self.seat_count
                            for key in ('chow_deltas', 'pon_deltas', 'closed_ready_deltas')}

    def legal_actions(self):
        actor, actions = super().legal_actions()
        if self.phase != 'action':
            return actor, actions
        player = self.players[actor]
        if player.closed_ready:
            return actor, [action for action in actions if action in (ZIMO, ('discard', self.drawn_tile))]
        return actor, actions + list(ready_discards(player, actions, self.options))

    def _response_actions(self):
        actor, actions = super()._response_actions()
        if self.players[actor].closed_ready or self.winners:
            actions = [action for action in actions if action in (PASS, RON)]
        return actor, actions

    def step(self, action):
        if action[0] != 'declare_ready':
            super().step(action)
            return
        actor, actions = self.legal_actions()
        if action not in actions:
            raise ValueError('Illegal paid-ready declaration')
        self.last_call = None
        self.players[actor].closed_ready = True
        payments = tuple(0 if seat == actor else 2 for seat in range(self.seat_count))
        self._transfer(actor, payments, self.side_deltas['closed_ready_deltas'])
        self._discard(actor, action[1])

    def _call_discard(self, offer):
        actor, kind, _ = self.claim
        super()._call_discard(offer)
        if kind in ('chi', 'peng'):
            key, share = ('chow_deltas', 1) if kind == 'chi' else ('pon_deltas', 2)
            payments = tuple(share if seat == offer.actor else 0 for seat in range(self.seat_count))
            self._transfer(actor, payments, self.side_deltas[key])
