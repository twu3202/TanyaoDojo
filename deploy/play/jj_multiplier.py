"""JJ's 2016 three-seat multiplier game and partly revealed ready hands."""

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations

from . import jj_three
from .honors import PASS, RON, ZIMO, HonorsGame, KongPoints, Meld, ready_discards, ready_waits
from .kawuxing import _best_shape


VARIANT = 'jj-three-multiplier'
MIN_FAAN = 0
WALL_RESERVE = 0
MATCH_HANDS = None
SWAP_LOCK = False
PASS_LOCK = False
OPTIONS = replace(jj_three.OPTIONS, chow=False, kong_points=KongPoints(1, 1, 2, single_direct_payer=True))
next_dealer = jj_three.next_dealer
liability = jj_three.liability


@dataclass(frozen=True)
class Score:
    multiplier: int
    patterns: tuple[str, ...]
    fan: None = None
    tier: None = None

    @property
    def share(self):
        return self.multiplier + 2


def faan(hand, melds, ctx):
    if ctx.kind == 'rob_concealed':
        return None
    shape = _best_shape(tuple(hand), tuple(melds), ctx.tile, False, 1, all_four_copies=True, stack_hook=False)
    if shape is None:
        return None
    patterns = list(shape)
    if ctx.kind == 'rob_kong':
        patterns.append((1, '抢杠'))
    if ctx.kong_chain and ctx.kind in ('zimo', 'ron'):
        patterns.append((1, '杠上花' if ctx.kind == 'zimo' else '杠上炮'))
    if ctx.kind == 'zimo' and ctx.last_live_draw:
        patterns.append((1, '海底'))
    return Score(2 ** sum(fan for fan, _ in patterns), tuple(name for _, name in patterns) or ('平胡',))


def settle_hand(game, winners, offer):
    revealed = [p.open_ready_hand is not None for p in game.players]
    outcomes = []
    for winner in winners:
        score = winner.score
        payers = [i for i in range(game.seat_count) if i != winner.actor] if offer is None else [offer.actor]
        amounts = [0] * game.seat_count
        for payer in payers:
            factor = 2 if revealed[payer] or revealed[winner.actor] else 1
            amounts[payer] = score.multiplier * factor + 2
        if any(revealed[i] for i in [winner.actor, *payers]):
            score = replace(score, patterns=score.patterns + ('亮倒',))
        outcomes.append((score, tuple(amounts)))
    return outcomes


def _waits(hand, melds):
    return ready_waits(tuple(hand), tuple(melds), OPTIONS.tile_kinds)


@lru_cache(maxsize=8192)
def _hideable_pungs(hand, melds):
    expected = _waits(hand, melds)
    triplets = [tile for tile, count in enumerate(hand) if count == 3]
    if not expected:
        return ()
    for count in range(len(triplets), 0, -1):
        for candidate in combinations(triplets, count):
            visible = list(hand)
            for tile in candidate:
                visible[tile] -= 3
            if _waits(visible, melds + tuple(Meld('triplet', tile) for tile in candidate)) == expected:
                return candidate
    return ()


class JJMultiplierGame(HonorsGame):
    def __init__(self, seed, dealer, round_wind, *, rules):
        self.passed_forms = [set() for _ in range(OPTIONS.seat_count)]
        super().__init__(seed, dealer, round_wind, rules=rules)
        self.declarations = []
        self.dark_seats = set()
        self.hidden_pungs = [set() for _ in self.players]
        self.ready_waits = [() for _ in self.players]
        self.ready_deltas = [0] * self.seat_count

    def _draw(self, actor, kong_chain=0, provider=None):
        if self.remaining:
            self.passed_forms[actor].clear()
        super()._draw(actor, kong_chain, provider)

    def hidden_ready_tiles(self, seat):
        return 3 * len(self.hidden_pungs[seat])

    def _public_waits(self, seat):
        player = self.players[seat]
        if player.open_ready_hand is None:
            return ()
        anonymous = (Meld('triplet', -1),) * len(self.hidden_pungs[seat])
        return _waits(player.open_ready_hand, tuple(player.melds) + anonymous)

    def _safe_discards(self, actor, actions):
        blocked = {tile for seat in range(self.seat_count) if seat != actor for tile in self._public_waits(seat)}
        if any(kind == 'discard' and tile not in blocked for kind, tile in actions):
            return [move for move in actions if move[0] != 'discard' or move[1] not in blocked]
        return actions

    def _keeps_waits(self, actor, tile, *, closed):
        player = self.players[actor]
        hand = player.hand.copy()
        hand[tile] -= 4 if closed else 3
        meld = Meld('gang_an' if closed else 'gang_ming', tile)
        return _waits(hand, (*player.melds, meld)) == self.ready_waits[actor]

    def legal_actions(self):
        actor, actions = super().legal_actions()
        if self.phase != 'action':
            return actor, actions
        player = self.players[actor]
        if player.open_ready_hand is not None:
            drawn = self.drawn_tile
            kept = [move for move in actions if move in (ZIMO, ('discard', drawn))]
            if (drawn in self.hidden_pungs[actor] and ('ankan', drawn) in actions
                    and self._keeps_waits(actor, drawn, closed=True)):
                kept.append(('ankan', drawn))
            return actor, kept

        safe = self._safe_discards(actor, actions)
        if self.remaining <= 3:
            return actor, safe
        declarations = list(ready_discards(player, safe, self.options))
        candidates = []
        hand = player.hand.copy()
        for _, tile in declarations:
            hand[tile] -= 1
            if _hideable_pungs(tuple(hand), tuple(player.melds)):
                candidates.append(('declare_dark', tile))
            hand[tile] += 1
        return actor, safe + declarations + candidates

    def _response_actions(self):
        actor, actions = super()._response_actions()
        if RON in actions and self._win(actor, self.offer)[0].patterns in self.passed_forms[actor]:
            actions.remove(RON)
        if self.winners:
            return actor, [move for move in actions if move in (PASS, RON)]
        if self.players[actor].open_ready_hand is not None:
            actions = [move for move in actions if move in (PASS, RON) or (
                move[0] == 'zhigang' and self.offer.tile in self.hidden_pungs[actor]
                and self._keeps_waits(actor, self.offer.tile, closed=False)
            )]
        return actor, actions

    def step(self, action):
        kind, tile = action
        if kind not in ('declare_ready', 'declare_dark'):
            super().step(action)
            return
        actor, actions = self.legal_actions()
        if action not in actions:
            raise ValueError('Illegal reveal declaration')
        self.last_call = None
        self._discard(actor, tile)
        player = self.players[actor]
        self.ready_waits[actor] = _waits(player.hand, player.melds)
        visible = player.hand.copy()
        if kind == 'declare_dark':
            hidden = _hideable_pungs(tuple(visible), tuple(player.melds))
            self.hidden_pungs[actor].update(hidden)
            self.dark_seats.add(actor)
            for pung in hidden:
                visible[pung] -= 3
        player.open_ready_hand = tuple(visible)
        self.declarations.append(actor)

    def _respond(self, actor, action, actions):
        if RON in actions and action != RON:
            self.passed_forms[actor].add(self._win(actor, self.offer)[0].patterns)
        super()._respond(actor, action, actions)

    def _call_discard(self, offer):
        actor, kind, _ = self.claim
        super()._call_discard(offer)
        if kind == 'zhigang':
            self.hidden_pungs[actor].discard(offer.tile)

    def _finish_own_kong(self, offer):
        super()._finish_own_kong(offer)
        self.hidden_pungs[offer.actor].discard(offer.tile)

    def _kong_factor(self, meld):
        return 2 if self.kong_chain else 1

    def _end(self, reason):
        super()._end(reason)
        if reason == 'draw' and self.declarations:
            first = self.declarations[0]
            payments = tuple(int(seat == first) for seat in range(self.seat_count))
            for seat in range(self.seat_count):
                if seat not in self.declarations:
                    self._transfer(seat, payments, self.ready_deltas)
