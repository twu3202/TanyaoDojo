"""Declared waits and the four private indicators used by Harbin Mahjong."""

from dataclasses import dataclass
from functools import lru_cache

from .honors import (
    PASS, RON, TERMINALS, HonorsGame, KongPoints, Meld, TableOptions, standard_shapes,
)
from .honors_bonus import Score
from .wildcard_shapes import without_tile


RED = 33
READY_CALLS = {'ready_chi': 'chi', 'ready_pon': 'peng'}


def wait_pattern(tile, shapes):
    if any(any(m.kind == 'chi' and ((m.tile % 9 == 0 and tile == m.tile + 2)
                                    or (m.tile % 9 == 6 and tile == m.tile)) for m in sets)
           for _, sets in shapes):
        return '边张'
    if any(any(m.kind == 'chi' and tile == m.tile + 1 for m in sets) for _, sets in shapes):
        return '夹张'
    return '单钓' if any(tile == pair for pair, _ in shapes) else None


@dataclass(frozen=True)
class NortheastRules:
    VARIANT: str
    MATCH_HANDS = 8
    MIN_FAAN = 0
    PASS_LOCK = False
    SWAP_LOCK = True
    OPEN_FOR_STANDARD = False

    @staticmethod
    def next_dealer(dealer, winners, provider):
        return winners[0] if winners else dealer

    @staticmethod
    def liability(melds, last):
        return ()

    @staticmethod
    def closed(melds):
        return all(m.kind == 'gang_an' for m in melds)

    @lru_cache(maxsize=8192)
    def shapes(self, hand, melds):
        if not self.rack_allowed(hand, melds):
            return ()
        return tuple((pair, sets) for pair, sets in standard_shapes(hand, len(melds))
                     if self.shape_allowed(pair, (*melds, *sets)))

    @lru_cache(maxsize=8192)
    def waits(self, hand, melds):
        if not self.ready_allowed(hand, melds):
            return ()
        used = [0] * 34
        for meld in melds:
            for tile in meld.tiles:
                used[tile] += 1
        return tuple(tile for tile in self.OPTIONS.tile_kinds if hand[tile] + used[tile] < 4
                     and self.natural_win(hand[:tile] + (hand[tile] + 1,) + hand[tile + 1:], melds, tile))

    def ready_allowed(self, hand, melds):
        return True

    def natural_win(self, hand, melds, tile):
        return bool(self.shapes(hand, melds))

    def faan(self, hand, melds, ctx):
        # The heuristic can estimate a naturally winning wait from its own tiles.
        return Score(1, ()) if self.natural_win(tuple(hand), tuple(melds), ctx.tile) else None


class HarbinRules(NortheastRules):
    OPTIONS = TableOptions(tile_kinds=(*range(27), RED), seven_pairs=False, orphans=False,
                           concealed_kong_faces=0, pass_win_all=True, kong_requires_live_tile=True,
                           kong_points=KongPoints(1, 1, 2, single_direct_payer=True))
    WALL_RESERVE = 4
    OPEN_FOR_STANDARD = True

    def rack_allowed(self, hand, melds):
        if len(melds) >= 4 or self.closed(melds):
            return False
        faces = {tile for tile, count in enumerate(hand) if count} | {tile for meld in melds for tile in meld.tiles}
        return len({tile // 9 for tile in faces if tile < 27}) >= 2 and bool(faces & (TERMINALS | {RED}))

    ready_allowed = rack_allowed

    @staticmethod
    def shape_allowed(pair, sets):
        return any(m.kind == 'chi' for m in sets) and (pair == RED or any(m.kind != 'chi' for m in sets))

    @staticmethod
    def settle_hand(game, winners, offer):
        return [(win.score, tuple(win.score.share if seat != win.actor and (offer is None or seat == offer.actor)
                                  else 0 for seat in range(game.seat_count))) for win in winners]


HARBIN = HarbinRules('harbin-four-bao')


class BaoGame(HonorsGame):
    ALLOW_READY_KONGS = False

    def __init__(self, seed, dealer, round_wind, *, rules):
        self.declared_waits = [() for _ in range(rules.OPTIONS.seat_count)]
        self.ready_river = [None for _ in self.declared_waits]
        super().__init__(seed, dealer, round_wind, rules=rules)

    @staticmethod
    def _ready_discards(rules, hand, melds, forbidden):
        return tuple((tile, waits) for tile, count in enumerate(hand) if count and tile not in forbidden
                     and (waits := rules.waits(without_tile(hand, tile), melds)))

    def _ready_choices(self, actor):
        p = self.players[actor]
        return self._ready_discards(self.rules, tuple(p.hand), tuple(p.melds), self.forbidden_discards)

    def _keeps_waits(self, actor, action):
        kind, tile = action
        if tile != self.drawn_tile or kind not in ('ankan', 'bugang'):
            return False
        p = self.players[actor]
        hand = tuple(p.hand)
        if kind == 'ankan':
            rest = without_tile(hand, tile, 4)
            melds = (*p.melds, Meld('gang_an', tile))
        else:
            rest = without_tile(hand, tile)
            melds = tuple(Meld('gang_bu', tile, m.provider) if m.kind == 'peng' and m.tile == tile else m for m in p.melds)
        return self.rules.waits(rest, melds) == self.declared_waits[actor]

    def legal_actions(self):
        actor, actions = super().legal_actions()
        if self.phase != 'action' or self.offer is not None:
            return actor, actions
        if self.declared_waits[actor]:
            return actor, [a for a in actions if a[0] == 'zimo' or a == ('discard', self.drawn_tile)
                           or self.ALLOW_READY_KONGS and self._keeps_waits(actor, a)]
        if self.remaining:
            actions += [('declare_ready', tile) for tile, _ in self._ready_choices(actor)]
        return actor, actions

    def step(self, action):
        if action[0] != 'declare_ready':
            super().step(action)
            return
        actor, actions = self.legal_actions()
        if action not in actions:
            raise ValueError('Illegal ready discard')
        tile = action[1]
        self.declared_waits[actor] = dict(self._ready_choices(actor))[tile]
        self.ready_river[actor] = (len(self.discards[actor]), tile)
        self._declared(actor)
        self.last_call = None
        self._discard(actor, tile)

    def _declared(self, actor):
        pass

    def bao_view(self, actor):
        return ()

    def _call_discard(self, offer):
        if self.ready_river[offer.actor] == (len(self.discards[offer.actor]) - 1, offer.tile):
            self.ready_river[offer.actor] = None
        super()._call_discard(offer)

    def _response_actions(self):
        actor = self.responses[0]
        if self.winners:
            return actor, [PASS]
        actor, actions = super()._response_actions()
        if self.declared_waits[actor]:
            actions = [a for a in actions if a in (PASS, RON)]
        return actor, actions

    def _replacement_index(self):
        return self.rules.WALL_RESERVE

    def _waiting_kind(self, actor):
        waits = self.declared_waits[actor]
        if len(waits) != 1:
            return None
        p = self.players[actor]
        tile = waits[0]
        waiting = tuple(p.hand)
        if self.offer is None and self.drawn_tile is not None:
            waiting = without_tile(waiting, self.drawn_tile)
        augmented = waiting[:tile] + (waiting[tile] + 1,) + waiting[tile + 1:]
        return wait_pattern(tile, self.rules.shapes(augmented, tuple(p.melds)))


class HarbinGame(BaoGame):
    def __init__(self, seed, dealer, round_wind, *, rules=HARBIN):
        self.bao_owners = []
        self.ready_claim = None
        self.must_declare = None
        super().__init__(seed, dealer, round_wind, rules=rules)

    def _declared(self, actor):
        self.bao_owners.append(actor)
        self.must_declare = None

    def bao_view(self, actor):
        if not self.declared_waits[actor]:
            return ()
        return tuple(zip(self.bao_owners, self.wall, strict=False))

    def _response_actions(self):
        actor, actions = super()._response_actions()
        if self.winners or self.declared_waits[actor] or self.offer.kind != 'discard' or not self.remaining:
            return actor, actions
        p = self.players[actor]
        tile = self.offer.tile
        source = self.offer.actor
        hand = tuple(p.hand)
        for start, consumed, blocked in self._chows(p, tile, source):
            rest = without_tile(without_tile(hand, consumed[0]), consumed[1])
            if self._ready_discards(self.rules, rest, (*p.melds, Meld('chi', start, source)), blocked):
                actions.append(('ready_chi', start))
        if p.hand[tile] >= 2:
            rest = without_tile(hand, tile, 2)
            if self._ready_discards(self.rules, rest, (*p.melds, Meld('peng', tile, source)), frozenset()):
                actions.append(('ready_pon', None))
        return actor, actions

    def _respond(self, actor, action, actions):
        kind, argument = action
        if kind in READY_CALLS:
            if self.ready_claim is None:
                self.ready_claim = actor, READY_CALLS[kind], argument
            action = PASS
        super()._respond(actor, action, actions)

    def _finish_responses(self, offer):
        if self.ready_claim is not None and not self.winners:
            self.claim = self.ready_claim
            self.must_declare = self.claim[0]
        self.ready_claim = None
        super()._finish_responses(offer)

    def legal_actions(self):
        actor, actions = super().legal_actions()
        return (actor, [a for a in actions if a[0] == 'declare_ready']) if self.must_declare is not None else (actor, actions)

    def _score(self, actor, hand, melds, ctx):
        waits = self.declared_waits[actor]
        if not waits or ctx.kind == 'rob_concealed':
            return None
        middle = self._waiting_kind(actor) == '夹张'
        if ctx.kind == 'zimo':
            matching = [owner for owner, tile in self.bao_view(actor) if tile == ctx.tile]
            if matching:
                own = actor in matching
                exact = middle and ctx.tile == waits[0]
                pattern = ('大宝中宝' if own else '小宝中宝') if exact else ('摸大宝' if own else '摸小宝')
                points = (4 if own else 2) * (2 if middle else 1) * (2 if exact else 1)
                return Score(points, (pattern, '卡当') if middle else (pattern,))
        if ctx.tile not in waits or not self.rules.shapes(tuple(hand), melds):
            return None
        pattern = {'zimo': '自摸', 'ron': '平胡', 'rob_kong': '抢杠胡'}[ctx.kind]
        points = (1 if ctx.kind == 'ron' else 2) * (2 if middle else 1)
        return Score(points, (pattern, '卡当') if middle else (pattern,))
