"""Changchun's opening egg kongs and the shared, privately seen bao."""

from collections import Counter
from dataclasses import dataclass, replace
from random import Random

from .honors import DRAGONS, ORPHANS, PASS, KongPoints, Meld, Offer, TableOptions
from .northeast import BaoGame, NortheastRules, wait_pattern
from .wildcard_shapes import without_tile


CHICKEN = 18
EGG_GROUPS = {
    'egg_winds': (27, 28, 29, 30), 'egg_dragons': (31, 32, 33),
    'egg_ones': (0, 9, 18), 'egg_nines': (8, 17, 26),
}
BIG_KONGS = DRAGONS | {CHICKEN, 9}
BAO_WIN = ('win_bao', None)


@dataclass(frozen=True)
class Egg(Meld):
    material: tuple[int, ...] = ()
    substitutions: int = 0

    @property
    def tiles(self):
        return self.material


def egg_rack_faces(hand, melds):
    faces = {tile for tile, count in enumerate(hand) if count}
    for meld in melds:
        tiles = list(meld.tiles)
        if isinstance(meld, Egg):
            for _ in range(meld.substitutions):
                tiles.remove(CHICKEN)
        faces.update(tiles)
    return faces


@dataclass(frozen=True)
class ChangchunScore:
    fan: int
    patterns: tuple[str, ...]
    tier = None

    @property
    def share(self):
        return 2 ** (self.fan - 1)


class ChangchunRules(NortheastRules):
    OPTIONS = TableOptions(seven_pairs=False, orphans=False, concealed_kong_faces=0, pass_win_all=True,
                           kong_requires_live_tile=True, closed_live_tail=4, kong_points=KongPoints(1, 1, 2))
    WALL_RESERVE = 14

    @staticmethod
    def rack_allowed(hand, melds):
        if len(melds) == 4 and any(m.kind == 'chi' for m in melds):
            return False
        faces = egg_rack_faces(hand, melds)
        return len({tile // 9 for tile in faces if tile < 27}) == 3 and bool(faces & ORPHANS)

    @staticmethod
    def shape_allowed(pair, sets):
        return pair in DRAGONS or any(m.kind != 'chi' for m in sets)

    @staticmethod
    def payments(game, actor, provider, score):
        return tuple(score.share * (2 if game.dealer in (actor, seat) else 1) * (2 if seat == provider else 1)
                     if seat != actor else 0 for seat in range(game.seat_count))

    @classmethod
    def settle_hand(cls, game, winners, offer):
        return [(win.score, cls.payments(game, win.actor, None if offer is None else offer.actor, win.score))
                for win in winners]


CHANGCHUN = ChangchunRules('changchun-shared-bao')


class ChangchunGame(BaoGame):
    BAO_SLOT_COUNT = 12

    def __init__(self, seed, dealer, round_wind, *, rules=CHANGCHUN):
        self.kong_count = 0
        self.egg_extension = None
        self.egg_thief = None
        self.egg_racks = [None] * rules.OPTIONS.seat_count
        self.bao_rng = Random(seed)
        self.bao_slot = None
        self.bao_history = set()
        self.bao_seen = [None] * rules.OPTIONS.seat_count
        self.bao_pending_draw = None
        super().__init__(seed, dealer, round_wind, rules=rules)

    def _deal_opening(self):
        super()._deal_opening()
        self.egg_racks = [p.hand.copy() for p in self.players]

    @property
    def remaining(self):
        return len(self.wall) - self._replacement_index()

    def _replacement_index(self):
        return self.rules.WALL_RESERVE + 2 * max(0, self.kong_count - 1)

    def bao_view(self, actor):
        if self.bao_slot is None or self.bao_seen[actor] != self.bao_slot:
            return ()
        return ((None, self.wall[self.bao_slot]),)

    def _refresh_bao(self):
        public = Counter(tile for river in self.discards for tile in river)
        public.update(tile for p in self.players for m in p.melds if m.kind != 'gang_an' for tile in m.tiles)
        if self.bao_slot is not None and public[self.wall[self.bao_slot]] < 3:
            return
        choices = [index for index, tile in enumerate(self.wall[:self.BAO_SLOT_COUNT])
                   if index not in self.bao_history and public[tile] < 3]
        self.bao_slot = self.bao_rng.choice(choices) if choices else None
        if self.bao_slot is not None:
            self.bao_history.add(self.bao_slot)

    def _can_open_bao(self, actor):
        return True

    def bao_match_reason(self):
        return 'bao_peek_win'

    def _draw(self, actor, kong_chain=0, provider=None):
        if self.declared_waits[actor] and self.remaining and self._can_open_bao(actor):
            self._refresh_bao()
            if self.bao_slot is not None and self.bao_seen[actor] != self.bao_slot:
                self.bao_seen[actor] = self.bao_slot
                if self.wall[self.bao_slot] in self.declared_waits[actor]:
                    self.bao_pending_draw = kong_chain, provider
                    self.phase = 'bao_peek'
                    self.cur = actor
                    self.drawn_tile = None
                    return
        super()._draw(actor, kong_chain, provider)

    @staticmethod
    def egg_tiles(kind, missing):
        return tuple(CHICKEN if tile == missing else tile for tile in EGG_GROUPS[kind])

    def _opening_egg(self, kind, arg):
        tiles = self.egg_tiles(kind, arg)
        return Egg(kind, tiles[0], material=tiles, substitutions=int(arg is not None))

    def _new_eggs(self, actor):
        p = self.players[actor]
        opening = self.egg_racks[actor]
        if opening is None or len(p.melds) == 4:
            return []
        available = tuple(min(a, b) for a, b in zip(opening, p.hand))
        actions = []
        for kind, group in EGG_GROUPS.items():
            missing = tuple(tile for tile in group if not available[tile])
            if not missing:
                actions.append((kind, None))
            elif len(missing) == 1 and missing[0] != CHICKEN and available[CHICKEN] > (CHICKEN in group):
                actions.append((kind, missing[0]))
        return actions

    @staticmethod
    def _extended(meld, tile):
        return replace(meld, material=(*meld.tiles, tile),
                       substitutions=meld.substitutions + (tile == CHICKEN and tile not in EGG_GROUPS[meld.kind]))

    def _preserves_waits(self, actor, index, tile, extended):
        waits = self.declared_waits[actor]
        if not waits:
            return True
        if tile != self.drawn_tile:
            return False
        p = self.players[actor]
        melds = tuple(p.melds[:index] + [extended] + p.melds[index + 1:])
        return self.rules.waits(without_tile(tuple(p.hand), tile), melds) == waits

    def legal_actions(self):
        if self.phase == 'bao_peek':
            return self.cur, [PASS, BAO_WIN]
        actor, actions = super().legal_actions()
        if self.phase != 'action' or self.offer is not None or self.drawn_tile is None or self.remaining <= 4:
            return actor, actions
        if not self.declared_waits[actor]:
            actions += self._new_eggs(actor)
        if self.egg_racks[self.dealer] is not None and actor == self.dealer:
            return actor, [a for a in actions if a[0] != 'bugang']
        p = self.players[actor]
        for i, meld in enumerate(p.melds):
            if isinstance(meld, Egg):
                for tile in sorted({CHICKEN, *EGG_GROUPS[meld.kind]}):
                    if p.hand[tile] and self._preserves_waits(actor, i, tile, self._extended(meld, tile)):
                        actions.append(('extend_egg', (i, tile)))
            elif self.declared_waits[actor] and meld.kind == 'peng' and p.hand[meld.tile]:
                if self._preserves_waits(actor, i, meld.tile, Meld('gang_bu', meld.tile, meld.provider)):
                    actions.append(('bugang', meld.tile))
        return actor, actions

    def step(self, action):
        kind, arg = action
        if self.phase != 'bao_peek' and kind not in EGG_GROUPS and kind != 'extend_egg':
            super().step(action)
            return
        actor, actions = self.legal_actions()
        if action not in actions:
            raise ValueError('Illegal Changchun action')
        self.last_call = None
        if self.phase == 'bao_peek':
            pending, self.bao_pending_draw = self.bao_pending_draw, None
            if action == PASS:
                super()._draw(actor, *pending)
            else:
                tile = self.wall[self.bao_slot]
                p = self.players[actor]
                hand = p.hand.copy()
                hand[tile] += 1
                score = self._natural_score(hand, tuple(p.melds), tile, 'bao_peek', '对宝')
                self._settle_special(actor, None, self.bao_match_reason(), score, hand)
            return
        if kind == 'extend_egg':
            self.egg_extension, tile = arg
            self.uninterrupted = False
            self._open_offer(Offer(actor, tile, 'gang_bu', self.kong_chain))
            return
        meld = self._opening_egg(kind, arg)
        tiles = meld.tiles
        p = self.players[actor]
        for tile in tiles:
            p.hand[tile] -= 1
            self.egg_racks[actor][tile] -= 1
        p.melds.append(meld)
        self.uninterrupted = False
        self.last_call = actor, meld
        self._pay_kong(actor, meld)
        if len(tiles) == 4:
            self._draw(actor, kong_chain=self.kong_chain + 1)

    def _discard(self, actor, tile):
        self.egg_racks[actor] = None
        super()._discard(actor, tile)

    def _response_actions(self):
        actor, actions = super()._response_actions()
        if self.egg_extension is not None and not self.winners and not self.declared_waits[actor]:
            if self.players[actor].hand[self.offer.tile] >= 2:
                actions.append(('rob_egg_pon', None))
        return actor, actions

    def _respond(self, actor, action, actions):
        if action == ('rob_egg_pon', None):
            if self.egg_thief is None:
                self.egg_thief = actor
            action = PASS
        super()._respond(actor, action, actions)

    def _finish_responses(self, offer):
        if self.egg_extension is None:
            super()._finish_responses(offer)
            return
        index, self.egg_extension = self.egg_extension, None
        thief, self.egg_thief = self.egg_thief, None
        if self.winners:
            self._settle(self.winners, offer)
            return
        p = self.players[offer.actor]
        p.hand[offer.tile] -= 1
        if thief is not None:
            self.discards[offer.actor].append(offer.tile)
            self.claim = thief, 'peng', None
            self._call_discard(Offer(offer.actor, offer.tile, 'discard'))
            return
        meld = self._extended(p.melds[index], offer.tile)
        p.melds[index] = meld
        self.last_call = offer.actor, meld
        self._pay_kong(offer.actor, meld)
        self._draw(offer.actor, kong_chain=offer.kong_chain + 1)

    def _pay_kong(self, actor, meld):
        self.kong_count += 1
        if isinstance(meld, Egg):
            self._transfer(actor, tuple(int(seat != actor) for seat in range(self.seat_count)), self.kong_deltas)
        else:
            super()._pay_kong(actor, meld)

    def _kong_factor(self, meld):
        return 2 if meld.tile in BIG_KONGS else 1

    def _natural_score(self, hand, melds, tile, kind, bonus=None):
        hand = tuple(hand)
        shapes = self.rules.shapes(hand, melds)
        if not shapes:
            return None
        patterns = [('抢杠胡' if kind == 'rob_kong' else '平胡', 1)]
        if kind == 'zimo':
            patterns.append(('自摸', 1))
        if self.rules.closed(melds):
            patterns.append(('门清', 1))
        if hand[tile] and self.rules.waits(without_tile(hand, tile), melds) == (tile,):
            if label := wait_pattern(tile, shapes):
                patterns.append((label, 1))
        if any(all(m.kind != 'chi' for m in (*melds, *sets)) for _, sets in shapes):
            patterns.append(('飘胡', 2))
        if bonus:
            patterns.append((bonus, 2 if bonus == '对宝' else 1))
        fan = sum(value for _, value in patterns)
        return ChangchunScore(fan, tuple(label for label, _ in patterns))

    def _is_bao_draw(self, actor, tile):
        return self.bao_view(actor) == ((None, tile),)

    def _score(self, actor, hand, melds, ctx):
        if ctx.kind == 'rob_concealed':
            return None
        if ctx.kind == 'zimo' and self._is_bao_draw(actor, ctx.tile):
            waiting = without_tile(tuple(hand), ctx.tile)
            scores = []
            for tile in self.declared_waits[actor]:
                filled = waiting[:tile] + (waiting[tile] + 1,) + waiting[tile + 1:]
                score = self._natural_score(filled, melds, tile, ctx.kind, '摸宝')
                if score is not None:
                    scores.append(score)
            if scores:
                return max(scores, key=lambda score: score.share)
        return self._natural_score(hand, melds, ctx.tile, ctx.kind)
