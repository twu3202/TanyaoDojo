"""Dalian private bao and Tieling closed ready with cross-hand scoring."""

from collections import Counter
from dataclasses import replace
from random import Random

from .changchun import BAO_WIN, Egg, egg_rack_faces
from .honors import DRAGONS, ORPHANS, PASS, WINDS, TableOptions
from .liaoning_split import JianchangRules, SplitScore, three_suits
from .northeast import BaoGame, NortheastRules, wait_pattern
from .shenyang import SHENYANG, ShenyangGame, ShenyangRules
from .wildcard_shapes import without_tile


INSIDE_READY = 'declare_inside_ready'
DALIAN_PEEK = 'dalian_bao_peek_win'
BAO_SLOTS = tuple(range(0, 12, 2))


class DalianRules(NortheastRules):
    WALL_RESERVE = 14
    OPEN_FOR_STANDARD = True
    OPTIONS = TableOptions(orphans=False, concealed_kong_faces=0, pass_win_all=True)
    next_dealer = staticmethod(JianchangRules.next_dealer)

    @classmethod
    def rack_allowed(cls, hand, melds):
        faces = egg_rack_faces(hand, melds)
        return cls.ready_allowed(hand, melds) and three_suits(faces) and bool(faces & ORPHANS)

    @classmethod
    def ready_allowed(cls, hand, melds):
        return len(melds) < 4 and not cls.closed(melds)

    @staticmethod
    def shape_allowed(pair, sets):
        return any(m.kind == 'chi' for m in sets) and (pair in DRAGONS or any(m.kind != 'chi' for m in sets))

    def inside_waits(self, hand, melds, waits):
        inside = []
        for face in waits:
            filled = hand[:face] + (hand[face] + 1,) + hand[face + 1:]
            if wait_pattern(face, self.shapes(filled, melds)) in ('边张', '夹张'):
                inside.append(face)
        return tuple(inside)

    @classmethod
    def payments(cls, game, actor, provider, score):
        closed = [cls.closed(p.melds) for p in game.players]
        three_closed = all(closed[seat] for seat in range(game.seat_count) if seat != actor)
        return tuple(
            score.share * 2 ** (int(closed[seat]) + int(three_closed) + int(game.dealer in (actor, seat)))
            if seat != actor and provider in (None, seat) else 0 for seat in range(game.seat_count)
        )

    @classmethod
    def settle_hand(cls, game, winners, offer):
        provider = offer.actor if offer is not None else None
        return [(win.score, cls.payments(game, win.actor, provider, win.score)) for win in winners]


DALIAN = DalianRules('dalian-private-bao')


class DalianGame(BaoGame):
    ALLOW_READY_KONGS = True

    def __init__(self, seed, dealer, round_wind, *, rules=DALIAN):
        self.inside_ready = set()
        self.bao_rng = Random(seed)
        self.bao_slot = None
        self.bao_seen = [None] * rules.OPTIONS.seat_count
        self.bao_pending_draw = None
        self.discarded_faces = Counter()
        super().__init__(seed, dealer, round_wind, rules=rules)

    def bao_view(self, actor):
        if self.bao_slot is None or self.bao_seen[actor] != self.bao_slot:
            return ()
        face = self.wall[self.bao_slot]
        return () if self.discarded_faces[face] >= 3 else ((None, face),)

    def _draw(self, actor, kong_chain=0, provider=None):
        if self.declared_waits[actor] and self.remaining:
            if self.bao_slot is None or self.discarded_faces[self.wall[self.bao_slot]] >= 3:
                slots = [i for i in BAO_SLOTS if self.discarded_faces[self.wall[i]] < 3]
                self.bao_slot = self.bao_rng.choice(slots) if slots else None
            if self.bao_slot is not None and self.bao_seen[actor] != self.bao_slot:
                self.bao_seen[actor] = self.bao_slot
                if self.wall[self.bao_slot] in self.declared_waits[actor]:
                    self.bao_pending_draw = kong_chain, provider
                    self.cur, self.phase, self.drawn_tile = actor, 'bao_peek', None
                    return
        super()._draw(actor, kong_chain, provider)

    def _inside_choices(self, actor):
        player = self.players[actor]
        hand, melds = tuple(player.hand), tuple(player.melds)
        return tuple((face, inside) for face, waits in super()._ready_choices(actor)
                     if (inside := self.rules.inside_waits(without_tile(hand, face), melds, waits)))

    def _ready_choices(self, actor):
        return self._inside_choices(actor) if actor in self.inside_ready else super()._ready_choices(actor)

    def _locked_waits(self, actor, hand, melds):
        waits = super()._locked_waits(actor, hand, melds)
        return self.rules.inside_waits(hand, melds, waits) if actor in self.inside_ready else waits

    def legal_actions(self):
        if self.phase == 'bao_peek':
            return self.cur, [PASS, BAO_WIN]
        actor, actions = super().legal_actions()
        if self.phase != 'action' or self.offer is not None:
            return actor, actions
        melds = self.players[actor].melds
        if self.rules.closed(melds) or len(melds) == 3:
            actions = [a for a in actions if a[0] != 'ankan']
        if self.remaining and not self.declared_waits[actor] and actor not in self.inside_ready:
            actions += [(INSIDE_READY, face) for face, _ in self._inside_choices(actor)]
        return actor, actions

    def _response_actions(self):
        actor, actions = super()._response_actions()
        if len(self.players[actor].melds) == 3:
            actions = [a for a in actions if a[0] not in ('chi', 'peng', 'zhigang')]
        return actor, actions

    def step(self, action):
        if self.phase == 'bao_peek':
            if action not in (PASS, BAO_WIN):
                raise ValueError('Illegal Dalian bao action')
            pending, self.bao_pending_draw = self.bao_pending_draw, None
            self.last_call = None
            if action == PASS:
                super()._draw(self.cur, *pending)
            else:
                actor = self.cur
                player, tile = self.players[actor], self.wall[self.bao_slot]
                hand = player.hand.copy()
                hand[tile] += 1
                score = self._natural_score(actor, tuple(hand), tuple(player.melds), 'zimo', '冲宝')
                self._settle_special(actor, None, DALIAN_PEEK, score, hand)
            return
        if action[0] == INSIDE_READY:
            actor, actions = self.legal_actions()
            if action not in actions:
                raise ValueError('Illegal Dalian inside ready action')
            self.inside_ready.add(actor)
            action = 'declare_ready', action[1]
        super().step(action)

    def _discard(self, actor, tile):
        self.discarded_faces[tile] += 1
        super()._discard(actor, tile)

    def _natural_score(self, actor, hand, melds, kind, bao=None):
        if not self.rules.shapes(hand, melds):
            return None
        labels = ['自摸' if kind == 'zimo' else '点炮']
        if actor in self.inside_ready:
            labels.append('夹胡')
        if bao:
            labels.append(bao)
        return SplitScore(len(labels) + (1 if bao else 0), tuple(labels))

    def _score(self, actor, hand, melds, ctx):
        if ctx.kind == 'rob_concealed':
            return None
        waits = self.declared_waits[actor]
        hand = tuple(hand)
        if waits and ctx.kind == 'zimo' and self.bao_view(actor) == ((None, ctx.tile),):
            for target in waits:
                filled = list(hand)
                filled[ctx.tile] -= 1
                filled[target] += 1
                score = self._natural_score(actor, tuple(filled), melds, ctx.kind, '摸宝')
                if score:
                    return score
        if waits and ctx.tile not in waits:
            return None
        return self._natural_score(actor, hand, melds, ctx.kind)


class TielingRules(ShenyangRules):
    OPEN_FOR_STANDARD = False
    OPTIONS = replace(SHENYANG.OPTIONS, concealed_kong_faces=4, closed_live_tail=4)
    next_dealer = staticmethod(JianchangRules.next_dealer)

    @staticmethod
    def rack_allowed(hand, melds):
        faces = egg_rack_faces(hand, melds)
        return three_suits(faces) and bool(faces & ORPHANS)

    @classmethod
    def payments(cls, game, actor, provider, score):
        closed = [cls.closed(p.melds) for p in game.players]
        three_closed = all(closed[seat] for seat in range(game.seat_count) if seat != actor)
        return tuple(
            score.share * 2 ** (int(closed[seat]) + int(three_closed) + int(game.dealer in (actor, seat))
                                + int(seat == provider))
            if seat != actor else 0 for seat in range(game.seat_count)
        )


TIELING = TielingRules('tieling-closed-ready')


class TielingGame(ShenyangGame):
    def __init__(self, seed, dealer, round_wind, *, rules=TIELING, previous_draw=False):
        self.long_hair = previous_draw
        super().__init__(seed, dealer, round_wind, rules=rules)

    def _ready_choices(self, actor):
        return super()._ready_choices(actor) if self.rules.closed(self.players[actor].melds) else ()

    def _discard(self, actor, tile):
        if self.uninterrupted and all(river == ([] if seat == actor else [tile])
                                      for seat, river in enumerate(self.discards)):
            self.long_hair = True
        super()._discard(actor, tile)

    def legal_actions(self):
        actor, actions = super().legal_actions()
        blocked = {'extend_egg'}
        if self.phase == 'action' and not self.declared_waits[actor] and self.rules.closed(self.players[actor].melds):
            blocked.add('ankan')
        return actor, [a for a in actions if a[0] not in blocked]

    def _score(self, actor, hand, melds, ctx):
        closed = self.rules.closed(melds)
        waits = self.declared_waits[actor]
        if ctx.kind == 'rob_concealed' or (closed and not waits) or (waits and ctx.tile not in waits):
            return None
        candidates = []
        for pair, sets in self.rules.shapes(tuple(hand), melds):
            pungs = all(m.kind != 'chi' for m in (*sets, *melds))
            if pungs:
                single = pair == ctx.tile
                candidates.append((3 + int(single), '单吊飘胡' if single else '对倒飘胡', sets))
            elif len(melds) < 4:
                eye = wait_pattern(ctx.tile, ((pair, sets),)) is not None
                candidates.append((int(eye), '夹胡' if eye else '屁胡', sets))
        if not candidates:
            return None

        common = [('自摸', int(ctx.kind == 'zimo')), ('闷听', int(closed)), ('长毛', int(self.long_hair)),
                  ('海底捞月', int(ctx.kind == 'zimo' and self.split_draw)),
                  ('杠上开花', int(ctx.kind == 'zimo' and ctx.kong_chain > 0)),
                  ('流泪', int(ctx.kind == 'ron' and ctx.kong_chain > 0))]
        other_eggs = {m.kind for seat, player in enumerate(self.players) if seat != actor
                      for m in player.melds if isinstance(m, Egg)}
        faces = Counter({t: n for t, n in enumerate(hand) if n})
        faces.update(t for m in melds if m.kind in ('chi', 'peng') for t in m.tiles)
        common.extend(('四归一', 1) for n in faces.values() if n == 4)
        for m in melds:
            if isinstance(m, Egg):
                common.append(('旋风杠', 2))
            elif m.kind.startswith('gang'):
                conceal = m.kind == 'gang_an'
                common.append(('暗杠' if conceal else '明杠', 1 + int(conceal) + int(m.tile in DRAGONS)))
        if len(melds) == 4 and ctx.tile < 27 and all(m.tile // 9 != ctx.tile // 9 for m in melds):
            common.append(('定向飘', 1))

        outcomes = []
        for fan, label, sets in candidates:
            shapes = [(label, fan), *common]
            for m in (*sets, *melds):
                if m.kind not in ('peng', 'triplet'):
                    continue
                if m.tile in DRAGONS:
                    shapes.append(('箭刻', 1))
                    if 'egg_dragons' in other_eggs:
                        shapes.append(('喜杠加番', 2))
                elif m.tile in WINDS and 'egg_winds' in other_eggs:
                    shapes.append(('旋风碰番', 1))
            outcomes.append(SplitScore(sum(n for _, n in shapes), (label, *(name for name, n in shapes[1:] if n))))
        return max(outcomes, key=lambda s: s.fan)


DALIAN_TIELING_ROOMS = (DALIAN, TIELING)
