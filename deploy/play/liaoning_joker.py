"""Panjin and Anshan tables using an opening indicator and a protected joker."""

from collections import Counter

from .hangzhou import split_wild
from .honors import DRAGONS, ORPHANS, TERMINALS, KongPoints, TableOptions
from .liaoning_split import JianchangRules, SplitDrawGame, SplitScore, rack_faces, three_suits
from .northeast import NortheastRules
from .wildcard_shapes import can_form_pairs, can_form_standard, wildcard_decompositions


CHICKEN = 18
DRAGON_ROTATION = (33, 32, 31)


def next_face(indicator):
    if indicator < 27:
        return indicator // 9 * 9 + (indicator + 1) % 9
    if indicator < 31:
        return 27 + (indicator - 26) % 4
    return DRAGON_ROTATION[(DRAGON_ROTATION.index(indicator) + 1) % 3]


def open_gun(wildcard):
    return wildcard // 9 * 9 + 6 if wildcard < 27 else None


class LiaoningJokerRules(NortheastRules):
    require_burst = False
    STOP_AFTER_JOKER_DISCARD = False
    JOKER_KONG = False
    next_dealer = staticmethod(JianchangRules.next_dealer)

    @staticmethod
    def layouts(natural, wild, melds, assignable):
        if not can_form_standard(natural, wild):
            return ()
        return wildcard_decompositions(natural, wild, len(melds), assignable)


class PanjinRules(LiaoningJokerRules):
    WALL_RESERVE = 8
    JOKER_KONG = True
    OPTIONS = TableOptions(orphans=False, seven_pairs=True, closed_live_tail=4,
                           concealed_kong_faces=0, kong_points=KongPoints(1, 1, 2))
    ASSIGNABLE = tuple(t for t in OPTIONS.tile_kinds if t not in TERMINALS)

    def faan(self, hand, melds, ctx, *, wildcard):
        if ctx.kind != 'zimo' and (ctx.kind != 'ron' or ctx.tile == wildcard):
            return None
        natural, wild = split_wild(hand, wildcard)
        faces = rack_faces(natural, melds) - {wildcard}
        if not three_suits(faces) or not faces & ORPHANS:
            return None

        candidates = []
        if not melds and sum(hand) == 14 and can_form_pairs(natural, wild):
            if all(natural[tile] % 2 == 0 for tile in TERMINALS):
                candidates.append((1, ('七小对',)))
        for _, assignments, shapes in self.layouts(natural, wild, melds, self.ASSIGNABLE):
            winning_faces = assignments if ctx.tile == wildcard else (ctx.tile,)
            for pair, sets in shapes:
                pungs = Counter(m.tile for m in sets if m.kind == 'triplet')
                if any(natural[face] < 3 * count for face, count in pungs.items()):
                    continue
                full = (*melds, *sets)
                if not (pair >= 27 or any(m.kind != 'chi' for m in full)):
                    continue
                all_pungs = all(m.kind != 'chi' for m in full)
                if len(melds) == 4 and not all_pungs:
                    continue
                middle = any(m.kind == 'chi' and m.tile + 1 in winning_faces
                             and (ctx.tile == wildcard or natural[ctx.tile] > 3 * pungs[ctx.tile])
                             for m in sets)
                labels = ('飘胡' if all_pungs else '平胡',)
                candidates.append((int(all_pungs) + int(middle), (*labels, '夹胡') if middle else labels))
        if not candidates:
            return None
        fan, labels = max(candidates)
        # Printed chicken faces count even if the opening flip makes them jokers.
        chickens = hand[CHICKEN] + sum(m.tiles.count(CHICKEN) for m in melds)
        bonuses = (('穷胡', not wild),
                   ('鸡胡', 0 < chickens < 4), ('群鸡', chickens == 4),
                   ('杠开', ctx.kind == 'zimo' and ctx.kong_chain > 0),
                   ('杠流泪', ctx.kind == 'ron' and ctx.kong_chain > 0))
        active = tuple(label for label, applies in bonuses if applies)
        mode = '自摸' if ctx.kind == 'zimo' else '点炮'
        return SplitScore(fan + 1 + len(active) + int(chickens == 4), (*labels, mode, *active))

    @staticmethod
    def settle_hand(game, winners, offer):
        win, = winners
        bills = tuple(win.score.share * (2 if game.dealer in (seat, win.actor) else 1)
                      if seat != win.actor else 0 for seat in range(game.seat_count))
        if offer is not None:
            bills = tuple(sum(bills) if seat == offer.actor else 0 for seat in range(game.seat_count))
        return [(win.score, bills)]


class AnshanRules(LiaoningJokerRules):
    WALL_RESERVE = 14
    STOP_AFTER_JOKER_DISCARD = True
    OPTIONS = TableOptions(seven_pairs=True, concealed_kong_faces=0)

    def faan(self, hand, melds, ctx, *, wildcard):
        if ctx.kind == 'rob_concealed' or (ctx.kind != 'zimo' and ctx.tile == wildcard):
            return None
        natural, wild = split_wild(hand, wildcard)
        faces = rack_faces(natural, melds)
        if not faces & ORPHANS:
            return None
        candidates = []
        if not melds and sum(hand) == 14:
            if can_form_pairs(natural, wild):
                candidates.append((1, ('七小对',)))
            if three_suits(faces) and faces <= ORPHANS and len(ORPHANS - faces) <= wild:
                candidates.append((1, ('十三幺',)))
        if three_suits(faces):
            for _, _, shapes in self.layouts(natural, wild, melds, self.OPTIONS.tile_kinds):
                for pair, sets in shapes:
                    pungs = tuple(m for m in (*sets, *melds) if m.kind != 'chi')
                    if not pungs and pair not in DRAGONS:
                        continue
                    bonuses = (('二八将', pair < 27 and pair % 9 in (1, 7)),
                               ('大哥大', any(m.tile in DRAGONS for m in pungs)),
                               ('三叉', len(pungs) >= 3))
                    active = tuple(label for label, applies in bonuses if applies)
                    candidates.append((len(active), ('平胡', *active)))
        if not candidates:
            return None
        fan, labels = max(candidates)
        counts = Counter({t: n for t, n in enumerate(hand) if n})
        counts.update(tile for m in melds if m.kind in ('chi', 'peng') for tile in m.tiles)
        fours = sum(n == 4 for n in counts.values())
        gun = open_gun(wildcard)
        guns = counts[gun] + sum(m.tiles.count(gun) for m in melds if m.kind.startswith('gang'))
        kongs = tuple('暗杠' if m.kind == 'gang_an' else '明杠' for m in melds if m.kind.startswith('gang'))
        bonuses = (('站立', self.closed(melds)), ('穷胡', not wild), ('杠开', ctx.kind == 'zimo' and ctx.kong_chain > 0),
                   ('上墙头', ctx.kind == 'zimo' and ctx.last_live_draw))
        active = tuple(label for label, applies in bonuses if applies)
        return SplitScore(fan + len(active) + fours + guns + len(kongs),
                          (*labels, *active, *kongs, *('四归一',) * fours, *('明枪',) * guns))

    @staticmethod
    def settle_hand(game, winners, offer):
        win, = winners
        return [(win.score, tuple(
            win.score.share * (2 if game.dealer in (seat, win.actor) else 1)
            * (2 if offer is None or seat == offer.actor else 1) if seat != win.actor else 0
            for seat in range(game.seat_count)
        ))]


class LiaoningJokerGame(SplitDrawGame):
    def _prepare_wall(self):
        self.indicator_tile = self.wildcard_indicator = self.wall.pop(0)
        self.wildcard_tile = next_face(self.indicator_tile)
        self.stopped = set()

    def physical_counts(self):
        counts = super().physical_counts()
        counts[self.indicator_tile] += 1
        return counts

    def legal_actions(self):
        actor, actions = super().legal_actions()
        if self.phase != 'action' or self.offer is not None or self.split_draw:
            return actor, actions
        player = self.players[actor]
        has_natural = sum(player.hand) > player.hand[self.wildcard_tile]
        return actor, [a for a in actions if a[1] != self.wildcard_tile
                       or a[0] == 'ankan' and self.rules.JOKER_KONG or a[0] == 'discard' and not has_natural]

    def _chows(self, player, tile, provider):
        for start, used, block in super()._chows(player, tile, provider):
            if self.wildcard_tile in used:
                continue
            remaining = [face for face, count in enumerate(player.hand)
                         if face != self.wildcard_tile and count > used.count(face)]
            if not remaining or any(face not in block for face in remaining):
                yield start, used, block

    def _discard(self, actor, tile):
        if tile != self.wildcard_tile:
            super()._discard(actor, tile)
            return
        self.players[actor].hand[tile] -= 1
        self.discards[actor].append(tile)
        self.opening = self.uninterrupted = False
        if self.rules.STOP_AFTER_JOKER_DISCARD:
            self.stopped.add(actor)
        self._draw((actor + 1) % self.seat_count)

    def _score(self, actor, hand, melds, ctx):
        if actor in self.stopped:
            return None
        return self.rules.faan(hand, melds, ctx, wildcard=self.wildcard_tile)


class PanjinGame(LiaoningJokerGame):
    def _response_actions(self):
        actor, actions = super()._response_actions()
        if (self.offer.kind == 'discard' and not self.winners and self.remaining == self.options.closed_live_tail
                and self.players[actor].hand[self.offer.tile] == 3):
            actions.append(('zhigang', None))
        return actor, actions

    def _kong_factor(self, meld):
        return 32 if meld.tile == self.wildcard_tile else 1


PANJIN = PanjinRules('panjin-joker-kong')
ANSHAN = AnshanRules('anshan-open-guns')
LIAONING_JOKER_ROOMS = (PANJIN, ANSHAN)
