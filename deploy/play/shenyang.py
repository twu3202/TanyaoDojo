"""Shenyang's whirlwind scoring and discard-free final four draws."""

from collections import Counter
from dataclasses import dataclass, replace

from .changchun import ChangchunGame, ChangchunRules, Egg, egg_rack_faces
from .honors import DRAGONS, ORPHANS, PASS, ZIMO
from .northeast import wait_pattern
from .wildcard_shapes import without_tile


CAP = 64


@dataclass(frozen=True)
class ShenyangScore:
    fan: int
    patterns: tuple[str, ...]
    tier = None

    @property
    def share(self):
        return 2 ** self.fan


class ShenyangRules(ChangchunRules):
    OPTIONS = replace(ChangchunRules.OPTIONS, kong_points=None)
    WALL_RESERVE = 0
    OPEN_FOR_STANDARD = True

    @staticmethod
    def closed(melds):
        return not any(m.kind in ('chi', 'peng', 'gang_ming', 'gang_bu') for m in melds)

    @classmethod
    def rack_allowed(cls, hand, melds):
        faces = egg_rack_faces(hand, melds)
        return not cls.closed(melds) and len({t // 9 for t in faces if t < 27}) == 3 and bool(faces & ORPHANS)

    @classmethod
    def settle_hand(cls, game, winners, offer):
        return [(win.score, cls.payments(game, win.actor, None if offer is None else offer.actor, win.score))
                for win in winners]

    @classmethod
    def payments(cls, game, actor, provider, score):
        closed = [cls.closed(p.melds) for p in game.players]
        all_closed = all(closed[seat] for seat in range(game.seat_count) if seat != actor)
        return tuple(
            min(CAP, 2 ** (score.fan + (game.dealer in (actor, seat)) + (seat == provider) +
                           (2 if all_closed else int(closed[seat]))))
            if seat != actor and provider in (None, seat) else 0 for seat in range(game.seat_count)
        )


SHENYANG = ShenyangRules('shenyang-whirlwind')


class ShenyangGame(ChangchunGame):
    ALLOW_READY_KONGS = True
    EGG_ROB_PON = False

    def __init__(self, seed, dealer, round_wind, *, rules=SHENYANG):
        self.split_draw = False
        self.split_tiles = [[] for _ in range(rules.OPTIONS.seat_count)]
        super().__init__(seed, dealer, round_wind, rules=rules)

    def _replacement_index(self):
        return 0

    def _can_open_bao(self, actor):
        return False

    def _can_extend_on_turn(self, actor):
        return True

    def _new_eggs(self, actor):
        return [a for a in super()._new_eggs(actor) if a[1] is None and a[0] in ('egg_winds', 'egg_dragons')]

    def _pay_kong(self, actor, meld):
        # Kongs in this room are fan on the eventual winning hand.
        pass

    def _draw(self, actor, kong_chain=0, provider=None):
        self.split_draw = 0 < self.remaining <= 4
        player = self.players[actor]
        first = player.draws == 0 and self.egg_racks[actor] is not None
        super()._draw(actor, kong_chain, provider)
        if first and player.draws == 1:
            self.egg_racks[actor] = player.hand.copy()

    def physical_counts(self):
        counts = super().physical_counts()
        counts.update(t for tiles in self.split_tiles for t in tiles)
        return counts

    def legal_actions(self):
        actor, actions = super().legal_actions()
        if self.phase != 'action' or self.offer is not None:
            return actor, actions
        if self.split_draw:
            return actor, [PASS, ZIMO] if ZIMO in actions else [PASS]
        return actor, [a for a in actions if a[0] != 'extend_egg' or self._egg_extension_allowed(actor, a[1])]

    def _egg_extension_allowed(self, actor, argument):
        index, face = argument
        return (self.egg_racks[actor] is not None
                and self.players[actor].melds[index].kind == 'egg_winds' and face >= 27)

    def step(self, action):
        if self.split_draw and self.phase == 'action' and self.offer is None and action == PASS:
            actor, tile = self.cur, self.drawn_tile
            self.players[actor].hand[tile] -= 1
            self.split_tiles[actor].append(tile)
            self.last_call = None
            self._draw((actor + 1) % self.seat_count)
            return
        super().step(action)

    def _score(self, actor, hand, melds, ctx):
        if ctx.kind == 'rob_concealed':
            return None
        hand = tuple(hand)
        shapes = self.rules.shapes(hand, melds)
        if not shapes:
            return None
        tile = ctx.tile
        bonus = []
        if ctx.kind == 'zimo':
            bonus.append(('自摸', 1))
            if self.split_draw:
                bonus.append(('海底捞月', 1))
            if ctx.kong_chain:
                bonus.append(('杠上开花', 1))
        elif ctx.kind == 'rob_kong':
            bonus.append(('抢杠胡', 1))
        elif ctx.kong_chain:
            bonus.append(('流泪', 1))
        if hand[tile] and self.rules.waits(without_tile(hand, tile), melds) == (tile,):
            if pattern := wait_pattern(tile, shapes):
                bonus.append((pattern, 1))
        counts = Counter({t: count for t, count in enumerate(hand) if count})
        for meld in melds:
            if meld.kind in ('chi', 'peng'):
                counts.update(meld.tiles)
        if 4 in counts.values():
            bonus.append(('四归一', 1))
        exposed_whirlwind = {t for other, player in enumerate(self.players) if other != actor
                             for m in player.melds if isinstance(m, Egg) for t in m.tiles}
        for meld in melds:
            if isinstance(meld, Egg):
                bonus.append(('旋风杠', 2))
            elif meld.kind == 'peng':
                if meld.tile in DRAGONS:
                    bonus.append(('箭刻', 1))
                if meld.tile in exposed_whirlwind:
                    bonus.append(('旋风碰番', 1))
            elif meld.kind.startswith('gang_'):
                bonus.append(('暗杠' if meld.kind == 'gang_an' else '明杠',
                              (2 if meld.kind == 'gang_an' else 1) + (meld.tile in DRAGONS)))

        best = ()
        for _, sets in shapes:
            labels = [('箭刻', 1) for m in sets if m.kind == 'triplet' and m.tile in DRAGONS]
            if all(m.kind != 'chi' for m in (*sets, *melds)):
                labels.append(('飘胡', 3))
            if sum(value for _, value in labels) > sum(value for _, value in best):
                best = labels
        labels = [*bonus, *best]
        return ShenyangScore(sum(value for _, value in labels), tuple(label for label, _ in labels) or ('平胡',))
