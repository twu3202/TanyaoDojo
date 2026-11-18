"""Jilin City's capped treasure game and Shuangliao's multiplied pung game."""

from collections import Counter
from dataclasses import replace
from itertools import combinations

from .changchun import BIG_KONGS, CHICKEN, EGG_GROUPS, ChangchunGame, ChangchunRules, Egg, egg_rack_faces
from .honors import PASS, TERMINALS, WINDS, HonorsGame, KongPoints, TableOptions
from .honors_bonus import Score
from .northeast import NortheastRules, wait_pattern
from .wildcard_shapes import can_form_pairs, without_tile


JILIN_CAP = 16
JILIN_BASE = {'单贴': 1, '夹胡': 1, '边胡': 2, '飘胡': 4, '手把一': 1, '七小对': JILIN_CAP}
JILIN_WAITS = {'单钓': '单贴', '夹张': '夹胡', '边张': '边胡'}
JILIN_KONGS = {7: 8, CHICKEN: 8, 10: 4, 9: 2}


class JilinRules(ChangchunRules):
    OPTIONS = replace(ChangchunRules.OPTIONS, seven_pairs=True)

    @staticmethod
    def next_dealer(dealer, winners, provider):
        return (dealer + 1) % 4 if winners and dealer not in winners else dealer

    @staticmethod
    def rack_allowed(hand, melds):
        faces = egg_rack_faces(hand, melds)
        return len({tile // 9 for tile in faces if tile < 27}) in (1, 3) and bool(faces & (TERMINALS | WINDS))

    ready_allowed = rack_allowed

    @staticmethod
    def shape_allowed(pair, sets):
        return pair in WINDS or any(m.kind != 'chi' for m in sets)

    def pattern(self, hand, melds, tile):
        if not melds and sum(hand) == 14 and can_form_pairs(hand, 0) and self.rack_allowed(hand, melds):
            return '七小对'
        shapes = self.shapes(hand, melds)
        if not shapes:
            return None
        if any(all(m.kind != 'chi' for m in (*melds, *sets)) for _, sets in shapes):
            return '飘胡'
        if len(melds) == 4:
            return '手把一'
        return JILIN_WAITS.get(wait_pattern(tile, shapes))

    def natural_win(self, hand, melds, tile):
        return self.pattern(hand, melds, tile) is not None

    @classmethod
    def payments(cls, game, actor, provider, score):
        winner_closed = cls.closed(game.players[actor].melds)
        payments = []
        for seat, player in enumerate(game.players):
            points = score.share
            for doubles in (game.dealer in (actor, seat), seat == provider, winner_closed or cls.closed(player.melds)):
                if doubles:
                    points *= 2
            payments.append(min(JILIN_CAP, points) if seat != actor else 0)
        return tuple(payments)


JILIN = JilinRules('jilin-dice-bao')


class JilinGame(ChangchunGame):
    BAO_SLOT_COUNT = 6

    def __init__(self, seed, dealer, round_wind, *, rules=JILIN):
        self.bao_starter = None
        super().__init__(seed, dealer, round_wind, rules=rules)

    def _declared(self, actor):
        if self.bao_starter is None:
            self.bao_starter = actor

    def _can_open_bao(self, actor):
        return self.bao_starter == actor or bool(self.bao_history)

    def bao_match_reason(self):
        return 'tong_bao_win' if len(self.bao_history) == 1 else super().bao_match_reason()

    @staticmethod
    def egg_tiles(kind, arg):
        if kind != 'egg_winds':
            return ChangchunGame.egg_tiles(kind, arg)
        group, missing = arg
        return tuple(CHICKEN if tile == missing else tile for tile in group)

    def _opening_egg(self, kind, arg):
        if kind != 'egg_winds':
            return super()._opening_egg(kind, arg)
        tiles = self.egg_tiles(kind, arg)
        return Egg(kind, tiles[0], material=tiles, substitutions=int(arg[1] is not None))

    def _new_eggs(self, actor):
        actions = [action for action in super()._new_eggs(actor) if action[0] != 'egg_winds']
        player = self.players[actor]
        opening = self.egg_racks[actor]
        if opening is None or len(player.melds) == 4:
            return actions
        available = tuple(min(a, b) for a, b in zip(opening, player.hand))
        natural = []
        substitutes = []
        for group in combinations(EGG_GROUPS['egg_winds'], 3):
            missing = [tile for tile in group if not available[tile]]
            if not missing:
                natural.append(('egg_winds', (group, None)))
            elif len(missing) == 1 and available[CHICKEN]:
                substitutes.append(('egg_winds', (group, missing[0])))
        return [*natural, *actions, *substitutes]

    def _pay_kong(self, actor, meld):
        self.kong_count += 1
        if isinstance(meld, Egg):
            points = 3 if len(meld.tiles) == 3 else 1
        else:
            points = JILIN_KONGS.get(meld.tile, 1) * (2 if meld.kind == 'gang_an' else 1)
        self._transfer(actor, tuple(points if seat != actor else 0 for seat in range(self.seat_count)), self.kong_deltas)

    def _is_bao_draw(self, actor, tile):
        return super()._is_bao_draw(actor, tile) or (tile == CHICKEN and bool(self.bao_view(actor)))

    def _score(self, actor, hand, melds, ctx):
        if not self.declared_waits[actor] and ctx.opening not in ('天胡', '地胡'):
            return None
        return super()._score(actor, hand, melds, ctx)

    def _natural_score(self, hand, melds, tile, kind, bonus=None):
        hand = tuple(hand)
        pattern = self.rules.pattern(hand, melds, tile)
        if pattern is None:
            return None
        points = JILIN_BASE[pattern]
        labels = [pattern]
        if kind == 'zimo':
            labels.append('自摸')
            points *= 2
        if bonus == '摸宝':
            labels.append('搂宝')
            points *= 4
        elif bonus == '对宝':
            labels.append('通宝' if self.bao_match_reason() == 'tong_bao_win' else '对宝')
            points *= 8
        if len({tile // 9 for tile in egg_rack_faces(hand, melds) if tile < 27}) == 1:
            points *= 2
            labels.append('清一色')
        if len(melds) == 4 and pattern != '手把一':
            labels.append('手把一')
        if len(melds) == 4:
            points *= 2
        if self.rules.closed(melds):
            labels.append('闭门')
        return Score(min(JILIN_CAP, points), tuple(labels))


class ShuangliaoRules(NortheastRules):
    OPTIONS = TableOptions(orphans=False, concealed_kong_faces=0, pass_win_all=True,
                           kong_requires_live_tile=True, kong_points=KongPoints(2, 2, 4))
    WALL_RESERVE = 14

    @staticmethod
    def rack_allowed(hand, melds):
        return True

    @staticmethod
    def shape_allowed(pair, sets):
        return True

    @staticmethod
    def settle_hand(game, winners, offer):
        return [(win.score, ChangchunRules.payments(game, win.actor, None if offer is None else offer.actor, win.score))
                for win in winners]


SHUANGLIAO = ShuangliaoRules('shuangliao-pung-multipliers')


class ShuangliaoGame(HonorsGame):
    def _replacement_index(self):
        return self.rules.WALL_RESERVE

    def _response_actions(self):
        if self.winners:
            return self.responses[0], [PASS]
        return super()._response_actions()

    def _kong_factor(self, meld):
        return 2 if meld.tile in BIG_KONGS else 1

    def _score(self, actor, hand, melds, ctx):
        if ctx.kind == 'rob_concealed':
            return None
        hand = tuple(hand)
        shapes = self.rules.shapes(hand, melds)
        if not shapes:
            return None
        closed = self.rules.closed(melds)
        splits = [(*melds, *sets) for _, sets in shapes]
        all_pungs = any(all(m.kind != 'chi' for m in sets) for sets in splits)
        piao = all_pungs or any(any(m.kind != 'chi' for m in sets)
                                and all(m.kind != 'chi' or m.tile % 9 == 0 for m in sets) for sets in splits)
        rank, points = ('飘中飘', 16) if closed and all_pungs else ('飘胡', 4) if piao else ('平胡', 1)
        patterns = [rank]
        counts = Counter({tile: n for tile, n in enumerate(hand) if n})
        for meld in melds:
            if not meld.kind.startswith('gang_'):
                counts.update(meld.tiles)
        bonus = (
            ('自摸', ctx.kind == 'zimo'), ('带幺鸡', hand[CHICKEN] or any(CHICKEN in m.tiles for m in melds)),
            ('站立', closed), ('杠翻' if ctx.kong_chain else '四归一', ctx.kong_chain or 4 in counts.values()),
            ('夹胡', wait_pattern(ctx.tile, shapes) == '夹张' and self.rules.waits(without_tile(hand, ctx.tile), melds) == (ctx.tile,)),
            ('抢杠胡', ctx.kind == 'rob_kong'),
        )
        for label, earned in bonus:
            if earned:
                points *= 2
                patterns.append(label)
        return Score(points, tuple(patterns))
