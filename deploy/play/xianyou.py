"""Xianyou five-set Mahjong with White Dragon substitution, flowers, or fixed gold."""

from dataclasses import dataclass, replace

from .gold_opening import GoldOpening, split_gold
from .honors import PASS, KongPoints, WinContext
from .honors_bonus import Score
from .longyan import TRAVEL
from .sixteen import OPTIONS, SET_COUNT, SixteenGame, SixteenRules
from .wildcard_shapes import can_form_sets, can_form_standard, without_tile


WHITE = 31
GOLD_DOWNS = {3: ('three_gold', '三金倒'), 4: ('four_gold', '四金倒')}
GOLD_REASONS = {'three_gold': 'three_gold_win', 'four_gold': 'four_gold_win', 'claim_gold': 'gold_claim'}
OPTIONS = replace(OPTIONS, tile_kinds=(*range(27), WHITE), chow=False)


@dataclass(frozen=True)
class XianyouRules(SixteenRules):
    VARIANT: str
    OPTIONS = replace(OPTIONS, kong_points=KongPoints(3, 0, 0, single_direct_payer=True))
    WALL_RESERVE = 12
    HAS_FLOWERS = True
    FIXED_GOLD = None
    WHITE_PROXY = WHITE
    WHITE_FLOWER = False
    DISCARD_WINS = True

    def split(self, hand, gold):
        return split_gold(hand, gold, substitute=self.WHITE_PROXY)

    def can_form(self, hand, melds, gold):
        if sum(hand) != 3 * (SET_COUNT - len(melds)) + 2:
            return False
        return can_form_standard(*self.split(hand, gold))

    def can_win(self, hand, melds, ctx, *, gold):
        if ctx.kind != 'zimo' and (not self.DISCARD_WINS or ctx.kind not in ('ron', 'rob_kong') or ctx.tile == gold):
            return False
        return self.can_form(hand, melds, gold)

    def travel_wait(self, hand, melds, gold):
        if not hand[gold] or sum(hand) != 3 * (SET_COUNT - len(melds)) + 1:
            return False
        natural, wild = self.split(hand, gold)
        return can_form_sets(natural, wild - 1)

    def travel_level(self, hand, gold):
        return 3 if hand[gold] >= 3 else 1

    def score(self, hand, melds, ctx, gold, flowers, travel_level=0):
        natural, wild = self.split(hand, gold)
        if ctx.kind in GOLD_REASONS:
            pattern = '抢金' if ctx.kind == 'claim_gold' else GOLD_DOWNS[wild][1]
        elif ctx.kind == 'zimo':
            if travel_level:
                pattern = '三金游' if travel_level == 3 else '游金'
            elif wild == 2 and self.DISCARD_WINS and can_form_sets(natural, 0):
                pattern = '二金倒'
            else:
                pattern = '白胡' if not wild and self.WHITE_FLOWER else '自摸'
        else:
            pattern = '抢杠胡' if ctx.kind == 'rob_kong' else '平胡'
        points = self.points(pattern, wild, flowers, melds)
        return Score(points, (pattern, f'金牌x{wild}'))

    @staticmethod
    def points(pattern, wild, flowers, melds):
        # The direct kong is paid by its supplier when declared.
        kong = sum(2 if m.kind == 'gang_an' else m.kind == 'gang_bu' for m in melds)
        return {'游金': 4, '三金游': 6, '三金倒': 3, '四金倒': 10}.get(pattern, 2) + kong

    @staticmethod
    def payments(game, actor, provider, score):
        return tuple(score.share if seat != actor and provider in (None, seat) else 0
                     for seat in range(game.seat_count))


class XianyouFlowerRules(XianyouRules):
    OPTIONS = OPTIONS
    WHITE_PROXY = None
    WHITE_FLOWER = True

    @staticmethod
    def points(pattern, wild, flowers, melds):
        kong = sum(2 if m.kind == 'gang_an' else m.kind.startswith('gang_') for m in melds)
        base = 2 + wild + len(flowers) + kong + (3 if pattern == '白胡' else 0)
        factor = {'平胡': 1, '抢杠胡': 1, '三金倒': 3, '四金倒': 10, '游金': 4, '三金游': 6}
        return base * factor.get(pattern, 2)


class XianyouSelfDrawRules(XianyouRules):
    OPTIONS = OPTIONS
    FIXED_GOLD = WHITE
    HAS_FLOWERS = False
    WHITE_PROXY = None
    DISCARD_WINS = False

    def travel_level(self, hand, gold):
        return 1

    @staticmethod
    def points(pattern, wild, flowers, melds):
        if pattern == '四金倒':
            return 10
        kong = sum(2 if m.kind == 'gang_an' else m.kind.startswith('gang_') for m in melds)
        return 1 + kong + (3 + 2 * (wild - 1) if pattern == '游金' else wild)


XIANYOU_SUBSTITUTE = XianyouRules('xianyou-white-substitute')
XIANYOU_FLOWER = XianyouFlowerRules('xianyou-white-flower')
XIANYOU_SELF_DRAW = XianyouSelfDrawRules('xianyou-self-draw')
XIANYOU_ROOMS = (XIANYOU_SUBSTITUTE, XIANYOU_FLOWER, XIANYOU_SELF_DRAW)


class XianyouGame(GoldOpening, SixteenGame):
    def __init__(self, seed, dealer, round_wind, *, rules, dealer_streak=0):
        self.indicator_tile = None
        self.travel_owner = None
        self.travel_level = 0
        self.first_discard_seen = False
        self.excluded_tiles = frozenset({WHITE} if rules.WHITE_FLOWER else ())
        super().__init__(seed, dealer, round_wind, rules=rules, dealer_streak=dealer_streak)

    @property
    def EXTRA_FLOWERS(self):
        return super().EXTRA_FLOWERS if self.rules.HAS_FLOWERS else ()

    def _prepare_wall(self):
        super()._prepare_wall()
        if self.rules.WHITE_FLOWER:
            self.flower_kinds |= {WHITE}

    def _deal_opening(self):
        super()._deal_opening()
        self.wildcard_tile = self.rules.FIXED_GOLD
        if self.wildcard_tile is not None:
            return
        indicator = self._flip_gold_indicator(self._replacement_index())
        self.wildcard_tile = (indicator // 9) * 9 + (indicator + 1) % 9
        order = [(self.dealer + step) % self.seat_count for step in range(1, self.seat_count + 1)]
        counts = [player.hand[self.wildcard_tile] for player in self.players]
        entries = [(actor, GOLD_DOWNS[counts[actor]][0]) for actor in order if counts[actor] in GOLD_DOWNS]
        entries.extend((actor, None) for actor in order if counts[actor] < 3)
        self._queue_gold_choices(entries)

    def _qualifies_for_gold(self, hand):
        return self.rules.can_form(hand, (), self.wildcard_tile)

    def _finish_gold_opening(self, actor, kind, hand):
        ctx = WinContext(kind, self.wildcard_tile, 27, self.round_wind)
        score = self.rules.score(hand, self.players[actor].melds, ctx, self.wildcard_tile, self.flowers[actor])
        self._settle_special(actor, None, GOLD_REASONS[kind], score, hand)

    def _score(self, actor, hand, melds, ctx):
        gold = self.wildcard_tile
        if self.rules.can_win(hand, melds, ctx, gold=gold):
            level = self.travel_level if ctx.kind == 'zimo' and self.travel_owner == actor else 0
            return self.rules.score(hand, melds, ctx, gold, self.flowers[actor], level)
        return None

    def _gold_down(self, actor):
        count = self.players[actor].hand[self.wildcard_tile]
        return [(GOLD_DOWNS[count][0], None)] if count in GOLD_DOWNS else []

    def _response_actions(self):
        actor, actions = super()._response_actions()
        if self.travel_owner == actor:
            return actor, [PASS]
        if not self.rules.DISCARD_WINS and self.offer.kind == 'discard':
            actions += self._gold_down(actor)
        return actor, actions

    def legal_actions(self):
        actor, actions = super().legal_actions()
        if self.phase != 'action' or self.offer is not None:
            return actor, actions
        gold = self.wildcard_tile
        actions = [action for action in actions if action[1] != gold]
        if self.drawn_tile is None:
            return actor, actions
        if self.rules.FIXED_GOLD is None or self.first_discard_seen:
            actions += self._gold_down(actor)
        player = self.players[actor]
        if player.hand[gold] and self.travel_owner is None and self.remaining >= self.seat_count:
            hand = tuple(player.hand)
            actions += [(TRAVEL, tile) for kind, tile in actions if kind == 'discard'
                        and self.rules.travel_wait(without_tile(hand, tile), player.melds, gold)]
        return actor, actions

    def step(self, action):
        kind, tile = action
        if self.phase == 'gold_opening' or kind not in (*GOLD_REASONS, TRAVEL):
            super().step(action)
            return
        actor, actions = self.legal_actions()
        if action not in actions:
            raise ValueError('Illegal Xianyou gold action')
        self.last_call = None
        player = self.players[actor]
        if kind == TRAVEL:
            self._discard(actor, tile)
            self.travel_owner = actor
            self.travel_level = self.rules.travel_level(player.hand, self.wildcard_tile)
        else:
            self._finish_gold_opening(actor, kind, player.hand.copy())

    def _discard(self, actor, tile):
        self.first_discard_seen = True
        if self.travel_owner == actor:
            self.travel_owner = None
            self.travel_level = 0
        super()._discard(actor, tile)
