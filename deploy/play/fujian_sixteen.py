"""Nanping and Ningde five-set rooms over the shared flower and gold opening."""

from dataclasses import dataclass, replace
from itertools import combinations

from .gold_opening import GoldOpening, record_opening_follow, split_gold
from .honors import PASS, RON, KongPoints, WinContext, standard_shapes
from .honors_bonus import Score
from .sixteen import OPTIONS, SET_COUNT, SixteenGame, SixteenRules
from .wildcard_shapes import can_form_pairs, can_form_sets, can_form_standard, without_tile


WHITE = 31
RED = 33
SUITED = tuple(range(27))
OPTIONS = replace(OPTIONS, pass_win_all=True, pass_pon_all=True,
                  kong_points=KongPoints(1, 1, 2, single_direct_payer=True))


def seven_pairs_and_pung(natural, wild, melds):
    if melds:
        return len(melds) == 1 and melds[0].kind != 'chi' and can_form_pairs(natural, wild)
    if wild >= 3 and can_form_pairs(natural, wild - 3):
        return True
    for tile, count in enumerate(natural):
        for used in range(max(1, 3 - wild), min(count, 3) + 1):
            if can_form_pairs(without_tile(natural, tile, used), wild - 3 + used):
                return True
    return False


def kong_units(melds):
    return sum(2 if m.kind == 'gang_an' else 1 if m.kind.startswith('gang_') else 0 for m in melds)


@dataclass(frozen=True)
class FujianSixteenRules(SixteenRules):
    VARIANT: str
    OPTIONS = OPTIONS
    FIXED_GOLD = None
    WHITE_PROXY = None

    def split(self, hand, gold):
        return split_gold(hand, gold, substitute=self.WHITE_PROXY)

    def can_form(self, hand, melds, *, gold):
        if sum(hand) != 3 * (SET_COUNT - len(melds)) + 2:
            return False
        natural, wild = self.split(hand, gold)
        return can_form_standard(natural, wild) or (
            self.FIXED_GOLD is not None and seven_pairs_and_pung(natural, wild, melds)
        )

    def can_win(self, hand, melds, ctx, *, gold):
        if ctx.kind == 'rob_concealed' or (ctx.kind != 'zimo' and ctx.tile == gold):
            return False
        return (ctx.kind == 'zimo' and hand[gold] >= 3) or self.can_form(hand, melds, gold=gold)

    def awards(self, hand, melds, ctx, gold, flowers):
        natural, wild = self.split(hand, gold)
        drawn = ctx.kind in ('zimo', 'opening_heaven', 'three_gold', 'four_gold', 'claim_gold')
        stats = self.DRAW_AWARDS if drawn else self.DISCARD_AWARDS
        awards = [('平胡', 0)]
        opening = {
            'opening_heaven': '天胡', 'three_gold': '三金倒',
            'four_gold': '四金倒', 'claim_gold': '抢金',
        }
        if special := opening.get(ctx.kind):
            return [(special, stats[special])]
        if ctx.opening in stats and (ctx.opening != '天胡' or self.can_form(hand, melds, gold=gold)):
            awards.append((ctx.opening, stats[ctx.opening]))
        if drawn and wild >= 3:
            label = '四金倒' if wild == 4 else '三金倒'
            awards.append((label, stats.get(label, stats['三金倒'])))
        if '七对' in stats and seven_pairs_and_pung(natural, wild, melds):
            awards.append(('七对', stats['七对']))
        if wild >= 2 and can_form_sets(natural, wild - 2):
            awards.append(('金雀', stats['金雀']))
        if wild >= 3 and can_form_standard(natural, wild - 3):
            chows_only = wild == 3 and all(m.kind == 'chi' for m in melds) and any(
                all(m.kind == 'chi' for m in sets)
                for _, sets in standard_shapes(natural, len(melds), groups=SET_COUNT - 1)
            )
            label = '金龙' if chows_only else '金坎'
            if label in stats:
                awards.append((label, stats[label]))
        gold_kong = next((m for m in melds if m.tile == gold and m.kind.startswith('gang_')), None)
        if gold_kong and '金杠' in stats:
            awards.append(('金杠', self.gold_kong_award(gold_kong, stats)))
        faces = {t for t, n in enumerate(natural) if n} | {m.tile for m in melds if m.tile != gold}
        suits = {t // 9 for t in faces if t < 27}
        if len(suits) == 1 and self.can_form(hand, melds, gold=gold):
            label = '混一色' if any(t >= 27 for t in faces) else '清一色'
            if label in stats:
                awards.append((label, stats[label]))
        if len(flowers) == 1 and '一张花' in stats:
            awards.append(('一张花', stats['一张花']))
        if not flowers and not kong_units(melds) and '无花无杠' in stats:
            awards.append(('无花无杠', stats['无花无杠']))
        return awards

    @staticmethod
    def gold_kong_award(meld, stats):
        return stats['金杠']

    @staticmethod
    def payments(game, actor, provider, score):
        return tuple(score.share if seat != actor and provider in (None, seat) else 0 for seat in range(game.seat_count))


class NanpingRules(FujianSixteenRules):
    DISCARD_AWARDS = {'天胡': 30, '抢金': 30, '三金倒': 40, '金坎': 50, '金雀': 50, '金龙': 100,
                      '混一色': 100, '清一色': 200, '一张花': 15, '无花无杠': 30}
    DRAW_AWARDS = DISCARD_AWARDS
    WHITE_PROXY = WHITE

    def score(self, game, hand, melds, ctx, flowers):
        awards = self.awards(hand, melds, ctx, game.wildcard_tile, flowers)
        pattern, premium = max(awards, key=lambda award: award[1])
        drawn = ctx.kind not in ('ron', 'rob_kong')
        groups = sum(set(range(start, start + 4)).issubset(flowers) for start in (34, 38))
        flower_points = len(flowers) + 2 * groups
        points = (1 + flower_points + hand[game.wildcard_tile] + game.dealer_streak) * (2 if drawn else 1)
        return Score(points + premium, (pattern, f'花数x{flower_points}', f'金牌x{hand[game.wildcard_tile]}'))


class NanpingSuitedRules(NanpingRules):
    OPTIONS = replace(OPTIONS, tile_kinds=SUITED, kong_points=KongPoints(3, 1, 2, single_direct_payer=True))
    WALL_RESERVE = 12
    WHITE_PROXY = None
    DISCARD_AWARDS = {'天胡': 8, '抢金': 8, '金雀': 8, '三金倒': 20, '金坎': 30, '金龙': 30, '清一色': 50}
    DRAW_AWARDS = DISCARD_AWARDS

    def score(self, game, hand, melds, ctx, flowers):
        label, special = max(self.awards(hand, melds, ctx, game.wildcard_tile, flowers), key=lambda award: award[1])
        return Score(special or (2 if ctx.kind == 'zimo' else 1), (label,))


class NanpingRedRules(NanpingRules):
    OPTIONS = replace(OPTIONS, tile_kinds=SUITED + (RED,))
    WALL_RESERVE = 12
    WHITE_PROXY = None
    FIXED_GOLD = RED
    DISCARD_AWARDS = {'三金倒': 10, '四金倒': 20, '七对': 20, '地胡': 20, '天胡': 30,
                      '金雀': 20, '金坎': 30, '金龙': 30, '金杠': 80, '清一色': 40}
    DRAW_AWARDS = {**DISCARD_AWARDS, '金雀': 30, '金坎': 40, '金龙': 40, '金杠': 100, '清一色': 50}


class NingdeRules(FujianSixteenRules):
    OPTIONS = replace(OPTIONS, tile_kinds=SUITED + (WHITE,), pass_win_all=False, pass_pon_all=False, kong_points=None)
    WALL_RESERVE = 12
    FIXED_GOLD = WHITE
    DISCARD_AWARDS = {'三金倒': 10, '四金倒': 20, '七对': 10, '地胡': 20, '天胡': 20,
                      '金雀': 20, '金坎': 30, '金龙': 30, '金杠': 100, '清一色': 40}
    DRAW_AWARDS = DISCARD_AWARDS

    @staticmethod
    def gold_kong_award(meld, stats):
        return 100 if meld.kind == 'gang_an' else 200

    def score(self, game, hand, melds, ctx, flowers):
        awards = self.awards(hand, melds, ctx, game.wildcard_tile, flowers)
        wild = hand[WHITE]
        pattern, special = max((award for award in awards if award[0] != '七对' or not wild), key=lambda award: award[1])
        basic = (2 + wild + kong_units(melds)) * (1 if ctx.kind in ('ron', 'rob_kong') else 2)
        return Score(special or basic, (pattern, f'金牌x{wild}'))

    @staticmethod
    def payments(game, actor, provider, score):
        feeds = {}
        if '清一色' in score.patterns:
            for meld in game.players[actor].melds:
                if meld.provider is not None:
                    feeds[meld.provider] = feeds.get(meld.provider, 0) + 1
        bills = tuple(score.share + (game.dealer in (actor, seat)) if seat != actor else 0
                      for seat in range(game.seat_count))
        load = 4 if provider is None else 3
        payer = next((seat for seat, count in feeds.items() if count >= load), None)
        if payer is not None:
            total = sum(bills) if provider is None else bills[payer]
            return tuple(total if seat == payer else 0 for seat in range(game.seat_count))
        return tuple(amount if provider in (None, seat) else 0 for seat, amount in enumerate(bills))


NANPING_HONORS = NanpingRules('nanping-sixteen-honors')
NANPING_SUITED = NanpingSuitedRules('nanping-sixteen-suited')
NANPING_RED = NanpingRedRules('nanping-sixteen-red')
NINGDE = NingdeRules('ningde-white-gold')


class FujianSixteenGame(GoldOpening, SixteenGame):
    RECORD_FOLLOW = True

    def __init__(self, seed, dealer, round_wind, *, rules, dealer_streak=0):
        self.indicator_tile = None
        self.following = True
        self.follow_rounds = 0
        self.follow_deltas = [0] * rules.OPTIONS.seat_count
        self.balance_deltas = [0] * rules.OPTIONS.seat_count
        self.opening_three = None
        self.passed_heaven = False
        super().__init__(seed, dealer, round_wind, rules=rules, dealer_streak=dealer_streak)

    @property
    def EXTRA_FLOWERS(self):
        return () if self.rules is NINGDE else super().EXTRA_FLOWERS

    def _deal_opening(self):
        super()._deal_opening()
        seats = [(self.dealer + step) % self.seat_count for step in range(1, self.seat_count + 1)]
        self.wildcard_tile = self.rules.FIXED_GOLD
        if self.wildcard_tile is None:
            self._open_flipped_gold(seats)
            return
        for actor in seats:
            if self.players[actor].hand[self.wildcard_tile] == 4:
                entries = [(actor, 'four_gold')]
                if self._qualifies_for_gold(self.players[self.dealer].hand):
                    entries.insert(0, (self.dealer, 'opening_heaven'))
                self._queue_gold_choices(entries)
                break

    def _open_flipped_gold(self, seats):
        self._start_gold_opening(self._replacement_index(), seats, heaven=True)

    def _qualifies_for_gold(self, hand):
        return self.rules.can_form(hand, (), gold=self.wildcard_tile)

    def _finish_gold_opening(self, actor, kind, hand):
        reason = {'claim_gold': 'gold_claim', 'opening_heaven': 'suited_self_draw',
                  'three_gold': 'three_gold_win', 'four_gold': 'four_gold_win'}[kind]
        ctx = WinContext(kind, self.wildcard_tile, 27, self.round_wind)
        score = self.rules.score(self, hand, (), ctx, self.flowers[actor])
        self._settle_special(actor, None, reason, score, hand)

    def step(self, action):
        if self.phase == 'gold_opening' and self.gold_queue[0][1] == 'opening_heaven' and action == PASS:
            self.passed_heaven = True
        super().step(action)

    def _discard(self, actor, tile):
        super()._discard(actor, tile)
        if self.RECORD_FOLLOW and self.rules is not NINGDE:
            record_opening_follow(self, actor, tile)

    def _chows(self, player, tile, provider):
        if tile == self.wildcard_tile:
            return ()
        return (choice for choice in super()._chows(player, tile, provider) if self.wildcard_tile not in choice[1])

    def _response_actions(self):
        actor, actions = super()._response_actions()
        if self.offer.tile == self.wildcard_tile:
            actions = [action for action in actions if action == PASS or
                       (action[0] == 'zhigang' and self.rules.FIXED_GOLD is not None)]
        if not self.winners and self.rules.FIXED_GOLD is not None and self.offer.opening == '地胡':
            if self.players[actor].hand[self.wildcard_tile] == 3:
                actions.append(('three_gold', None))
        return actor, actions

    def _respond(self, actor, action, actions):
        if action[0] == 'three_gold':
            self.opening_three = actor
            action = PASS
        super()._respond(actor, action, actions)

    def _finish_responses(self, offer):
        actor = self.opening_three
        self.opening_three = None
        if actor is not None and not self.winners:
            self._finish_gold_opening(actor, 'three_gold', self.players[actor].hand.copy())
        else:
            super()._finish_responses(offer)

    def _score(self, actor, hand, melds, ctx):
        if ctx.opening == '天胡' and self.passed_heaven:
            return None
        if self.rules.can_win(hand, melds, ctx, gold=self.wildcard_tile):
            return self.rules.score(self, hand, melds, ctx, self.flowers[actor])
        return None

    def _end(self, reason):
        if self.rules is NINGDE and self.winners:
            losers = [actor for actor in range(self.seat_count) if actor not in self.winners]
            units = [p.hand[WHITE] + kong_units(p.melds) + 4 * any(m.tile == WHITE for m in p.melds) for p in self.players]
            for a, b in combinations(losers, 2):
                receiver, payer = (a, b) if units[a] >= units[b] else (b, a)
                amount = units[receiver] - units[payer]
                self._transfer(receiver, tuple(amount if seat == payer else 0 for seat in range(self.seat_count)),
                               self.balance_deltas)
        super()._end(reason)
