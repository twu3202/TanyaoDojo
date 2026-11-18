"""Zhejiang rooms with a flipped tile or its successor as the wildcard."""

from dataclasses import replace
from functools import lru_cache
from itertools import combinations
from math import prod

from .hangzhou import (FLOAT, WHITE, FOUR_HANGZHOU, HangzhouGame, HangzhouRules, HangzhouScore,
                       _score_shape, burst_wait, split_wild)
from .honors import ORPHANS, WINDS, Offer, chow_choices
from .wildcard_shapes import (can_form_all_pungs, can_form_pairs, can_form_pungs, can_form_standard,
                             without_tile)


TEN_WINDS = 'ten_winds'
ELEVEN_WINDS = 'eleven_winds'
XIAOSHAN_CAP = 128


def _pattern_score(patterns):
    return HangzhouScore(prod(multiplier for _, multiplier in patterns),
                         tuple(name for name, _ in patterns) or ('平胡',))


def _filled_families(natural, melds):
    used = {tile for tile, n in enumerate(natural) if n}
    used.update(tile for meld in melds for tile in meld.tiles)
    return {tile // 9 for tile in used if tile < 27}, any(tile >= 27 for tile in used)


def _pair_name(luxury, wild):
    if wild:
        return ('七对子', '豪华七对', '双豪华七对', '三豪华七对')[luxury]
    return ('清七对', '豪华清七对', '双豪华清七对', '三豪华清七对')[luxury]


class XiaoshanRules(HangzhouRules):
    @staticmethod
    def chow_swap_lock(melds, provider):
        return any(meld.provider == provider for meld in melds)

    def faan(self, hand, melds, ctx, *, floats=0, kongs_before_float=0, last_kong=None):
        if ctx.kind != 'zimo' or ctx.tile is None or not hand[ctx.tile] or sum(hand) != 14 - 3 * len(melds):
            return None
        natural, wild = split_wild(hand)
        before = without_tile(tuple(hand), ctx.tile)
        plain_burst, pairs_burst = burst_wait(before, len(melds))
        before_natural, before_wild = split_wild(before)
        can_pung = all(meld.kind != 'chi' for meld in melds)
        all_pungs = can_pung and can_form_all_pungs(natural, wild)
        pung_burst = plain_burst and can_pung and can_form_pungs(before_natural, before_wild - 1)
        suits, honors = _filled_families(natural, melds)

        if not suits:
            if all_pungs:
                return HangzhouScore(128, ('正风',))
            if ctx.opening or ctx.first_nondealer_draw:
                return HangzhouScore(128, ('天胡' if ctx.opening else '地胡',))
            return HangzhouScore(64, ('乱风',))

        common = []
        if len(suits) == 1 and not honors:
            common.append(('清一色', 8))
        if ctx.kong_chain:
            common.append(('豪杠开', 4) if last_kong == 'gang_an' else ('杠开', 2))

        def candidate(patterns, burst, pair_hand=False):
            if burst:
                name = ('爆头', '财飘', '双财飘')[floats]
                if pair_hand and not floats:
                    name = '七客'
                patterns = [*patterns, (name, 2 << floats)]
            score = _pattern_score([*patterns, *common])
            return score if not wild or score.share >= 2 else None

        scores = []
        if plain_burst or can_form_standard(natural, wild):
            single = [('大吊车', 2)] if len(melds) == 4 else []
            scores.append(candidate(single, plain_burst))
            if all_pungs or pung_burst:
                scores.append(candidate([('大对子', 2), *single], pung_burst))
        if not melds and can_form_pairs(natural, wild):
            luxury = sum(n == 4 for n in (before_natural if pairs_burst else natural))
            share = (4, 8, 16, 64)[luxury] * (1 if wild else 2)
            scores.append(candidate([(_pair_name(luxury, wild), share)], pairs_burst, pair_hand=True))

        if scores and (ctx.opening or ctx.first_nondealer_draw):
            return HangzhouScore(128, ('天胡' if ctx.opening else '地胡',))
        return max((score for score in scores if score is not None), key=lambda score: score.share, default=None)

XIAOSHAN = XiaoshanRules('xiaoshan-mahjong', replace(FOUR_HANGZHOU.OPTIONS, pass_pon_same=True),
                        feed_mode='two_total', flip_wildcard=True, floater_can_call=False,
                        open_kong_after_float=True)


class TenWindsGame(HangzhouGame):
    WIND_RESET = -1
    WIND_TARGET = 10
    WIND_ACTION = TEN_WINDS
    WIND_REASON = 'ten_winds_win'

    def __init__(self, seed, dealer, round_wind, *, rules, dealer_streak=0):
        self.ten_winds = [0] * rules.OPTIONS.seat_count
        self.ten_winds_winner = None
        super().__init__(seed, dealer, round_wind, rules=rules, dealer_streak=dealer_streak)

    def _record_wind(self, actor, tile):
        if self.ten_winds[actor] >= 0 and self._wind_tile(tile):
            self.ten_winds[actor] += 1
        else:
            self.ten_winds[actor] = self.WIND_RESET

    def _throw_tile(self, actor, tile):
        self._record_wind(actor, tile)
        super()._throw_tile(actor, tile)

    def _call_discard(self, offer):
        self.ten_winds[self.claim[0]] = self.WIND_RESET
        super()._call_discard(offer)

    def _pay_kong(self, actor, meld):
        if meld.kind != 'gang_an':
            self.ten_winds[actor] = self.WIND_RESET
        super()._pay_kong(actor, meld)

    def legal_actions(self):
        actor, actions = super().legal_actions()
        if (self.phase == 'action' and self.ten_winds[actor] >= self.WIND_TARGET - 1
                and self._can_declare_winds(actor)):
            actions.extend((self.WIND_ACTION, tile) for kind, tile in list(actions)
                           if kind in ('discard', FLOAT[0]) and self._wind_tile(tile))
        return actor, actions

    def _can_declare_winds(self, actor):
        return True

    def step(self, action):
        if action[0] != self.WIND_ACTION:
            return super().step(action)
        actor, legal = self.legal_actions()
        if action not in legal:
            raise ValueError('Illegal ten-winds declaration')
        self.last_call = None
        tile = action[1]
        self.players[actor].hand[tile] -= 1
        self.discards[actor].append(tile)
        self._record_wind(actor, tile)
        self.ten_winds_winner = actor
        self._settle([actor], None)
        self.reason_code = self.WIND_REASON

    def _ten_winds_score(self, actor):
        return HangzhouScore(4 << (self.ten_winds[actor] - 10), ('十风',))


class XiaoshanGame(TenWindsGame):
    def __init__(self, seed, dealer, round_wind, *, rules, dealer_streak=0):
        self.last_kong_kind = None
        super().__init__(seed, dealer, round_wind, rules=rules, dealer_streak=dealer_streak)
        self.dealer_multiplier = 4 << min(dealer_streak, 1)

    def _wind_tile(self, tile):
        return tile == WHITE or tile >= 27 or self.physical_tile(tile) == WHITE

    def _chows(self, player, tile, provider):
        return chow_choices(player.hand, tile, swap_lock=self.rules.chow_swap_lock(player.melds, provider))

    def _pay_kong(self, actor, meld):
        self.last_kong_kind = meld.kind
        super()._pay_kong(actor, meld)

    def _score(self, actor, hand, melds, ctx):
        if actor == self.ten_winds_winner:
            return self._ten_winds_score(actor)
        return self.rules.faan(hand, melds, ctx, floats=self.float_streak[actor], last_kong=self.last_kong_kind)

    def payments(self, actor, share, provider=None):
        base = super().payments(actor, share, provider)
        kongs = sum(meld.kind.startswith('gang_') for meld in self.players[actor].melds)
        return tuple(min(share + kongs, XIAOSHAN_CAP) if share else 0 for share in base)


_SPACED_RANKS = tuple(ranks for count in range(4) for ranks in combinations(range(9), count)
                      if all(second - first >= 3 for first, second in zip(ranks, ranks[1:])))


@lru_cache(maxsize=512)
def spaced_capacity(ranks):
    return max((len(candidate) for candidate in _SPACED_RANKS if all(rank in candidate for rank in ranks)), default=-1)


def seven_winds_shape(physical, wild):
    if max(physical) > 1 or physical[27:].count(0) > wild:
        return False
    capacities = [spaced_capacity(tuple(rank for rank in range(9) if physical[suit + rank]))
                  for suit in (0, 9, 18)]
    return min(capacities) >= 0 and sum(capacities) >= 7


def _orphans_and_seven_winds(physical, wild, unit):
    if not any(n for tile, n in enumerate(physical) if tile not in ORPHANS):
        missing = sum(not physical[tile] for tile in ORPHANS)
        if missing <= wild and max(physical) <= 2:
            yield ('不正规十三幺', unit) if wild else ('正规十三幺', 2 * unit)
    if seven_winds_shape(physical, wild):
        yield ('不正规七风', 2 * unit) if wild else ('正规七风', 4 * unit)


class SongyangRules(HangzhouRules):
    WALL_RESERVE = 20
    WIN_KINDS = ('zimo', 'rob_kong')
    ROB_MULTIPLIER = 4
    SPECIAL_UNIT = 2
    UNSTRUCTURED_HONORS = 8

    @staticmethod
    def _pairs_pattern(natural):
        return '七对子', 2

    def faan(self, hand, melds, ctx, *, floats=0, kongs_before_float=0, indicator=None):
        if (ctx.kind not in self.WIN_KINDS or ctx.tile is None or not hand[ctx.tile]
                or sum(hand) != 14 - 3 * len(melds)):
            return None
        natural, wild = split_wild(hand)
        suits, honors = _filled_families(natural, melds)
        before = without_tile(tuple(hand), ctx.tile)
        standard_burst, pairs_burst = burst_wait(before, len(melds)) if ctx.kind == 'zimo' else (False, False)
        color = [('全字', 8)] if not suits else ([('混一色', 2) if honors else ('清一色', 4)] if len(suits) == 1 else [])
        timing = [('杠上花', 2)] if ctx.kind == 'zimo' and ctx.kong_chain else []
        if ctx.kind == 'rob_kong':
            timing.append(('抢杠', self.ROB_MULTIPLIER))
        if ctx.opening:
            timing.append((ctx.opening, 2))

        def candidate(form, burst=False):
            bonus = []
            if burst:
                name = ('财神头', '财飘', '双财飘')[floats]
                bonus = [(name, 2 << floats)]
            return _pattern_score([*form, *color, *timing, *bonus])

        scores = []
        if standard_burst or can_form_standard(natural, wild):
            single = [('单钓', 2)] if len(melds) == 4 else []
            scores.append(candidate(single, standard_burst))
            if all(meld.kind != 'chi' for meld in melds) and can_form_all_pungs(natural, wild):
                before_natural, before_wild = split_wild(before)
                pung_burst = standard_burst and can_form_pungs(before_natural, before_wild - 1)
                scores.append(candidate([('大碰胡', 2), *single], pung_burst))
        elif not suits:
            scores.append(_pattern_score([('全字', self.UNSTRUCTURED_HONORS), *timing]))
        if not melds:
            if can_form_pairs(natural, wild):
                scores.append(candidate([self._pairs_pattern(natural)], pairs_burst))
            physical = list(natural)
            if indicator is not None and indicator != WHITE:
                physical[WHITE], physical[indicator] = physical[indicator], 0
            scores.extend(_pattern_score([special, *timing])
                          for special in _orphans_and_seven_winds(physical, wild, self.SPECIAL_UNIT))
            if ctx.kind == 'zimo' and (ctx.opening or ctx.first_nondealer_draw) and wild == 3:
                scores.append(HangzhouScore(2, ('三财神',)))
        return max(scores, key=lambda score: score.share, default=None)


SONGYANG = SongyangRules('songyang-mahjong', FOUR_HANGZHOU.OPTIONS, fixed_dealer=2,
                        feed_mode='room_liability', flip_wildcard=True, floater_can_call=False,
                        open_kong_after_float=True)


class SongyangGame(HangzhouGame):
    def _replacement_index(self):
        return self.rules.WALL_RESERVE

    def _can_take_win(self, actor, offer):
        return offer.kind == 'gang_bu' and self.float_cycle_owner is None

    def _score(self, actor, hand, melds, ctx):
        return self.rules.faan(hand, melds, ctx, floats=self.float_streak[actor], indicator=self.wildcard_indicator)

    def payments(self, actor, share, provider=None):
        normal = [0 if seat == actor else share * (2 if self.dealer in (actor, seat) else 1)
                  for seat in range(self.seat_count)]
        total = sum(normal)
        if provider is not None:
            return tuple(total if seat == provider else 0 for seat in range(self.seat_count))
        for caller, feeder in self.related_seats:
            if actor in (caller, feeder):
                liable, factor = (feeder, 1) if actor == caller else (caller, 2)
                normal[liable] = total * factor
                break
        return tuple(normal)


class SuichangRules(SongyangRules):
    WALL_RESERVE = 34
    WIN_KINDS = ('zimo', 'ron', 'rob_kong')
    ROB_MULTIPLIER = 2
    SPECIAL_UNIT = 1
    UNSTRUCTURED_HONORS = 4

    @staticmethod
    def _pairs_pattern(natural):
        luxury = sum(count == 4 for count in natural)
        return _pair_name(luxury, True), 2 << luxury

    @staticmethod
    def settle_hand(game, winners, offer):
        provider = offer.actor if offer is not None else None
        return [(winner.score, game.payments(winner.actor, winner.score.share, provider,
                                             rob_kong=winner.kind == 'rob_kong')) for winner in winners]


SUICHANG = SuichangRules('suichang-mahjong', FOUR_HANGZHOU.OPTIONS, fixed_dealer=1, flip_wildcard=True,
                        floater_can_call=False, open_kong_after_float=True)


class SuichangGame(SongyangGame):
    def _completed_kongs(self):
        return sum(meld.kind.startswith('gang_') for player in self.players for meld in player.melds)

    @property
    def remaining(self):
        return super().remaining + 2 * self._completed_kongs()

    def _replacement_index(self):
        return self.rules.WALL_RESERVE - 2 * self._completed_kongs()

    def _can_take_win(self, actor, offer):
        return self.float_cycle_owner is None and offer.kind in ('discard', 'gang_bu')

    def payments(self, actor, share, provider=None, *, rob_kong=False):
        if rob_kong:
            return tuple(4 * share if seat == provider else 0 for seat in range(self.seat_count))
        return tuple(0 if seat == actor else share * (2 if provider in (None, seat) else 1)
                     for seat in range(self.seat_count))


def next_wildcard(indicator):
    if indicator < 27:
        suit, rank = divmod(indicator, 9)
        return suit * 9 + (rank + 1) % 9
    if indicator in WINDS:
        return 27 + (indicator - 26) % 4
    return 31 + (indicator - 32) % 3  # Red → Green → White → Red.


_FOUR_WILDCARDS = HangzhouScore(10, ('四财神',))


class DeqingRules(HangzhouRules):
    MULTIPLE_WINS = True
    HEAVEN_SHARE = 20
    NATURAL_FACTOR = 1
    ROB_PAYER_MULTIPLIER = 3
    ROB_USE_HIGHEST = True
    DEALER_DOUBLE_PATTERNS = ('地胡',)

    @staticmethod
    def four_wildcards(dealer):
        return _FOUR_WILDCARDS

    def faan(self, hand, melds, ctx, *, wildcard=WHITE, floats=0, kongs_before_float=0, seen_wildcard=False):
        if ctx.kind not in ('zimo', 'ron', 'rob_kong') or (ctx.kind == 'ron' and ctx.opening != '地胡'):
            return None
        if ctx.tile is None or not hand[ctx.tile] or sum(hand) != 14 - 3 * len(melds):
            return None
        natural, wild = split_wild(hand, wildcard)
        if ctx.kind == 'zimo' and (ctx.opening or ctx.first_nondealer_draw) and wild == 4:
            return self.four_wildcards(ctx.seat_wind == 27)
        pairs = not melds and can_form_pairs(natural, wild)
        burst, pair_burst = (False, False)
        if ctx.kind == 'zimo':
            before = without_tile(tuple(hand), ctx.tile)
            burst, pair_burst = burst_wait(before, len(melds), wildcard)
        if not (burst or pair_burst or pairs or can_form_standard(natural, wild)):
            return None
        base = self.NATURAL_FACTOR if not (seen_wildcard or wild) else 1
        if ctx.kind == 'rob_kong':
            return HangzhouScore(base << (1 + ctx.kong_chain), ('抢杠',))
        if ctx.opening == '天胡':
            return HangzhouScore(self.HEAVEN_SHARE, ('天胡',))
        if ctx.opening == '地胡' or ctx.first_nondealer_draw:
            return HangzhouScore(5, ('地胡',))
        score = _score_shape(base, (), burst or pair_burst, ctx.kong_chain, floats, kongs_before_float)
        pairs_share = 2 * base
        return (HangzhouScore(pairs_share, ('七对子',))
                if pairs and not (burst or pair_burst) and score.share <= pairs_share else score)

    @classmethod
    def settle_hand(cls, game, winners, offer):
        if offer is not None and offer.kind == 'gang_bu':
            highest = max(winner.score.share for winner in winners) if cls.ROB_USE_HIGHEST else None
            outcomes = []
            for winner in winners:
                share = highest if highest is not None else winner.score.share
                bill = tuple(cls.ROB_PAYER_MULTIPLIER * share if seat == offer.actor else 0
                             for seat in range(game.seat_count))
                outcomes.append((replace(winner.score, share=share), bill))
            return outcomes
        winning_seats = {winner.actor for winner in winners}
        outcomes = []
        for winner in winners:
            score = winner.score
            double_dealer = any(pattern in cls.DEALER_DOUBLE_PATTERNS for pattern in score.patterns)
            bill = tuple(0 if seat in winning_seats else score.share * (2 if double_dealer and seat == game.dealer else 1)
                         for seat in range(game.seat_count))
            outcomes.append((score, bill))
        return outcomes


DEQING = DeqingRules('deqing-wukang', FOUR_HANGZHOU.OPTIONS, fixed_dealer=1, flip_wildcard=True,
                     floater_can_call=False, open_kong_after_float=True)


class DeqingGame(TenWindsGame):
    WIND_RESET = 0

    def __init__(self, seed, dealer, round_wind, *, rules, dealer_streak=0):
        super().__init__(seed, dealer, round_wind, rules=rules, dealer_streak=dealer_streak)
        self.opening_score = None
        for offset in range(self.seat_count):
            seat = (dealer + offset) % self.seat_count
            self.opening_score = self._deal_score(seat)
            if self.opening_score is not None:
                self._settle([seat], None)
                break

    def _deal_score(self, actor):
        if self.players[actor].hand[self.wildcard_tile] == 4:
            return self.rules.four_wildcards(actor == self.dealer)
        return None

    def _flip_wildcard(self):
        self.wildcard_indicator = self.indicator_tile = self.wall.pop(0)
        self.wildcard_tile = next_wildcard(self.wildcard_indicator)
        self.seen_wildcard = [bool(player.hand[self.wildcard_tile]) for player in self.players]

    def _draw(self, actor, kong_chain=0, provider=None):
        super()._draw(actor, kong_chain, provider)
        if self.wildcard_indicator is not None and self.phase == 'action' and self.drawn_tile == self.wildcard_tile:
            self.seen_wildcard[actor] = True

    def physical_tile(self, tile):
        return tile

    def _wind_tile(self, tile):
        return tile in WINDS or tile == self.wildcard_tile

    def _chows(self, player, tile, provider):
        return ((start, take, block) for start, take, block in super()._chows(player, tile, provider)
                if self.wildcard_tile not in take)

    def _open_offer(self, offer):
        if offer.kind == 'discard' and self.uninterrupted and self.players[self.dealer].draws == 1:
            offer = replace(offer, opening='地胡')
        super()._open_offer(offer)

    def _can_take_win(self, actor, offer):
        return self.float_cycle_owner is None and (
            offer.kind == 'gang_bu' or (actor != self.dealer and offer.opening == '地胡'))

    def _score(self, actor, hand, melds, ctx):
        if actor == self.ten_winds_winner:
            return self._ten_winds_score(actor)
        if self.opening_score is not None:
            return self.opening_score
        return self.rules.faan(hand, melds, ctx, wildcard=self.wildcard_tile, floats=self.float_streak[actor],
                               kongs_before_float=self.float_kongs[actor], seen_wildcard=self.seen_wildcard[actor])


_OPENING_TEN_WINDS = '起手十风'


class DeqingXinshiRules(DeqingRules):
    HEAVEN_SHARE = 10
    NATURAL_FACTOR = 2
    ROB_PAYER_MULTIPLIER = 6
    ROB_USE_HIGHEST = False
    DEALER_DOUBLE_PATTERNS = ('地胡', '四财神', _OPENING_TEN_WINDS)

    @staticmethod
    def four_wildcards(dealer):
        return _FOUR_WILDCARDS if dealer else HangzhouScore(5, ('四财神',))


DEQING_XINSHI = DeqingXinshiRules('deqing-xinshi', DEQING.OPTIONS, fixed_dealer=1, flip_wildcard=True,
                                  floater_can_call=False, open_kong_after_float=True)


class DeqingXinshiGame(DeqingGame):
    def __init__(self, seed, dealer, round_wind, *, rules, dealer_streak=0):
        self.pon_upgrade = None
        super().__init__(seed, dealer, round_wind, rules=rules, dealer_streak=dealer_streak)

    def _wind_count(self, hand):
        return sum(count for tile, count in enumerate(hand) if self._wind_tile(tile))

    def _deal_score(self, actor):
        four_wildcards = super()._deal_score(actor)
        if four_wildcards is not None:
            return four_wildcards
        if self._wind_count(self.players[actor].hand) >= 10:
            return HangzhouScore(10 if actor == self.dealer else 5, (_OPENING_TEN_WINDS,))
        return None

    def _win(self, actor, offer=None, winner_count=0):
        player = self.players[actor]
        if (offer is not None and offer.kind == 'discard' and offer.opening == '地胡'
                and offer.actor == self.dealer and player.draws == 0 and offer.tile in WINDS
                and offer.tile != self.wildcard_tile and self._wind_count(player.hand) == 9):
            hand = player.hand.copy()
            hand[offer.tile] += 1
            return HangzhouScore(5, (_OPENING_TEN_WINDS,)), hand, 'ron'
        return super()._win(actor, offer, winner_count)

    def _call_discard(self, offer):
        actor, kind, _ = self.claim
        super()._call_discard(offer)
        self.pon_upgrade = offer.tile if kind == 'peng' and self.players[actor].hand[offer.tile] else None

    def _throw_tile(self, actor, tile):
        self.pon_upgrade = None
        super()._throw_tile(actor, tile)

    def _immediate_kong(self):
        if self.phase == 'action' and self.drawn_tile is None and self.pon_upgrade is not None and self.remaining:
            return 'bugang', self.pon_upgrade
        return None

    def legal_actions(self):
        actor, actions = super().legal_actions()
        kong = self._immediate_kong()
        if kong is not None:
            actions.append(kong)
        return actor, actions

    def step(self, action):
        kong = self._immediate_kong()
        if kong is not None and action == kong:
            self.pon_upgrade = None
            self.last_call = None
            self._finish_own_kong(Offer(self.cur, action[1], 'gang_bu'))
        else:
            super().step(action)
