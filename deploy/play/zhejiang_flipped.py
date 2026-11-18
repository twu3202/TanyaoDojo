"""Xiaoshan and Songyang rooms over the common flipped-wildcard game."""

from dataclasses import replace
from functools import lru_cache
from itertools import combinations
from math import prod

from .hangzhou import (FLOAT, WHITE, FOUR_HANGZHOU, HangzhouGame, HangzhouRules, HangzhouScore,
                       burst_wait, split_wild)
from .honors import ORPHANS, chow_choices
from .wildcard_shapes import (can_form_all_pungs, can_form_pairs, can_form_pungs, can_form_standard,
                             without_tile)


TEN_WINDS = 'ten_winds'
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


class XiaoshanGame(HangzhouGame):
    def __init__(self, seed, dealer, round_wind, *, rules, dealer_streak=0):
        self.ten_winds = [0] * rules.OPTIONS.seat_count
        self.ten_winds_winner = None
        self.last_kong_kind = None
        super().__init__(seed, dealer, round_wind, rules=rules, dealer_streak=dealer_streak)
        self.dealer_multiplier = 4 << min(dealer_streak, 1)

    def _wind_tile(self, tile):
        return tile == WHITE or tile >= 27 or self.physical_tile(tile) == WHITE

    def _record_wind(self, actor, tile):
        if self.ten_winds[actor] >= 0 and self._wind_tile(tile):
            self.ten_winds[actor] += 1
        else:
            self.ten_winds[actor] = -1

    def _throw_tile(self, actor, tile):
        self._record_wind(actor, tile)
        super()._throw_tile(actor, tile)

    def _call_discard(self, offer):
        self.ten_winds[self.claim[0]] = -1
        super()._call_discard(offer)

    def _chows(self, player, tile, provider):
        return chow_choices(player.hand, tile, swap_lock=self.rules.chow_swap_lock(player.melds, provider))

    def _pay_kong(self, actor, meld):
        self.last_kong_kind = meld.kind
        if meld.kind != 'gang_an':
            self.ten_winds[actor] = -1
        super()._pay_kong(actor, meld)

    def legal_actions(self):
        actor, actions = super().legal_actions()
        if self.phase == 'action' and self.ten_winds[actor] >= 9:
            actions.extend((TEN_WINDS, tile) for kind, tile in list(actions)
                           if kind in ('discard', FLOAT[0]) and self._wind_tile(tile))
        return actor, actions

    def step(self, action):
        if action[0] != TEN_WINDS:
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
        self.reason_code = 'ten_winds_win'

    def _score(self, actor, hand, melds, ctx):
        if actor == self.ten_winds_winner:
            return HangzhouScore(4 << (self.ten_winds[actor] - 10), ('十风',))
        return self.rules.faan(hand, melds, ctx, floats=self.float_streak[actor], last_kong=self.last_kong_kind)

    def payments(self, actor, share, provider=None):
        base = super().payments(actor, share, provider)
        kongs = sum(meld.kind.startswith('gang_') for meld in self.players[actor].melds)
        return tuple(min(share + kongs, XIAOSHAN_CAP) if share else 0 for share in base)


_SPACED_RANKS = tuple(ranks for count in range(4) for ranks in combinations(range(9), count)
                      if all(second - first >= 3 for first, second in zip(ranks, ranks[1:])))


@lru_cache(maxsize=512)
def _spaced_capacity(ranks):
    return max((len(candidate) for candidate in _SPACED_RANKS if all(rank in candidate for rank in ranks)), default=-1)


def _seven_winds(physical, wild):
    if max(physical) > 1 or physical[27:].count(0) > wild:
        return False
    capacities = [_spaced_capacity(tuple(rank for rank in range(9) if physical[suit + rank]))
                  for suit in (0, 9, 18)]
    return min(capacities) >= 0 and sum(capacities) >= 7


def _songyang_specials(physical, wild):
    if not any(n for tile, n in enumerate(physical) if tile not in ORPHANS):
        missing = sum(not physical[tile] for tile in ORPHANS)
        if missing <= wild and max(physical) <= 2:
            yield ('不正规十三幺', 2) if wild else ('正规十三幺', 4)
    if _seven_winds(physical, wild):
        yield ('不正规七风', 4) if wild else ('正规七风', 8)


class SongyangRules(HangzhouRules):
    WALL_RESERVE = 20

    def faan(self, hand, melds, ctx, *, floats=0, kongs_before_float=0, indicator=None):
        if ctx.kind not in ('zimo', 'rob_kong') or ctx.tile is None or not hand[ctx.tile] or sum(hand) != 14 - 3 * len(melds):
            return None
        natural, wild = split_wild(hand)
        suits, honors = _filled_families(natural, melds)
        before = without_tile(tuple(hand), ctx.tile)
        standard_burst, pairs_burst = burst_wait(before, len(melds)) if ctx.kind == 'zimo' else (False, False)
        color = [('全字', 8)] if not suits else ([('混一色', 2) if honors else ('清一色', 4)] if len(suits) == 1 else [])
        timing = [('杠上花', 2)] if ctx.kind == 'zimo' and ctx.kong_chain else []
        if ctx.kind == 'rob_kong':
            timing.append(('抢杠', 4))
        if ctx.opening:
            timing.append(('天胡', 2))

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
            scores.append(candidate([]))
        if not melds:
            if can_form_pairs(natural, wild):
                scores.append(candidate([('七对子', 2)], pairs_burst))
            physical = list(natural)
            if indicator is not None and indicator != WHITE:
                physical[WHITE], physical[indicator] = physical[indicator], 0
            scores.extend(_pattern_score([special, *timing]) for special in _songyang_specials(physical, wild))
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
