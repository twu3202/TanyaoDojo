"""Dongyang's flipped wildcard, optional White Dragon substitution and shared discard payments."""

from dataclasses import replace
from functools import lru_cache

from .hangzhou import FOUR_HANGZHOU, WHITE, HangzhouGame, HangzhouRules, HangzhouScore, split_wild
from .honors import WINDS
from .wildcard_shapes import can_form_all_pungs, can_form_pairs, can_form_sets, can_form_standard, without_tile
from .zhejiang_flipped import spaced_capacity


@lru_cache(maxsize=16384)
def white_assignments(natural, wildcard):
    choices = [natural]
    for count in range(1, natural[WHITE] + 1):
        remapped = list(natural)
        remapped[WHITE] -= count
        remapped[wildcard] += count
        choices.append(tuple(remapped))
    return tuple(choices)


def burst_forms(hand, meld_count, wildcard):
    natural, wild = split_wild(hand, wildcard)
    if not wild or sum(hand) != 13 - 3 * meld_count:
        return False, False
    choices = white_assignments(natural, wildcard)
    return (any(can_form_sets(counts, wild - 1) for counts in choices),
            not meld_count and any(can_form_pairs(counts, wild - 1) for counts in choices))


def _loose_honors(natural, wild, minimum_honors, exact):
    if max(natural) > 1:
        return False
    suits = [tuple(rank for rank in range(9) if natural[start + rank]) for start in (0, 9, 18)]
    if exact:
        capacities = [3 if len({rank % 3 for rank in ranks}) <= 1 else -1 for ranks in suits]
    else:
        capacities = [spaced_capacity(ranks) for ranks in suits]
    honors = sum(natural[27:])
    return min(capacities) >= 0 and max(minimum_honors, honors, 14 - sum(capacities)) <= min(7, honors + wild)


def _loose_scores(natural, wild, wildcard):
    if _loose_honors(natural, wild, 5, False):
        yield HangzhouScore(1, ('十三不搭',))
    if _loose_honors(natural, wild, 5, True):
        yield HangzhouScore(2, ('真十三不搭',))
    seven_honors = natural
    if wildcard in WINDS:
        if natural[WHITE] != 2:
            return
        seven_honors = white_assignments(natural, wildcard)[1]
    if _loose_honors(seven_honors, wild, 7, True):
        yield HangzhouScore(4, ('真十三不搭', '七风'))
    elif _loose_honors(seven_honors, wild, 7, False):
        yield HangzhouScore(2, ('七风',))


class DongyangRules(HangzhouRules):
    @staticmethod
    def next_dealer(dealer, winners, provider):
        return winners[0] if winners and winners[0] != dealer else (dealer + 1) % 4

    def faan(self, hand, melds, ctx, *, wildcard=WHITE, floats=0, kongs_before_float=0):
        self_draw = ctx.kind == 'zimo'
        if (ctx.kind not in ('zimo', 'ron') or ctx.tile is None or not hand[ctx.tile]
                or sum(hand) != 14 - 3 * len(melds)):
            return None
        natural, wild = split_wild(hand, wildcard)
        choices = white_assignments(natural, wildcard)
        before = without_tile(tuple(hand), ctx.tile)
        burst, pair_burst = burst_forms(before, len(melds), wildcard) if self_draw else (False, False)
        standard = burst or any(can_form_standard(counts, wild) for counts in choices)
        pairs = not melds and (pair_burst or any(can_form_pairs(counts, wild) for counts in choices))
        scores = list(_loose_scores(natural, wild, wildcard)) if not melds else []

        if pairs:
            scores.append(HangzhouScore(4 if pair_burst else 2, ('七客',) if pair_burst else ('七对子',)))
        meld_tiles = {tile for meld in melds for tile in meld.tiles}
        if not any(natural[:27]) and all(tile >= 27 for tile in meld_tiles):
            structured = can_form_standard(natural, wild) or (
                wildcard >= 27 and any(can_form_standard(counts, wild) for counts in choices))
            scores.append(HangzhouScore(16, ('正风',)) if structured else HangzhouScore(8, ('乱风',)))
        if standard:
            for counts in choices:
                used = meld_tiles | {tile for tile, count in enumerate(counts) if count}
                if used and max(used) < 27 and len({tile // 9 for tile in used}) == 1 and can_form_standard(counts, wild):
                    scores.append(HangzhouScore(4, ('清一色',)))
                    break
            if len(melds) == 3 and all(meld.kind in ('peng', 'gang_bu') for meld in melds):
                if any(can_form_all_pungs(counts, wild) for counts in choices):
                    scores.append(HangzhouScore(2, ('大碰胡',)))
            if len(melds) == 4 and not before[wildcard] and not self_draw:
                scores.append(HangzhouScore(2, ('全求人',)))
            if self_draw:
                scores.append(HangzhouScore(2, ('爆头',)) if burst else HangzhouScore(1, ('平胡',)))

        if (scores or standard) and ctx.opening:
            scores.insert(0, HangzhouScore(10, (ctx.opening,)))
        if self_draw and (scores or standard):
            if ctx.kong_chain:
                share = 4 if (burst or pair_burst) and ctx.kong_provider is not None else 2
                scores.append(HangzhouScore(share, ('杠暴',) if burst or pair_burst else ('杠开',)))
            if (burst or pair_burst) and floats:
                kong_float = bool(ctx.kong_chain or kongs_before_float)
                share = 2 + floats + (kong_float and wild + floats == 3)
                scores.append(HangzhouScore(share, ('杠飘',) if kong_float else ('财飘',)))
        if self_draw and wild == 3:
            scores.append(HangzhouScore(1, ('三财神',)))
        return max(scores, key=lambda score: score.share, default=None)

    @staticmethod
    def settle_hand(game, winners, offer):
        provider = offer.actor if offer is not None else None
        outcomes = []
        for winner in winners:
            share = winner.score.share
            payments = [share if provider is None else share / 2] * game.seat_count
            if provider is not None:
                payments[provider] = share
            payments[winner.actor] = 0
            outcomes.append((winner.score, tuple(payments)))
        return outcomes


DONGYANG = DongyangRules('dongyang-mahjong', replace(FOUR_HANGZHOU.OPTIONS, host_deals_first=False),
                         fixed_dealer=1, flip_wildcard=True, floater_can_call=False, open_kong_after_float=True)


class DongyangGame(HangzhouGame):
    def _flip_wildcard(self):
        self.wildcard_indicator = self.indicator_tile = self.wildcard_tile = self.wall.pop(0)

    def physical_tile(self, tile):
        return tile

    def _has_float_wait(self, hand, meld_count):
        return any(burst_forms(hand, meld_count, self.wildcard_tile))

    def _chows(self, player, tile, provider):
        return ((start, take, block) for start, take, block in super()._chows(player, tile, provider)
                if self.wildcard_tile not in take)

    def _can_take_win(self, actor, offer):
        return offer.kind == 'discard' and self.float_cycle_owner is None

    def _score(self, actor, hand, melds, ctx):
        return self.rules.faan(hand, melds, ctx, wildcard=self.wildcard_tile, floats=self.float_streak[actor],
                               kongs_before_float=self.float_kongs[actor])
