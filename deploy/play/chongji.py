"""Bianfeng Chongji ordinary points and Seven Gaps rooms."""

from dataclasses import replace
from functools import lru_cache

from .dongyang import isolated_shape
from .hangzhou import FOUR_HANGZHOU, WHITE, HangzhouRules, HangzhouScore, burst_wait, split_wild
from .honors import ZIMO, KongPoints
from .wildcard_shapes import can_form_all_pungs, can_form_pairs, can_form_standard, without_tile
from .zhejiang_flipped import ELEVEN_WINDS, TenWindsGame, next_wildcard


FULL_PAYMENT = 100
FULL_PATTERNS = ('十一风', '三杠全缴')


@lru_cache(maxsize=4096)
def seven_gap_pairs(natural, wild):
    if any(natural[27:]):
        return False
    unpaired = 0
    for start in (0, 9, 18):
        for parity in (0, 1):
            pending = 0
            for rank in range(parity, 9, 2):
                count = natural[start + rank]
                matched = min(pending, count)
                unpaired += pending - matched
                pending = count - matched
            unpaired += pending
    return unpaired <= wild and (wild - unpaired) % 2 == 0


def _kong_pair_pattern(natural, wild, melds):
    n = len(melds)
    if not 1 <= n <= 3 or any(not meld.kind.startswith('gang_') for meld in melds):
        return None
    singles = sum(count % 2 for count in natural)
    matched = min(singles, wild)
    pairs = sum(count // 2 for count in natural) + matched + (wild - matched) // 2
    return ('一杠一达', '二杠二达', '三杠三达')[n - 1] if pairs >= 7 - 2 * n else None


class ChongjiRules(HangzhouRules):
    SWAP_LOCK = True
    PON_SWAP_LOCK = True
    SEVEN_GAPS = False

    @staticmethod
    def next_dealer(dealer, winners, provider):
        return dealer if winners and winners[0] == dealer else (dealer + 1) % 4

    def faan(self, hand, melds, ctx, *, wildcard=WHITE, floated=False):
        if (ctx.kind not in ('zimo', 'ron', 'rob_kong') or ctx.tile is None or not hand[ctx.tile]
                or sum(hand) != 14 - 3 * len(melds)):
            return None
        self_draw = ctx.kind == 'zimo'
        ordinary_ron = ctx.kind == 'ron' and ctx.opening != '地胡'
        natural, wild = split_wild(hand, wildcard)
        kongs = sum(meld.kind.startswith('gang_') for meld in melds)
        if self_draw and kongs >= 3:
            return HangzhouScore(FULL_PAYMENT, ('三杠全缴',))
        before = without_tile(tuple(hand), ctx.tile)
        burst = (self_draw or ctx.kind == 'rob_kong') and any(burst_wait(before, len(melds), wildcard))
        standard = can_form_standard(natural, wild)
        pairs = not melds and can_form_pairs(natural, wild)
        isolated = not melds and isolated_shape(natural, wild)
        gaps = self.SEVEN_GAPS and not melds and seven_gap_pairs(natural, wild)
        kong_pairs = _kong_pair_pattern(natural, wild, melds)
        used = {tile for tile, count in enumerate(natural) if count}
        used.update(tile for meld in melds for tile in meld.tiles)
        suits = {tile // 9 for tile in used if tile < 27}
        all_honors = not suits
        honors = any(tile >= 27 for tile in used)
        pure = standard and len(suits) == 1 and not honors
        single = standard and len(melds) == 4
        four_wild = self_draw and wild == 4
        if ordinary_ron and (single or pure or (pairs and not wild)):
            return None
        if not (standard or pairs or isolated or gaps or kong_pairs or all_honors or four_wild):
            return None

        bonuses = []
        if not wild:
            bonuses.append(('无财神', 2))
        if pairs:
            bonuses.append(('七对子', 1) if wild else ('清七对', 10))
        if isolated:
            bonuses.append(('十三不搭', 1))
            if sum(natural[27:]) == 7:
                bonuses.append(('七风', 1))
        if gaps:
            bonuses.append(('七嵌', 1))
        if kong_pairs:
            bonuses.append((kong_pairs, 1))
        if all_honors:
            bonuses.append(('全风向', 100))
        if standard and len(suits) == 1:
            bonuses.append(('清一色', 10) if pure else ('混一色', 1))
        if standard and all(meld.kind != 'chi' for meld in melds) and can_form_all_pungs(natural, wild):
            bonuses.append(('大对子', 1))
        if single:
            bonuses.append(('单钓', 10))
        if burst and (floated or single):
            bonuses.append(('抛龙', 10))
        if four_wild:
            bonuses.append(('四财神', 10))
        if ctx.opening:
            bonuses.append((ctx.opening, 10))
        if ctx.kind == 'rob_kong':
            bonuses.append(('抢杠', 0))
        if self_draw and ctx.kong_chain:
            bonuses.append(('杠暴', 10))
        if self_draw and ctx.last_live_draw:
            bonuses.append(('海底', 10))
        return HangzhouScore(sum(amount for _, amount in bonuses), tuple(name for name, _ in bonuses) or ('平胡',))

    @staticmethod
    def settle_hand(game, winners, offer):
        outcomes = []
        for winner in winners:
            score = winner.score
            full = any(pattern in FULL_PATTERNS for pattern in score.patterns)
            share = FULL_PAYMENT if full else game.dealer_stake + score.share
            payments = [share] * game.seat_count
            if offer is not None and offer.kind == 'gang_bu':
                payments = [3 * share if seat == offer.actor else 0 for seat in range(game.seat_count)]
            elif offer is not None and '地胡' not in score.patterns:
                payments = [(share + 1) // 2] * game.seat_count
                payments[offer.actor] = share
            payments[winner.actor] = 0
            outcomes.append((replace(score, share=share), tuple(payments)))
        return outcomes


class SevenGapsRules(ChongjiRules):
    SEVEN_GAPS = True


_OPTIONS = replace(FOUR_HANGZHOU.OPTIONS, host_deals_first=False, chow_lock_extensions=False,
                   allow_locked_chow_discard=True)
CHONGJI = ChongjiRules('chongji-ten-ten', _OPTIONS, fixed_dealer=1, flip_wildcard=True)
CHONGJI_SEVEN = SevenGapsRules('chongji-seven-gaps', replace(_OPTIONS, kong_points=KongPoints(20, 20, 20)),
                              fixed_dealer=1, flip_wildcard=True)


class ChongjiGame(TenWindsGame):
    WIND_TARGET = 11
    WIND_ACTION = ELEVEN_WINDS
    WIND_REASON = 'eleven_winds_win'
    WIND_RESET = 0

    def __init__(self, seed, dealer, round_wind, *, rules, dealer_streak=0):
        self.dealer_stake = 11 + dealer_streak * 10
        self.skipped_draw_win = set()
        self.discarded_for_burst = [False] * rules.OPTIONS.seat_count
        super().__init__(seed, dealer, round_wind, rules=rules, dealer_streak=dealer_streak)

    def _flip_wildcard(self):
        self.wildcard_indicator = self.indicator_tile = self.wall.pop(0)
        self.wildcard_tile = next_wildcard(self.wildcard_indicator)

    def physical_tile(self, tile):
        return tile

    def _has_float_wait(self, hand, meld_count):
        # Throwing a wildcard alone never starts a forced-discard circuit in this room.
        return False

    def _wind_tile(self, tile):
        return tile >= 27

    def _can_declare_winds(self, actor):
        return not self.players[actor].melds

    def _ten_winds_score(self, actor):
        return HangzhouScore(FULL_PAYMENT, ('十一风',))

    def _can_take_win(self, actor, offer):
        return actor not in self.skipped_draw_win and offer.kind in ('discard', 'gang_bu')

    def _chows(self, player, tile, provider):
        return ((start, take, block) for start, take, block in super()._chows(player, tile, provider)
                if self.wildcard_tile not in take)

    def _call_discard(self, offer):
        actor, kind, _ = self.claim
        self.discarded_for_burst[actor] = False
        super()._call_discard(offer)
        if kind == 'peng':
            self.forbidden_discards = frozenset((offer.tile,))
        if not any(count and tile not in self.forbidden_discards
                   for tile, count in enumerate(self.players[actor].hand)):
            self.forbidden_discards = frozenset()

    def _draw(self, actor, kong_chain=0, provider=None):
        self.skipped_draw_win.discard(actor)
        super()._draw(actor, kong_chain, provider)

    def step(self, action):
        if self.phase == 'action' and action[0] == 'discard':
            actor, actions = self.legal_actions()
            if action in actions:
                player = self.players[actor]
                skip = ZIMO in actions
                if skip:
                    self.skipped_draw_win.add(actor)
                before = without_tile(tuple(player.hand), action[1])
                self.discarded_for_burst[actor] = skip and any(burst_wait(before, len(player.melds), self.wildcard_tile))
        super().step(action)

    def _pay_kong(self, actor, meld):
        points = self.options.kong_points
        if points is not None and meld.kind == 'gang_ming':
            self.ten_winds[actor] = self.WIND_RESET
            bill = tuple(0 if seat == actor else points.direct if seat == meld.provider else points.direct // 2
                         for seat in range(self.seat_count))
            self._transfer(actor, bill, self.kong_deltas)
        else:
            super()._pay_kong(actor, meld)

    def _score(self, actor, hand, melds, ctx):
        if actor == self.ten_winds_winner:
            return self._ten_winds_score(actor)
        return self.rules.faan(hand, melds, ctx, wildcard=self.wildcard_tile, floated=self.discarded_for_burst[actor])
