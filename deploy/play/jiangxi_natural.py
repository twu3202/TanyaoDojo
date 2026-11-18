"""Yichun and Pingxiang tables with natural tiles and different win gates."""

from dataclasses import replace

from .honors import PASS, HonorsGame, KongPoints, TableOptions, WinContext, seven_pairs, standard_shapes
from .honors_bonus import Score
from .jiangxi import HonorChowGame, hand_copies
from .jiangxi_shapes import can_form_spaced, honor_standard
from .northeast import NortheastRules
from .wildcard_shapes import can_form_all_pungs


TRUE_PAIRS = frozenset(t for t in range(27) if t % 9 in (1, 4, 7))
SAME_SUIT_FEED = 'same_suit_chows'


def single_numbered_suit(faces):
    return bool(faces) and max(faces) < 27 and min(faces) // 9 == max(faces) // 9


class YichunRules(NortheastRules):
    WALL_RESERVE = 0
    SWAP_LOCK = False
    OPTIONS = TableOptions(seven_pairs=True, orphans=False, concealed_kong_faces=0,
                           pass_win_same=True, pass_win_clear_by_call=True, kong_requires_live_tile=True,
                           kong_points=KongPoints(1, 1, 2, single_direct_payer=True, refund_on_draw=True))

    @staticmethod
    def faan(hand, melds, ctx):
        if ctx.kind == 'rob_concealed' or sum(hand) != 14 - 3 * len(melds):
            return None
        hand = tuple(hand)
        standard = honor_standard(hand, 0)
        faces = hand_copies(hand, melds)
        candidates = []
        if min(faces) >= 27:
            candidates.append(Score(12 if standard else 6, ('字一色成阙' if standard else '字一色',)))
        if standard:
            patterns = ['碰碰胡'] if all(m.kind != 'chi' for m in melds) and can_form_all_pungs(hand, 0) else []
            points = 3 if patterns else 1
            if single_numbered_suit(faces):
                patterns.append('清一色')
                points *= 5
            candidates.append(Score(points, tuple(patterns or ['平胡'])))
        if not melds and can_form_spaced(hand, 0):
            stars = all(hand[27:])
            candidates.append(Score(3 if stars else 2, ('七星十三烂' if stars else '十三烂',)))
        if seven_pairs(hand, melds):
            quads = hand.count(4)
            label = ('小七对', '豪华七对', '双豪华七对', '三豪华七对')[quads]
            candidates.append(Score(4 * 2 ** quads, (label,)))
        if not candidates:
            return None

        if ctx.opening in ('天胡', '地胡'):
            return Score(8, (ctx.opening,))
        score = max(candidates, key=lambda row: row.share)
        deltas = []
        if score.patterns[0] not in ('字一色', '字一色成阙') and NortheastRules.closed(melds):
            deltas.append('门清')
        if sum(hand) == 2 and not any(m.kind == 'gang_an' for m in melds):
            deltas.append('单钓')
        if ctx.kind == 'rob_kong':
            deltas.append('抢杠胡')
        elif ctx.kong_chain:
            deltas.append('杠开' if ctx.kind == 'zimo' else '杠上炮')
        if ctx.last_live_draw:
            deltas.append('海底捞月' if ctx.kind == 'zimo' else '海底炮')
        return Score(score.share * 2 ** len(deltas), (*score.patterns, *deltas))

    @staticmethod
    def settle_hand(game, winners, offer):
        return [(win.score, tuple(win.score.share if i != win.actor and (offer is None or i == offer.actor) else 0
                                 for i in range(game.seat_count))) for win in winners]

    @staticmethod
    def next_dealer(dealer, winners, provider):
        return winners[0] if len(winners) == 1 else provider if winners else dealer


YICHUN = YichunRules('yichun-honor-chows')


class PingxiangRules(NortheastRules):
    WALL_RESERVE = 0
    SWAP_LOCK = False
    OPTIONS = TableOptions(seven_pairs=True, orphans=False, concealed_kong_faces=0,
                           kong_requires_live_tile=True,
                           kong_points=KongPoints(1, 1, 1, single_direct_payer=True, refund_on_draw=True,
                                                  added_payer_from_pong=True))

    @staticmethod
    def faan(hand, melds, ctx):
        if ctx.kind == 'rob_concealed' or sum(hand) != 14 - 3 * len(melds):
            return None

        faces = hand_copies(hand, melds)
        pure = single_numbered_suit(faces)
        all_honors = min(faces) >= 27
        no_chow = all(m.kind != 'chi' for m in melds)
        opening = ctx.opening in ('天胡', '地胡')
        candidates = []
        if pure or no_chow:
            for pair, sets in standard_shapes(hand, len(melds)):
                true_pair = pair in TRUE_PAIRS
                pungs = no_chow and true_pair and all(m.kind == 'triplet' for m in sets)
                if pure:
                    patterns = ('清一色',) + (('真将',) if true_pair else ()) + (('碰碰胡',) if pungs else ())
                    candidates.append(Score(16 if pungs else 8 if true_pair else 4, patterns))
                elif pungs:
                    candidates.append(Score(4, ('碰碰胡', '真将')))
                elif true_pair and (ctx.kind == 'zimo' or opening):
                    candidates.append(Score(1, ('平胡', '真将')))
        if seven_pairs(hand, melds):
            quads = hand.count(4)
            label = ('七对', '龙七', '双龙七', '三龙七')[quads]
            patterns = (label,) + (('清一色',) if pure else ('字一色',) if all_honors else ())
            candidates.append(Score(max(8 if pure or all_honors else 4, 4 * 2 ** quads), patterns))
        if not candidates:
            return None
        if opening:
            return Score(32, (ctx.opening,))
        score = max(candidates, key=lambda row: row.share)
        if ctx.kind == 'rob_kong':
            score = replace(score, patterns=(*score.patterns, '抢杠胡'))
        return score

    @staticmethod
    def liability(melds, last):
        if last.kind != 'chi':
            return ()
        matches = sum(m.kind == 'chi' and m.provider == last.provider and m.tile // 9 == last.tile // 9 for m in melds)
        return (SAME_SUIT_FEED,) if matches == 3 else ()

    @staticmethod
    def settle_hand(game, winners, offer):
        win, = winners
        score = win.score
        payer = None if offer is None else offer.actor
        feeder = next((seat for pattern, seat in game.players[win.actor].liabilities if pattern == SAME_SUIT_FEED), None)
        payments = [0] * game.seat_count
        if feeder is not None and win.kind != 'rob_kong':
            payments[feeder] = score.share * (game.seat_count - 1 if payer is None else 1)
            score = replace(score, patterns=(*score.patterns, '包牌'))
        elif payer is not None:
            payments[payer] = score.share
        else:
            payments = [0 if i == win.actor else score.share for i in range(game.seat_count)]
        return [(score, tuple(payments))]

    def natural_win(self, hand, melds, tile):
        return self.faan(hand, melds, WinContext('zimo', tile, 27, 27)) is not None

    @staticmethod
    def next_dealer(dealer, winners, provider):
        return winners[0] if winners else (dealer + 1) % 4


PINGXIANG = PingxiangRules('pingxiang-true-pair')


class PingxiangGame(HonorsGame):
    def __init__(self, seed, dealer, round_wind, *, rules=PINGXIANG):
        self.dealer_ready_deltas = [0] * rules.OPTIONS.seat_count
        self.first_discards = {}
        super().__init__(seed, dealer, round_wind, rules=rules)

    def _response_actions(self):
        if self.winners:
            return self.responses[0], [PASS]
        return super()._response_actions()

    def _discard(self, actor, tile):
        if actor not in self.first_discards:
            self.first_discards[actor] = tile
            if len(self.first_discards) == self.seat_count and len(set(self.first_discards.values())) == 1:
                self._pay_echo(self.dealer)
        super()._discard(actor, tile)

    def _end(self, reason):
        if reason == 'draw':
            p = self.players[self.dealer]
            if self.rules.waits(tuple(p.hand), tuple(p.melds)):
                bills = tuple(int(seat != self.dealer) for seat in range(self.seat_count))
                self._transfer(self.dealer, bills, self.dealer_ready_deltas)
        super()._end(reason)
