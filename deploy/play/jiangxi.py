"""Golden Island Nanchang and Tongcheng Shangrao, with separate reward ledgers."""

from collections import Counter
from dataclasses import dataclass, replace
from random import Random

from .hangzhou import split_wild
from .honors import ORPHANS, PASS, HonorsGame, KongPoints, Meld, TableOptions
from .jiangxi_shapes import HONOR_CHOW_ID, HONOR_CHOWS, can_form_spaced, honor_sets, honor_standard, split_jing
from .liaoning_joker import next_face
from .liaoning_split import JianchangRules, SplitScore
from .northeast import NortheastRules
from .wildcard_shapes import can_form_all_pungs, can_form_pairs, can_form_sets, can_form_standard, without_tile


@dataclass(frozen=True)
class HonorChow(Meld):
    @property
    def tiles(self):
        return HONOR_CHOWS[self.tile - HONOR_CHOW_ID]


@dataclass(frozen=True)
class JingScore:
    base: int
    extra: int
    patterns: tuple[str, ...]
    fan = None
    tier = None

    @property
    def share(self):
        return self.base + self.extra


def jing_chows(hand, offered):
    if offered < 27:
        return
    for i, group in enumerate(HONOR_CHOWS):
        if offered not in group:
            continue
        take = tuple(face for face in group if face != offered)
        if all(hand[face] for face in take):
            yield HONOR_CHOW_ID + i, take, frozenset()


def hand_copies(hand, melds):
    counts = Counter({face: n for face, n in enumerate(hand) if n})
    counts.update(t for meld in melds for t in meld.tiles)
    return counts


def deal_bonus(game, raw, ledger, sponsors=()):
    single = sum(bool(amount) for amount in raw) == 1
    for actor, amount in enumerate(raw):
        if not amount:
            continue
        base = amount * (2 if single else 1)
        total = base * max(1, amount - 3)
        boost = total - base
        sponsor = sponsors[actor] if sponsors else None
        if sponsor is None:
            bills = tuple(total if seat != actor else 0 for seat in range(game.seat_count))
        else:
            bills = tuple(base + (game.seat_count - 1) * boost if seat == sponsor else base if seat != actor else 0
                          for seat in range(game.seat_count))
        game._transfer(actor, bills, ledger)


class NanchangRules(NortheastRules):
    require_burst = False
    WALL_RESERVE = 34
    SWAP_LOCK = False
    OPTIONS = TableOptions(orphans=False, seven_pairs=True, concealed_kong_faces=4, dealer_echo=True, echo_share=5,
                           kong_points=KongPoints(2, 2, 2), pass_win_higher_all=True, pass_win_clear_by_call=True)
    next_dealer = staticmethod(JianchangRules.next_dealer)

    def faan(self, hand, melds, ctx, *, wildcard):
        if ctx.kind == 'rob_concealed':
            return None
        hand = tuple(hand)
        jing = (wildcard, next_face(wildcard))
        natural, wild = split_jing(hand, jing)
        if sum(hand) != 14 - 3 * len(melds):
            return None

        modes = []
        if honor_standard(natural, wild):
            modes.append((1, '平胡', honor_standard(hand, 0)))
        if not melds:
            if can_form_pairs(natural, wild):
                modes.append((2, '大七对' if max(hand) == 4 else '小七对', can_form_pairs(hand, 0)))
            if can_form_spaced(natural, wild):
                modes.append((2, '十三烂', can_form_spaced(hand, 0)))
            if all(hand[t] for t in range(27, 34)):
                retained, flexible = split_jing(hand, jing, keep_honors=True)
                if can_form_spaced(retained, flexible):
                    modes.append((4, '七星十三烂', can_form_spaced(hand, 0)))
        if not modes:
            return None

        hanging = False
        if ctx.kind == 'zimo':
            before = without_tile(hand, ctx.tile)
            hanging = any(before[t] and honor_sets(without_tile(before, t), 0) for t in jing)
        factor, event = 1, None
        if ctx.opening in ('天胡', '地胡'):
            factor, event = 20, ctx.opening
        elif hanging and ctx.first_nondealer_draw:
            factor, event = 20, '精吊地胡'
        elif ctx.kind == 'zimo':
            factor, event = (4, '杠开') if ctx.kong_chain else (2, '自摸')
        elif ctx.kind == 'rob_kong':
            event = '抢杠胡'
        outcomes = []
        for multiplier, shape, german in modes:
            labels = [shape] + ([event] if event else [])
            if german or hanging:
                labels.append('德国' if german else '精吊')
            special = german or hanging
            outcomes.append(JingScore(factor * multiplier * (2 if special else 1), 5 if special else 0, tuple(labels)))
        return max(outcomes, key=lambda score: score.share)

    @staticmethod
    def payments(game, actor, provider, score):
        return tuple(
            score.base * (2 if game.dealer in (actor, seat) else 1)
            * ((4 if '抢杠胡' in score.patterns else 2) if seat == provider else 1) + score.extra
            if seat != actor else 0 for seat in range(game.seat_count)
        )

    def settle_hand(self, game, winners, offer):
        provider = offer.actor if offer else None
        outcomes = []
        for win in winners:
            score = win.score
            if '德国' in score.patterns and not any(
                    game.jing_credit(win.hand if seat == win.actor else p.hand, p.melds, game.indicator_tile)
                    for seat, p in enumerate(game.players)):
                score = replace(score, base=score.base * 2,
                                patterns=tuple('德中德' if name == '德国' else name for name in score.patterns))
            outcomes.append((score, self.payments(game, win.actor, provider, score)))
        return outcomes


NANCHANG = NanchangRules('nanchang-double-jing')


class NanchangGame(HonorsGame):
    def __init__(self, seed, dealer, round_wind, *, rules=NANCHANG):
        self.flip_rng = Random(seed)
        self.lower_deltas = [0] * rules.OPTIONS.seat_count
        self.jing_deltas = [0] * rules.OPTIONS.seat_count
        self.chong_sponsors = [None] * rules.OPTIONS.seat_count
        super().__init__(seed, dealer, round_wind, rules=rules)

    def _prepare_wall(self):
        self.flip_roll = self.flip_rng.randint(1, 6) + self.flip_rng.randint(1, 6)
        slot = 2 * (self.flip_roll - 1)
        self.indicator_tile, self.lower_indicator_tile = self.wall[slot:slot + 2]
        self.wildcard_tile = self.indicator_tile

    def _deal_opening(self):
        super()._deal_opening()
        raw = [self.jing_credit(p.hand, p.melds, self.lower_indicator_tile) for p in self.players]
        deal_bonus(self, raw, self.lower_deltas)

    def _replacement_index(self):
        return self.rules.WALL_RESERVE

    @staticmethod
    def jing_credit(hand, melds, face):
        counts = hand_copies(hand, melds)
        return 2 * counts[face] + counts[next_face(face)]

    def _chow_meld(self, start, provider):
        return HonorChow('chi', start, provider) if start >= HONOR_CHOW_ID else super()._chow_meld(start, provider)

    def _chows(self, player, tile, provider):
        yield from super()._chows(player, tile, provider)
        yield from jing_chows(player.hand, tile)

    def _kong_factor(self, meld):
        return 5 if meld.tile == next_face(self.indicator_tile) else 1

    def legal_actions(self):
        actor, actions = super().legal_actions()
        if self.opening and self.offer is None:
            actions = [a for a in actions if a[0] not in ('ankan', 'bugang')]
        return actor, actions

    def _response_actions(self):
        if self.winners:
            return self.responses[0], [PASS]
        return super()._response_actions()

    def _pass_win_value(self, actor, offer, score):
        return sum(self.rules.payments(self, actor, offer.actor, score))

    def _discard(self, actor, tile):
        player = self.players[actor]
        if self.jing_credit(without_tile(tuple(player.hand), tile), player.melds, self.indicator_tile) < 5:
            self.chong_sponsors[actor] = None
        super()._discard(actor, tile)

    def _sponsor_crossing(self, actor, offer):
        player = self.players[actor]
        before = self.jing_credit(player.hand, player.melds, self.indicator_tile)
        increase = 2 if offer.tile == self.indicator_tile else int(offer.tile == next_face(self.indicator_tile))
        if before < 5 <= before + increase:
            self.chong_sponsors[actor] = offer.actor

    def _call_discard(self, offer):
        actor = self.claim[0]
        self._sponsor_crossing(actor, offer)
        skipped = (offer.actor + 1) % self.seat_count
        while skipped != actor:
            self.players[skipped].declined_win_share = -1
            skipped = (skipped + 1) % self.seat_count
        super()._call_discard(offer)

    def _score(self, actor, hand, melds, ctx):
        return self.rules.faan(hand, melds, ctx, wildcard=self.indicator_tile)

    def _settle(self, winners, offer):
        if offer is not None:
            for actor in winners:
                self._sponsor_crossing(actor, offer)
        super()._settle(winners, offer)

    def _end(self, reason):
        raw = [self.jing_credit(self.revealed.get(seat, p.hand), p.melds, self.indicator_tile)
               for seat, p in enumerate(self.players)]
        deal_bonus(self, raw, self.jing_deltas, self.chong_sponsors)
        super()._end(reason)


class ShangraoRules(NortheastRules):
    require_burst = False
    WALL_RESERVE = 14
    OPTIONS = TableOptions(orphans=False, seven_pairs=True, concealed_kong_faces=0,
                           kong_points=KongPoints(1, 1, 2))

    def faan(self, hand, melds, ctx, *, wildcard):
        if ctx.kind != 'zimo':
            return None
        hand = tuple(hand)
        natural, wild = split_wild(hand, wildcard)
        if sum(hand) != 14 - 3 * len(melds):
            return None
        candidates = []
        if not melds and can_form_pairs(natural, wild):
            candidates.append((2 if max(hand) == 4 else 1, '豪华七对' if max(hand) == 4 else '七小对', True))
        if not melds and can_form_spaced(natural, wild):
            candidates.append((1, '十三浪', False))
        if can_form_standard(natural, wild):
            all_pungs = all(m.kind != 'chi' for m in melds) and can_form_all_pungs(natural, wild)
            candidates.append((int(all_pungs), '碰碰胡' if all_pungs else '平胡', all_pungs))
        if not candidates:
            return None

        terminals = all(t in ORPHANS for t in hand_copies(natural, melds))
        closed = self.closed(melds)
        luxury = closed and any(m.kind == 'gang_an' for m in melds)
        modes = []
        for fan, label, can_be_terminals in candidates:
            patterns = [label]
            if terminals and can_be_terminals:
                patterns.append('长九幺' if closed else '九幺')
                fan += 2 if closed else 1
                if luxury:
                    patterns.append('豪华长九幺')
                    fan += 1
            modes.append((fan, label, patterns))
        fan, _, patterns = max(modes)
        before = without_tile(hand, ctx.tile)
        if before[wildcard] == 1 and can_form_sets(without_tile(before, wildcard), 0):
            patterns.append('宝吊头')
            fan += 1
        if ctx.kong_chain:
            patterns.append('杠开')
            fan += 1
        return SplitScore(fan, tuple(patterns))

    @staticmethod
    def settle_hand(game, winners, offer):
        win, = winners
        return [(win.score, tuple(0 if i == win.actor else win.score.share for i in range(game.seat_count)))]


SHANGRAO = ShangraoRules('shangrao-bao-self-draw')


class ShangraoGame(HonorsGame):
    def __init__(self, seed, dealer, round_wind, *, rules=SHANGRAO):
        self.flip_rng = Random(seed)
        super().__init__(seed, dealer, round_wind, rules=rules)

    def _prepare_wall(self):
        self.flip_roll = self.flip_rng.randint(1, 6)
        self.indicator_tile = self.wall[2 * (self.flip_roll - 1)]
        self.wildcard_tile = next_face(self.indicator_tile)

    def _replacement_index(self):
        return self.rules.WALL_RESERVE

    def _score(self, actor, hand, melds, ctx):
        return self.rules.faan(hand, melds, ctx, wildcard=self.wildcard_tile)
