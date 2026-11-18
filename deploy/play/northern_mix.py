"""Tianjin dual jokers and Beijing's closed-hand discard wins."""

from dataclasses import replace
from functools import lru_cache
from math import prod
from random import Random

from .hangzhou import split_wild
from .honors import PASS, ZIMO, HonorsGame, KongPoints, TableOptions
from .honors_bonus import Score
from .jiangxi_shapes import split_jing
from .liaoning_joker import next_face
from .northeast import NortheastRules
from .wildcard_shapes import can_form_all_pungs, can_form_pairs, can_form_sets, can_form_standard, without_tile


FIVE = 4
FIVE_SIDES = (3, 5)
RAISE_PHASE = 'dealer_raise'
RAISE_LEVELS = (1, 2, 4)


def after_ranks(natural, wild, required, *, exact_used=None):
    """Consume printed ranks and spend jokers on missing ranks."""
    rest = list(natural)
    spent = 0
    for tile in required:
        if rest[tile]:
            rest[tile] -= 1
        else:
            spent += 1
    if spent > wild:
        return
    if exact_used is None or spent == exact_used:
        yield tuple(rest), wild - spent
    elif spent < exact_used == 1 and wild >= 1:
        for tile in set(required):
            rest[tile] += 1
            yield tuple(rest), wild - 1
            rest[tile] -= 1


def with_ranks(natural, wild, required, *, pairs=True, exact_used=None):
    solver = can_form_standard if pairs else can_form_sets
    return any(solver(rest, jokers) for rest, jokers in after_ranks(natural, wild, required, exact_used=exact_used))


@lru_cache(maxsize=512)
def dragon_ranks(suit, melds, *, draw_in_dragon=None):
    """Show the part of 123/456/789 still required in concealed tiles."""
    start = 9 * suit
    exposed = {m.tile for m in melds if m.kind == 'chi'}
    runs = (start, start + 3, start + 6)
    if draw_in_dragon is not None:
        if not start <= draw_in_dragon < start + 9:
            return None
        drawn_run = start + 3 * ((draw_in_dragon - start) // 3)
        exposed.discard(drawn_run)
    ranks = tuple(t for run in runs if run not in exposed
                  for t in range(run, run + 3) if t != draw_in_dragon)
    return ranks if len(ranks) + int(draw_in_dragon is not None) <= 3 * (4 - len(melds)) else None


def middle_five_dragon(pre, wild, suit, melds):
    outside = dragon_ranks(suit, melds)
    if outside is not None and len(outside) + 3 <= 3 * (4 - len(melds)):
        if with_ranks(pre, wild, (*outside, *FIVE_SIDES)):
            return True
    inside = dragon_ranks(suit, melds, draw_in_dragon=FIVE)
    return inside is not None and with_ranks(pre, wild, inside)


class TianjinRules(NortheastRules):
    WALL_RESERVE = 0
    OPTIONS = TableOptions(chow=False, orphans=False, dealer_echo=True, kong_requires_live_tile=True,
                           concealed_kong_faces=4, kong_points=KongPoints(1, 1, 2))

    @staticmethod
    def mixed_faces(indicator):
        return indicator, next_face(indicator)

    def faan(self, hand, melds, ctx, *, indicator):
        if ctx.kind != 'zimo' or sum(hand) != 14 - 3 * len(melds):
            return None
        mixed = self.mixed_faces(indicator)
        hand = tuple(hand)
        natural, wild = split_jing(hand, mixed)
        if not can_form_standard(natural, wild):
            return None
        pre, before = split_jing(without_tile(hand, ctx.tile), mixed)
        single = before >= 1 and can_form_sets(pre, before - 1)
        double = before >= 2 and can_form_standard(pre, before - 2)
        catch = (ctx.tile == FIVE or ctx.tile in mixed) and with_ranks(pre, before, FIVE_SIDES)
        clean = not wild and all(m.tile not in mixed for m in melds)
        options = [(2 if clean else 1, ('素提溜',) if clean else ('提溜',))]
        if single or double:
            options.append((2, ('双混吊' if double else '混吊',)))
        if catch:
            bonus = ('素提溜',) if clean else ('双混吊',) if double else ()
            options.append((3 * (2 if bonus else 1), ('捉五', *bonus)))

        for suit in range(3):
            ranks = dragon_ranks(suit, melds)
            if ranks is None:
                continue
            part = with_ranks(natural, wild, ranks)
            catch_dragon = catch and middle_five_dragon(pre, before, suit, melds)
            if not part and not catch_dragon:
                continue
            native = indicator < 27 and indicator // 9 == suit
            color = ('本混龙',) if native else ('龙',)
            bonus = ('素提溜',) if clean else ()
            factor = 2 ** (native + clean)
            if part:
                options.append((4 * factor, (*color, *bonus)))
                if before >= 2 and with_ranks(pre, before - 2, ranks):
                    options.append((8 * factor, (*color, '双混吊')))
                elif before >= 1 and with_ranks(pre, before - 1, ranks, pairs=False):
                    options.append((8 * factor, (*color, '混吊')))
            if catch_dragon:
                double_dragon = before >= 2 and with_ranks(pre, before - 2, ranks)
                options.append((7 * factor * (2 if double_dragon else 1),
                                (*color, '捉五', *bonus, *(('双混吊',) if double_dragon else ()))))
        points, names = max(options, key=lambda row: row[0])
        if ctx.kong_chain:
            points *= 2
            names += ('杠开',)
        if ctx.opening == '天胡':
            points += 8
            names += ('天胡',)
        return Score(points, names) if points >= 2 else None

    @staticmethod
    def settle_hand(game, winners, offer):
        win, = winners
        bills = tuple(win.score.share * game.dealer_factor(win.actor, seat) if seat != win.actor else 0
                      for seat in range(game.seat_count))
        return [(win.score, bills)]


TIANJIN = TianjinRules('tianjin-double-hun')


class BeijingRules(NortheastRules):
    WALL_RESERVE = 0
    SWAP_LOCK = False
    OPTIONS = TableOptions(seven_pairs=True, orphans=False, concealed_kong_faces=1,
                           kong_requires_live_tile=True, kong_points=KongPoints(1, 1, 2))

    def faan(self, hand, melds, ctx, *, wildcard, sea=False):
        if sum(hand) != 14 - 3 * len(melds):
            return None
        hand = tuple(hand)
        natural, wild = split_wild(hand, wildcard)
        closed = self.closed(melds)
        if ctx.kind != 'zimo' and (not closed or wild >= 3):
            return None
        if wild == 4:
            return Score(20, ('混杠',))

        pairs = not melds and can_form_pairs(natural, wild)
        standard = can_form_standard(natural, wild)
        if not pairs and not standard:
            return None

        pre, before = split_wild(without_tile(hand, ctx.tile), wildcard)
        distinct = {t // 9 for t, n in enumerate(natural) if n}
        distinct.update(t // 9 for m in melds for t in m.tiles)
        pure = len(distinct) == 1 and 3 not in distinct
        common = [('门清', 2)] if closed else []
        if wild == 0:
            common.append(('素', 2))
        if sea:
            common.append(('海捞', 2))
        if ctx.kind in ('rob_kong', 'rob_concealed'):
            common.append(('抢暗杠' if ctx.kind == 'rob_concealed' else '抢杠胡', 2))
        elif ctx.kong_chain and ctx.kind == 'zimo':
            common.append(('杠瓷', 4))

        catch = ctx.tile == FIVE and with_ranks(pre, before, FIVE_SIDES) and self.only_five(pre, before, hand, melds, wildcard)
        options = []
        if standard and ctx.kind != 'rob_concealed':
            shape = [('碰碰胡', 2)] if all(m.kind != 'chi' for m in melds) and can_form_all_pungs(natural, wild) else []
            options.append(shape)
            if catch:
                options.append([('捉五魁', 2)])
        if pairs and ctx.kind != 'rob_concealed':
            luxury = natural.count(4)
            options.append([(('小七对', '豪华七对', '双豪华七对', '三豪华七对')[luxury], 2 ** (luxury + 1)),
                            *([('捉五魁', 2)] if ctx.tile == FIVE else [])])
        for suit in range(3):
            robbing = ctx.kind == 'rob_concealed'
            ranks = dragon_ranks(suit, melds, draw_in_dragon=ctx.tile if robbing else None)
            if ranks is None:
                continue
            rest, jokers = (pre, before) if robbing else (natural, wild)
            extra = [('本混龙', 4)] if jokers == 1 and with_ranks(rest, jokers, ranks, exact_used=1) else (
                    [('一条龙', 2)] if with_ranks(rest, jokers, ranks) else [])
            if not extra:
                continue
            options.append(extra)
            if catch and not robbing and middle_five_dragon(pre, before, suit, melds):
                options.append([*extra, ('捉五魁', 2)])
        if not options:
            return None
        if ctx.opening in ('天胡', '地胡'):
            return Score(20, (ctx.opening,))

        if pure:
            common.append(('清一色', 4))
        best = max(options, key=lambda tags: prod(n for _, n in tags))
        factors = [*best, *common]
        return Score(prod(n for _, n in factors), tuple(name for name, _ in factors) or ('平胡',))

    @staticmethod
    def only_five(pre, before, hand, melds, wildcard):
        shown = [t for m in melds for t in m.tiles]
        for t in range(34):
            if t == FIVE or hand[t] + shown.count(t) >= 4:
                continue
            row, count = (pre, before + 1) if t == wildcard else (without_tile(pre, t, -1), before)
            if can_form_standard(row, count) or (not melds and can_form_pairs(row, count)):
                return False
        return True

    @staticmethod
    def settle_hand(game, winners, offer):
        win, = winners
        for meld in game.players[win.actor].melds:
            if meld.kind.startswith('gang'):
                game._transfer(win.actor, game._kong_payments(win.actor, meld), game.kong_deltas)
        score = win.score
        if game.upstairs:
            score = replace(score, share=score.share * 2, patterns=(*score.patterns, '上楼'))
        bills = tuple(0 if seat == win.actor else score.share * (2 if game.dealer in (seat, win.actor)
                      and '混杠' not in score.patterns else 1) for seat in range(game.seat_count))
        if offer is not None:
            bills = tuple(sum(bills) if seat == offer.actor else 0 for seat in range(game.seat_count))
        return [(score, bills)]


BEIJING = BeijingRules('beijing-upstairs-mixed')


class NorthernMixGame(HonorsGame):
    DICE = 1

    def __init__(self, seed, dealer, round_wind, *, rules):
        self.flip_rng = Random(seed)
        super().__init__(seed, dealer, round_wind, rules=rules)

    def _prepare_wall(self):
        self.flip_roll = sum(self.flip_rng.randint(1, 6) for _ in range(self.DICE))
        self.reserve = 2 * self.flip_roll
        self.indicator_tile = self.wildcard_indicator = self.wall[self.reserve - 2]
        self.wildcard_tile = next_face(self.indicator_tile)

    @property
    def remaining(self):
        return len(self.wall) - self.reserve

    def _replacement_index(self):
        return self.reserve

    def _response_actions(self):
        if self.winners:
            return self.responses[0], [PASS]
        return super()._response_actions()


class TianjinGame(NorthernMixGame):
    DICE = 2

    def __init__(self, seed, dealer, round_wind, *, rules=TIANJIN):
        self.raises = {dealer: 1}
        self.sealed = set()
        super().__init__(seed, dealer, round_wind, rules=rules)

    @property
    def mixed_faces(self):
        return self.rules.mixed_faces(self.indicator_tile)

    def _deal_opening(self):
        self.phase, self.cur = RAISE_PHASE, (self.dealer + 1) % self.seat_count

    def legal_actions(self):
        if self.phase == RAISE_PHASE:
            return self.cur, [(RAISE_PHASE, n) for n in RAISE_LEVELS]
        return super().legal_actions()

    def step(self, action):
        if self.phase != RAISE_PHASE:
            super().step(action)
            return
        if action not in self.legal_actions()[1]:
            raise ValueError('Illegal Tianjin dealer raise')
        self.raises[self.cur] = action[1]
        if len(self.raises) == self.seat_count:
            super()._deal_opening()
        else:
            self.cur = (self.cur + 1) % self.seat_count

    def _response_actions(self):
        if self.offer.tile in self.mixed_faces:
            return self.responses[0], [PASS]
        return super()._response_actions()

    def _discard(self, actor, tile):
        if tile in self.mixed_faces:
            self.sealed.add(actor)
        super()._discard(actor, tile)

    def _score(self, actor, hand, melds, ctx):
        return None if actor in self.sealed else self.rules.faan(hand, melds, ctx, indicator=self.indicator_tile)

    def dealer_factor(self, actor, opponent):
        return 2 * self.raises[opponent if actor == self.dealer else actor] if self.dealer in (actor, opponent) else 1

    def _kong_payments(self, actor, meld):
        share = 8 if meld.kind == 'gang_an' and meld.tile in self.mixed_faces else 2 if meld.kind == 'gang_an' else 1
        return tuple(share * self.dealer_factor(actor, seat) if seat != actor else 0 for seat in range(self.seat_count))


class BeijingGame(NorthernMixGame):
    def __init__(self, seed, dealer, round_wind, *, rules=BEIJING, dealer_streak=0, previous_draw=False):
        self.upstairs = previous_draw or dealer_streak == 3
        self.burn_sequence = []
        super().__init__(seed, dealer, round_wind, rules=rules)

    def legal_actions(self):
        actor, actions = super().legal_actions()
        if self.phase == 'action' and self.offer is None:
            if ZIMO in actions and self.players[actor].hand[self.wildcard_tile] == 4:
                return actor, [ZIMO]
            actions = [a for a in actions if a != ('discard', self.wildcard_tile)]
        return actor, actions

    def _chows(self, player, tile, provider):
        for start, take, block in super()._chows(player, tile, provider):
            if self.wildcard_tile not in take:
                yield start, take, block

    def _response_actions(self):
        actor, actions = super()._response_actions()
        if self.offer.kind != 'discard':
            return actor, actions
        player = self.players[actor]
        nonwild = sum(player.hand) - player.hand[self.wildcard_tile]
        if self.offer.tile == self.wildcard_tile:
            return actor, [PASS]
        if nonwild <= 2:
            actions = [a for a in actions if a[0] not in ('chi', 'peng')]
        return actor, actions

    def _discard(self, actor, tile):
        if actor == self.dealer:
            self.burn_sequence = [tile]
        elif self.burn_sequence and actor == (self.dealer + len(self.burn_sequence)) % self.seat_count:
            self.burn_sequence.append(tile)
        else:
            self.burn_sequence.clear()
        super()._discard(actor, tile)

    def _finish_responses(self, offer):
        if (offer.kind == 'discard' and self.claim is None and not self.winners
                and (len(self.burn_sequence) == self.seat_count or len(self.burn_sequence) == 3 and self.remaining == 0)
                and len(set(self.burn_sequence)) == 1):
            self._end('draw')
            self.reason_code = 'beijing_burn_draw'
            return
        super()._finish_responses(offer)

    def _call_discard(self, offer):
        self.burn_sequence.clear()
        super()._call_discard(offer)

    def _finish_own_kong(self, offer):
        self.burn_sequence.clear()
        super()._finish_own_kong(offer)

    def _settle(self, winners, offer):
        super()._settle(winners, offer)
        if offer is not None and offer.kind == 'gang_an':
            self.reason_code = 'beijing_dragon_rob'

    def _score(self, actor, hand, melds, ctx):
        return self.rules.faan(hand, melds, ctx, wildcard=self.wildcard_tile, sea=len(self.wall) <= 18)

    def _pay_kong(self, actor, meld):
        # Only the winner is paid for completed kongs, at settlement.
        pass

    def _kong_payments(self, actor, meld):
        return tuple(n * (2 if self.upstairs else 1) for n in super()._kong_payments(actor, meld))
