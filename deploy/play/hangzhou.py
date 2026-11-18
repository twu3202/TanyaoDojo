"""White-dragon wildcard rooms with burst, float, and Hangzhou dealer scoring."""

from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache

from .honors import PASS, HonorsGame, TableOptions
from .wildcard_shapes import can_form_pairs, can_form_sets, can_form_standard, without_tile

WHITE = 31
FLOAT = ('float_wildcard', WHITE)


@dataclass(frozen=True)
class HangzhouScore:
    share: int
    patterns: tuple[str, ...]
    fan: None = None
    tier: None = None


@dataclass(frozen=True)
class HangzhouRules:
    VARIANT: str
    OPTIONS: TableOptions
    fixed_dealer: bool = False
    three_feeds: bool = False

    MATCH_HANDS = 8
    MIN_FAAN = 0
    WALL_RESERVE = 0
    PASS_LOCK = False
    SWAP_LOCK = False
    WILDCARD = WHITE

    @staticmethod
    def next_dealer(dealer, winners, provider):
        return winners[0] if winners else dealer

    @staticmethod
    def liability(melds, meld):
        return ()  # Feed order and the float circuit are adjudicated by HangzhouGame.

    @staticmethod
    def faan(hand, melds, ctx):
        return score_hand(hand, len(melds), ctx)

    @staticmethod
    def settle_hand(game, winners, offer):
        return [(winner.score, game.payments(winner.actor, winner.score.share)) for winner in winners]


_OPTIONS = TableOptions(seven_pairs=True, orphans=False, concealed_kong_faces=0, show_winds=False,
                        host_deals_first=True, kong_requires_live_tile=True)
THREE_HANGZHOU = HangzhouRules('hangzhou-three-player', replace(_OPTIONS, seat_count=3, chow=False),
                              fixed_dealer=True)
FOUR_HANGZHOU = HangzhouRules('hangzhou-three-feeds', _OPTIONS, three_feeds=True)
HANGZHOU_ROOMS = (THREE_HANGZHOU, FOUR_HANGZHOU)


def split_wild(hand):
    return tuple(hand[:WHITE]) + (0,) + tuple(hand[WHITE + 1:]), hand[WHITE]


@lru_cache(maxsize=16384)
def burst_wait(hand, meld_count):
    """A 13-tile hand has a wildcard reserved to pair with the next draw."""
    natural, wild = split_wild(hand)
    if not wild or sum(hand) != 13 - 3 * meld_count:
        return False, False
    return can_form_sets(natural, wild - 1), not meld_count and can_form_pairs(natural, wild - 1)


def score_hand(hand, meld_count, ctx, floats=0, kongs_before_float=0):
    if ctx.kind != 'zimo' or ctx.tile is None or not hand[ctx.tile] or sum(hand) != 14 - 3 * meld_count:
        return None
    natural, wild = split_wild(hand)
    before = without_tile(tuple(hand), ctx.tile)
    burst, pair_burst = burst_wait(before, meld_count)
    scores = []
    if burst or can_form_standard(natural, wild):
        scores.append(_score_shape(1, (), burst, ctx.kong_chain, floats, kongs_before_float))
    if not meld_count and can_form_pairs(natural, wild):
        pair_hand, _ = split_wild(before) if pair_burst else (natural, wild)
        luxury = sum(count == 4 for count in pair_hand)
        if not wild:
            name = ('清七对', '豪华清七对', '双豪华清七对', '三豪华清七对')[luxury]
        else:
            name = ('七对子', '豪华七对', '双豪华七对', '三豪华七对')[luxury]
        scores.append(_score_shape(1 << (1 + luxury + (not wild)), (name,), pair_burst,
                                   ctx.kong_chain, floats, kongs_before_float))
    return max(scores, key=lambda score: score.share, default=None)


def _score_shape(base, patterns, burst, kong_chain, floats, kongs_before_float):
    if not burst:
        names = ('平胡', '杠开', '二连杠开', '三连杠开', '四连杠开')
        return HangzhouScore(base << kong_chain, patterns + ((names[kong_chain],) if kong_chain or not patterns else ()))
    kongs = kong_chain + kongs_before_float
    if floats and kongs:
        labels = ('飘杠',) if kong_chain else ('杠飘',)
        if floats > 1 or kongs > 1:
            labels = (('财飘', '双财飘', '三财飘')[floats - 1],
                      ('杠开', '二连杠开', '三连杠开', '四连杠开')[kongs - 1])
    elif floats:
        labels = (('财飘', '双财飘', '三财飘')[floats - 1],)
    elif kongs:
        labels = (('杠暴', '二连杠暴', '三连杠暴', '四连杠暴')[kongs - 1],)
    else:
        labels = ('爆头',)
    if patterns:
        labels = ('七客',) + (() if labels == ('爆头',) else labels)
        if base > 2:
            labels = patterns + labels
    return HangzhouScore(base << (1 + floats + kongs), labels)


class HangzhouGame(HonorsGame):
    def __init__(self, seed, dealer, round_wind, *, rules, dealer_streak=0):
        self.float_cycle_owner = None
        self.floated = set()
        self.float_streak = [0] * rules.OPTIONS.seat_count
        self.float_kongs = [0] * rules.OPTIONS.seat_count
        self.feeds = Counter()
        self.liability_feeds = Counter()
        self.dealer_multiplier = 8 if rules.fixed_dealer else 2 << min(dealer_streak, 2)
        super().__init__(seed, dealer, round_wind, rules=rules)

    def _score(self, actor, hand, melds, ctx):
        return score_hand(hand, len(melds), ctx, self.float_streak[actor], self.float_kongs[actor])

    def _draw(self, actor, kong_chain=0, provider=None):
        if actor == self.float_cycle_owner and not kong_chain:
            self.float_cycle_owner = None
        super()._draw(actor, kong_chain, provider)

    def legal_actions(self):
        actor, actions = super().legal_actions()
        if self.phase != 'action':
            return actor, actions
        actions = [(kind, tile) for kind, tile in actions
                   if (kind not in ('ankan', 'bugang') or tile != WHITE)
                   and (kind != 'bugang' or (actor not in self.floated and self.float_cycle_owner is None))]
        if self.float_cycle_owner is not None and self.float_cycle_owner != actor:
            actions = [(kind, tile) for kind, tile in actions if kind != 'discard' or tile == self.drawn_tile]
        elif self.drawn_tile is not None and self.remaining and ('discard', WHITE) in actions:
            player = self.players[actor]
            if any(burst_wait(without_tile(tuple(player.hand), WHITE), len(player.melds))):
                actions[actions.index(('discard', WHITE))] = FLOAT
        return actor, actions

    def step(self, action):
        if action != FLOAT:
            return super().step(action)
        actor, actions = self.legal_actions()
        if FLOAT not in actions:
            raise ValueError('Illegal wildcard float')
        self.last_call = None
        self.floated.add(actor)
        self.float_streak[actor] += 1
        self.float_kongs[actor] += self.kong_chain
        self.float_cycle_owner = actor
        super()._discard(actor, WHITE)

    def _discard(self, actor, tile):
        self.float_streak[actor] = 0
        self.float_kongs[actor] = 0
        super()._discard(actor, tile)

    def _response_actions(self):
        actor = self.responses[0]
        provider = self.offer.actor
        during_float = self.float_cycle_owner is not None
        if (self.offer.tile == WHITE or self.offer.kind != 'discard'
                or (during_float and actor != self.float_cycle_owner)
                or (not self.rules.three_feeds and actor in self.floated)):
            return actor, [PASS]
        if actor != self.dealer and provider != self.dealer and self.feeds[actor, provider] >= 2:
            return actor, [PASS]
        _, actions = super()._response_actions()
        if actor in self.floated:
            actions = [action for action in actions if action[0] != 'zhigang']
        return actor, actions

    def _call_discard(self, offer):
        actor = self.claim[0]
        if self.rules.three_feeds:
            direction = actor, offer.actor
            self.feeds[direction] += 1
            if self.dealer in direction and self.float_cycle_owner != actor:
                self.liability_feeds[direction] += 1
                if self.liability_feeds[direction] == 3:
                    self.related_seats.append(direction)
        if self.float_cycle_owner == actor:
            self.float_cycle_owner = None
        super()._call_discard(offer)

    def payments(self, actor, share):
        ordinary = tuple(0 if seat == actor else share * (self.dealer_multiplier if self.dealer in (actor, seat) else 1)
                         for seat in range(self.seat_count))
        for caller, provider in self.related_seats:
            if actor in (caller, provider):
                payer, factor = (provider, 1) if actor == caller else (caller, 2)
                return tuple(sum(ordinary) * factor if seat == payer else 0 for seat in range(self.seat_count))
        return ordinary
