"""Hangzhou wildcard rooms with burst, float, and dealer scoring."""

from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache

from .honors import PASS, RON, HonorsGame, TableOptions
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
    fixed_dealer: int | None = None
    feed_mode: str = 'none'
    require_burst: bool = False
    open_kong_after_float: bool = False
    dealer_discard_wins: bool = False
    guard_dealer: bool = False
    flip_wildcard: bool = False
    floater_can_call: bool = True

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

    def faan(self, hand, melds, ctx, *, floats=0, kongs_before_float=0):
        return score_hand(hand, len(melds), ctx, floats, kongs_before_float,
                          require_burst=self.require_burst, allow_ron=self.dealer_discard_wins)

    @staticmethod
    def settle_hand(game, winners, offer):
        provider = offer.actor if offer is not None else None
        return [(winner.score, game.payments(winner.actor, winner.score.share, provider)) for winner in winners]


_OPTIONS = TableOptions(seven_pairs=True, orphans=False, concealed_kong_faces=0, show_winds=False,
                        host_deals_first=True, kong_requires_live_tile=True)
THREE_HANGZHOU = HangzhouRules('hangzhou-three-player', replace(_OPTIONS, seat_count=3, chow=False),
                              fixed_dealer=8)
FOUR_HANGZHOU = HangzhouRules('hangzhou-three-feeds', _OPTIONS, feed_mode='dealer_liability')
TONGLU = HangzhouRules('tonglu-mahjong', _OPTIONS, feed_mode='two_each', require_burst=True,
                       open_kong_after_float=True)
CHUNAN = HangzhouRules('chunan-mahjong', _OPTIONS, fixed_dealer=8, feed_mode='two_total',
                       dealer_discard_wins=True, guard_dealer=True)
TWO_HANGZHOU = HangzhouRules('hangzhou-two-player-flip', replace(_OPTIONS, seat_count=2),
                              fixed_dealer=1, flip_wildcard=True, open_kong_after_float=True,
                              floater_can_call=False)
HANGZHOU_ROOMS = (THREE_HANGZHOU, FOUR_HANGZHOU, TONGLU, CHUNAN, TWO_HANGZHOU)


def split_wild(hand):
    return tuple(hand[:WHITE]) + (0,) + tuple(hand[WHITE + 1:]), hand[WHITE]


@lru_cache(maxsize=16384)
def burst_wait(hand, meld_count):
    """A 13-tile hand has a wildcard reserved to pair with the next draw."""
    natural, wild = split_wild(hand)
    if not wild or sum(hand) != 13 - 3 * meld_count:
        return False, False
    return can_form_sets(natural, wild - 1), not meld_count and can_form_pairs(natural, wild - 1)


def score_hand(hand, meld_count, ctx, floats=0, kongs_before_float=0, *, require_burst=False, allow_ron=False):
    self_draw = ctx.kind == 'zimo'
    if not (self_draw or (allow_ron and ctx.kind in ('ron', 'rob_kong'))):
        return None
    if ctx.tile is None or not hand[ctx.tile] or sum(hand) != 14 - 3 * meld_count:
        return None
    natural, wild = split_wild(hand)
    before = without_tile(tuple(hand), ctx.tile)
    burst, pair_burst = burst_wait(before, meld_count) if self_draw else (False, False)
    ordinary_allowed = not require_burst or not wild
    kongs = ctx.kong_chain if self_draw else 0
    scores = []
    if burst or (ordinary_allowed and can_form_standard(natural, wild)):
        scores.append(_score_shape(1, (), burst, kongs, floats, kongs_before_float))
    if not meld_count and (pair_burst or (ordinary_allowed and can_form_pairs(natural, wild))):
        pair_hand, _ = split_wild(before) if pair_burst else (natural, wild)
        luxury = sum(count == 4 for count in pair_hand)
        if not wild:
            name = ('清七对', '豪华清七对', '双豪华清七对', '三豪华清七对')[luxury]
        else:
            name = ('七对子', '豪华七对', '双豪华七对', '三豪华七对')[luxury]
        scores.append(_score_shape(1 << (1 + luxury + (not wild)), (name,), pair_burst,
                                   kongs, floats, kongs_before_float))
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
        self.dealer_multiplier = rules.fixed_dealer or 2 << min(dealer_streak, 2)
        self.wildcard_indicator = None
        super().__init__(seed, dealer, round_wind, rules=rules)
        if rules.flip_wildcard:
            self._flip_wildcard()

    def _flip_wildcard(self):
        # Inside the rules engine WHITE still means a joker. Exchange the actual flip with WHITE
        # so the original white dragons can be used as the indicator's natural tile in any call.
        self.wildcard_indicator = self.wall.pop(0)
        self.wall[:] = map(self.physical_tile, self.wall)
        for player in self.players:
            hand = player.hand
            indicator = self.wildcard_indicator
            hand[WHITE], hand[indicator] = hand[indicator], hand[WHITE]
        self.drawn_tile = self.physical_tile(self.drawn_tile)

    def physical_tile(self, tile):
        if self.wildcard_indicator is not None:
            if tile == WHITE:
                return self.wildcard_indicator
            if tile == self.wildcard_indicator:
                return WHITE
        return tile

    def physical_counts(self):
        counts = super().physical_counts()
        if self.wildcard_indicator is None:
            return counts
        physical = Counter({self.physical_tile(tile): n for tile, n in counts.items()})
        physical[self.wildcard_indicator] += 1
        return physical

    def _score(self, actor, hand, melds, ctx):
        return self.rules.faan(hand, melds, ctx, floats=self.float_streak[actor],
                               kongs_before_float=self.float_kongs[actor])

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
                   and (kind != 'bugang' or self._open_kong_allowed(actor))]
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
        self._throw_tile(actor, WHITE)

    def _discard(self, actor, tile):
        self.float_streak[actor] = 0
        self.float_kongs[actor] = 0
        self._throw_tile(actor, tile)

    def _throw_tile(self, actor, tile):
        super()._discard(actor, tile)

    def _can_take_win(self, actor, offer):
        return (self.rules.dealer_discard_wins and self.dealer in (actor, offer.actor)
                and self.float_cycle_owner is None and offer.kind in ('discard', 'gang_bu'))

    def _response_actions(self):
        actor, offer = self.responses[0], self.offer
        during_float = self.float_cycle_owner is not None
        can_win = self._can_take_win(actor, offer)
        can_call = (offer.kind == 'discard'
                    and (not during_float or (self.rules.floater_can_call and actor == self.float_cycle_owner))
                    and (self.options.chow or actor not in self.floated))
        if offer.tile == WHITE or self.winners or not (can_win or can_call):
            return actor, [PASS]
        _, actions = super()._response_actions()
        actions = [action for action in actions if action == PASS or (action == RON and can_win)
                   or (action not in (PASS, RON) and can_call and self._can_call(actor, offer, action[0]))]
        return actor, actions

    def _open_kong_allowed(self, actor):
        return ((actor not in self.floated or self.rules.open_kong_after_float)
                and (self.float_cycle_owner is None or actor == self.float_cycle_owner))

    def _can_call(self, actor, offer, kind):
        provider = offer.actor
        mode = self.rules.feed_mode
        if kind == 'zhigang' and not self._open_kong_allowed(actor):
            return False
        if mode == 'two_total' or (mode == 'dealer_liability' and self.dealer not in (actor, provider)):
            if self.feeds[actor, provider] >= 2:
                return False
        if mode == 'two_each':
            matching = sum(meld.provider == provider and (meld.kind == 'chi') == (kind == 'chi')
                           for meld in self.players[actor].melds)
            if matching >= 2:
                return False
        if kind == 'peng' and self.rules.guard_dealer and actor == (self.dealer - 1) % self.seat_count:
            return provider == (actor - 1) % self.seat_count or bool(self.players[actor].hand[WHITE])
        return True

    def _call_discard(self, offer):
        actor = self.claim[0]
        mode = self.rules.feed_mode
        if mode in ('dealer_liability', 'room_liability', 'two_total'):
            direction = actor, offer.actor
            self.feeds[direction] += 1
            liability = mode == 'room_liability' or (mode == 'dealer_liability' and self.dealer in direction)
            if liability and self.float_cycle_owner != actor:
                self.liability_feeds[direction] += 1
                if self.liability_feeds[direction] == 3:
                    self.related_seats.append(direction)
        if self.float_cycle_owner == actor:
            self.float_cycle_owner = None
        super()._call_discard(offer)

    def payments(self, actor, share, provider=None):
        if provider is not None:
            return tuple(share * self.dealer_multiplier if seat == provider else 0 for seat in range(self.seat_count))
        ordinary = tuple(0 if seat == actor else share * (self.dealer_multiplier if self.dealer in (actor, seat) else 1)
                         for seat in range(self.seat_count))
        for caller, provider in self.related_seats:
            if actor in (caller, provider):
                payer, factor = (provider, 1) if actor == caller else (caller, 2)
                return tuple(sum(ordinary) * factor if seat == payer else 0 for seat in range(self.seat_count))
        return ordinary
