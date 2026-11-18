"""Hand shapes and calls for three- and four-seat Mahjong rooms."""

from collections import Counter, deque
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
import random

TILE_COUNT = 34
WINDS = frozenset(range(27, 31))
DRAGONS = frozenset(range(31, 34))
TERMINALS = frozenset((0, 8, 9, 17, 18, 26))
ORPHANS = TERMINALS | WINDS | DRAGONS
COPIES = Counter({tile: 4 for tile in range(TILE_COUNT)})
PASS = ('pass', None)
RON = ('ron', None)
ZIMO = ('zimo', None)
CALL_PRIORITY = {'chi': 1, 'peng': 2, 'zhigang': 3}


@dataclass(frozen=True)
class KongPoints:
    direct: int
    added: int
    concealed: int
    single_direct_payer: bool = False
    refund_on_draw: bool = False
    added_payer_from_pong: bool = False


@dataclass(frozen=True)
class TableOptions:
    tile_kinds: tuple[int, ...] = tuple(range(TILE_COUNT))
    chow: bool = True
    seven_pairs: bool = False
    orphans: bool = True
    drawn_added_kong_only: bool = False
    concealed_kong_faces: int = 4
    host_deals_first: bool = False
    show_winds: bool = True
    pass_win_same: bool = False
    pass_win_all: bool = False
    pass_pon_all: bool = False
    pass_pon_same: bool = False
    pass_win_higher_same: bool = False
    pass_win_higher_all: bool = False
    kong_points: KongPoints | None = None
    horse_counts: tuple[int, int, int] = (0, 0, 0)  # discard, self-draw, rob added kong
    dealer_echo: bool = False
    winner_echo: bool = False
    echo_share: int = 1
    reciprocal_liability: bool = False
    chow_lock_extensions: bool = True
    pass_win_clear_by_call: bool = False
    kong_requires_live_tile: bool = False
    first_liability_only: frozenset[str] = frozenset()
    exchange_count: int = 0
    closed_live_tail: int = 0
    seat_count: int = 4
    allow_locked_chow_discard: bool = False


DEFAULT_OPTIONS = TableOptions()


@dataclass(frozen=True)
class Meld:
    kind: str
    tile: int  # The start of a chow, otherwise the repeated tile.
    provider: int | None = None

    @property
    def tiles(self):
        if self.kind == 'chi':
            return (self.tile, self.tile + 1, self.tile + 2)
        return (self.tile,) * (3 if self.kind in ('peng', 'triplet') else 4)


@dataclass(frozen=True)
class WinContext:
    kind: str
    tile: int
    seat_wind: int
    round_wind: int
    kong_chain: int = 0
    opening: str | None = None
    first_nondealer_draw: bool = False
    kong_provider: int | None = None
    last_live_draw: bool = False
    winner_count: int = 0
    late_wall: bool = False
    ready: bool = False


@dataclass(frozen=True)
class Offer:
    actor: int
    tile: int
    kind: str
    kong_chain: int = 0
    opening: str | None = None


@dataclass
class Player:
    hand: list[int] = field(default_factory=lambda: [0] * TILE_COUNT)
    melds: list[Meld] = field(default_factory=list)
    declined_wins: set[int] = field(default_factory=set)
    declined_calls: set[int] = field(default_factory=set)
    declined_win_values: dict[int, int] = field(default_factory=dict)
    declined_win_share: int = -1
    liabilities: list[tuple[str, int]] = field(default_factory=list)
    draws: int = 0
    score: int = 0
    open_ready_hand: tuple[int, ...] | None = None
    closed_ready: bool = False


@dataclass(frozen=True)
class Win:
    actor: int
    provider: int | None
    kind: str
    fan: int | None
    points: int
    patterns: tuple[str, ...]
    payments: tuple[int, ...]
    tier: str | None = None


@dataclass(frozen=True)
class WinningHand:
    actor: int
    score: object
    hand: list[int]
    kind: str


def chow_choices(hand, tile, *, swap_lock=True, lock_extensions=True, allow_locked_discard=False):
    """Yield (sequence start, consumed tiles, forbidden next discards)."""
    if tile >= 27:
        return
    suit = tile // 9 * 9
    for start in range(max(suit, tile - 2), min(tile, suit + 6) + 1):
        consumed = tuple(other for other in range(start, start + 3) if other != tile)
        if not all(hand[other] for other in consumed):
            continue
        forbidden = {tile} if swap_lock else set()
        if swap_lock and lock_extensions and tile == start and start < suit + 6:
            forbidden.add(start + 3)
        if swap_lock and lock_extensions and tile == start + 2 and start > suit:
            forbidden.add(start - 1)
        if any(count > (other in consumed) for other, count in enumerate(hand) if other not in forbidden):
            yield start, consumed, frozenset(forbidden)
        elif allow_locked_discard and sum(hand) > len(consumed):
            yield start, consumed, frozenset()


@lru_cache(maxsize=32768)
def _sets(counts):
    """Exact decompositions of leftover tiles after taking the pair."""
    first = next((i for i, count in enumerate(counts) if count), None)
    if first is None:
        return ((),)
    rest = list(counts)
    options = []
    if counts[first] >= 3:
        rest[first] -= 3
        options.extend((Meld('triplet', first),) + tail for tail in _sets(tuple(rest)))
        rest[first] += 3
    if first < 27 and first % 9 <= 6 and counts[first + 1] and counts[first + 2]:
        for tile in (first, first + 1, first + 2):
            rest[tile] -= 1
        options.extend((Meld('chi', first),) + tail for tail in _sets(tuple(rest)))
    return tuple(options)


def standard_shapes(hand, open_sets):
    """Yield all four-set-and-a-pair interpretations, counting declared sets."""
    if sum(hand) != 3 * (4 - open_sets) + 2:
        return
    for tile, count in enumerate(hand):
        if count < 2:
            continue
        rest = list(hand)
        rest[tile] -= 2
        for sets in _sets(tuple(rest)):
            yield tile, sets


def middle_five(sets, tile):
    return tile < 27 and tile % 9 == 4 and Meld('chi', tile - 1) in sets


def thirteen_orphans(hand, melds):
    return not melds and sum(hand) == 14 and all(hand[t] for t in ORPHANS)


def seven_pairs(hand, melds):
    return not melds and sum(hand) == 14 and all(count % 2 == 0 for count in hand)


@lru_cache(maxsize=16384)
def ready_waits(hand, melds, tile_kinds, allow_seven_pairs=True):
    used = Counter(tile for meld in melds for tile in meld.tiles)
    augmented = list(hand)
    waits = []
    for tile in tile_kinds:
        if hand[tile] + used[tile] >= 4:
            continue
        augmented[tile] += 1
        if ((allow_seven_pairs and seven_pairs(augmented, melds)) or
                next(standard_shapes(augmented, len(melds)), None) is not None):
            waits.append(tile)
        augmented[tile] -= 1
    return tuple(waits)


def ready_discards(player, actions, options):
    hand = player.hand.copy()
    melds = tuple(player.melds)
    for kind, tile in actions:
        if kind == 'discard':
            hand[tile] -= 1
            if ready_waits(tuple(hand), melds, options.tile_kinds, options.seven_pairs):
                yield 'declare_ready', tile
            hand[tile] += 1


class HonorsGame:
    """First-win state machine for rooms using suited and honor tiles."""

    def __init__(self, seed, dealer, round_wind, *, rules):
        self.rules = rules
        self.options = getattr(rules, 'OPTIONS', DEFAULT_OPTIONS)
        self.seat_count = self.options.seat_count
        self.dealer = dealer
        self.round_wind = round_wind
        self.next_dealer = dealer
        self.tile_supply = self._tile_supply()
        self.wall = list(self.tile_supply.elements())
        random.Random(seed).shuffle(self.wall)
        self.players = [Player() for _ in range(self.seat_count)]
        self.discards = [[] for _ in self.players]
        self.cur = dealer
        self.phase = 'action'
        self.drawn_tile = None
        self.kong_chain = 0
        self.kong_provider = None
        self.opening = True
        self.uninterrupted = True
        self.forbidden_discards = frozenset()
        self.offer = None
        self.responses = deque()
        self.claim = None
        self.last_call = None
        self.winners = []
        self.win_details = []
        self.horse_tiles = []
        self.horse_details = []
        self.wake_draws = []
        self.wake_tiles = []
        self.wake_details = []
        self.bilateral_horses = []
        self.echo_pending = False
        self.exchanges = []
        self.kong_deltas = [0] * self.seat_count
        self.echo_deltas = [0] * self.seat_count
        self.related_seats = []
        self.revealed = {}
        self.claimed_tile = None
        self.reason_code = None
        self._prepare_wall()
        for _ in range(13):
            for step in range(self.seat_count):
                actor = (dealer + step) % self.seat_count
                self.players[actor].hand[self._deal_tile(actor)] += 1
        self._draw(dealer)
        if self.options.exchange_count:
            self.phase = 'exchange'
            self.cur = 0
            self.drawn_tile = None

    def scores(self):
        return [player.score for player in self.players]

    def _tile_supply(self):
        return Counter({tile: COPIES[tile] for tile in self.options.tile_kinds})

    def _prepare_wall(self):
        pass

    def _deal_tile(self, actor):
        return self.wall.pop()

    @property
    def remaining(self):
        return len(self.wall) + len(self.horse_tiles) + len(self.wake_draws) - self.rules.WALL_RESERVE

    def physical_counts(self):
        counts = Counter(self.wall)
        counts.update(self.horse_tiles)
        counts.update(self.wake_draws)
        for player, river in zip(self.players, self.discards):
            counts.update(river)
            counts.update({tile: count for tile, count in enumerate(player.hand) if count})
            for meld in player.melds:
                counts.update(meld.tiles)
        if self.claimed_tile is not None:
            counts[self.claimed_tile] += 1
        return counts

    def _draw(self, actor, kong_chain=0, provider=None):
        tile = self._take_tile(actor, bool(kong_chain))
        if tile is None:
            self._end('draw')
            return
        player = self.players[actor]
        player.hand[tile] += 1
        player.draws += 1
        player.declined_calls.clear()
        player.declined_wins.clear()
        player.declined_win_values.clear()
        player.declined_win_share = -1
        self.cur = actor
        self.phase = 'action'
        self.drawn_tile = tile
        self.kong_chain = kong_chain
        self.kong_provider = provider
        self.forbidden_discards = frozenset()

    def _take_tile(self, actor, replacement):
        if self.remaining:
            return self.wall.pop(self._replacement_index() if replacement else -1)
        return None

    def _replacement_index(self):
        return self.rules.WALL_RESERVE if any(self.options.horse_counts) else 0

    def _win(self, actor, offer=None, winner_count=0):
        player = self.players[actor]
        hand = player.hand.copy()
        seat_wind = 27 + (actor - self.dealer) % self.seat_count
        ready = player.open_ready_hand is not None
        if offer is None:
            ctx = WinContext('zimo', self.drawn_tile, seat_wind,
                             self.round_wind, self.kong_chain, '天胡' if self.opening else None,
                             self.uninterrupted and actor != self.dealer and player.draws == 1,
                             self.kong_provider, self.remaining == 0,
                             late_wall=self.remaining <= self.options.closed_live_tail, ready=ready)
        else:
            hand[offer.tile] += 1
            kind = {'discard': 'ron', 'gang_bu': 'rob_kong', 'gang_an': 'rob_concealed'}[offer.kind]
            ctx = WinContext(kind, offer.tile, seat_wind,
                             self.round_wind, offer.kong_chain, offer.opening,
                             last_live_draw=self.remaining == 0, winner_count=winner_count, ready=ready)
        return self._score(actor, hand, tuple(player.melds), ctx), hand, ctx.kind

    def _score(self, actor, hand, melds, ctx):
        return self.rules.faan(hand, melds, ctx)

    def legal_actions(self):
        if self.phase == 'over':
            return None, []
        if self.phase == 'exchange':
            tiles = [tile for tile, count in enumerate(self.players[self.cur].hand) for _ in range(count)]
            groups = sorted(set(combinations(tiles, self.options.exchange_count)))
            return self.cur, [('exchange', group) for group in groups]
        if self.offer is not None:
            return self._response_actions()

        player = self.players[self.cur]
        actions = [('discard', tile) for tile, count in enumerate(player.hand)
                   if count and tile not in self.forbidden_discards]
        if self.drawn_tile is not None:
            if self._win(self.cur)[0] is not None:
                actions.append(ZIMO)
            if (self.remaining > self.options.closed_live_tail
                    or not (self.rules.WALL_RESERVE or self.options.kong_requires_live_tile)):
                actions.extend(('ankan', tile) for tile, count in enumerate(player.hand) if count == 4)
                actions.extend(('bugang', meld.tile) for meld in player.melds
                               if meld.kind == 'peng' and player.hand[meld.tile]
                               and (not self.options.drawn_added_kong_only or meld.tile == self.drawn_tile))
        return self.cur, actions

    def _response_actions(self):
        offer = self.offer
        actor = self.responses[0]
        player = self.players[actor]
        actions = [PASS]
        if offer.tile not in player.declined_wins and not (self.options.pass_win_all and player.declined_wins):
            score, _, _ = self._win(actor, offer)
            if score is not None and score.share > max(player.declined_win_share,
                                                        player.declined_win_values.get(offer.tile, -1)):
                actions.append(RON)
        if (offer.kind == 'discard' and self.remaining > self.options.closed_live_tail
                and (not self.rules.PASS_LOCK or offer.tile not in player.declined_calls)):
            pon_locked = offer.tile in player.declined_calls or (
                self.options.pass_pon_all and bool(player.declined_calls)
            )
            if player.hand[offer.tile] >= 2 and not pon_locked:
                actions.append(('peng', None))
            if player.hand[offer.tile] == 3:
                actions.append(('zhigang', None))
            if self.options.chow and actor == (offer.actor + 1) % self.seat_count:
                actions.extend(('chi', start) for start, _, _ in self._chows(player, offer.tile, offer.actor))
        return actor, actions

    def _chows(self, player, tile, provider):
        return chow_choices(player.hand, tile, swap_lock=self.rules.SWAP_LOCK,
                            lock_extensions=self.options.chow_lock_extensions,
                            allow_locked_discard=self.options.allow_locked_chow_discard)

    def step(self, action):
        actor, actions = self.legal_actions()
        if action not in actions:
            raise ValueError('Illegal honors-room action')
        self.last_call = None
        kind, tile = action
        if kind == 'exchange':
            self._exchange(tile)
        elif self.offer is not None:
            self._respond(actor, action, actions)
        elif kind == 'discard':
            self._discard(actor, tile)
        elif kind == 'zimo':
            self._settle([actor], None)
        else:
            kong = 'gang_an' if kind == 'ankan' else 'gang_bu'
            self.uninterrupted = False
            self._open_offer(Offer(actor, tile, kong, self.kong_chain))

    def _discard(self, actor, tile):
        self.players[actor].hand[tile] -= 1
        self.discards[actor].append(tile)
        if self.uninterrupted and all(river == [tile] for river in self.discards):
            if self.options.dealer_echo:
                self._pay_echo(self.dealer)
            elif self.options.winner_echo:
                self.echo_pending = True
        self._open_offer(Offer(actor, tile, 'discard', self.kong_chain, '地胡' if self.opening else None))

    def _exchange(self, tiles):
        self.exchanges.append(tiles)
        if len(self.exchanges) < len(self.players):
            self.cur += 1
            return
        for seat, player in enumerate(self.players):
            for tile in self.exchanges[seat]:
                player.hand[tile] -= 1
            for tile in self.exchanges[seat - 1]:
                player.hand[tile] += 1
        self.drawn_tile = self.exchanges[self.dealer - 1][-1]
        self.exchanges.clear()
        self.cur = self.dealer
        self.phase = 'action'

    def _pay_echo(self, payer):
        for seat in range(self.seat_count):
            delta = self.options.echo_share * (1 - self.seat_count if seat == payer else 1)
            self.echo_deltas[seat] += delta
            self.players[seat].score += delta

    def _open_offer(self, offer):
        self.offer = offer
        self.responses = deque((offer.actor + step) % self.seat_count for step in range(1, self.seat_count))
        self.claim = None
        self.phase = 'response' if offer.kind == 'discard' else 'rob_kong'
        self.drawn_tile = None
        self.opening = False

    def _respond(self, actor, action, actions):
        offer = self.offer
        kind, argument = action
        player = self.players[actor]
        self.responses.popleft()
        if action == RON:
            self.winners.append(actor)
        elif RON in actions:
            if self.rules.PASS_LOCK or self.options.pass_win_same or self.options.pass_win_all:
                player.declined_wins.add(offer.tile)
            elif self.options.pass_win_higher_same:
                score, _, _ = self._win(actor, offer)
                player.declined_win_values[offer.tile] = score.share
            elif self.options.pass_win_higher_all:
                score, _, _ = self._win(actor, offer)
                player.declined_win_share = score.share
        if kind in CALL_PRIORITY:
            if self.claim is None or CALL_PRIORITY[kind] > CALL_PRIORITY[self.claim[1]]:
                self.claim = actor, kind, argument
        elif self.rules.PASS_LOCK and action == PASS and any(k in CALL_PRIORITY for k, _ in actions):
            player.declined_calls.add(offer.tile)
        if kind in ('pass', 'chi') and (self.options.pass_pon_same or self.options.pass_pon_all) and ('peng', None) in actions:
            player.declined_calls.add(offer.tile)
        if self.responses:
            return

        self.offer = None
        if self.winners:
            self._settle(self.winners, offer)
        elif offer.kind != 'discard':
            self._finish_own_kong(offer)
        elif self.claim is not None:
            self._call_discard(offer)
        else:
            self._draw((offer.actor + 1) % self.seat_count)

    def _call_discard(self, offer):
        actor, kind, argument = self.claim
        player = self.players[actor]
        self.discards[offer.actor].pop()
        if kind == 'chi':
            consumed, forbidden = next((take, block) for start, take, block in self._chows(player, offer.tile, offer.actor)
                                       if start == argument)
            meld = Meld('chi', argument, offer.actor)
        else:
            consumed = (offer.tile,) * (2 if kind == 'peng' else 3)
            forbidden = frozenset()
            meld = Meld('peng' if kind == 'peng' else 'gang_ming', offer.tile, offer.actor)
        for tile in consumed:
            player.hand[tile] -= 1
        player.melds.append(meld)
        liabilities = self.rules.liability(player.melds, meld)
        for pattern in liabilities:
            entry = pattern, offer.actor
            if pattern in self.options.first_liability_only and any(entry in p.liabilities for p in self.players):
                continue
            player.liabilities.append(entry)
        if self.options.reciprocal_liability and liabilities:
            related = (actor, offer.actor)
            if related not in self.related_seats and related[::-1] not in self.related_seats:
                self.related_seats.append(related)
        self.uninterrupted = False
        self.last_call = actor, meld
        self.forbidden_discards = forbidden
        self.cur = actor
        self.phase = 'action'
        self.kong_chain = 0
        self.kong_provider = None
        if self.options.pass_win_clear_by_call:
            player.declined_wins.clear()
            player.declined_win_values.clear()
            player.declined_win_share = -1
        if kind == 'zhigang':
            self._pay_kong(actor, meld)
            self._draw(actor, kong_chain=1, provider=meld.provider)

    def _finish_own_kong(self, offer):
        player = self.players[offer.actor]
        tile = offer.tile
        if offer.kind == 'gang_bu':
            index = next(i for i, meld in enumerate(player.melds) if meld.kind == 'peng' and meld.tile == tile)
            meld = Meld('gang_bu', tile, player.melds[index].provider)
            player.melds[index] = meld
            player.hand[tile] -= 1
        else:
            meld = Meld('gang_an', tile)
            player.melds.append(meld)
            player.hand[tile] -= 4
        self.last_call = offer.actor, meld
        self._pay_kong(offer.actor, meld)
        self._draw(offer.actor, kong_chain=offer.kong_chain + 1)

    def _transfer(self, actor, payments, ledger=None):
        for seat, amount in enumerate(payments):
            self.players[seat].score -= amount
        self.players[actor].score += sum(payments)
        if ledger is not None:
            for seat, amount in enumerate(payments):
                ledger[seat] -= amount
            ledger[actor] += sum(payments)

    def _pay_kong(self, actor, meld):
        points = self.options.kong_points
        if points is None:
            return
        share = {'gang_ming': points.direct, 'gang_bu': points.added, 'gang_an': points.concealed}[meld.kind]
        share *= self._kong_factor(meld)
        if (meld.kind == 'gang_ming' and points.single_direct_payer
                or meld.kind == 'gang_bu' and points.added_payer_from_pong) and meld.provider is not None:
            payers = (meld.provider,)
        else:
            payers = tuple(seat for seat in range(self.seat_count) if seat != actor)
        self._transfer(actor, tuple(share if seat in payers else 0 for seat in range(self.seat_count)), self.kong_deltas)

    def _kong_factor(self, meld):
        return 1

    def _liabilities_for(self, actor):
        if not self.options.reciprocal_liability:
            return self.players[actor].liabilities
        return [('three_feeds', partner if caller == actor else caller)
                for caller, partner in self.related_seats if actor in (caller, partner)]

    def _apply_horses(self, actor, kind, score):
        counts = self.options.horse_counts
        count = counts[{'ron': 0, 'zimo': 1, 'rob_kong': 2, 'rob_concealed': 2}[kind]]
        if not count:
            return score
        if not self.horse_tiles:
            self.horse_tiles = self.wall[:count]
            del self.wall[:count]
        hits = sum(self.rules.horse_hit(tile, actor, self.dealer) for tile in self.horse_tiles)
        adjusted = self.rules.with_horses(score, hits)
        self.horse_details.append((actor, count, hits, adjusted.share, adjusted.share - score.share))
        return adjusted

    def _settle(self, winners, offer):
        self.winners = list(winners)
        if self.echo_pending:
            self._pay_echo(self.winners[0])
        provider = None if offer is None else offer.actor
        hands = []
        for actor in self.winners:
            score, hand, kind = self._win(actor, offer, winner_count=len(self.winners))
            self.revealed[actor] = hand
            hands.append(WinningHand(actor, score, hand, kind))
        settle_hand = getattr(self.rules, 'settle_hand', None)
        if settle_hand is None:
            outcomes = [self._settle_winner(winner, provider) for winner in hands]
        else:
            outcomes = settle_hand(self, hands, offer)
        for winner, (score, payments) in zip(hands, outcomes, strict=True):
            self._transfer(winner.actor, payments)
            self.win_details.append(Win(winner.actor, provider, winner.kind, score.fan, score.share,
                                        score.patterns, payments, score.tier))
        if offer is not None:
            if offer.kind == 'discard':
                self.discards[provider].pop()
            else:
                self.players[provider].hand[offer.tile] -= 1
            self.claimed_tile = offer.tile
        reason = {'zimo': 'suited_self_draw', 'ron': 'suited_discard_win',
                  'rob_kong': 'rob_kong_win', 'rob_concealed': 'rob_concealed_win'}
        self._end(reason[self.win_details[0].kind])

    def _settle_winner(self, winner, provider):
        score = self._apply_horses(winner.actor, winner.kind, winner.score)
        payments = self.rules.settlement(score, winner.kind, winner.actor, provider,
                                         self._liabilities_for(winner.actor))
        return score, payments

    def _end(self, reason):
        self.phase = 'over'
        self.drawn_tile = None
        self.reason_code = reason
        points = self.options.kong_points
        if points is not None and points.refund_on_draw and reason == 'draw':
            for player, delta in zip(self.players, self.kong_deltas):
                player.score -= delta
            self.kong_deltas = [0] * self.seat_count
        provider = self.win_details[0].provider if self.win_details else None
        self.next_dealer = self.rules.next_dealer(self.dealer, self.winners, provider)
