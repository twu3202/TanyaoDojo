"""Hand shapes and calls for four-player 34-kind Mahjong rooms."""

from collections import Counter, deque
from dataclasses import dataclass, field
from functools import lru_cache
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
    pass_win_all: bool = False
    pass_pon_all: bool = False
    pass_pon_same: bool = False
    pass_win_higher_same: bool = False
    kong_points: KongPoints | None = None
    horse_counts: tuple[int, int, int] = (0, 0, 0)  # discard, self-draw, rob added kong
    dealer_echo: bool = False
    reciprocal_liability: bool = False


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
    liabilities: list[tuple[str, int]] = field(default_factory=list)
    draws: int = 0
    score: int = 0


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


def chow_choices(hand, tile, *, swap_lock=True):
    """Yield (sequence start, consumed tiles, forbidden next discards)."""
    if tile >= 27:
        return
    suit = tile // 9 * 9
    for start in range(max(suit, tile - 2), min(tile, suit + 6) + 1):
        consumed = tuple(other for other in range(start, start + 3) if other != tile)
        if not all(hand[other] for other in consumed):
            continue
        forbidden = {tile} if swap_lock else set()
        if swap_lock and tile == start and start < suit + 6:
            forbidden.add(start + 3)
        if swap_lock and tile == start + 2 and start > suit:
            forbidden.add(start - 1)
        if any(count > (other in consumed) for other, count in enumerate(hand) if other not in forbidden):
            yield start, consumed, frozenset(forbidden)


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


def thirteen_orphans(hand, melds):
    return not melds and sum(hand) == 14 and all(hand[t] for t in ORPHANS)


def seven_pairs(hand, melds):
    return not melds and sum(hand) == 14 and all(count % 2 == 0 for count in hand)


class HonorsGame:
    """First-win state machine for four-player games using suited and honor tiles."""

    def __init__(self, seed, dealer, round_wind, *, rules):
        self.rules = rules
        self.options = getattr(rules, 'OPTIONS', DEFAULT_OPTIONS)
        self.dealer = dealer
        self.round_wind = round_wind
        self.next_dealer = dealer
        self.tile_supply = Counter({tile: COPIES[tile] for tile in self.options.tile_kinds})
        self.wall = list(self.tile_supply.elements())
        random.Random(seed).shuffle(self.wall)
        self.players = [Player() for _ in range(4)]
        self.discards = [[] for _ in self.players]
        self.cur = dealer
        self.phase = 'action'
        self.drawn_tile = None
        self.kong_chain = 0
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
        self.kong_deltas = [0] * 4
        self.echo_deltas = [0] * 4
        self.related_seats = []
        self.revealed = {}
        self.claimed_tile = None
        self.reason_code = None
        for _ in range(13):
            for step in range(4):
                self.players[(dealer + step) % 4].hand[self.wall.pop()] += 1
        self._draw(dealer)

    def scores(self):
        return [player.score for player in self.players]

    @property
    def remaining(self):
        return len(self.wall) + len(self.horse_tiles) - self.rules.WALL_RESERVE

    def physical_counts(self):
        counts = Counter(self.wall)
        counts.update(self.horse_tiles)
        for player, river in zip(self.players, self.discards):
            counts.update(river)
            counts.update({tile: count for tile, count in enumerate(player.hand) if count})
            for meld in player.melds:
                counts.update(meld.tiles)
        if self.claimed_tile is not None:
            counts[self.claimed_tile] += 1
        return counts

    def _draw(self, actor, kong_chain=0):
        if not self.remaining:
            self._end('draw')
            return
        replacement = self.rules.WALL_RESERVE if any(self.options.horse_counts) else 0
        tile = self.wall.pop(replacement if kong_chain else -1)
        player = self.players[actor]
        player.hand[tile] += 1
        player.draws += 1
        player.declined_calls.clear()
        player.declined_wins.clear()
        player.declined_win_values.clear()
        self.cur = actor
        self.phase = 'action'
        self.drawn_tile = tile
        self.kong_chain = kong_chain
        self.forbidden_discards = frozenset()

    def _win(self, actor, offer=None):
        player = self.players[actor]
        hand = player.hand.copy()
        if offer is None:
            ctx = WinContext('zimo', self.drawn_tile, 27 + (actor - self.dealer) % 4,
                             self.round_wind, self.kong_chain, '天胡' if self.opening else None,
                             self.uninterrupted and actor != self.dealer and player.draws == 1)
        else:
            hand[offer.tile] += 1
            kind = {'discard': 'ron', 'gang_bu': 'rob_kong', 'gang_an': 'rob_concealed'}[offer.kind]
            ctx = WinContext(kind, offer.tile, 27 + (actor - self.dealer) % 4,
                             self.round_wind, offer.kong_chain, offer.opening)
        return self.rules.faan(hand, tuple(player.melds), ctx), hand, ctx.kind

    def legal_actions(self):
        if self.phase == 'over':
            return None, []
        if self.offer is not None:
            return self._response_actions()

        player = self.players[self.cur]
        actions = [('discard', tile) for tile, count in enumerate(player.hand)
                   if count and tile not in self.forbidden_discards]
        if self.drawn_tile is not None:
            if self._win(self.cur)[0] is not None:
                actions.append(ZIMO)
            if self.remaining or not self.rules.WALL_RESERVE:
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
            if score is not None and score.share > player.declined_win_values.get(offer.tile, -1):
                actions.append(RON)
        if (offer.kind == 'discard' and self.remaining
                and (not self.rules.PASS_LOCK or offer.tile not in player.declined_calls)):
            pon_locked = offer.tile in player.declined_calls or (
                self.options.pass_pon_all and bool(player.declined_calls)
            )
            if player.hand[offer.tile] >= 2 and not pon_locked:
                actions.append(('peng', None))
            if player.hand[offer.tile] == 3:
                actions.append(('zhigang', None))
            if self.options.chow and actor == (offer.actor + 1) % 4:
                actions.extend(('chi', start) for start, _, _ in self._chows(player, offer.tile))
        return actor, actions

    def _chows(self, player, tile):
        return chow_choices(player.hand, tile, swap_lock=self.rules.SWAP_LOCK)

    def step(self, action):
        actor, actions = self.legal_actions()
        if action not in actions:
            raise ValueError('Illegal honors-room action')
        self.last_call = None
        kind, tile = action
        if self.offer is not None:
            self._respond(actor, action, actions)
        elif kind == 'discard':
            self.players[actor].hand[tile] -= 1
            self.discards[actor].append(tile)
            if self.options.dealer_echo and self.uninterrupted and all(river == [tile] for river in self.discards):
                for seat in range(4):
                    self.echo_deltas[seat] = -3 if seat == self.dealer else 1
                    self.players[seat].score += self.echo_deltas[seat]
            self._open_offer(Offer(actor, tile, 'discard', self.kong_chain, '地胡' if self.opening else None))
        elif kind == 'zimo':
            self._settle([actor], None)
        else:
            kong = 'gang_an' if kind == 'ankan' else 'gang_bu'
            self.uninterrupted = False
            self._open_offer(Offer(actor, tile, kong, self.kong_chain))

    def _open_offer(self, offer):
        self.offer = offer
        self.responses = deque((offer.actor + step) % 4 for step in (1, 2, 3))
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
            if self.rules.PASS_LOCK or self.options.pass_win_all:
                player.declined_wins.add(offer.tile)
            elif self.options.pass_win_higher_same:
                score, _, _ = self._win(actor, offer)
                player.declined_win_values[offer.tile] = score.share
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
            self._draw((offer.actor + 1) % 4)

    def _call_discard(self, offer):
        actor, kind, argument = self.claim
        player = self.players[actor]
        self.discards[offer.actor].pop()
        if kind == 'chi':
            consumed, forbidden = next((take, block) for start, take, block in self._chows(player, offer.tile)
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
        player.liabilities.extend((pattern, offer.actor) for pattern in liabilities)
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
        if kind == 'zhigang':
            self._pay_kong(actor, meld)
            self._draw(actor, kong_chain=1)

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
        if meld.kind == 'gang_ming' and points.single_direct_payer:
            payers = (meld.provider,)
        else:
            payers = tuple(seat for seat in range(4) if seat != actor)
        self._transfer(actor, tuple(share if seat in payers else 0 for seat in range(4)), self.kong_deltas)

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
        self.horse_details.append((actor, count, hits, score.share * (hits + 1), score.share * hits))
        return self.rules.with_horses(score, hits)

    def _settle(self, winners, offer):
        self.winners = list(winners)
        provider = None if offer is None else offer.actor
        for actor in self.winners:
            score, hand, kind = self._win(actor, offer)
            self.revealed[actor] = hand
            score = self._apply_horses(actor, kind, score)
            payments = self.rules.settlement(score, kind, actor, provider, self._liabilities_for(actor))
            self._transfer(actor, payments)
            self.win_details.append(Win(actor, provider, kind, score.fan, score.share,
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

    def _end(self, reason):
        self.phase = 'over'
        self.drawn_tile = None
        self.reason_code = reason
        points = self.options.kong_points
        if points is not None and points.refund_on_draw and reason == 'draw':
            for player, delta in zip(self.players, self.kong_deltas):
                player.score -= delta
            self.kong_deltas = [0] * 4
        provider = self.win_details[0].provider if self.win_details else None
        self.next_dealer = self.rules.next_dealer(self.dealer, self.winners, provider)
