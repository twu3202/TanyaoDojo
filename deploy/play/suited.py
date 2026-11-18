"""Four-player, first-win settlement engine for 108 suited tiles."""

from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import random

from sichuan.reference_impl import is_hu

from .honors import standard_shapes
from .suited_regional import (WinContext, all_triplets, datang_payment, guilin_payment,
                              lishui_payment, pook_payment, seven_pairs, shaanxi_payment,
                              wake_award, wenxi_payment)

TILE_COUNT = 27
COPIES = Counter({tile: 4 for tile in range(TILE_COUNT)})
PASS = ('pass', None)
RON = ('ron', None)


class Profile(Enum):
    WENZHOU = 'wenzhou-fangpao'
    CHANGSHA = 'changsha-zhuanzhuan'
    TAISHUN = 'taishun'
    XUANHUA = 'xuanhua'
    SHAANXI = 'shaanxi'
    LISHUI = 'lishui-fangpao'
    WENXI = 'wenxi'
    GUILIN = 'guilin-zhuanzhuan'
    POOK = 'pook-dazhong'
    DATANG = 'datang-tuidaohu'


class PassLock(Enum):
    NONE = 'none'
    SUIT_UNTIL_DRAW = 'suit_until_draw'
    UNTIL_CONTROL = 'until_control'
    UNTIL_DISCARD = 'until_discard'
    UNTIL_DRAW = 'until_draw'
    TILE_UNTIL_CONTROL = 'tile_until_control'
    SCORE_UNTIL_DRAW = 'score_until_draw'


@dataclass(frozen=True)
class Rules:
    pass_lock: PassLock
    direct_kong_payers: int
    host_deals_first: bool = False
    nearest_robber_only: bool = False
    nearest_discard_winner_only: bool = False
    direct_kong_share: int = 1
    refund_kongs_on_draw: bool = False
    allow_seven_pairs: bool = True
    award_wake_tiles: bool = False
    regional_payment: Callable | None = None
    declare_ready: bool = False
    require_ready_to_win: bool = False
    ready_open_kong: bool = False
    passed_pon: bool = False


RULES = {
    Profile.WENZHOU: Rules(PassLock.SUIT_UNTIL_DRAW, direct_kong_payers=1),
    Profile.CHANGSHA: Rules(PassLock.NONE, direct_kong_payers=0),
    Profile.TAISHUN: Rules(PassLock.UNTIL_CONTROL, direct_kong_payers=1, host_deals_first=True),
    Profile.XUANHUA: Rules(PassLock.UNTIL_DISCARD, direct_kong_payers=3, host_deals_first=True,
                           nearest_robber_only=True),
    Profile.SHAANXI: Rules(PassLock.TILE_UNTIL_CONTROL, direct_kong_payers=1,
                            nearest_robber_only=True, nearest_discard_winner_only=True,
                            allow_seven_pairs=False, regional_payment=shaanxi_payment),
    Profile.LISHUI: Rules(PassLock.NONE, direct_kong_payers=1, direct_kong_share=3,
                           refund_kongs_on_draw=True, regional_payment=lishui_payment),
    Profile.WENXI: Rules(PassLock.NONE, direct_kong_payers=1,
                          refund_kongs_on_draw=True, regional_payment=wenxi_payment),
    Profile.GUILIN: Rules(PassLock.UNTIL_DRAW, direct_kong_payers=1, direct_kong_share=3,
                           host_deals_first=True, award_wake_tiles=True, regional_payment=guilin_payment),
    Profile.POOK: Rules(PassLock.SCORE_UNTIL_DRAW, direct_kong_payers=1, direct_kong_share=3,
                         regional_payment=pook_payment, declare_ready=True, passed_pon=True),
    Profile.DATANG: Rules(PassLock.UNTIL_DISCARD, direct_kong_payers=1, direct_kong_share=3,
                           host_deals_first=True, refund_kongs_on_draw=True, regional_payment=datang_payment,
                           declare_ready=True, require_ready_to_win=True, ready_open_kong=True),
}


@dataclass
class Player:
    hand: list[int] = field(default_factory=lambda: [0] * TILE_COUNT)
    melds: list[tuple[str, int]] = field(default_factory=list)
    declined_tiles: set[int] = field(default_factory=set)
    declined_win_points: int = 0
    declined_pon_tiles: set[int] = field(default_factory=set)
    ready_waits: frozenset[int] = field(default_factory=frozenset)
    ready_discard: int | None = None
    draws: int = 0
    score: int = 0


@dataclass(frozen=True)
class Offer:
    actor: int
    tile: int
    kind: str
    after_kong: bool = False
    unpaid_kong: bool = False


@dataclass(frozen=True)
class Win:
    actor: int
    provider: int | None
    kind: str
    fan: int | None
    patterns: tuple[str, ...]
    points: int
    payments: tuple[int, ...] | None = None


@lru_cache(maxsize=16384)
def _waits(tiles, melds):
    """Structural waits, excluding fifth copies already in this player's hand and melds."""
    hand = list(tiles)
    owned = Counter()
    for kind, tile in melds:
        owned[tile] += 3 if kind == 'peng' else 4
    waits = set()
    for tile in range(TILE_COUNT):
        if hand[tile] + owned[tile] >= 4:
            continue
        hand[tile] += 1
        if is_hu(hand, len(melds), -1):
            waits.add(tile)
        hand[tile] -= 1
    return frozenset(waits)


def wenzhou_fan(hand, melds, kind, *, opening=None, after_kong=False):
    """Compute the maximum applicable base fan, then the applicable event bonus."""
    if opening is not None:
        return 6, (opening,)

    meld_suits = {tile // 9 for _, tile in melds}
    flush = len(meld_suits | {tile // 9 for tile, count in enumerate(hand) if count}) == 1
    if seven_pairs(hand, melds):
        dragon = 4 in hand
        base = {(False, False): (3, '七对'), (False, True): (5, '清七对'),
                (True, False): (5, '龙七对'), (True, True): (6, '清龙七对')}
        fan, pattern = base[dragon, flush]
    elif len(melds) == 4:
        fan, pattern = (5, '清金钩钩') if len(meld_suits) == 1 else (3, '金钩钩')
    else:
        triplets = all_triplets(hand, melds)
        base = {(False, False): (1, '平胡'), (False, True): (3, '清一色'),
                (True, False): (2, '对对胡'), (True, True): (4, '清对')}
        fan, pattern = base[triplets, flush]

    bonus = {'zimo': '杠上花' if after_kong else '自摸',
             'ron': '杠上炮' if after_kong else None, 'rob_kong': '抢杠'}[kind]
    return (fan + 1, (pattern, bonus)) if bonus else (fan, (pattern,))


def taishun_points(hand, melds, kind, *, opening=None, after_kong=False):
    if opening is not None:
        return (opening,), 6
    if kind == 'ron' and after_kong:
        return ('杠上炮',), 3

    suits = {tile // 9 for _, tile in melds} | {tile // 9 for tile, count in enumerate(hand) if count}
    pairs = seven_pairs(hand, melds)
    if pairs and 4 in hand:
        base, pattern = 6, '龙七对'
    elif pairs:
        base, pattern = 3, '七对'
    elif len(suits) == 1:
        base, pattern = 3, '清一色'
    elif all_triplets(hand, melds):
        base, pattern = 2, '对对胡'
    else:
        base, pattern = 1, '平胡'
    if kind == 'ron':
        return (pattern,), base
    if kind == 'rob_kong':
        return (pattern, '抢杠'), 4 * base
    if after_kong:
        return (pattern, '杠上花'), 4 * base
    return (pattern, '自摸'), 2 * base


def xuanhua_points(melds, kind, *, heavenly=False):
    if heavenly:
        return ('天胡',), 6
    points, patterns = 1, ['平胡']
    if all(meld == 'gang_an' for meld, _ in melds):
        points *= 2
        patterns.append('门清')
    if kind != 'ron':
        points *= 2
        patterns.append('抢杠' if kind == 'rob_kong' else '自摸')
    return tuple(patterns), points


class SuitedGame:
    copies = COPIES

    def __init__(self, seed, dealer=None, *, profile, rules=None):
        self.profile = profile
        self.rules = RULES[profile] if rules is None else rules
        self.rng = random.Random(seed)
        if dealer is None:
            dealer = 0 if self.rules.host_deals_first else self.rng.randrange(4)
        self.dealer = dealer
        self.next_dealer = self.dealer
        self.wall = list(self.copies.elements())
        self.rng.shuffle(self.wall)
        self.players = [Player() for _ in range(4)]
        self.discards = [[] for _ in self.players]
        self.cur = self.dealer
        self.last_drawer = self.dealer
        self.phase = 'action'
        self.drawn_tile = None
        self.post_pon_tile = None
        self.replacement_draw = False
        self.kong_provider = None
        self.kong_deltas = [0] * 4
        self.opening = True
        self.discard_count = 0
        self.offer = None
        self.responses = deque()
        self.claim = None
        self.last_call = None
        self.winners = []
        self.win_details = []
        self.wake_indicators = []
        self.wake_results = {}
        self.revealed = {}
        self.claimed_tile = None
        self.reason_code = None
        for player in self.players:
            for _ in range(13):
                player.hand[self.wall.pop()] += 1
        self._draw(self.dealer)

    def scores(self):
        return [player.score for player in self.players]

    def physical_counts(self):
        counts = Counter(self.wall)
        counts.update(self.wake_indicators)
        for player, river in zip(self.players, self.discards):
            counts.update(river)
            counts.update({tile: count for tile, count in enumerate(player.hand) if count})
            for kind, tile in player.melds:
                counts[tile] += 3 if kind == 'peng' else 4
        if self.claimed_tile is not None:
            counts[self.claimed_tile] += 1
        return counts

    def _draw(self, actor, *, replacement=False, provider=None):
        if not self.wall:
            self._end('draw')
            return
        tile = self.wall.pop(0 if replacement else -1)
        player = self.players[actor]
        player.hand[tile] += 1
        player.draws += 1
        player.declined_win_points = 0
        player.declined_pon_tiles.clear()
        if self.rules.pass_lock in (PassLock.SUIT_UNTIL_DRAW, PassLock.UNTIL_CONTROL,
                                    PassLock.UNTIL_DRAW, PassLock.TILE_UNTIL_CONTROL):
            player.declined_tiles.clear()
        self.cur = actor
        self.last_drawer = actor
        self.phase = 'action'
        self.drawn_tile = tile
        self.post_pon_tile = None
        self.replacement_draw = replacement
        self.kong_provider = provider

    def _response_hand(self, actor, tile):
        hand = self.players[actor].hand.copy()
        hand[tile] += 1
        return hand

    def _discard_win_locked(self, player, tile):
        if self.rules.pass_lock is PassLock.SUIT_UNTIL_DRAW:
            return any(previous // 9 == tile // 9 for previous in player.declined_tiles)
        if self.rules.pass_lock is PassLock.TILE_UNTIL_CONTROL:
            return tile in player.declined_tiles
        return bool(player.declined_tiles)

    def _can_win(self, hand, melds):
        if not self.rules.allow_seven_pairs and seven_pairs(hand, melds):
            return next(standard_shapes(hand + [0] * 7, len(melds)), None) is not None
        return is_hu(hand, len(melds), -1)

    def _ready_discards(self, player, discards):
        melds = tuple(player.melds)
        hand = player.hand.copy()
        for _, tile in discards:
            hand[tile] -= 1
            if _waits(tuple(hand), melds):
                yield 'declare_ready', tile
            hand[tile] += 1

    def _preserves_waits(self, player, kind, tile):
        hand = player.hand.copy()
        melds = player.melds.copy()
        hand[tile] -= {'gang_an': 4, 'gang_bu': 1, 'gang_ming': 3}[kind]
        if kind == 'gang_bu':
            melds.remove(('peng', tile))
        melds.append((kind, tile))
        return _waits(tuple(hand), tuple(melds)) == player.ready_waits

    def _win_context(self, actor, kind, after_kong, provider=None, *, tiles_left=None, hits=0):
        player = self.players[actor]
        opening = None
        if self.opening and kind == 'zimo' and actor == self.dealer and player.draws == 1:
            opening = '天胡'
        elif self.opening and kind == 'ron' and self.discard_count == 1:
            opening = '地胡'
        return WinContext(
            actor, provider, self.dealer, kind, opening, after_kong, self.kong_provider,
            len(self.wall) if tiles_left is None else tiles_left, len(self.winners), hits,
            bool(player.ready_waits), provider is not None and bool(self.players[provider].ready_waits),
            tuple(self.winners),
        )

    def _offer_points(self, actor, offer=None):
        player = self.players[actor]
        if offer is None:
            hand = player.hand
            context = self._win_context(actor, 'zimo', self.replacement_draw)
        else:
            hand = self._response_hand(actor, offer.tile)
            context = self._win_context(actor, offer.kind, offer.after_kong, offer.actor)
        return self.rules.regional_payment(hand, player.melds, context)[1]

    def legal_actions(self):
        if self.phase == 'over':
            return None, []
        if self.offer is not None:
            actor = self.responses[0]
            player = self.players[actor]
            tile = self.offer.tile
            acts = [PASS]
            hand = self._response_hand(actor, tile)
            checks_pass = self.offer.kind == 'ron' or self.profile is Profile.DATANG
            locked = checks_pass and self._discard_win_locked(player, tile)
            if not locked and (not self.rules.require_ready_to_win or player.ready_waits) and (
                self._can_win(hand, player.melds)
            ):
                if self.rules.pass_lock is not PassLock.SCORE_UNTIL_DRAW or (
                    self._offer_points(actor, self.offer) > player.declined_win_points
                ):
                    acts.append(RON)
            if self.offer.kind == 'ron' and self.wall:
                if player.hand[tile] >= 2 and not player.ready_waits and tile not in player.declined_pon_tiles:
                    acts.append(('peng', None))
                if player.hand[tile] == 3 and (not player.ready_waits or (
                    self.rules.ready_open_kong and self._preserves_waits(player, 'gang_ming', tile)
                )):
                    acts.append(('zhigang', None))
            return actor, acts

        player = self.players[self.cur]
        acts = [('discard', tile) for tile, count in enumerate(player.hand)
                if count and (not player.ready_waits or tile == self.drawn_tile)]
        if self.rules.declare_ready and not player.ready_waits:
            acts.extend(list(self._ready_discards(player, acts)))
        if self.drawn_tile is not None:
            if (not self.rules.require_ready_to_win or player.ready_waits) and self._can_win(player.hand, player.melds):
                acts.append(('zimo', None))
            if self.wall:
                acts.extend(('ankan', tile) for tile, count in enumerate(player.hand) if count == 4 and (
                    not player.ready_waits or (tile == self.drawn_tile and self._preserves_waits(player, 'gang_an', tile))
                ))
                if ('peng', self.drawn_tile) in player.melds and (
                    not player.ready_waits or self._preserves_waits(player, 'gang_bu', self.drawn_tile)
                ):
                    acts.append(('bugang', self.drawn_tile))
        elif self.profile is Profile.TAISHUN and self.post_pon_tile is not None and self.wall:
            if player.hand[self.post_pon_tile]:
                acts.append(('bugang', self.post_pon_tile))
        return self.cur, acts

    def step(self, action):
        actor, acts = self.legal_actions()
        if action not in acts:
            raise ValueError('Illegal suited mahjong action')
        self.last_call = None
        kind, tile = action
        if self.offer is not None:
            self._respond(actor, kind, acts)
        elif kind in ('discard', 'declare_ready'):
            player = self.players[actor]
            if self.rules.pass_lock is PassLock.SCORE_UNTIL_DRAW and ('zimo', None) in acts:
                player.declined_win_points = self._offer_points(actor)
            player.hand[tile] -= 1
            if kind == 'declare_ready':
                player.ready_waits = _waits(tuple(player.hand), tuple(player.melds))
                player.ready_discard = len(self.discards[actor])
            if self.rules.pass_lock is PassLock.UNTIL_DISCARD:
                player.declined_tiles.clear()
            self.discard_count += 1
            self.discards[actor].append(tile)
            self._offer(Offer(actor, tile, 'ron', self.replacement_draw))
        elif kind == 'zimo':
            self._settle([actor], None, 'zimo', self.replacement_draw)
        elif kind == 'ankan':
            self._kong(actor, tile, 'gang_an')
        elif kind == 'bugang':
            unpaid = self.profile is Profile.TAISHUN and self.drawn_tile is None
            self._offer(Offer(actor, tile, 'rob_kong', unpaid_kong=unpaid))

    def _offer(self, offer):
        self.offer = offer
        self.responses = deque((offer.actor + step) % 4 for step in (1, 2, 3))
        self.claim = None
        self.phase = 'response' if offer.kind == 'ron' else 'rob_kong'
        self.drawn_tile = None
        self.post_pon_tile = None
        self.replacement_draw = False
        self.kong_provider = None

    def _respond(self, actor, kind, acts):
        offer = self.offer
        player = self.players[actor]
        self.responses.popleft()
        if kind == 'ron':
            self.winners.append(actor)
            nearest = self.rules.nearest_robber_only if offer.kind == 'rob_kong' else self.rules.nearest_discard_winner_only
            if nearest:
                self.responses.clear()
        elif RON in acts:
            if self.rules.pass_lock is PassLock.SCORE_UNTIL_DRAW:
                player.declined_win_points = self._offer_points(actor, offer)
            elif self.rules.pass_lock is not PassLock.NONE and (
                offer.kind == 'ron' or self.profile is Profile.DATANG
            ):
                player.declined_tiles.add(offer.tile)
        if self.rules.passed_pon and ('peng', None) in acts and kind not in ('peng', 'zhigang'):
            player.declined_pon_tiles.add(offer.tile)
        if kind in ('peng', 'zhigang') and not self.winners:
            if self.claim is None or (self.claim[1] == 'peng' and kind == 'zhigang'):
                self.claim = actor, kind
        if self.responses:
            return

        self.offer = None
        if self.winners:
            self._settle(self.winners, offer, offer.kind, offer.after_kong)
        elif offer.kind == 'rob_kong':
            self._kong(offer.actor, offer.tile, 'gang_bu', unpaid=offer.unpaid_kong)
        elif self.claim is None:
            self._draw((offer.actor + 1) % 4)
        else:
            caller, call = self.claim
            self.discards[offer.actor].pop()
            if call == 'peng':
                player = self.players[caller]
                player.hand[offer.tile] -= 2
                player.melds.append(('peng', offer.tile))
                player.declined_pon_tiles.clear()
                player.declined_win_points = 0
                if self.rules.pass_lock in (PassLock.UNTIL_CONTROL, PassLock.TILE_UNTIL_CONTROL):
                    player.declined_tiles.clear()
                self.post_pon_tile = offer.tile
                self.last_call = caller, 'peng'
                self.phase = 'action'
                self.cur = caller
                self.opening = False
            else:
                self._kong(caller, offer.tile, 'gang_ming', provider=offer.actor)

    def _pay(self, payer, winner, points):
        self.players[payer].score -= points
        self.players[winner].score += points

    def _kong_factor(self, provider):
        return 1

    def _kong(self, actor, tile, kind, provider=None, *, unpaid=False):
        player = self.players[actor]
        if kind == 'gang_bu':
            player.melds.remove(('peng', tile))
        player.hand[tile] -= {'gang_ming': 3, 'gang_bu': 1, 'gang_an': 4}[kind]
        player.melds.append((kind, tile))
        self.last_call = actor, kind
        self.opening = False
        if self.rules.direct_kong_payers and not unpaid:
            split_direct = (self.profile is Profile.DATANG and provider is not None
                            and self.players[provider].ready_waits)
            payers = (provider,) if provider is not None and self.rules.direct_kong_payers == 1 and not split_direct else (
                seat for seat in range(4) if seat != actor
            )
            points = {'gang_ming': 1 if split_direct else self.rules.direct_kong_share,
                      'gang_an': 2, 'gang_bu': 1}[kind]
            points *= self._kong_factor(provider)
            for payer in payers:
                self._pay(payer, actor, points)
                self.kong_deltas[payer] -= points
                self.kong_deltas[actor] += points
        self._draw(actor, replacement=True, provider=provider)

    def _score_win(self, actor, hand, kind, after_kong):
        player = self.players[actor]
        if self.profile is Profile.CHANGSHA:
            kongs = sum(kind.startswith('gang') for kind, _ in player.melds)
            return None, ('底分',) + ('带杠胡',) * bool(kongs), 1 + kongs

        opening = None
        if self.opening and not self.discards[actor]:
            if actor == self.dealer and kind == 'zimo' and player.draws == 1:
                opening = '天胡'
            elif self.profile is Profile.TAISHUN and kind == 'ron' and self.discard_count == 1:
                opening = '地胡'
            elif (self.profile is Profile.WENZHOU and actor != self.dealer and kind != 'rob_kong'
                  and player.draws == (1 if kind == 'zimo' else 0)):
                opening = '地胡'
        if self.profile is Profile.TAISHUN:
            patterns, points = taishun_points(hand, player.melds, kind, opening=opening, after_kong=after_kong)
            return None, patterns, points
        if self.profile is Profile.XUANHUA:
            patterns, points = xuanhua_points(player.melds, kind, heavenly=opening == '天胡')
            return None, patterns, points
        fan, patterns = wenzhou_fan(hand, player.melds, kind, opening=opening, after_kong=after_kong)
        return fan, patterns, 2 ** (fan - 1)

    def _settle(self, winners, offer, kind, after_kong):
        self.winners = list(winners)
        tiles_left = len(self.wall)
        for actor in self.winners:
            hand = self.players[actor].hand.copy() if offer is None else self._response_hand(actor, offer.tile)
            self.revealed[actor] = hand
            provider = None if offer is None else offer.actor
            if self.rules.regional_payment:
                self._settle_regional(actor, hand, provider, kind, after_kong, tiles_left)
                continue
            fan, patterns, points = self._score_win(actor, hand, kind, after_kong)
            if kind == 'rob_kong' and self.profile is Profile.TAISHUN:
                self._pay(provider, actor, 3 * points)
            elif provider is None or (kind == 'rob_kong' and self.profile is Profile.XUANHUA):
                for payer in range(4):
                    if payer != actor:
                        self._pay(payer, actor, points)
            else:
                self._pay(provider, actor, points)
            self.win_details.append(Win(actor, provider, kind, fan, patterns, points))
        if offer is not None:
            if offer.kind == 'ron':
                self.discards[offer.actor].pop()
            else:
                self.players[offer.actor].hand[offer.tile] -= 1
            self.claimed_tile = offer.tile
        self._end({'zimo': 'suited_self_draw', 'ron': 'suited_discard_win', 'rob_kong': 'rob_kong_win'}[kind])

    def _settle_regional(self, actor, hand, provider, kind, after_kong, tiles_left):
        player = self.players[actor]
        hits = 0
        if self.rules.award_wake_tiles:
            count, hits = wake_award(hand, player.melds, self.wall, self.wake_indicators)
            self.wake_results[actor] = count, hits
        ctx = self._win_context(actor, kind, after_kong, provider, tiles_left=tiles_left, hits=hits)
        patterns, points, shares = self.rules.regional_payment(hand, player.melds, ctx)
        for payer, amount in enumerate(shares):
            if amount:
                self._pay(payer, actor, amount)
        self.win_details.append(Win(actor, provider, kind, None, patterns, points, shares))

    def _end(self, reason):
        self.phase = 'over'
        self.drawn_tile = None
        self.reason_code = reason
        if reason == 'draw' and self.rules.refund_kongs_on_draw:
            for player, amount in zip(self.players, self.kong_deltas):
                player.score -= amount
        if self.profile is Profile.CHANGSHA:
            self.next_dealer = self.rng.choice([seat for seat in range(4) if seat not in self.winners])
        elif self.win_details:
            last = self.win_details[0]
            self.next_dealer = last.actor if len(self.winners) == 1 else last.provider
        elif self.profile is Profile.DATANG:
            self.next_dealer = self.last_drawer
