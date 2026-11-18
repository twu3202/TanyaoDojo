"""Four-player, first-win settlement engine for 108 suited tiles."""

from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
import random

from sichuan.reference_impl import is_hu

TILE_COUNT = 27
COPIES = Counter({tile: 4 for tile in range(TILE_COUNT)})
PASS = ('pass', None)
RON = ('ron', None)


class Profile(Enum):
    WENZHOU = 'wenzhou-fangpao'
    CHANGSHA = 'changsha-zhuanzhuan'


@dataclass
class Player:
    hand: list[int] = field(default_factory=lambda: [0] * TILE_COUNT)
    melds: list[tuple[str, int]] = field(default_factory=list)
    declined_suits: set[int] = field(default_factory=set)
    draws: int = 0
    score: int = 0


@dataclass(frozen=True)
class Offer:
    actor: int
    tile: int
    kind: str
    after_kong: bool = False


@dataclass(frozen=True)
class Win:
    actor: int
    provider: int | None
    kind: str
    fan: int
    patterns: tuple[str, ...]
    points: int


def wenzhou_fan(hand, melds, kind, *, opening=None, after_kong=False):
    """Compute the maximum applicable base fan, then the applicable event bonus."""
    if opening is not None:
        return 6, (opening,)

    meld_suits = {tile // 9 for _, tile in melds}
    flush = len(meld_suits | {tile // 9 for tile, count in enumerate(hand) if count}) == 1
    if not melds and all(count % 2 == 0 for count in hand):
        dragon = 4 in hand
        base = {(False, False): (3, '七对'), (False, True): (5, '清七对'),
                (True, False): (5, '龙七对'), (True, True): (6, '清龙七对')}
        fan, pattern = base[dragon, flush]
    elif len(melds) == 4:
        fan, pattern = (5, '清金钩钩') if len(meld_suits) == 1 else (3, '金钩钩')
    else:
        triplets = sorted(count for count in hand if count) == [2] + [3] * (4 - len(melds))
        base = {(False, False): (1, '平胡'), (False, True): (3, '清一色'),
                (True, False): (2, '对对胡'), (True, True): (4, '清对')}
        fan, pattern = base[triplets, flush]

    bonus = {'zimo': '杠上花' if after_kong else '自摸',
             'ron': '杠上炮' if after_kong else None, 'rob_kong': '抢杠'}[kind]
    return (fan + 1, (pattern, bonus)) if bonus else (fan, (pattern,))


class SuitedGame:
    def __init__(self, seed, dealer=None, *, profile):
        self.profile = profile
        self.rng = random.Random(seed)
        self.dealer = self.rng.randrange(4) if dealer is None else dealer
        self.next_dealer = self.dealer
        self.wall = list(COPIES.elements())
        self.rng.shuffle(self.wall)
        self.players = [Player() for _ in range(4)]
        self.discards = [[] for _ in self.players]
        self.cur = self.dealer
        self.phase = 'action'
        self.drawn_tile = None
        self.replacement_draw = False
        self.opening = True
        self.offer = None
        self.responses = deque()
        self.claim = None
        self.last_call = None
        self.winners = []
        self.win_details = []
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
        for player, river in zip(self.players, self.discards):
            counts.update(river)
            counts.update({tile: count for tile, count in enumerate(player.hand) if count})
            for kind, tile in player.melds:
                counts[tile] += 3 if kind == 'peng' else 4
        if self.claimed_tile is not None:
            counts[self.claimed_tile] += 1
        return counts

    def _draw(self, actor, *, replacement=False):
        if not self.wall:
            self._end('draw')
            return
        tile = self.wall.pop(0 if replacement else -1)
        player = self.players[actor]
        player.hand[tile] += 1
        player.draws += 1
        player.declined_suits.clear()
        self.cur = actor
        self.phase = 'action'
        self.drawn_tile = tile
        self.replacement_draw = replacement

    def _response_hand(self, actor, tile):
        hand = self.players[actor].hand.copy()
        hand[tile] += 1
        return hand

    def legal_actions(self):
        if self.phase == 'over':
            return None, []
        if self.offer is not None:
            actor = self.responses[0]
            player = self.players[actor]
            tile = self.offer.tile
            acts = [PASS]
            locked = self.offer.kind == 'ron' and tile // 9 in player.declined_suits
            if not locked and is_hu(self._response_hand(actor, tile), len(player.melds), -1):
                acts.append(RON)
            if self.offer.kind == 'ron' and self.wall:
                if player.hand[tile] >= 2:
                    acts.append(('peng', None))
                if player.hand[tile] == 3:
                    acts.append(('zhigang', None))
            return actor, acts

        player = self.players[self.cur]
        acts = [('discard', tile) for tile, count in enumerate(player.hand) if count]
        if self.drawn_tile is not None:
            if is_hu(player.hand, len(player.melds), -1):
                acts.append(('zimo', None))
            if self.wall:
                acts.extend(('ankan', tile) for tile, count in enumerate(player.hand) if count == 4)
                if ('peng', self.drawn_tile) in player.melds:
                    acts.append(('bugang', self.drawn_tile))
        return self.cur, acts

    def step(self, action):
        actor, acts = self.legal_actions()
        if action not in acts:
            raise ValueError('Illegal suited mahjong action')
        self.last_call = None
        kind, tile = action
        if self.offer is not None:
            self._respond(actor, kind, RON in acts)
        elif kind == 'discard':
            self.players[actor].hand[tile] -= 1
            self.discards[actor].append(tile)
            self._offer(Offer(actor, tile, 'ron', self.replacement_draw))
        elif kind == 'zimo':
            self._settle([actor], None, 'zimo', self.replacement_draw)
        elif kind == 'ankan':
            self._kong(actor, tile, 'gang_an')
        elif kind == 'bugang':
            self._offer(Offer(actor, tile, 'rob_kong'))

    def _offer(self, offer):
        self.offer = offer
        self.responses = deque((offer.actor + step) % 4 for step in (1, 2, 3))
        self.claim = None
        self.phase = 'response' if offer.kind == 'ron' else 'rob_kong'
        self.drawn_tile = None
        self.replacement_draw = False

    def _respond(self, actor, kind, could_win):
        offer = self.offer
        self.responses.popleft()
        if kind == 'ron':
            self.winners.append(actor)
        elif could_win and offer.kind == 'ron' and self.profile is Profile.WENZHOU:
            self.players[actor].declined_suits.add(offer.tile // 9)
        if kind in ('peng', 'zhigang') and not self.winners:
            if self.claim is None or (self.claim[1] == 'peng' and kind == 'zhigang'):
                self.claim = actor, kind
        if self.responses:
            return

        self.offer = None
        if self.winners:
            self._settle(self.winners, offer, offer.kind, offer.after_kong)
        elif offer.kind == 'rob_kong':
            self._kong(offer.actor, offer.tile, 'gang_bu')
        elif self.claim is None:
            self._draw((offer.actor + 1) % 4)
        else:
            caller, call = self.claim
            self.discards[offer.actor].pop()
            if call == 'peng':
                player = self.players[caller]
                player.hand[offer.tile] -= 2
                player.melds.append(('peng', offer.tile))
                self.last_call = caller, 'peng'
                self.phase = 'action'
                self.cur = caller
                self.opening = False
            else:
                self._kong(caller, offer.tile, 'gang_ming', provider=offer.actor)

    def _pay(self, payer, winner, points):
        self.players[payer].score -= points
        self.players[winner].score += points

    def _kong(self, actor, tile, kind, provider=None):
        player = self.players[actor]
        if kind == 'gang_bu':
            player.melds.remove(('peng', tile))
        player.hand[tile] -= {'gang_ming': 3, 'gang_bu': 1, 'gang_an': 4}[kind]
        player.melds.append((kind, tile))
        self.last_call = actor, kind
        self.opening = False
        if self.profile is Profile.WENZHOU:
            if provider is not None:
                self._pay(provider, actor, 1)
            else:
                for payer in range(4):
                    if payer != actor:
                        self._pay(payer, actor, 2 if kind == 'gang_an' else 1)
        self._draw(actor, replacement=True)

    def _score_win(self, actor, hand, kind, after_kong):
        player = self.players[actor]
        if self.profile is Profile.CHANGSHA:
            kongs = sum(kind.startswith('gang') for kind, _ in player.melds)
            return 1 + kongs, ('底分',) + ('带杠胡',) * bool(kongs), 1 + kongs

        opening = None
        if self.opening and not self.discards[actor]:
            if actor == self.dealer and kind == 'zimo' and player.draws == 1:
                opening = '天胡'
            elif actor != self.dealer and kind != 'rob_kong' and player.draws == (1 if kind == 'zimo' else 0):
                opening = '地胡'
        fan, patterns = wenzhou_fan(hand, player.melds, kind, opening=opening, after_kong=after_kong)
        return fan, patterns, 2 ** (fan - 1)

    def _settle(self, winners, offer, kind, after_kong):
        self.winners = list(winners)
        for actor in self.winners:
            hand = self.players[actor].hand.copy() if offer is None else self._response_hand(actor, offer.tile)
            self.revealed[actor] = hand
            fan, patterns, points = self._score_win(actor, hand, kind, after_kong)
            provider = None if offer is None else offer.actor
            payers = [seat for seat in range(4) if seat != actor] if provider is None else [provider]
            for payer in payers:
                self._pay(payer, actor, points)
            self.win_details.append(Win(actor, provider, kind, fan, patterns, points))
        if offer is not None:
            if offer.kind == 'ron':
                self.discards[offer.actor].pop()
            else:
                self.players[offer.actor].hand[offer.tile] -= 1
            self.claimed_tile = offer.tile
        self._end({'zimo': 'suited_self_draw', 'ron': 'suited_discard_win', 'rob_kong': 'rob_kong_win'}[kind])

    def _end(self, reason):
        self.phase = 'over'
        self.drawn_tile = None
        self.reason_code = reason
        if self.profile is Profile.CHANGSHA:
            self.next_dealer = self.rng.choice([seat for seat in range(4) if seat not in self.winners])
        elif self.win_details:
            last = self.win_details[0]
            self.next_dealer = last.actor if len(self.winners) == 1 else last.provider
