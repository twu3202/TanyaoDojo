"""Two-suit continuous-win play over the suited discard and kong engine."""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from .honors import Meld, seven_pairs, standard_shapes
from .suited import PASS, RON, PassLock, Rules, SuitedGame, Win, _waits


VARIANT = 'kaxinwu-blood-flow'
RULES = Rules(PassLock.NONE, direct_kong_payers=1, direct_kong_share=2, host_deals_first=True, passed_pon=True)
TILES = range(9, 27)
ZIMO = ('zimo', None)
PON = ('peng', None)
OPEN_KONG = ('zhigang', None)
KONG_TILES = {'gang_ming': 3, 'gang_an': 4, 'gang_bu': 1}
SEVEN_PAIRS = ('七对', '龙七对', '双龙七对', '三龙七对')
KONG_BLOOM = ('杠上花', '双杠上花', '三杠上花', '四杠上花')


@lru_cache(maxsize=8192)
def _base_score(hand, melds, winning_tile):
    """Best basic pattern plus pure suit, before roots or win-event fan."""
    suits = {tile // 9 for _, tile in melds} | {tile // 9 for tile, count in enumerate(hand) if count}
    if seven_pairs(hand, melds):
        quads = hand.count(4)
        fan, pattern = 2 + quads, SEVEN_PAIRS[quads]
        roots = 0
    else:
        fan, pattern = 1, '平胡'
        middle_five = Meld('chi', winning_tile - 1) if winning_tile % 9 == 4 else None
        for _, sets in standard_shapes(hand, len(melds)):
            if middle_five is not None and middle_five in sets:
                fan, pattern = 2, '卡心五'
                break
            if all(group.kind == 'triplet' for group in sets):
                fan, pattern = 2, '对对胡'
        roots = hand.count(4)
    if len(suits) == 1:
        return fan + 1, (pattern, '清一色'), roots
    return fan, (pattern,), roots


@dataclass(frozen=True, kw_only=True)
class FlowWin(Win):
    tile: int


class KaxinwuGame(SuitedGame):
    copies = Counter({tile: 4 for tile in TILES})

    def __init__(self, seed, dealer=None):
        self.kong_chain = 0
        self.won_ranks = [set() for _ in range(4)]
        self.won_tiles = []
        self.win_counts = [0] * 4
        self.best_scored_shape = [0] * 4
        self.ready_deltas = [0] * 4
        self.passed_wins = [{} for _ in range(4)]
        self.passed_self_draw = [None] * 4
        self.first_equal_seen = [False] * 4
        self.forbidden_added_kongs = [set() for _ in range(4)]
        super().__init__(seed, dealer, profile=VARIANT, rules=RULES)

    def physical_counts(self):
        counts = super().physical_counts()
        counts.update(self.won_tiles)
        return counts

    def _clear_win_passes(self, actor):
        self.passed_wins[actor].clear()
        self.passed_self_draw[actor] = None
        self.first_equal_seen[actor] = False

    def _draw(self, actor, *, replacement=False, provider=None):
        if self.wall:
            if replacement:
                self.kong_chain = 1 if provider is not None else self.kong_chain + 1
            else:
                self.kong_chain = 0
            self._clear_win_passes(actor)
        super()._draw(actor, replacement=replacement, provider=provider)

    def _kong_factor(self, provider):
        return 2 if provider is None and self.kong_chain else 1

    def _score(self, hand, melds, tile, kind, chain=0):
        ready_fan, patterns, roots = _base_score(tuple(hand), tuple(melds), tile)
        if roots:
            patterns += (f'鬼x{roots}',)
        fan = ready_fan + roots
        if kind == 'rob_kong':
            return fan + 1, patterns + ('抢杠',), ready_fan
        if kind == 'ron' and chain:
            return fan + 1, patterns + ('杠上炮',), ready_fan
        if kind == 'zimo' and chain:
            return fan + chain, patterns + (KONG_BLOOM[chain - 1],), ready_fan
        return fan, patterns, ready_fan

    def _offer_fan(self, actor, offer):
        hand = self._response_hand(actor, offer.tile)
        return self._score(hand, self.players[actor].melds, offer.tile, offer.kind, offer.after_kong)[0]

    def _can_take_offer(self, actor, offer):
        hand = self._response_hand(actor, offer.tile)
        if not self._can_win(hand, self.players[actor].melds):
            return False
        fan = self._offer_fan(actor, offer)
        if fan <= self.passed_wins[actor].get(offer.tile, -1):
            return False
        skipped = self.passed_self_draw[actor]
        return skipped is None or fan > skipped or (fan == skipped and not self.first_equal_seen[actor])

    def _wait_tiles(self, hand, melds):
        return _waits(tuple(hand), tuple(melds)).intersection(TILES)

    def _keeps_won_waits(self, actor, kind, tile):
        player = self.players[actor]
        hand, melds = player.hand.copy(), player.melds.copy()
        hand[tile] -= KONG_TILES[kind]
        if kind == 'gang_bu':
            melds.remove(('peng', tile))
        melds.append((kind, tile))
        # The operator's example allows the waits to narrow if every won rank remains.
        return self.won_ranks[actor] <= self._wait_tiles(hand, melds)

    def legal_actions(self):
        if self.phase == 'over':
            return None, []
        if self.offer is not None:
            actor = self.responses[0]
            return actor, self._offered_actions(actor)
        actor = self.cur
        player = self.players[actor]
        locked = bool(self.won_ranks[actor])
        acts = [('discard', tile) for tile in TILES
                if player.hand[tile] and (not locked or tile == self.drawn_tile)]
        if self.drawn_tile is not None and self._can_win(player.hand, player.melds):
            acts.append(ZIMO)
        if self.wall:
            for tile in TILES if not locked else (self.drawn_tile,):
                if tile is None:
                    continue
                if player.hand[tile] == 4 and (not locked or self._keeps_won_waits(actor, 'gang_an', tile)):
                    acts.append(('ankan', tile))
                if player.hand[tile] and ('peng', tile) in player.melds and tile not in self.forbidden_added_kongs[actor]:
                    if not locked or self._keeps_won_waits(actor, 'gang_bu', tile):
                        acts.append(('bugang', tile))
        return actor, acts

    def _offered_actions(self, actor):
        offer = self.offer
        player = self.players[actor]
        acts = [RON] if self._can_take_offer(actor, offer) else []
        if offer.kind != 'ron' or self.winners or not self.wall:
            return [PASS, *acts]
        tile = offer.tile
        locked = bool(self.won_ranks[actor])
        if player.hand[tile] == 3 and (not locked or self._keeps_won_waits(actor, 'gang_ming', tile)):
            acts.append(OPEN_KONG)
            if locked:
                return acts
        if not locked and player.hand[tile] >= 2 and tile not in player.declined_pon_tiles:
            acts.append(PON)
        return [PASS, *acts]

    def step(self, action):
        if self.offer is None and action[0] == 'discard':
            actor, acts = self.legal_actions()
            if action in acts and ZIMO in acts:
                player = self.players[actor]
                fan, _, _ = self._score(player.hand, player.melds, self.drawn_tile, 'zimo', self.kong_chain)
                self.passed_self_draw[actor] = fan
        super().step(action)

    def _respond(self, actor, kind, acts):
        offer = self.offer
        if RON in acts:
            fan = self._offer_fan(actor, offer)
            if self.passed_self_draw[actor] == fan:
                self.first_equal_seen[actor] = True
            if kind != 'ron':
                self.passed_wins[actor][offer.tile] = fan
        super()._respond(actor, kind, acts)
        if self.last_call is not None and self.last_call[1] == 'peng':
            caller, _ = self.last_call
            self.kong_chain = 0
            self._clear_win_passes(caller)
            if self.players[caller].hand[offer.tile] == 1:
                self.forbidden_added_kongs[caller].add(offer.tile)

    def _settle(self, winners, offer, kind, after_kong):
        provider = None if offer is None else offer.actor
        tile = self.drawn_tile if offer is None else offer.tile
        chain = self.kong_chain if kind == 'zimo' else int(after_kong)
        for actor in winners:
            player = self.players[actor]
            hand = player.hand if offer is None else self._response_hand(actor, tile)
            fan, patterns, base = self._score(hand, player.melds, tile, kind, chain)
            points = 2 ** fan
            payers = [seat for seat in range(4) if seat != actor] if provider is None else [provider]
            payments = tuple(points if seat in payers else 0 for seat in range(4))
            for payer in payers:
                self._pay(payer, actor, points)
            self.win_details.append(FlowWin(actor, provider, kind, fan, patterns, points, payments, tile=tile))
            self.win_counts[actor] += 1
            self.won_ranks[actor].add(tile)
            self.best_scored_shape[actor] = max(base, self.best_scored_shape[actor])
        self.won_tiles.append(tile)
        if kind == 'ron':
            self.discards[provider].pop()
        else:
            source = winners[0] if provider is None else provider
            self.players[source].hand[tile] -= 1
        next_actor = provider if len(winners) > 1 else ((winners[0] if provider is None else provider) + 1) % 4
        self.winners = []
        self._draw(next_actor)

    def _ready_value(self, actor):
        if self.win_counts[actor]:
            return 2 ** self.best_scored_shape[actor]
        player = self.players[actor]
        waits = self._wait_tiles(player.hand, player.melds)
        return max((2 ** _base_score(tuple(self._response_hand(actor, tile)), tuple(player.melds), tile)[0]
                    for tile in waits), default=0)

    def _end(self, reason):
        values = [self._ready_value(actor) for actor in range(4)]
        for payer, value in enumerate(values):
            if value:
                continue
            for receiver, points in enumerate(values):
                if points:
                    self._pay(payer, receiver, points)
                    self.ready_deltas[payer] -= points
                    self.ready_deltas[receiver] += points
        self.reason_code = 'repeat_win_end'
        self.phase = 'over'
        self.drawn_tile = None
        self.next_dealer = self.last_drawer
