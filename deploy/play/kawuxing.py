"""Three-seat Suizhou, Xiangyang and Shiyan Ka Wu Xing profiles."""

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import ClassVar

from .honors import (DRAGONS, PASS, RON, ZIMO, HonorsGame, KongPoints, TableOptions,
                     middle_five, ready_discards, seven_pairs, standard_shapes)


TILE_KINDS = (*range(9, 27), *sorted(DRAGONS))
PAIR_PATTERNS = ('七对', '龙七对', '双龙七对', '三龙七对')
KONG_POINTS = KongPoints(2, 1, 2, single_direct_payer=True)


@dataclass(frozen=True)
class Score:
    fan: int
    patterns: tuple[str, ...]
    ready: bool
    tier: ClassVar[None] = None

    @property
    def share(self):
        return 2 ** self.fan


def _set_patterns(sets, melds, pair, winning_tile, middle_five_fan, stack_hook):
    pungs = {meld.tile for meld in sets + melds if meld.kind != 'chi'}
    hook = len(melds) == 4 and all(meld.kind != 'gang_an' for meld in melds)
    patterns = []
    if len(pungs) == 4 and (stack_hook or not hook):
        patterns.append((1, '对对胡'))
    if DRAGONS <= pungs:
        patterns.append((3, '大三元'))
    elif pair in DRAGONS and len(pungs & DRAGONS) == 2:
        patterns.append((2, '小三元'))
    if middle_five(sets, winning_tile):
        patterns.append((middle_five_fan, '卡五星'))
    if hook:
        patterns.append((2, '手抓一'))
    return patterns


def _four_copies(hand, melds, winning_tile, broad):
    kinds = range(len(hand)) if broad else (winning_tile,)
    opens = {meld.tile for meld in melds if meld.kind == 'peng'}
    hidden = sum(hand[tile] == 4 for tile in kinds)
    exposed = sum(tile in opens and hand[tile] == 1 for tile in kinds)
    return [(2, '暗四归一')] * hidden + [(1, '明四归一')] * exposed


@lru_cache(maxsize=16384)
def _best_shape(hand, melds, winning_tile, broad_on_big, middle_five_fan,
                *, all_four_copies=False, stack_hook=True, max_pair_quads=3, dragon_pairs=False):
    used = {tile for tile, count in enumerate(hand) if count}
    used.update(tile for meld in melds for tile in meld.tiles)
    flush = all(tile < 27 for tile in used) and len({tile // 9 for tile in used}) == 1
    common = [(2, '清一色')] if flush else []
    candidates = []
    # A seven-pairs hand uses its own shape and never scores the same quad as four-in-one.
    if seven_pairs(hand, melds):
        quads = hand.count(4)
        name = '三元七对' if dragon_pairs and not quads and all(hand[tile] >= 2 for tile in DRAGONS) else PAIR_PATTERNS[quads]
        candidates.append(common + [(2 + min(quads, max_pair_quads), name)])
    else:
        for pair, sets in standard_shapes(hand, len(melds)):
            shape = _set_patterns(sets, melds, pair, winning_tile, middle_five_fan, stack_hook)
            extras = _four_copies(hand, melds, winning_tile, all_four_copies or flush or (broad_on_big and bool(shape)))
            candidates.append(common + shape + extras)
    if not candidates:
        return None
    return max(candidates, key=lambda patterns: sum(fan for fan, _ in patterns))


@dataclass(frozen=True)
class KaWuXingRules:
    VARIANT: str
    OPTIONS: TableOptions
    xiangyang: bool = False
    nearest_winner: bool = False
    open_ready: bool = True
    bill_first_ready: bool = True
    rob_concealed: bool = False
    min_ron_fan: int = 0
    middle_five_fan: int = 2
    MIN_FAAN: ClassVar[int] = 0
    WALL_RESERVE: ClassVar[int] = 0
    MATCH_HANDS: ClassVar[None] = None
    SWAP_LOCK: ClassVar[bool] = False
    PASS_LOCK: ClassVar[bool] = False

    def shape(self, hand, melds, winning_tile):
        return _best_shape(hand, melds, winning_tile, self.xiangyang, self.middle_five_fan)

    def faan(self, hand, melds, ctx):
        if ctx.kind == 'rob_concealed' and not self.rob_concealed:
            return None
        shape = self.shape(tuple(hand), tuple(melds), ctx.tile)
        if shape is None:
            return None
        patterns = list(shape)
        if ctx.ready:
            patterns.append((1, '亮倒'))
        if ctx.kind in ('rob_kong', 'rob_concealed'):
            patterns.append((1, '抢杠'))
        if ctx.kind == 'zimo':
            if ctx.kong_chain:
                patterns.append((1, '杠上花'))
            if ctx.last_live_draw:
                patterns.append((1, '海底'))
        if not self.xiangyang and ctx.kind == 'ron' and ctx.kong_chain:
            patterns.append((1, '杠上炮'))
        fan = sum(fan for fan, _ in patterns)
        if ctx.kind == 'ron' and fan < self.min_ron_fan:
            return None
        if self.xiangyang and fan > 3:
            fan = 3
            patterns.append((0, '八倍封顶'))
        return Score(fan, tuple(pattern for _, pattern in patterns) or ('平胡',), ctx.ready)

    def settle_hand(self, game, winners, offer):
        provider_ready = offer is not None and game.players[offer.actor].open_ready_hand is not None
        outcomes = []
        for winner in winners:
            score = winner.score
            if offer is None:
                payments = tuple(score.share if seat != winner.actor else 0 for seat in range(game.seat_count))
            else:
                covered = winner.kind == 'ron' and score.ready and not (self.xiangyang and provider_ready)
                amount = score.share * (game.seat_count - 1 if covered else 1)
                payments = tuple(amount if seat == offer.actor else 0 for seat in range(game.seat_count))
            outcomes.append((score, payments))
        return outcomes

    @staticmethod
    def liability(melds, last):
        return ()

    @staticmethod
    def next_dealer(dealer, winners, provider):
        if not winners:
            return dealer
        return winners[0] if len(winners) == 1 else provider


def _options(*, xiangyang):
    return TableOptions(tile_kinds=TILE_KINDS, seat_count=3, chow=False, seven_pairs=True, orphans=False,
                        show_winds=False, drawn_added_kong_only=True, kong_requires_live_tile=True,
                        kong_points=KONG_POINTS, pass_win_all=xiangyang, pass_win_higher_all=not xiangyang)


SUIZHOU = KaWuXingRules('suizhou-kawuxing', _options(xiangyang=False), rob_concealed=True)
XIANGYANG = KaWuXingRules('xiangyang-kawuxing', _options(xiangyang=True), xiangyang=True, nearest_winner=True)
SHIYAN = KaWuXingRules('shiyan-kawuxing', replace(SUIZHOU.OPTIONS, pass_win_higher_all=False),
                      open_ready=False, min_ron_fan=1, middle_five_fan=1)


class KaWuXingGame(HonorsGame):
    def __init__(self, seed, dealer, round_wind, *, rules):
        self.first_ready = None
        self.ready_deltas = [0] * rules.OPTIONS.seat_count
        super().__init__(seed, dealer, round_wind, rules=rules)

    def legal_actions(self):
        actor, actions = super().legal_actions()
        if self.phase != 'action' or not self.rules.open_ready:
            return actor, actions
        player = self.players[actor]
        if player.open_ready_hand is not None:
            return actor, [action for action in actions if action in (ZIMO, ('discard', self.drawn_tile))]
        return actor, actions + list(ready_discards(player, actions, self.options))

    def _response_actions(self):
        actor, actions = super()._response_actions()
        if self.players[actor].open_ready_hand is not None or self.winners:
            return actor, [action for action in actions if action in (PASS, RON)]
        tile = self.offer.tile
        if self.remaining == 0 and self.offer.kind == 'discard' and self.players[actor].hand[tile] >= 2:
            actions.append(('peng', None))
        return actor, actions

    def step(self, action):
        if action[0] != 'declare_ready':
            super().step(action)
            return
        actor, actions = self.legal_actions()
        if action not in actions:
            raise ValueError('Illegal open-ready declaration')
        tile = action[1]
        self.last_call = None
        self._discard(actor, tile)
        self.players[actor].open_ready_hand = tuple(self.players[actor].hand)
        if self.first_ready is None:
            self.first_ready = actor

    def _respond(self, actor, action, actions):
        if action == RON and self.rules.nearest_winner:
            self.responses.clear()
            self.responses.append(actor)
        super()._respond(actor, action, actions)

    def _end(self, reason):
        super()._end(reason)
        if reason == 'draw' and self.first_ready is not None and self.rules.bill_first_ready:
            payments = tuple(int(seat == self.first_ready) for seat in range(self.seat_count))
            for seat in range(self.seat_count):
                if seat != self.first_ready:
                    self._transfer(seat, payments, self.ready_deltas)
