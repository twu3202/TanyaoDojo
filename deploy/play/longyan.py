"""Longyan gold-wildcard Mahjong with follow payments and traveling gold."""

from dataclasses import dataclass

from .flower_wall import FlowerWall
from .gold_opening import GoldOpening
from .honors import PASS, RON, HonorsGame, KongPoints, TableOptions, thirteen_orphans
from .honors_bonus import Score
from .wildcard_shapes import can_form_pairs, can_form_sets, can_form_standard, without_tile


WHITE = 31
TRAVEL = 'travel_gold'
TRAVEL_PATTERNS = ('游金', '双游', '三游')


def split_longyan(hand, gold):
    natural = list(hand)
    wild = natural[gold]
    natural[gold] = 0
    if gold != WHITE:
        natural[gold], natural[WHITE] = natural[WHITE], 0
    return tuple(natural), wild


def _shape(hand, melds, gold):
    if sum(hand) != 14 - 3 * len(melds):
        return None
    natural, wild = split_longyan(hand, gold)
    if not melds and can_form_pairs(natural, wild):
        return '七对'
    return '平胡' if can_form_standard(natural, wild) else None


def travel_wait(hand, melds, gold):
    if not hand[gold] or sum(hand) != 13 - 3 * len(melds):
        return False
    natural, wild = split_longyan(hand, gold)
    return can_form_sets(natural, wild - 1) or (not melds and can_form_pairs(natural, wild - 1))


@dataclass(frozen=True)
class LongyanRules:
    VARIANT: str
    half_self_draw: bool = False

    OPTIONS = TableOptions(chow=False, orphans=False, seven_pairs=True,
                           concealed_kong_faces=1, kong_requires_live_tile=True,
                           kong_points=KongPoints(1, 1, 2))
    WALL_RESERVE = 12
    MATCH_HANDS = 8
    MIN_FAAN = 0
    PASS_LOCK = False
    SWAP_LOCK = False

    def can_win(self, hand, melds, ctx, *, gold):
        if ctx.kind == 'rob_concealed' or (ctx.kind == 'ron' and (ctx.tile == gold or self.half_self_draw)):
            return False
        if ctx.kind == 'zimo' and hand[gold] == 3:
            return True
        return bool(_shape(hand, melds, gold) or (self.half_self_draw and thirteen_orphans(hand, melds)))

    @staticmethod
    def liability(melds, last):
        return ()

    @staticmethod
    def next_dealer(dealer, winners, provider):
        return (dealer + 1) % 4 if winners and dealer not in winners else dealer

    @staticmethod
    def payments(game, actor, provider, score):
        return tuple(score.share * (2 if game.dealer in (actor, seat) else 1) if seat != actor else 0
                     for seat in range(game.seat_count))

    def settle_hand(self, game, winners, offer):
        winner, = winners
        score = winner.score
        payments = list(self.payments(game, winner.actor, None, score))
        if winner.kind == 'ron':
            payments[offer.actor] += 1
        if winner.kind == 'rob_kong' and self.half_self_draw:
            payments = [0] * game.seat_count
            payments[offer.actor] = score.share * (6 if game.dealer in (winner.actor, offer.actor) else 4)
        return [(score, tuple(payments))]


LONGYAN = LongyanRules('longyan-new-chapter')
LONGYAN_HALF = LongyanRules('longyan-half-self-draw', half_self_draw=True)


class LongyanGame(GoldOpening, FlowerWall, HonorsGame):
    def __init__(self, seed, dealer, round_wind, *, rules):
        self.flower_kong_deltas = [0] * rules.OPTIONS.seat_count
        self.follow_deltas = [0] * rules.OPTIONS.seat_count
        self.follow_rounds = 0
        self.following = True
        self.passed_kongs = [set() for _ in range(rules.OPTIONS.seat_count)]
        self.travel_owner = None
        self.travel_level = 0
        self.gold_owner = None
        self.gold_count = 0
        self.replacement_kong = None
        self.passed_opening_heaven = False
        super().__init__(seed, dealer, round_wind, rules=rules)

    def _replacement_index(self):
        return -1

    def _deal_opening(self):
        self._deal_flower_racks(13)
        order = [(self.dealer + i) % self.seat_count for i in range(1, self.seat_count + 1)]
        self._start_gold_opening(-1, order, heaven=self.rules.half_self_draw)

    def _qualifies_for_gold(self, hand):
        return bool(_shape(hand, (), self.wildcard_tile))

    def _finish_gold_opening(self, actor, kind, hand):
        points, pattern, reason = {
            'three_gold': (4, '三金倒', 'three_gold_win'),
            'claim_gold': (4, '抢金', 'gold_claim'),
            'opening_heaven': (8, '天胡', 'suited_self_draw'),
        }[kind]
        self._settle_special(actor, None, reason, Score(points, (pattern,)), hand)

    def _reveal_flower(self, actor, tile):
        def awards(flowers):
            if self.rules.half_self_draw:
                return sum(set(range(start, start + 4)).issubset(flowers) for start in (34, 38))
            return max(0, len(flowers) - 3)

        flowers = self.flowers[actor]
        before = awards(flowers)
        super()._reveal_flower(actor, tile)
        if awards(flowers) > before:
            amounts = self.rules.payments(self, actor, None, Score(1, ()))
            self._transfer(actor, amounts, self.flower_kong_deltas)

    def _pay_kong(self, actor, meld):
        share = 2 if meld.kind == 'gang_an' else 1
        self._transfer(actor, self.rules.payments(self, actor, None, Score(share, ())), self.kong_deltas)

    def _draw(self, actor, kong_chain=0, provider=None):
        self.replacement_kong = self.last_call[1].kind if kong_chain and self.last_call else None
        super()._draw(actor, kong_chain, provider)

    def _response_actions(self):
        if self.winners or self.offer.tile == self.wildcard_tile:
            return self.responses[0], [PASS]
        actor, actions = super()._response_actions()
        if self.travel_level >= 2 and actor != self.travel_owner:
            return actor, [PASS]
        if self.rules.half_self_draw:
            if self.gold_owner not in (None, actor):
                actions = [a for a in actions if a[0] != 'peng']
            if self.offer.tile in self.passed_kongs[actor]:
                actions = [a for a in actions if a[0] != 'zhigang']
        return actor, actions

    def _respond(self, actor, action, actions):
        if self.rules.half_self_draw and ('zhigang', None) in actions and action != ('zhigang', None):
            self.passed_kongs[actor].add(self.offer.tile)
        super()._respond(actor, action, actions)

    def legal_actions(self):
        actor, actions = super().legal_actions()
        if self.phase != 'action' or self.offer is not None or self.drawn_tile is None:
            return actor, actions
        if self.rules.half_self_draw:
            actions = [a for a in actions if a[0] not in ('bugang', 'ankan') or a[1] not in self.passed_kongs[actor]]
        if self.travel_level >= 2 and actor != self.travel_owner:
            return actor, [a for a in actions if a[0] != 'bugang' and (self.travel_level == 3 or a[0] != 'ankan')]
        if self.travel_owner not in (None, actor) or self.travel_level >= 3 or self.remaining < self.seat_count:
            return actor, actions
        player = self.players[actor]
        hand = tuple(player.hand)
        gold = self.wildcard_tile
        actions += [
            (TRAVEL, tile) for kind, tile in actions if kind == 'discard'
            and (not self.travel_level or self.drawn_tile == gold == tile)
            and travel_wait(without_tile(hand, tile), player.melds, gold)
        ]
        return actor, actions

    def step(self, action):
        if action[0] != TRAVEL:
            if self.phase == 'gold_opening' and self.gold_queue[0][1] and action == PASS:
                self.passed_opening_heaven = True
            super().step(action)
            return
        actor, actions = self.legal_actions()
        if action not in actions:
            raise ValueError('Illegal traveling gold action')
        self.last_call = None
        self.travel_owner = actor
        self.travel_level += 1
        self._throw(actor, action[1])

    def _discard(self, actor, tile):
        if actor == self.travel_owner:
            self.travel_owner = None
            self.travel_level = 0
        self._throw(actor, tile)

    def _throw(self, actor, tile):
        if tile == self.wildcard_tile:
            self.gold_count = self.gold_count + 1 if self.gold_owner == actor else 1
            self.gold_owner = actor
        elif self.gold_owner == actor:
            self.gold_owner = None
            self.gold_count = 0
        super()._discard(actor, tile)
        if not self.following or not self.uninterrupted:
            return
        river = self.discards[actor]
        number = len(river) - 1
        if actor != self.dealer and tile != self.discards[self.dealer][number]:
            self.following = False
        elif actor == (self.dealer - 1) % self.seat_count:
            share = (2 if tile < 27 else 1) * 2 ** self.follow_rounds
            self.follow_rounds += 1
            delta = [share if seat != self.dealer else -3 * share for seat in range(self.seat_count)]
            for seat, amount in enumerate(delta):
                self.players[seat].score += amount
                self.follow_deltas[seat] += amount

    def _score(self, actor, hand, melds, ctx):
        drew = ctx.kind == 'zimo'
        if drew and ctx.opening == '天胡' and self.passed_opening_heaven:
            return None
        if self.travel_level >= 2 and actor != self.travel_owner:
            if not drew or (self.travel_level == 3 and self.replacement_kong != 'gang_an'):
                return None
        gold = self.wildcard_tile
        traveled = drew and actor == self.travel_owner and travel_wait(without_tile(tuple(hand), ctx.tile), melds, gold)
        if self.gold_owner not in (None, actor):
            exceptional_rob = self.rules.half_self_draw and self.gold_count == 1 and ctx.kind == 'rob_kong' and thirteen_orphans(hand, melds)
            if not drew and not exceptional_rob:
                return None
            if drew and self.rules.half_self_draw:
                if self.gold_count == 1 and not (ctx.kong_chain or traveled or hand[gold] == 3):
                    return None
                if self.gold_count > 1 and self.replacement_kong != 'gang_an':
                    return None
        if not self.rules.can_win(hand, melds, ctx, gold=gold):
            return None
        if ctx.kind == 'rob_kong':
            return Score(2, ('抢杠胡',))
        shape = _shape(hand, melds, gold)
        if drew and ctx.opening == '天胡' and shape:
            return Score(8, ('天胡',))
        if drew and self.rules.half_self_draw and thirteen_orphans(hand, melds):
            return Score(16, ('十三幺',))
        if traveled:
            return Score(2 ** (self.travel_level + 1), (TRAVEL_PATTERNS[self.travel_level - 1],))
        if drew and hand[gold] == 3:
            return Score(4, ('三金倒',))
        return Score(2 if drew else 1, (shape,))
