"""Four-seat Sichuan exchange and Yunnan blood battle rooms."""
from itertools import combinations

from sichuan.reference_impl import FAN_CAP, _decomp_all_kotsu, is_hu

from .errors import ClientError
from .sichuan_game import WebGame


SICHUAN_EXCHANGE = 'sichuan-exchange'
YUNNAN = 'yunnan-xuezhan'
EXCHANGE_DIRECTIONS = {1: 'clockwise', -1: 'counterclockwise', 2: 'opposite'}


def blood_fan(hand, melds, context, *, yunnan):
    """Return uncapped fan and display patterns; caps apply to payments."""
    counts = hand[:]
    for kind, tile in melds:
        counts[tile] += 4 if kind.startswith('gang') else 3
    tiles = [tile for tile, count in enumerate(counts) if count]
    flush = len({tile // 9 for tile in tiles}) == 1
    pairs = not melds and all(count % 2 == 0 for count in hand)
    roots = counts.count(4)
    patterns = []

    if pairs:
        dragon_fan = roots if yunnan else int(bool(roots))
        name = ('七对', '龙七对', '双龙七对', '三龙七对')[dragon_fan]
        patterns.append((2 + dragon_fan + 2 * flush, ('清' if flush else '') + name))
        roots = 0
    elif yunnan and len(melds) == 4 and roots == 4:
        patterns.append((6 + 2 * flush, '清十八罗汉' if flush else '十八罗汉'))
        roots = 0
    elif len(melds) == 4:
        # Sichuan's hook retains the triplets fan. The pure hook replaces the flush.
        if yunnan:
            patterns.append((3, '清金钩钩') if flush else (1, '金钩钩'))
        elif flush:
            patterns.extend(((4, '清金钩钓'), (1, '对对胡')))
        else:
            patterns.extend(((1, '金钩钓'), (1, '对对胡')))
    elif _decomp_all_kotsu(hand, None, len(melds)):
        patterns.append((3, '清对') if flush else (1, '对对胡'))
    elif flush:
        patterns.append((2, '清一色'))

    if roots:
        patterns.append((roots, f'根x{roots}'))
    if not yunnan:
        if all(kind == 'gang_an' for kind, _ in melds):
            patterns.append((1, '门清'))
        if all(tile % 9 not in (0, 8) for tile in tiles):
            patterns.append((1, '中张'))
        for key, points, name in (
            ('zimo', 1, '自摸'), ('heaven', 3, '天胡'), ('earth', 2, '地胡'),
            ('hai_di', 1, '扫底胡'), ('final_discard', 1, '海底炮'),
        ):
            if context.get(key):
                patterns.append((points, name))
    for key, name in (('qiang_gang', '抢杠'), ('gang_shang_hua', '杠上花'), ('gang_shang_pao', '杠上炮')):
        if context.get(key):
            patterns.append((1, name))
    return sum(fan for fan, _ in patterns), tuple(name for _, name in patterns) or ('平胡',)


class BloodBattleGame(WebGame):
    def __init__(self, seed, variant, dealer=0):
        super().__init__(seed)
        self.yunnan = variant == YUNNAN
        self.reveal_winners = not self.yunnan
        self.dealer = self.next_dealer = dealer
        self.opening_tile = self._draw_to(self.players[dealer])
        self.exchange_groups = []
        self.exchange_direction = None
        self.winning_discards = {}
        self.shown_claimed_discards = []
        self.passed_win = [-1] * 4
        self.passed_pon = [set() for _ in self.players]
        self.unpaid_kongs = [set() for _ in self.players]
        self.draw_counts = [0] * 4
        self.opening_uninterrupted = True
        self.discard_count = 0
        self.final_discard = False
        self.hai_di = False
        self.exhausted = False
        self.win_details = []
        self.pig_deltas = [0] * 4
        self.ready_deltas = [0] * 4
        self.refund_deltas = [0] * 4
        if not self.yunnan:
            self.phase = 'exchange'

    def legal_actions(self):
        if self.phase == 'exchange':
            seat = len(self.exchange_groups)
            hand = self.players[seat].hand
            groups = []
            for start in (0, 9, 18):
                suit = [tile for tile in range(start, start + 9) for _ in range(hand[tile])]
                groups.extend(sorted(set(combinations(suit, 3))))
            return seat, [('exchange', group) for group in groups]

        seat, actions = super().legal_actions()
        if self.phase == 'action':
            actions = self._own_actions(seat, actions)
        if self.phase == 'response':
            actions = self._response_actions(seat, actions)
        return seat, actions

    def _own_actions(self, seat, actions):
        # The reference requires clearing the void suit before any own kong.
        # These rooms permit a kong of a different suit on a draw.
        available = [action for action in actions if action[0] in ('discard', 'zimo')]
        if self.drawn_tile is None or not self.wall:
            return available
        player = self.players[seat]
        for tile, count in enumerate(player.hand):
            if tile // 9 == player.void:
                continue
            if count == 4 and (self.yunnan or tile == self.drawn_tile):
                available.append(('ankan', tile))
            if count == 1 and ('peng', tile) in player.melds:
                if self.yunnan or tile == self.drawn_tile or tile in self.unpaid_kongs[seat]:
                    available.append(('bugang', tile))
        return available

    def _response_actions(self, seat, actions):
        tile = self.pending_discard[1]
        forbidden = set()
        if not self.wall:
            forbidden.add('zhigang')
        if tile in self.passed_pon[seat]:
            forbidden.update(('peng', 'zhigang'))
        if ('ron', None) in actions:
            context = self._discard_context()
            if not self._can_win(seat, self._with_tile(seat, tile), context):
                forbidden.add('ron')
        return [action for action in actions if action[0] not in forbidden]

    def _with_tile(self, seat, tile):
        hand = self.players[seat].hand[:]
        hand[tile] += 1
        return hand

    def _fan(self, seat, hand, context):
        return blood_fan(hand, self.players[seat].melds, context, yunnan=self.yunnan)

    def _can_win(self, seat, hand, context):
        threshold = self.passed_win[seat]
        if threshold == -1:
            return True
        if not self.yunnan and (context.get('qiang_gang') or context.get('gang_shang_pao')):
            return True
        return self._fan(seat, hand, context)[0] > threshold

    def _can_rob(self, seat, tile):
        return super()._can_rob(seat, tile) and self._can_win(
            seat, self._with_tile(seat, tile), {'qiang_gang': True},
        )

    def _discard_context(self):
        return {
            'gang_shang_pao': self.discard_was_gang,
            'final_discard': self.final_discard,
            'earth': self.opening_uninterrupted and self.discard_count == 1,
        }

    def _draw_context(self, seat):
        first_draw = self.draw_counts[seat] == 1 and self.opening_uninterrupted
        return {
            'zimo': True,
            'gang_shang_hua': self.last_draw_was_gang,
            'hai_di': self.hai_di,
            'heaven': first_draw and seat == self.dealer,
            'earth': first_draw and seat != self.dealer,
        }

    def _clear_passes(self, seat):
        self.passed_win[seat] = -1
        self.passed_pon[seat].clear()

    def _begin_turn(self, seat, from_gang=False):
        super()._begin_turn(seat, from_gang)
        if self.phase == 'action':
            self.draw_counts[seat] += 1
            self._clear_passes(seat)

    def _exchange(self, group):
        self.exchange_groups.append(group)
        if len(self.exchange_groups) < 4:
            return
        offset = self.rng.choice(tuple(EXCHANGE_DIRECTIONS))
        self.exchange_direction = EXCHANGE_DIRECTIONS[offset]
        for seat, player in enumerate(self.players):
            outgoing = self.exchange_groups[seat]
            incoming = self.exchange_groups[(seat - offset) % 4]
            for tile in outgoing:
                player.hand[tile] -= 1
            for tile in incoming:
                player.hand[tile] += 1
            if seat == self.dealer and self.opening_tile in outgoing:
                self.opening_tile = incoming[-1]
        self.exchange_groups.clear()
        self.phase = 'void'

    def _declare_void(self, seat, suit):
        self.players[seat].void = suit
        self.void_declared += 1
        if self.void_declared == 4:
            self.cur = self.dealer
            self.drawn_tile = self.opening_tile
            self.draw_counts[self.dealer] = 1
            self.phase = 'action'

    def step(self, action):
        seat, actions = self.legal_actions()
        if action not in actions:
            raise ClientError('illegal_action')
        kind, arg = action
        phase = self.phase
        if phase == 'exchange':
            self._exchange(arg)
            return
        if phase == 'void':
            self._declare_void(seat, arg)
            return
        if phase == 'action':
            if kind == 'discard':
                if ('zimo', None) in actions:
                    self.passed_win[seat] = self._fan(seat, self.players[seat].hand, self._draw_context(seat))[0]
                self.discard_count += 1
                self.final_discard = self.drawn_tile is not None and self.hai_di
            if kind in ('ankan', 'bugang'):
                self.opening_uninterrupted = False
        elif phase == 'response':
            provider, tile = self.pending_discard
            if ('ron', None) in actions and kind != 'ron':
                self.passed_win[seat] = self._fan(seat, self._with_tile(seat, tile), self._discard_context())[0]
            if self.yunnan and ('peng', None) in actions and kind == 'pass':
                self.passed_pon[seat].add(tile)
        elif kind == 'pass':
            self.passed_win[seat] = self._fan(seat, self._with_tile(seat, self.rob_tile), {'qiang_gang': True})[0]

        super().step(action)
        if phase == 'response' and not self.response_queue:
            if self.ron_winners:
                if self.yunnan:
                    self.shown_claimed_discards.append(tile)
                self._set_multiple_win_dealer(self.ron_winners, provider)
            elif self.phase == 'action' and self.drawn_tile is None:
                self._after_pon(self.cur, tile)
        elif phase == 'rob_kong' and not self.rob_queue:
            self._set_multiple_win_dealer(self.rob_winners, self.rob_actor)

    def _set_multiple_win_dealer(self, winners, provider):
        if not self.exhausted and len(winners) > 1 and len(winners) == len(self.hu_order):
            self.next_dealer = provider

    def _after_pon(self, seat, tile):
        self.opening_uninterrupted = False
        self._clear_passes(seat)
        if not self.yunnan and self.wall and self.players[seat].hand[tile] == 1:
            self.unpaid_kongs[seat].add(tile)

    def _gang_money(self, ganger, kind, provider=None):
        self.opening_uninterrupted = False
        tile = self.players[ganger].melds[-1][1]
        if kind != 'gang_bu' or tile not in self.unpaid_kongs[ganger]:
            super()._gang_money(ganger, kind, provider)

    def _settle_hu(self, winner, hand_counts, ctx, discarder=None):
        if self.phase == 'response':
            ctx = self._discard_context()
            self.winning_discards[winner] = self.pending_discard[1]
        if self.phase == 'rob_kong':
            self.winning_discards[winner] = self.rob_tile
        if ctx.get('zimo'):
            ctx = self._draw_context(winner)
        fan, names = self._fan(winner, hand_counts, ctx)
        fan = min(fan, FAN_CAP)
        amount = 2 ** fan
        payers = [i for i in self.alive_players() if i != winner] if discarder is None else [discarder]
        payments = [0] * 4
        for payer in payers:
            self._pay(payer, winner, amount)
            payments[payer] = amount
        if not self.hu_order:
            self.next_dealer = winner
        self.players[winner].hu = True
        self.players[winner].hu_fan = fan
        self.revealed_hands[winner] = hand_counts[:]
        self.hu_order.append((winner, fan, names))
        self.win_details.append({'actor': winner, 'fan': fan, 'patterns': names, 'payments': payments})

    def _waits_max_fan(self, seat):
        player = self.players[seat]
        counts = player.hand[:]
        for kind, tile in player.melds:
            counts[tile] += 4 if kind.startswith('gang') else 3
        fans = []
        for tile, count in enumerate(counts):
            if count >= 4:
                continue
            hand = self._with_tile(seat, tile)
            if is_hu(hand, len(player.melds), player.void):
                fans.append(min(self._fan(seat, hand, {})[0], FAN_CAP))
        return max(fans, default=None)

    def _settlement_payment(self, deltas, payer, receiver, amount):
        self._pay(payer, receiver, amount)
        deltas[payer] -= amount
        deltas[receiver] += amount

    def _liuju_settle(self):
        self.exhausted = True
        self.next_dealer = self.dealer
        waits = {seat: self._waits_max_fan(seat) for seat in self.alive_players()}
        unready = [seat for seat, fan in waits.items() if fan is None]
        if self.yunnan:
            pigs = {
                seat for seat in unready
                if any(count and tile // 9 == self.players[seat].void
                       for tile, count in enumerate(self.players[seat].hand))
            }
            for payer in sorted(pigs):
                for receiver in range(4):
                    if receiver not in pigs:
                        self._settlement_payment(self.pig_deltas, payer, receiver, 2 ** FAN_CAP)
        for payer in unready:
            for receiver, fan in waits.items():
                if fan is not None:
                    self._settlement_payment(self.ready_deltas, payer, receiver, 2 ** fan)
        for payer, receiver, amount in self.gang_ledger:
            if receiver in unready:
                self._settlement_payment(self.refund_deltas, receiver, payer, amount)
        self.phase = 'over'

    def settlement(self):
        kong_deltas = [0] * 4
        for payer, receiver, amount in self.gang_ledger:
            kong_deltas[payer] -= amount
            kong_deltas[receiver] += amount
        return {
            'win_details': self.win_details,
            'kong_deltas': kong_deltas,
            'pig_deltas': self.pig_deltas,
            'ready_deltas': self.ready_deltas,
            'refund_deltas': self.refund_deltas,
        }
