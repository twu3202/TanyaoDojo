"""Lipu Mahjong with three-feed liability and cancellable kong payments."""

from .honors import HonorsGame, KongPoints, TableOptions, seven_pairs, standard_shapes
from .honors_bonus import Score


class LipuRules:
    VARIANT = 'lipu-kong-liability'
    OPTIONS = TableOptions(seven_pairs=True, orphans=False, host_deals_first=True, pass_win_all=True,
                           pass_pon_all=True, pass_win_clear_by_call=True, reciprocal_liability=True,
                           kong_requires_live_tile=True, show_winds=False,
                           kong_points=KongPoints(1, 1, 1, single_direct_payer=True, added_payer_from_pong=True))
    MATCH_HANDS = 8
    MIN_FAAN = 0
    WALL_RESERVE = 0
    PASS_LOCK = False
    SWAP_LOCK = True

    @staticmethod
    def faan(hand, melds, ctx, *, kong_kind=None, kong_provider=None):
        if ctx.kind == 'rob_concealed':
            return None
        shapes = tuple(standard_shapes(hand, len(melds)))
        pairs = seven_pairs(hand, melds)
        if not shapes and not pairs:
            return None

        provider = kong_provider if ctx.kind == 'zimo' and ctx.kong_chain else None
        self_draw = ctx.kind == 'zimo' and provider is None
        faces = {tile for tile, count in enumerate(hand) if count} | {m.tile for m in melds}
        one_suit = max(faces) < 27 and len({tile // 9 for tile in faces}) == 1
        closed = self_draw and all(m.kind == 'gang_an' for m in melds)

        def scored(patterns):
            if one_suit:
                patterns.append(('清一色', 2 if self_draw else 3))
            points = max((n for _, n in patterns), default=1) * 2 ** max(0, len(patterns) - 1)
            patterns = [name for name, _ in patterns] or ['平胡']
            if closed:
                points *= 2
                patterns.append('门清')
            if ctx.kind == 'rob_kong':
                points *= 7
                patterns.append('抢杠胡')
            elif ctx.kong_chain and ctx.kind == 'ron':
                points *= 9 if kong_kind == 'gang_an' else 7
                patterns.append('杠上炮')
            elif ctx.kong_chain and ctx.kind == 'zimo':
                points *= 3 if self_draw else 7
                patterns.append('杠上开花' if self_draw else '点杠杠开')
            return Score(points, tuple(patterns), provider)

        scores = []
        if shapes:
            patterns = []
            no_chow = all(m.kind != 'chi' for m in melds)
            if len(melds) == 4:
                big = no_chow
                patterns.append(('大吊车' if big else '小吊车', (4 if big else 2) if self_draw else (6 if big else 3)))
            elif no_chow and any(all(m.kind == 'triplet' for m in sets) for _, sets in shapes):
                patterns.append(('碰碰胡', 2 if self_draw else 3))
            scores.append(scored(patterns))
        if pairs:
            quads = min(hand.count(4), 2)
            name = ('七对', '豪华七对', '超豪华七对')[quads]
            scores.append(scored([(name, (2 if self_draw else 3) * 2 ** quads)]))
        return max(scores, key=lambda score: score.share)

    @staticmethod
    def liability(melds, last):
        count = sum(m.provider == last.provider for m in melds)
        return ('three_feeds',) if count == 3 else ()

    @staticmethod
    def next_dealer(dealer, winners, provider):
        if len(winners) > 1:
            return provider
        return winners[0] if winners else dealer

    @staticmethod
    def settle_hand(game, winners, offer):
        if (offer is None and game.kong_chain) or (offer is not None and offer.kind == 'discard' and offer.kong_chain):
            game._reverse_last_kong()
        outcomes = []
        for winner in winners:
            actor, score = winner.actor, winner.score
            provider = score.kong_provider if offer is None else offer.actor
            partners = [seat for _, seat in game._liabilities_for(actor)]
            payments = [0] * game.seat_count
            if provider is not None:
                for seat in (*partners, provider):
                    payments[seat] = score.share
            elif partners:
                partners.sort(key=lambda seat: (seat - actor) % game.seat_count)
                share, remainder = divmod((game.seat_count - 1) * score.share, len(partners))
                for i, seat in enumerate(partners):
                    payments[seat] = share + (i < remainder)
            else:
                for seat in range(game.seat_count):
                    if seat != actor:
                        payments[seat] = score.share
            outcomes.append((score, tuple(payments)))
        return outcomes


LIPU = LipuRules()


class LipuGame(HonorsGame):
    def __init__(self, seed, dealer, round_wind, *, rules):
        self.sealed = set()
        self.last_kong = None
        self.kong_refunds = [0] * rules.OPTIONS.seat_count
        super().__init__(seed, dealer, round_wind, rules=rules)

    def _score(self, actor, hand, melds, ctx):
        if actor in self.sealed:
            return None
        kong = self.last_kong[1] if ctx.kong_chain else None
        return self.rules.faan(hand, melds, ctx, kong_kind=kong.kind if kong else None,
                               kong_provider=kong.provider if kong else None)

    def _call_discard(self, offer):
        caller, _, _ = self.claim
        super()._call_discard(offer)
        feeds = [m for m in self.players[caller].melds if m.provider == offer.actor]
        if len(feeds) == 4 or sum(m.kind in ('peng', 'gang_bu') for m in feeds) >= 3:
            self.sealed.add(offer.actor)

    def _pay_kong(self, actor, meld):
        payments = self._kong_payments(actor, meld) if actor not in self.sealed else (0,) * self.seat_count
        self.last_kong = actor, meld, payments
        self._transfer(actor, payments, self.kong_deltas)

    def _reverse_last_kong(self):
        actor, _, payments = self.last_kong
        undo = tuple(-n for n in payments)
        self._transfer(actor, undo, self.kong_refunds)
        for seat, refund in enumerate(self.kong_refunds):
            self.kong_deltas[seat] += refund
