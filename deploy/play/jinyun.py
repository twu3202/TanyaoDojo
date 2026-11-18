"""Jinyun classic and Cantonese-style mahjong."""

from dataclasses import replace

from .hangzhou import FOUR_HANGZHOU, WHITE, HangzhouGame, HangzhouRules, split_wild
from .honors import ORPHANS, PASS, TILE_COUNT, HonorsGame, KongPoints, TableOptions, seven_pairs, standard_shapes, thirteen_orphans
from .honors_bonus import Score
from .wildcard_shapes import can_form_pairs, can_form_standard, wildcard_decompositions


_TILES = tuple(range(TILE_COUNT))


def _faces(hand, melds):
    return {tile for tile, count in enumerate(hand) if count} | {tile for meld in melds for tile in meld.tiles}


def _all_honors(faces):
    return min(faces) >= 27


def _pure_suit(faces):
    return max(faces) < 27 and min(faces) // 9 == max(faces) // 9


def _wild_orphans(natural, wild):
    return (all(not count or (tile in ORPHANS and count <= 2) for tile, count in enumerate(natural))
            and sum(not natural[tile] for tile in ORPHANS) <= wild)


class JinyunClassicRules(HangzhouRules):
    WALL_RESERVE = 33

    @staticmethod
    def can_win(hand, melds, ctx):
        if ctx.kind not in ('zimo', 'ron', 'rob_kong') or ctx.tile is None or not hand[ctx.tile]:
            return False
        if (ctx.kind != 'zimo' and ctx.tile == WHITE) or sum(hand) != 14 - 3 * len(melds):
            return False
        natural, wild = split_wild(hand)
        return can_form_standard(natural, wild) or (
            not melds and (can_form_pairs(natural, wild) or _wild_orphans(natural, wild)))

    def faan(self, hand, melds, ctx):
        if not self.can_win(hand, melds, ctx):
            return None
        natural, wild = split_wild(hand)
        self_draw = ctx.kind in ('zimo', 'rob_kong')
        events = []
        if ctx.opening == '天胡' and ctx.kind == 'zimo':
            events.append(('天胡', 4))
        if ctx.opening == '地胡' and ctx.kind == 'ron':
            events.append(('地胡', 4))
        if ctx.kind == 'zimo':
            if wild == 3:
                events.append(('财神会', 4))
            if ctx.kong_chain:
                events.append(('杠上开花', 4))
            if ctx.last_live_draw:
                events.append(('海底捞月', 4))

        face_value = 4 if self_draw else 2

        def scored(faces, shape_patterns):
            if _all_honors(faces):
                return Score(40, ('全字',))
            specials = [*events, *((pattern, face_value) for pattern in shape_patterns)]
            if _pure_suit(faces):
                specials.append(('清一色', face_value))
            if specials:
                return Score(sum(points for _, points in specials), tuple(pattern for pattern, _ in specials))
            return Score(2 if self_draw else 1, ('平胡',))

        candidates = []
        if not melds and _wild_orphans(natural, wild):
            candidates.append(scored(ORPHANS, ('十三幺',)))
        if not melds and can_form_pairs(natural, wild):
            # All unused wildcards can pair in a suit (or in honors) if the natural tiles already fit.
            faces = _faces(natural, melds)
            candidates.append(scored(faces, ('七对',)))
        for filled, _, shapes in wildcard_decompositions(natural, wild, len(melds), _TILES):
            patterns = []
            if len(melds) == 4:
                patterns.append('单钓')
            if all(meld.kind != 'chi' for meld in melds) and any(
                    all(meld.kind == 'triplet' for meld in concealed) for _, concealed in shapes):
                patterns.append('大对子')
            candidates.append(scored(_faces(filled, melds), patterns))
        return max(candidates, key=lambda score: score.share)

    @staticmethod
    def settle_hand(game, winners, offer):
        winner, = winners
        payments = tuple(winner.score.share * (2 if game.dealer in (winner.actor, seat) else 1)
                         if seat != winner.actor else 0 for seat in range(game.seat_count))
        return [(winner.score, payments)]


class JinyunClassicGame(HangzhouGame):
    def __init__(self, seed, dealer, round_wind, *, rules, dealer_streak=0):
        self.follow_deltas = [0] * rules.OPTIONS.seat_count
        super().__init__(seed, dealer, round_wind, rules=rules, dealer_streak=dealer_streak)

    def _has_float_wait(self, hand, meld_count):
        return False

    def _can_take_win(self, actor, offer):
        return offer.kind in ('discard', 'gang_bu')

    def _score(self, actor, hand, melds, ctx):
        return self.rules.faan(hand, melds, ctx)

    def _replacement_index(self):
        return self.rules.WALL_RESERVE

    def _throw_tile(self, actor, tile):
        super()._throw_tile(actor, tile)
        if not self.uninterrupted or actor != (self.dealer - 1) % self.seat_count:
            return
        if any(len(river) != 1 for river in self.discards):
            return
        first = self.discards[self.dealer][0]
        if any(river[0] != first for seat, river in enumerate(self.discards) if seat != actor):
            return
        payer = self.dealer if tile in (first, WHITE) else actor
        share = 2 if tile == first else 1
        self.follow_deltas = [share * (1 - self.seat_count if seat == payer else 1) for seat in range(self.seat_count)]
        for player, delta in zip(self.players, self.follow_deltas, strict=True):
            player.score += delta

    def _end(self, reason):
        super()._end(reason)
        if reason == 'draw' and any(meld.kind.startswith('gang_') for seat, player in enumerate(self.players)
                                    if seat != self.dealer for meld in player.melds):
            self.next_dealer = (self.dealer + 1) % self.seat_count


class JinyunCantoneseRules:
    VARIANT = 'jinyun-cantonese'
    OPTIONS = TableOptions(chow=False, seven_pairs=True, concealed_kong_faces=0, drawn_added_kong_only=True,
                           kong_points=KongPoints(3, 1, 2, single_direct_payer=True, refund_on_draw=True))
    MATCH_HANDS = 8
    MIN_FAAN = 0
    WALL_RESERVE = 9
    PASS_LOCK = False
    SWAP_LOCK = False

    @staticmethod
    def faan(hand, melds, ctx):
        if ctx.kind not in ('zimo', 'rob_kong'):
            return None
        if thirteen_orphans(hand, melds):
            return Score(30, ('十三幺',))
        pairs = seven_pairs(hand, melds)
        shapes = list(standard_shapes(hand, len(melds)))
        if not (pairs or shapes):
            return None
        if ctx.opening == '天胡' and ctx.kind == 'zimo':
            return Score(20, ('天胡',))
        faces = _faces(hand, melds)
        if _all_honors(faces):
            return Score(4, ('全字',))
        if _pure_suit(faces):
            return Score(4, ('清一色',))
        if len(melds) == 4:
            return Score(4, ('单钓',))
        if pairs:
            return Score(4, ('龙七对',)) if 4 in hand else Score(2, ('七对',))
        if any(all(meld.kind == 'triplet' for meld in concealed) for _, concealed in shapes):
            return Score(2, ('大对子',))
        return Score(1, ('平胡',))

    @staticmethod
    def next_dealer(dealer, winners, provider):
        return winners[0] if winners else dealer

    @staticmethod
    def liability(melds, meld):
        return ()

    @staticmethod
    def settle_hand(game, winners, offer):
        winner, = winners
        provider = offer.actor if offer is not None else game.kong_provider
        if provider is None:
            payments = tuple(winner.score.share if seat != winner.actor else 0 for seat in range(game.seat_count))
        else:
            payments = tuple((game.seat_count - 1) * winner.score.share if seat == provider else 0
                             for seat in range(game.seat_count))
        return [(winner.score, payments)]


class JinyunCantoneseGame(HonorsGame):
    def _score(self, actor, hand, melds, ctx):
        score = super()._score(actor, hand, melds, ctx)
        if score is not None and ctx.kong_provider is not None:
            return replace(score, patterns=(*score.patterns, '点杠杠开'))
        return score

    def _replacement_index(self):
        return self.rules.WALL_RESERVE

    def _response_actions(self):
        if self.winners:
            return self.responses[0], [PASS]
        return super()._response_actions()

    def _end(self, reason):
        self.kong_refunds = [-points for points in self.kong_deltas] if reason == 'draw' else [0] * self.seat_count
        super()._end(reason)


JINYUN = JinyunClassicRules('jinyun-classic', replace(FOUR_HANGZHOU.OPTIONS, orphans=True, show_winds=True,
                                                     host_deals_first=False, kong_points=KongPoints(1, 1, 2)),
                            fixed_dealer=2, flip_wildcard=True)
JINYUN_CANTONESE = JinyunCantoneseRules()
