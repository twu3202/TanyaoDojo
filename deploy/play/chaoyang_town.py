"""Chaoyang Town egg claims, opposite-end liability and kong recovery."""

from dataclasses import replace

from .changchun import CHICKEN, EGG_GROUPS, ChangchunGame, Egg, egg_rack_faces
from .honors import DRAGONS, ORPHANS, PASS, RON, TERMINALS, KongPoints
from .liaoning_split import SplitScore, three_suits
from .northeast import wait_pattern
from .shenyang import SHENYANG, ShenyangGame, ShenyangRules


ROB_FED_KONG = 'rob_fed_kong'
KONG_BASE = {CHICKEN: 2, 9: 2, 10: 4, 7: 4, 13: 8, 24: 8, **dict.fromkeys(DRAGONS, 2)}


class ChaoyangRules(ShenyangRules):
    OPEN_FOR_STANDARD = False
    OPTIONS = replace(SHENYANG.OPTIONS, chow_lock_extensions=False, pass_win_clear_by_call=True,
                      kong_points=KongPoints(1, 1, 2))
    @staticmethod
    def next_dealer(dealer, winners, provider):
        return dealer if not winners or winners[0] == dealer else (dealer + 1) % 4

    @staticmethod
    def rack_allowed(hand, melds):
        faces = egg_rack_faces(hand, melds)
        return three_suits(faces) and bool(faces & ORPHANS)

    def winning_shapes(self, hand, melds, tile):
        for pair, sets in self.shapes(hand, melds):
            yield pair, (*melds, *sets), sets, False
        for kind in ('egg_dragons', 'egg_ones', 'egg_nines'):
            group = EGG_GROUPS[kind]
            if tile not in group or not all(hand[t] for t in group):
                continue
            rest = list(hand)
            for t in group:
                rest[t] -= 1
            egg = Egg(kind, group[0], material=group)
            augmented = (*melds, egg)
            for pair, sets in self.shapes(tuple(rest), augmented):
                yield pair, (*augmented, *sets), sets, True

    def natural_win(self, hand, melds, tile):
        return next(self.winning_shapes(tuple(hand), melds, tile), None) is not None

    @classmethod
    def payments(cls, game, actor, provider, score):
        closed = [cls.closed(p.melds) for p in game.players]
        all_closed = all(closed[seat] for seat in range(game.seat_count) if seat != actor)
        bills = tuple(
            score.share * 2 ** (int(closed[actor]) + int(closed[seat]) + int(all_closed)
                                + int(game.dealer in (actor, seat)) + int(seat == provider)
                                + int(seat == provider and '抢杠胡' in score.patterns))
            if seat != actor else 0 for seat in range(game.seat_count)
        )
        if provider is None:
            return bills
        bill = sum(bills) if provider == game.reverse_provider else bills[provider]
        return tuple(bill if seat == provider else 0 for seat in range(game.seat_count))


CHAOYANG_TOWN = ChaoyangRules('chaoyang-town-eggs')


class ChaoyangGame(ShenyangGame):
    EGG_ROB_PON = True
    SINGLE_WINNER = False

    def __init__(self, seed, dealer, round_wind, *, rules=CHAOYANG_TOWN):
        self.reverse_end = None
        self.reverse_provider = None
        self.fed_kong_offer = None
        super().__init__(seed, dealer, round_wind, rules=rules)

    def _ready_choices(self, actor):
        return ()

    def _new_eggs(self, actor):
        return [action for action in ChangchunGame._new_eggs(self, actor) if action[1] is None]

    def _egg_extension_allowed(self, actor, argument):
        index, face = argument
        meld = self.players[actor].melds[index]
        return face == self.drawn_tile and face in EGG_GROUPS[meld.kind]

    def _discard(self, actor, tile):
        self.reverse_provider = actor if tile == self.reverse_end else None
        self.reverse_end = None
        super()._discard(actor, tile)

    def _call_discard(self, offer):
        actor, kind, start = self.claim
        self.reverse_end = None
        if kind == 'chi':
            reverse = start + 3 if offer.tile == start else start - 1 if offer.tile == start + 2 else -1
            if reverse >= 0 and reverse // 9 == start // 9:
                self.reverse_end = reverse
        super()._call_discard(offer)
        if kind == 'peng':
            self.forbidden_discards = frozenset((offer.tile,))

    def _response_actions(self):
        actor, actions = super()._response_actions()
        if self.winners:
            return actor, [a for a in actions if a in (PASS, RON)]
        if (self.offer.kind == 'discard' and not self.winners and self.remaining == 4
                and self.players[actor].hand[self.offer.tile] == 3):
            actions.append(('zhigang', None))
        return actor, actions

    def _finish_responses(self, offer):
        if not self.winners and offer.kind == 'discard' and self.claim and self.claim[1] == 'zhigang':
            if self._win(offer.actor, replace(offer, kind='gang_bu'))[0] is not None:
                self.fed_kong_offer = offer
                self.cur, self.phase, self.drawn_tile = offer.actor, ROB_FED_KONG, None
                return
        super()._finish_responses(offer)

    def legal_actions(self):
        if self.phase == ROB_FED_KONG:
            return self.cur, [PASS, (ROB_FED_KONG, self.fed_kong_offer.tile)]
        return super().legal_actions()

    def step(self, action):
        if self.phase != ROB_FED_KONG:
            super().step(action)
            return
        if action not in self.legal_actions()[1]:
            raise ValueError('Illegal recovery of a kong-fed tile')
        offer, self.fed_kong_offer = self.fed_kong_offer, None
        if action == PASS:
            self._call_discard(offer)
            return
        score, hand, _ = self._win(offer.actor, replace(offer, kind='gang_bu'))
        self.discards[offer.actor].pop()
        self.claimed_tile = offer.tile
        self._settle_special(offer.actor, self.claim[0], 'rob_kong_win', score, hand)

    def _pay_kong(self, actor, meld):
        if isinstance(meld, Egg):
            base = 1 if len(meld.tiles) > len(EGG_GROUPS[meld.kind]) else 4
        else:
            base = KONG_BASE.get(meld.tile, 1) * (2 if meld.kind == 'gang_an' else 1)
        amounts = tuple(base * (2 if self.dealer in (actor, seat) else 1) if seat != actor else 0
                        for seat in range(self.seat_count))
        if meld.kind == 'gang_ming' and self.reverse_provider == meld.provider:
            amounts = tuple(sum(amounts) if seat == meld.provider else 0 for seat in range(self.seat_count))
        self._transfer(actor, amounts, self.kong_deltas)

    def _score(self, actor, hand, melds, ctx):
        if ctx.kind == 'rob_concealed':
            return None
        hand = tuple(hand)
        open_sets = sum(m.kind in ('chi', 'peng', 'gang_ming', 'gang_bu') for m in melds)
        extra = []
        if open_sets >= 3:
            extra.append(('四副落', 2) if open_sets == 4 else ('三副落', 1))
        if ctx.kind == 'zimo':
            extra.append(('自摸', 1))
            if ctx.kong_chain:
                extra.append(('杠上开花', 1))
        elif ctx.kind == 'rob_kong':
            extra.append(('抢杠胡', 0))
        old_faces = egg_rack_faces(hand, melds) - {ctx.tile}
        lone_terminal = (ctx.tile in TERMINALS and hand[ctx.tile] == 1 and not old_faces & ORPHANS
                         and not any(ctx.tile in m.tiles for m in melds))
        choices = []
        for pair, full, sets, egg_middle in self.rules.winning_shapes(hand, melds, ctx.tile):
            if all(m.kind != 'chi' for m in full):
                labels = [('飘胡', 3)]
            else:
                eye = egg_middle or lone_terminal or wait_pattern(ctx.tile, ((pair, sets),)) is not None
                labels = [('蛋夹' if egg_middle else '夹胡', 1)] if eye else [('屁胡', 0)]
                if pair in DRAGONS and any(m.kind == 'triplet' and m.tile == ctx.tile for m in sets):
                    labels.extend(('丁胡', 2) for _ in range(1 + int(ctx.tile in DRAGONS)))
            labels.extend(extra)
            choices.append(SplitScore(sum(n for _, n in labels), tuple(name for name, _ in labels)))
        return max(choices, key=lambda score: score.share, default=None)
