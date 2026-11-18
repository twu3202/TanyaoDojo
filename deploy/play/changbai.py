"""Changbai's pre-deal piao and visible scoring on both sides of a bill."""

from collections import Counter
from dataclasses import dataclass

from .honors import ORPHANS, PASS, RON, HonorsGame, KongPoints, Meld, TableOptions, seven_pairs, standard_shapes, thirteen_orphans
from .liaoning_split import SplitDrawGame, rack_faces
from .northeast import BaoGame, NortheastRules, wait_pattern
from .wildcard_shapes import without_tile


PIAO_VALUES = (0, 2, 4, 6)


@dataclass(frozen=True)
class ChangbaiScore:
    base: int
    fan: int
    patterns: tuple[str, ...]
    tier = None

    @property
    def share(self):
        return self.base * (2 ** self.fan)


def numbered_counts(melds):
    return Counter(t for m in melds if m.kind in ('chi', 'peng') for t in m.tiles if t < 27)


def public_fan(melds):
    repeats = Counter(m.tile for m in melds if m.kind == 'chi')
    four_in_one = sum(n == 4 for n in numbered_counts(melds).values())
    return four_in_one + sum(min(2, n - 1) for n in repeats.values())


class ChangbaiRules(NortheastRules):
    WALL_RESERVE = 14
    OPTIONS = TableOptions(seven_pairs=True, concealed_kong_faces=0, closed_live_tail=4,
                           kong_points=KongPoints(1, 1, 2), pass_win_all=True)

    @staticmethod
    def next_dealer(dealer, winners, provider):
        return dealer if not winners or winners[0] == dealer else (dealer + 1) % 4

    @staticmethod
    def rack_allowed(hand, melds):
        return True

    @staticmethod
    def shape_allowed(pair, sets):
        return True

    def natural_win(self, hand, melds, tile):
        pairs = seven_pairs(hand, melds) and any(hand[t] for t in ORPHANS)
        return pairs or thirteen_orphans(hand, melds) or super().natural_win(hand, melds, tile)

    def faan(self, hand, melds, ctx):
        if ctx.kind == 'rob_concealed':
            return None
        hand = tuple(hand)
        public = numbered_counts(melds)
        all_faces = public + Counter({t: n for t, n in enumerate(hand[:27]) if n})
        if ctx.kind != 'zimo' and ctx.tile < 27:
            public[ctx.tile] += 1
        common = [('明四归一' if public[t] == 4 else '暗四归一', 1) for t, n in all_faces.items() if n == 4]
        faces = rack_faces(hand, melds)
        suits = {t // 9 for t in faces if t < 27}
        has_honors = any(t >= 27 for t in faces)
        if not suits or len(suits) == 1:
            common.append(('混一色' if suits and has_honors else '清一色', 2))
        if ctx.kong_chain and ctx.kind in ('zimo', 'ron'):
            common.append(('杠上开花' if ctx.kind == 'zimo' else '杠后炮', 2))
        if ctx.kind == 'rob_kong':
            common.append(('抢杠胡', 0))
        if ctx.kind == 'zimo':
            common.append(('自摸', 1))

        candidates = []
        if seven_pairs(hand, melds) and faces & ORPHANS:
            candidates.append((1, [('七小对', 2)]))
        if thirteen_orphans(hand, melds):
            candidates.append((13, [('十三幺', 0)]))
        shown_chows = Counter(m.tile for m in melds if m.kind == 'chi')
        for pair, sets in standard_shapes(hand, len(melds)):
            patterns = []
            if all(m.kind != 'chi' for m in (*melds, *sets)):
                patterns.append(('飘胡', 2 if self.closed(melds) else 1))
            if wait_pattern(ctx.tile, ((pair, sets),)) is not None:
                patterns.append(('夹胡', 0))
                if ctx.tile == pair and pair in ORPHANS:
                    patterns.append(('单吊幺九', 1))
            if len(melds) == 4 and not self.closed(melds):
                patterns.append(('手把一', 1))
            repeats = shown_chows + Counter(m.tile for m in sets if m.kind == 'chi')
            for t, n in repeats.items():
                if n < 2:
                    continue
                amount = min(2, n - 1)
                exposed = shown_chows[t] >= amount + 1
                name = '两般高' if amount == 2 else '一般高'
                patterns.append((name + ('（明）' if exposed else '（暗）'), amount))
            candidates.append((2 if ('夹胡', 0) in patterns else 1, patterns))
        if not candidates:
            return None
        closed = self.closed(melds)
        scores = []
        for base, labels in candidates:
            if closed and labels != [('七小对', 2)]:
                labels = [*labels, ('门清', 1)]
            patterns = (*labels, *common)
            scores.append(ChangbaiScore(base, sum(n for _, n in patterns),
                                         tuple(name for name, _ in patterns) or ('平胡',)))
        return max(scores, key=lambda score: score.share)

    @classmethod
    def settle_hand(cls, game, winners, offer):
        provider = offer.actor if offer else None
        outcomes = []
        for win in winners:
            bills = tuple(
                win.score.share * 2 ** (public_fan(p.melds) + int(cls.closed(p.melds)) + int(game.dealer in (win.actor, seat)))
                if seat != win.actor and provider in (None, seat) else 0
                for seat, p in enumerate(game.players)
            )
            piao = tuple(game.piao[seat] + game.piao[win.actor] if bill else 0 for seat, bill in enumerate(bills))
            game._transfer(win.actor, piao, game.piao_deltas)
            outcomes.append((win.score, bills))
        return outcomes


CHANGBAI = ChangbaiRules('changbai-piao-ready')


class ChangbaiGame(SplitDrawGame, BaoGame):
    ALLOW_READY_KONGS = True

    def __init__(self, seed, dealer, round_wind, *, rules=CHANGBAI):
        self.piao = []
        self.piao_deltas = [0] * rules.OPTIONS.seat_count
        self.draw_deltas = [0] * rules.OPTIONS.seat_count
        self.has_kong = False
        super().__init__(seed, dealer, round_wind, rules=rules)

    def _deal_opening(self):
        self.phase, self.cur = 'piao', 0

    @property
    def remaining(self):
        return len(self.wall) - self._replacement_index()

    def _replacement_index(self):
        return 0 if self.has_kong else self.rules.WALL_RESERVE

    def _pay_kong(self, actor, meld):
        self.has_kong = True
        super()._pay_kong(actor, meld)

    def _ready_choices(self, actor):
        return tuple((face, wait) for face, wait in super()._ready_choices(actor) if len(wait) == 1)

    def _response_actions(self):
        actor, actions = HonorsGame._response_actions(self)
        if self.winners:
            return actor, [a for a in actions if a in (PASS, RON)]
        if self.declared_waits[actor]:
            player = self.players[actor]
            can_kong = ('zhigang', None) in actions and not self.winners
            if can_kong:
                face = self.offer.tile
                rest = without_tile(tuple(player.hand), face, 3)
                melds = (*player.melds, Meld('gang_ming', face, self.offer.actor))
                can_kong = self.rules.waits(rest, melds) == self.declared_waits[actor]
            actions = [a for a in actions if a in (PASS, RON) or a == ('zhigang', None) and can_kong]
        return actor, actions

    def legal_actions(self):
        if self.phase == 'piao':
            return self.cur, [('piao', value) for value in PIAO_VALUES]
        return super().legal_actions()

    def step(self, action):
        if self.phase != 'piao':
            super().step(action)
            return
        if action not in self.legal_actions()[1]:
            raise ValueError('Illegal Changbai piao')
        self.piao.append(action[1])
        if len(self.piao) == self.seat_count:
            super()._deal_opening()
        else:
            self.cur += 1

    def _score(self, actor, hand, melds, ctx):
        if ctx.tile not in self.declared_waits[actor]:
            return None
        return self.rules.faan(hand, melds, ctx)

    def _end(self, reason):
        if reason == 'draw':
            for seat, player in enumerate(self.players):
                delta = -2 * (self.seat_count - 1) if seat == self.dealer else 2
                self.draw_deltas[seat] = delta
                player.score += delta
        super()._end(reason)
