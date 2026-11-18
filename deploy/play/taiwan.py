"""Taiwanese sixteen-tile Mahjong with flower wins and per-payer dealer tai."""

from collections import Counter

from .honors import DRAGONS, PASS, RON, WINDS, ZIMO, WinContext, standard_shapes
from .honors_bonus import Score
from .sixteen import SET_COUNT, SixteenGame, SixteenRules


def _two_sided(chows, tile):
    return any((tile == m.tile and m.tile % 9 < 6) or (tile == m.tile + 2 and m.tile % 9 > 0)
               for m in chows)


def _unique_wait(hand, melds, tile):
    rest = list(hand)
    rest[tile] -= 1
    exposed = Counter(face for m in melds for face in m.tiles)
    for other, count in enumerate(rest):
        if other != tile and count + exposed[other] < 4:
            rest[other] += 1
            another = any(standard_shapes(rest, len(melds), groups=SET_COUNT))
            rest[other] -= 1
            if another:
                return False
    return True


class TaiwanRules(SixteenRules):
    VARIANT = 'taiwan-sixteen'
    BASE = 3

    @staticmethod
    def can_win(hand, melds, ctx):
        return ctx.kind != 'rob_concealed' and any(standard_shapes(hand, len(melds), groups=SET_COUNT))

    def faan(self, hand, melds, ctx, *, flowers=(), flower_draw=False):
        if ctx.kind == 'rob_concealed':
            return None
        shapes = tuple(standard_shapes(hand, len(melds), groups=SET_COUNT))
        if not shapes:
            return None
        if ctx.opening in ('天胡', '地胡'):
            return Score(self.BASE + (32 if ctx.opening == '天胡' else 16), (ctx.opening,))

        self_draw = ctx.kind == 'zimo'
        closed = all(m.kind == 'gang_an' for m in melds)
        unique = _unique_wait(hand, melds, ctx.tile)
        all_from_others = not self_draw and len(melds) == SET_COUNT and not any(m.kind == 'gang_an' for m in melds)
        awards = []
        if closed and self_draw:
            awards.append(('门清自摸', 3))
        elif closed:
            awards.append(('门清', 1))
        elif self_draw:
            awards.append(('自摸', 1))
        if all_from_others:
            awards.append(('全求人', 2))
        elif unique:
            awards.append(('独听', 1))
        if self_draw and (flower_draw or (ctx.kong_chain and ctx.kong_provider is None)):
            awards.append(('杠上开花', 1))
        if ctx.kind == 'rob_kong':
            awards.append(('抢杠胡', 1))
        if ctx.last_live_draw:
            awards.append(('海底捞月' if self_draw else '河底捞鱼', 1))

        flower_set = set(flowers)
        seat = ctx.seat_wind - 27
        for start in (34, 38):
            if start + seat in flower_set:
                awards.append(('正花', 1))
            if set(range(start, start + 4)) <= flower_set:
                awards.append(('花杠', 2))
        faces = {tile for tile, count in enumerate(hand) if count} | {m.tile for m in melds}
        suits = {tile // 9 for tile in faces if tile < 27}
        honors = any(tile >= 27 for tile in faces)
        if not suits:
            awards.append(('字一色', 8))
        elif len(suits) == 1:
            awards.append(('混一色', 4) if honors else ('清一色', 8))

        scores = []
        for pair, concealed in shapes:
            patterns = awards.copy()
            chows = tuple(m for m in concealed if m.kind == 'chi')
            open_chows = any(m.kind == 'chi' for m in melds)
            if not chows and not open_chows:
                patterns.append(('碰碰胡', 4))
            if ctx.kind == 'ron' and not flowers and not honors and not unique and len(chows) + sum(
                    m.kind == 'chi' for m in melds) == SET_COUNT and _two_sided(chows, ctx.tile):
                patterns.append(('平胡', 2))

            pungs = {m.tile for m in (*melds, *concealed) if m.kind != 'chi'}
            if DRAGONS <= pungs:
                patterns.append(('大三元', 8))
            elif pair in DRAGONS and len(DRAGONS & pungs) == 2:
                patterns.append(('小三元', 4))
            else:
                patterns.extend((name, 1) for tile, name in ((31, '白板'), (32, '青发'), (33, '红中')) if tile in pungs)
            if WINDS <= pungs:
                patterns.append(('大四喜', 16))
            else:
                if pair in WINDS and len(WINDS & pungs) == 3:
                    patterns.append(('小四喜', 8))
                if ctx.seat_wind in pungs:
                    patterns.append(('门风', 1))
                if ctx.round_wind in pungs:
                    patterns.append(('圈风', 1))
            private = sum(m.kind == 'triplet' for m in concealed) + sum(m.kind == 'gang_an' for m in melds)
            if not self_draw and pair != ctx.tile and ctx.tile in {m.tile for m in concealed if m.kind == 'triplet'}:
                private -= not any(ctx.tile in m.tiles for m in chows)
            if private >= 3:
                patterns.append((('三暗刻', '四暗刻', '五暗刻')[private - 3], (2, 5, 8)[private - 3]))
            tai = sum(n for _, n in patterns)
            scores.append(Score(self.BASE + tai, (*(name for name, _ in patterns), f'台数x{tai}')))
        return max(scores, key=lambda result: result.share)

    @staticmethod
    def payments(game, actor, provider, score):
        add = 1 + 2 * game.dealer_streak
        return tuple(score.share + add * (game.dealer in (actor, seat))
                     if seat != actor and provider in (None, seat) else 0 for seat in range(game.seat_count))


TAIWAN = TaiwanRules()


class TaiwanGame(SixteenGame):
    def __init__(self, seed, dealer, round_wind, *, rules, dealer_streak=0):
        self.passed_win = set()
        self.unlock_on_discard = set()
        self.flower_declined = False
        self.flower_provider = None
        self.flower_resume = None
        super().__init__(seed, dealer, round_wind, rules=rules, dealer_streak=dealer_streak)

    def _deal_opening(self):
        super()._deal_opening()
        self._offer_flower_win()

    def _offer_flower_win(self):
        if self.flower_declined or self.winners or sum(map(len, self.flowers)) != 8:
            return
        for actor, flowers in enumerate(self.flowers):
            if len(flowers) >= 7 and actor not in self.passed_win:
                self.flower_provider = next((i for i, other in enumerate(self.flowers) if i != actor and other), None)
                self.flower_resume = self.phase, self.cur
                self.phase = 'flower_win'
                self.cur = actor
                return

    def legal_actions(self):
        if self.phase == 'flower_win':
            return self.cur, [PASS, ('flower_hu', None)]
        return super().legal_actions()

    def step(self, action):
        if self.phase == 'flower_win':
            if action == PASS:
                self.flower_declined = True
                self.phase, self.cur = self.flower_resume
            elif action == ('flower_hu', None):
                label = '八仙过海' if self.flower_provider is None else '七抢一'
                score = Score(self.rules.BASE + 8, (label, '台数x8'))
                self._settle_special(self.cur, self.flower_provider, 'flower_hu', score, self.players[self.cur].hand.copy())
            else:
                raise ValueError('Illegal flower win action')
            return
        if self.offer is None and action[0] == 'discard':
            actor, actions = self.legal_actions()
            if action in actions and ZIMO in actions:
                self.passed_win.add(actor)
        super().step(action)

    def _respond(self, actor, action, actions):
        if RON in actions and action != RON:
            self.passed_win.add(actor)
        super()._respond(actor, action, actions)

    def _draw(self, actor, kong_chain=0, provider=None):
        self.unlock_on_discard.discard(actor)
        super()._draw(actor, kong_chain, provider)
        if not kong_chain and actor in self.passed_win and self.drawn_tile is not None:
            player = self.players[actor]
            ctx = WinContext('zimo', self.drawn_tile, 27, 27)
            if not self.rules.can_win(player.hand, player.melds, ctx):
                self.unlock_on_discard.add(actor)
        self._offer_flower_win()

    def _discard(self, actor, tile):
        super()._discard(actor, tile)
        if actor in self.unlock_on_discard:
            self.unlock_on_discard.remove(actor)
            self.passed_win.discard(actor)
            self._offer_flower_win()

    def _score(self, actor, hand, melds, ctx):
        if actor in self.passed_win:
            return None
        return self.rules.faan(hand, melds, ctx, flowers=self.flowers[actor], flower_draw=bool(self.draw_flowers))
