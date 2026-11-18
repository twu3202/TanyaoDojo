"""Xiaogan Ka Wu Xing with private opening piao choices and a rank horse."""

from dataclasses import replace

from .kawuxing import SUIZHOU, KaWuXingGame, KaWuXingRules, _best_shape


class XiaoganRules(KaWuXingRules):
    BONUS_MODE = 'xiaogan_rank'

    def shape(self, hand, melds, winning_tile):
        return _best_shape(hand, melds, winning_tile, False, 1, max_pair_quads=1, dragon_pairs=True)

    def faan(self, hand, melds, ctx):
        score = super().faan(hand, melds, ctx)
        if score is not None and ctx.kind == 'zimo' and ctx.opening == '天胡':
            patterns = tuple(pattern for pattern in score.patterns if pattern != '平胡')
            return replace(score, fan=score.fan + 1, patterns=patterns + ('天胡',))
        return score

    def settle_hand(self, game, winners, offer):
        outcomes = super().settle_hand(game, winners, offer)
        for winner, (score, base) in zip(winners, outcomes, strict=True):
            actor = winner.actor
            piao = tuple(game.piao[actor] + game.piao[seat] if payment else 0 for seat, payment in enumerate(base))
            game._transfer(actor, piao, game.piao_deltas)
            if offer is None and score.ready and game.wall:
                tile = game.wall.pop(0)
                rank = tile % 9 + 1 if tile < 27 else 10
                game.horse_tiles.append(tile)
                game.horse_details.append((actor, 1, 1, rank, rank))
                horse = tuple(rank if payment else 0 for payment in base)
                game._transfer(actor, horse, game.horse_deltas)
        return outcomes


XIAOGAN = XiaoganRules('xiaogan-kawuxing',
                       replace(SUIZHOU.OPTIONS, chow=True, pass_win_higher_all=False),
                       nearest_winner=True, bill_first_ready=False, middle_five_fan=1)


class XiaoganGame(KaWuXingGame):
    def __init__(self, seed, dealer, round_wind, *, rules):
        super().__init__(seed, dealer, round_wind, rules=rules)
        self.piao = []
        self.piao_deltas = [0] * self.seat_count
        self.horse_deltas = [0] * self.seat_count
        self.opening_draw = self.drawn_tile
        self.drawn_tile = None
        self.cur = 0
        self.phase = 'piao'

    def legal_actions(self):
        if self.phase == 'piao':
            return self.cur, [('piao', value) for value in range(4)]
        return super().legal_actions()

    def step(self, action):
        if self.phase != 'piao':
            super().step(action)
            return
        if action not in self.legal_actions()[1]:
            raise ValueError('Illegal piao choice')
        self.piao.append(action[1])
        if len(self.piao) == self.seat_count:
            self.phase = 'action'
            self.cur = self.dealer
            self.drawn_tile = self.opening_draw
        else:
            self.cur += 1
