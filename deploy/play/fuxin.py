"""Fuxin announced wins, two settlement schedules and concealed ready discards."""

from dataclasses import replace

from .honors import DRAGONS, ORPHANS, KongPoints, TableOptions
from .honors_bonus import Score
from .northeast import BaoGame, NortheastRules
from .wildcard_shapes import can_form_pairs


OPTIONS = TableOptions(orphans=False, pass_win_all=True, concealed_kong_faces=0,
                       kong_requires_live_tile=True, kong_points=KongPoints(1, 1, 2))


class FuxinRules(NortheastRules):
    WALL_RESERVE = 14
    DARK_READY = False
    OPEN_FOR_STANDARD = True
    OPTIONS = replace(OPTIONS, seven_pairs=True, dealer_echo=True, echo_share=5)

    @classmethod
    def rack_allowed(cls, hand, melds):
        faces = {t for t, count in enumerate(hand) if count} | {t for m in melds for t in m.tiles}
        return len({t // 9 for t in faces if t < 27}) == 3 and bool(faces & ORPHANS)

    @staticmethod
    def shape_allowed(pair, sets):
        return pair in DRAGONS or any(m.kind != 'chi' for m in sets)

    def seven_pairs(self, hand, melds):
        return not melds and sum(hand) == 14 and can_form_pairs(hand, 0) and self.rack_allowed(hand, melds)

    def natural_win(self, hand, melds, tile):
        return self.seven_pairs(hand, melds) or (not self.closed(melds) and super().natural_win(hand, melds, tile))

    def score(self, hand, melds, ctx):
        pairs = self.seven_pairs(hand, melds)
        middle_five = ctx.tile < 27 and ctx.tile % 9 == 4 and any(
            any(m.kind == 'chi' and m.tile == ctx.tile - 1 for m in sets) for _, sets in self.shapes(hand, melds)
        )
        base = 2 if pairs else 1
        labels = ['七小对' if pairs else '平胡']
        if ctx.kind == 'zimo':
            base *= 2
            labels.append('自摸')
        if middle_five:
            labels.append('扎五')
        return Score(base, tuple(labels))

    @staticmethod
    def settle_hand(game, winners, offer):
        outcomes = []
        for win in winners:
            provider = offer.actor if offer else None
            pay = [win.score.share if seat != win.actor else 0 for seat in range(game.seat_count)]
            if provider is not None and not game.declared_waits[provider]:
                pay[provider] *= 2
            if '扎五' in win.score.patterns:
                if provider is None:
                    pay = [p + 10 if seat != win.actor else 0 for seat, p in enumerate(pay)]
                else:
                    pay[provider] += 20
            outcomes.append((win.score, tuple(pay)))
        return outcomes


class FuxinFaceRules(NortheastRules):
    WALL_RESERVE = 14
    DARK_READY = True
    OPTIONS = replace(OPTIONS, chow=False)

    @staticmethod
    def rack_allowed(hand, melds):
        return True

    @staticmethod
    def shape_allowed(pair, sets):
        return True

    @staticmethod
    def tile_points(tile):
        return tile % 9 + 1 if tile < 27 else 6

    def natural_win(self, hand, melds, tile):
        return self.tile_points(tile) >= 3 and super().natural_win(hand, melds, tile)

    def score(self, hand, melds, ctx):
        drawn = ctx.kind == 'zimo'
        points = self.tile_points(ctx.tile) * (2 if drawn else 1)
        return Score(points, ('自摸' if drawn else '平胡',)) if points >= 6 else None

    @staticmethod
    def settle_hand(game, winners, offer):
        outcomes = []
        for win in winners:
            provider = offer.actor if offer else None
            pays_all = provider is not None and not game.declared_waits[provider]
            pay = tuple(win.score.share * (game.seat_count - 1) if seat == provider else 0
                        for seat in range(game.seat_count)) if pays_all else tuple(
                win.score.share if seat != win.actor else 0 for seat in range(game.seat_count)
            )
            score = Score(win.score.share, (*win.score.patterns, '一包三')) if pays_all else win.score
            outcomes.append((score, pay))
        return outcomes


FUXIN_ZAWU = FuxinRules('fuxin-middle-five')
FUXIN_DARK = FuxinFaceRules('fuxin-hidden-ready')
FUXIN_ROOMS = (FUXIN_ZAWU, FUXIN_DARK)


class FuxinGame(BaoGame):
    ALLOW_READY_KONGS = True

    def __init__(self, seed, dealer, round_wind, *, rules):
        self.folded_ready_discards = [None] * rules.OPTIONS.seat_count
        super().__init__(seed, dealer, round_wind, rules=rules)

    def physical_counts(self):
        counts = super().physical_counts()
        counts.update(t for t in self.folded_ready_discards if t is not None)
        return counts

    def legal_actions(self):
        actor, actions = super().legal_actions()
        if (self.rules is FUXIN_ZAWU and self.phase == 'action' and self.offer is None
                and self.rules.closed(self.players[actor].melds)):
            actions = [a for a in actions if a[0] != 'ankan']
        return actor, actions

    def _discard(self, actor, tile):
        if self.rules.DARK_READY and self.ready_river[actor] == (len(self.discards[actor]), tile):
            self.ready_river[actor] = None
            self.players[actor].hand[tile] -= 1
            self.folded_ready_discards[actor] = tile
            self.opening = self.uninterrupted = False
            self.offer = None
            self._draw((actor + 1) % self.seat_count)
            return
        super()._discard(actor, tile)

    def _score(self, actor, hand, melds, ctx):
        if ctx.kind == 'rob_concealed' or ctx.tile not in self.declared_waits[actor]:
            return None
        hand = tuple(hand)
        if self.rules.natural_win(hand, melds, ctx.tile):
            return self.rules.score(hand, melds, ctx)
        return None
