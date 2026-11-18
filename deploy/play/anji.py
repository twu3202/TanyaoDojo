"""Anji Mahjong with fifteen replaceable flowers in a 144-tile wall."""

from dataclasses import replace

from .hangzhou import FOUR_HANGZHOU, WHITE, HangzhouGame, HangzhouRules, HangzhouScore, burst_wait, split_wild
from .honors import TILE_COUNT
from .wildcard_shapes import can_form_pairs, can_form_standard, without_tile
from .zhejiang_flipped import next_wildcard


FLOWER_TILES = {tile: f'F{tile - TILE_COUNT + 1}' for tile in range(TILE_COUNT, TILE_COUNT + 8)}
NORTH = 30
EAST = 27
DRAGON_SUCCESSOR = {32: 33, 33: 32}


class AnjiRules(HangzhouRules):
    def faan(self, hand, melds, ctx, *, wildcard=WHITE, excluded=frozenset(), flowers=0,
             flower_draw=False, floats=0, kongs_before_float=0):
        zimo = ctx.kind == 'zimo'
        if not (zimo or (ctx.kind == 'ron' and ctx.opening == '地胡')):
            return None
        if ctx.tile is None or not hand[ctx.tile] or sum(hand) != 14 - 3 * len(melds):
            return None
        natural, wild = split_wild(hand, wildcard)
        if zimo and wild == 4:
            return HangzhouScore(10, ('四财神',))

        before = without_tile(tuple(hand), ctx.tile)
        burst = zimo and any(burst_wait(before, len(melds), wildcard, excluded))
        kong = zimo and ctx.kong_chain > 0
        kong_burst = burst and (kong or flower_draw)
        if wild == 3 and not (kong or kong_burst or (burst and floats)):
            return None
        pairs = not melds and can_form_pairs(natural, wild)
        if not (burst or pairs or can_form_standard(natural, wild, excluded)):
            return None

        base = 1 if wild else 2
        if burst:
            scores = [HangzhouScore(2, ('七客',) if pairs else ('爆头',))]
        else:
            scores = [HangzhouScore(base, ('七对子',) if pairs else ('平胡',))]
        if kong:
            scores.append(HangzhouScore(3 if wild else 5, ('杠开',)))
        if kong_burst:
            scores.append(HangzhouScore(5, ('杠暴',)))
        if burst and floats:
            groups = kongs_before_float + int(kong or flower_draw)
            patterns = [('财飘', '双财飘', '三财飘')[floats - 1]]
            if groups:
                patterns.append('飘杠' if kong or flower_draw else '杠飘')
            four = wild + floats == 4
            if four:
                patterns.insert(0, '四财神')
            scores.append(HangzhouScore(10 * (floats + four) + 5 * groups, tuple(patterns)))
        if len(melds) == 4 and not before[wildcard]:
            scores.append(HangzhouScore(5, ('大吊车',)))
        if not wild and not flowers:
            scores.append(HangzhouScore(10, ('无花无财神',)))
        used = [tile for tile, count in enumerate(natural) if count]
        used.extend(tile for meld in melds for tile in meld.tiles)
        if used and len({tile // 9 for tile in used}) == 1 and max(used) < 27:
            scores.append(HangzhouScore(10, ('清一色',)))
        opening = ctx.opening or ('地胡' if ctx.first_nondealer_draw else None)
        if opening:
            scores.insert(0, HangzhouScore(10, (opening,)))
        return max(scores, key=lambda score: score.share)

    @staticmethod
    def settle_hand(game, winners, offer):
        return [(winner.score, tuple(0 if seat == winner.actor else winner.score.share
                                     for seat in range(game.seat_count))) for winner in winners]


ANJI = AnjiRules('anji-fifteen-flowers', FOUR_HANGZHOU.OPTIONS, fixed_dealer=1, floater_can_call=False,
                 open_kong_after_float=True)


class AnjiGame(HangzhouGame):
    def _tile_supply(self):
        supply = super()._tile_supply()
        supply.update(FLOWER_TILES.keys())
        return supply

    def _prepare_wall(self):
        self.flowers = [[] for _ in self.players]
        self.drew_through_flower = False
        self.wildcard_indicator = self.indicator_tile = self.wall.pop(0)
        indicator = self.wildcard_indicator
        special = indicator == WHITE or indicator in FLOWER_TILES
        self.excluded_tiles = frozenset((WHITE, NORTH if special else indicator))
        self.wildcard_tile = EAST if special else DRAGON_SUCCESSOR.get(indicator, next_wildcard(indicator))
        self.flower_kinds = self.excluded_tiles.union(FLOWER_TILES)

    def physical_tile(self, tile):
        return tile

    def physical_counts(self):
        counts = super().physical_counts()
        for flowers in self.flowers:
            counts.update(flowers)
        return counts

    def _deal_tile(self, actor):
        return self._take_tile(actor, False)

    def _take_tile(self, actor, replacement):
        while (tile := super()._take_tile(actor, replacement)) is not None:
            if tile not in self.flower_kinds:
                return tile
            self.flowers[actor].append(tile)
            self.drew_through_flower = True
            replacement = True
        return None

    def _draw(self, actor, kong_chain=0, provider=None):
        self.drew_through_flower = False
        super()._draw(actor, kong_chain, provider)

    def _float_kong_count(self):
        return int(bool(self.kong_chain or self.drew_through_flower))

    def _score(self, actor, hand, melds, ctx):
        return self.rules.faan(hand, melds, ctx, wildcard=self.wildcard_tile, excluded=self.excluded_tiles,
                               flowers=len(self.flowers[actor]), flower_draw=self.drew_through_flower,
                               floats=self.float_streak[actor], kongs_before_float=self.float_kongs[actor])

    def _chows(self, player, tile, provider):
        return ((start, take, block) for start, take, block in super()._chows(player, tile, provider)
                if self.wildcard_tile not in take)

    def _can_take_win(self, actor, offer):
        return self.float_cycle_owner is None and actor != self.dealer and offer.opening == '地胡'

    def _open_offer(self, offer):
        if offer.kind != 'discard':
            self.opening = False
            self._finish_own_kong(offer)
            return
        if self.uninterrupted and self.players[self.dealer].draws == 1:
            offer = replace(offer, opening='地胡')
        super()._open_offer(offer)

    def legal_actions(self):
        actor, actions = super().legal_actions()
        if (self.phase == 'action' and self.drawn_tile is None and self.remaining and self._open_kong_allowed(actor)):
            player = self.players[actor]
            actions.extend(('bugang', meld.tile) for meld in player.melds
                           if meld.kind == 'peng' and player.hand[meld.tile])
        return actor, actions
