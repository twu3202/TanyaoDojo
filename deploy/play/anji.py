"""Anji Mahjong rooms with public flower replacement in a 144-tile wall."""

from dataclasses import replace

from .hangzhou import FOUR_HANGZHOU, WHITE, HangzhouGame, HangzhouRules, HangzhouScore, burst_wait, split_wild
from .honors import TILE_COUNT
from .wildcard_shapes import can_form_pairs, can_form_standard, without_tile
from .zhejiang_flipped import next_wildcard, seven_winds_shape


FLOWER_TILES = {tile: f'F{tile - TILE_COUNT + 1}' for tile in range(TILE_COUNT, TILE_COUNT + 8)}
NORTH = 30
EAST = 27
DRAGON_SUCCESSOR = {32: 33, 33: 32}


class AnjiRules(HangzhouRules):
    EARTH_RON = True
    ADD_KONG_AFTER_CALL = True
    THREE_REQUIRES_EVENT = True
    SEVEN_WINDS = False
    PAIR_SHARES = (2, 1)  # Natural, then wildcard-assisted.
    NO_FLOWER_SHARE = 10
    FLOWER_POINTS = 0

    def faan(self, hand, melds, ctx, *, wildcard=WHITE, excluded=frozenset(), flowers=0,
             flower_draw=False, floats=0, kongs_before_float=0):
        zimo = ctx.kind == 'zimo'
        if not (zimo or (self.EARTH_RON and ctx.kind == 'ron' and ctx.opening == '地胡')):
            return None
        if ctx.tile is None or not hand[ctx.tile] or sum(hand) != 14 - 3 * len(melds):
            return None
        natural, wild = split_wild(hand, wildcard)
        if zimo and wild == 4:
            return self._add_flower_points(HangzhouScore(10, ('四财神',)), flowers)

        before = without_tile(tuple(hand), ctx.tile)
        burst = zimo and any(burst_wait(before, len(melds), wildcard, excluded))
        kong = zimo and ctx.kong_chain > 0
        kong_burst = burst and (kong or flower_draw)
        if self.THREE_REQUIRES_EVENT and wild == 3 and not (kong or kong_burst or (burst and floats)):
            return None
        pairs = not melds and can_form_pairs(natural, wild)
        seven_winds = self.SEVEN_WINDS and not melds and seven_winds_shape(natural, wild)
        if not (burst or pairs or seven_winds or can_form_standard(natural, wild, excluded)):
            return None

        if burst:
            scores = [HangzhouScore(2, ('七客',) if pairs else ('爆头',))]
        elif seven_winds:
            scores = [HangzhouScore(2, ('十三不搭',))]
        elif pairs:
            scores = [HangzhouScore(self.PAIR_SHARES[bool(wild)], ('七对子',))]
        else:
            scores = [HangzhouScore(1 if wild else 2, ('平胡',))]
        replacement = self._replacement_score(wild, flowers, burst, ctx.kong_chain if zimo else 0, flower_draw)
        if replacement is not None:
            scores.append(replacement)
        if burst and floats:
            groups = kongs_before_float + self.replacement_units(ctx.kong_chain, flower_draw)
            patterns = [('财飘', '双财飘', '三财飘')[floats - 1]]
            if groups:
                patterns.append('飘杠' if kong or flower_draw else '杠飘')
            four = wild + floats == 4
            if four:
                patterns.insert(0, '四财神')
            scores.append(HangzhouScore(10 * (floats + four) + 5 * groups, tuple(patterns)))
        if len(melds) == 4:
            single = self._single_wait(before[wildcard], flowers, kong_burst, ctx.kong_chain)
            if single is not None:
                scores.append(single)
        if not wild and not flowers:
            scores.append(HangzhouScore(self.NO_FLOWER_SHARE, ('无花无财神',)))
        used = [tile for tile, count in enumerate(natural) if count]
        used.extend(tile for meld in melds for tile in meld.tiles)
        if used and len({tile // 9 for tile in used}) == 1 and max(used) < 27:
            scores.append(HangzhouScore(10, ('清一色',)))
        opening = ctx.opening or ('地胡' if ctx.first_nondealer_draw else None)
        if opening:
            scores.insert(0, HangzhouScore(10, (opening,)))
        return self._add_flower_points(max(scores, key=lambda score: score.share), flowers)

    @staticmethod
    def replacement_units(kong_chain, flower_draw):
        return int(bool(kong_chain or flower_draw))

    @staticmethod
    def _replacement_score(wild, flowers, burst, kong_chain, flower_draw):
        if burst and (kong_chain or flower_draw):
            return HangzhouScore(5, ('杠暴',))
        if kong_chain:
            return HangzhouScore(3 if wild else 5, ('杠开',))
        return None

    @staticmethod
    def _single_wait(single_wild, flowers, kong_burst, kong_chain):
        return None if single_wild else HangzhouScore(5, ('大吊车',))

    def _add_flower_points(self, score, flowers):
        if self.FLOWER_POINTS and flowers:
            return replace(score, share=score.share + flowers * self.FLOWER_POINTS,
                           patterns=(*score.patterns, f'花牌x{flowers}'))
        return score

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
        self.draw_flowers = 0
        self.wildcard_indicator = self.indicator_tile = self.wall.pop(0)
        indicator = self.wildcard_indicator
        special = indicator == WHITE or indicator in FLOWER_TILES
        self.excluded_tiles = self._extra_flowers(indicator, special)
        self.wildcard_tile = EAST if special else DRAGON_SUCCESSOR.get(indicator, next_wildcard(indicator))
        self.flower_kinds = self.excluded_tiles.union(FLOWER_TILES)

    @staticmethod
    def _extra_flowers(indicator, special):
        return frozenset((WHITE, NORTH if special else indicator))

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
            self.draw_flowers += 1
            replacement = True
        return None

    def _draw(self, actor, kong_chain=0, provider=None):
        self.draw_flowers = 0
        super()._draw(actor, kong_chain, provider)

    def _float_kong_count(self):
        return self.rules.replacement_units(self.kong_chain, self.draw_flowers)

    def _score(self, actor, hand, melds, ctx):
        return self.rules.faan(hand, melds, ctx, wildcard=self.wildcard_tile, excluded=self.excluded_tiles,
                               flowers=len(self.flowers[actor]), flower_draw=self.draw_flowers,
                               floats=self.float_streak[actor], kongs_before_float=self.float_kongs[actor])

    def _chows(self, player, tile, provider):
        return ((start, take, block) for start, take, block in super()._chows(player, tile, provider)
                if self.wildcard_tile not in take)

    def _can_take_win(self, actor, offer):
        return self.rules.EARTH_RON and self.float_cycle_owner is None and actor != self.dealer and offer.opening == '地胡'

    def _open_offer(self, offer):
        if offer.kind != 'discard':
            self.opening = False
            self._finish_own_kong(offer)
            return
        if self.rules.EARTH_RON and self.uninterrupted and self.players[self.dealer].draws == 1:
            offer = replace(offer, opening='地胡')
        super()._open_offer(offer)

    def legal_actions(self):
        actor, actions = super().legal_actions()
        if (self.rules.ADD_KONG_AFTER_CALL and self.phase == 'action' and self.drawn_tile is None
                and self.remaining and self._open_kong_allowed(actor)):
            player = self.players[actor]
            actions.extend(('bugang', meld.tile) for meld in player.melds
                           if meld.kind == 'peng' and player.hand[meld.tile])
        return actor, actions


class AnjiTeaHouseRules(AnjiRules):
    EARTH_RON = False
    ADD_KONG_AFTER_CALL = False
    THREE_REQUIRES_EVENT = False
    SEVEN_WINDS = True
    PAIR_SHARES = (5, 2)
    NO_FLOWER_SHARE = 5
    FLOWER_POINTS = 1

    @staticmethod
    def replacement_units(kong_chain, flower_draw):
        return kong_chain + flower_draw

    @staticmethod
    def _replacement_score(wild, flowers, burst, kong_chain, flower_draw):
        if burst and (kong_chain or flower_draw):
            return HangzhouScore(5 * max(1, kong_chain), ('杠暴',))
        if kong_chain:
            return HangzhouScore(3 if wild else 5 if flowers else 10, ('杠开',))
        return None

    @staticmethod
    def _single_wait(single_wild, flowers, kong_burst, kong_chain):
        share = 10 if single_wild or not flowers else 5
        if kong_burst and kong_chain:
            return HangzhouScore(share + 5 * kong_chain, ('大吊车', '杠暴'))
        return HangzhouScore(share, ('大吊车',))


ANJI_TEA_HOUSE = AnjiTeaHouseRules('anji-tea-house', replace(ANJI.OPTIONS, host_deals_first=False), fixed_dealer=1,
                                   floater_can_call=False, open_kong_after_float=True)


class AnjiTeaHouseGame(AnjiGame):
    @staticmethod
    def _extra_flowers(indicator, special):
        return frozenset()

    def _score(self, actor, hand, melds, ctx):
        first_draw = (ctx.kind == 'zimo' and actor != self.dealer
                      and self.players[actor].draws == 1 and not ctx.kong_chain)
        return super()._score(actor, hand, melds, replace(ctx, first_nondealer_draw=first_draw))
