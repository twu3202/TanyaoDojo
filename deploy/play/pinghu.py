"""Pinghu flower-count scoring with a flipped wildcard."""

from dataclasses import replace

from .flower_wall import FLOWER_TILES, FlowerWall
from .hangzhou import FOUR_HANGZHOU, HangzhouRules, split_wild
from .honors import DRAGONS, TILE_COUNT, WINDS
from .honors_bonus import Score
from .jiaxing import PrintedWildcardGame
from .wildcard_shapes import can_form_standard, wildcard_decompositions


_TILES = tuple(range(TILE_COUNT))
_SEASONS = frozenset(range(TILE_COUNT, TILE_COUNT + 4))
_PLANTS = frozenset(FLOWER_TILES) - _SEASONS


def _set_flowers(meld):
    if meld.kind == 'chi':
        return 0
    return (meld.tile in WINDS) + (2 if meld.kind == 'gang_an' else int(meld.kind.startswith('gang_')))


class PinghuRules(HangzhouRules):
    WALL_RESERVE = 16

    @staticmethod
    def can_win(hand, melds, ctx, *, wildcard):
        return (ctx.kind in ('zimo', 'ron', 'rob_kong') and ctx.tile is not None and hand[ctx.tile]
                and (ctx.kind == 'zimo' or ctx.tile != wildcard) and sum(hand) == 14 - 3 * len(melds)
                and can_form_standard(*split_wild(hand, wildcard)))

    def faan(self, hand, melds, ctx, *, wildcard, flowers=(), flower_draw=False):
        if not self.can_win(hand, melds, ctx, wildcard=wildcard):
            return None
        natural, wild = split_wild(hand, wildcard)
        fixed = []
        if not wild:
            fixed.append('无百搭')
        elif wild == 3:
            fixed.append('三百搭')
        if len(melds) == 4:
            fixed.append('单钓')
        if (ctx.kind == 'zimo' and (ctx.kong_chain or flower_draw)) or ctx.kind == 'rob_kong':
            fixed.append('杠开花开')
        four_flowers = sum(group.issubset(flowers) for group in (_SEASONS, _PLANTS))
        four_flowers += sum(natural[tile] == 4 for tile in DRAGONS)
        four_flowers += sum(meld.tile in DRAGONS and meld.kind.startswith('gang_') for meld in melds)
        fixed_flowers = len(flowers) + sum(_set_flowers(meld) for meld in melds)
        opened = {tile for meld in melds for tile in meld.tiles}
        best = None
        for filled, _, shapes in wildcard_decompositions(natural, wild, len(melds), _TILES):
            faces = opened | {tile for tile, count in enumerate(filled) if count}
            suits = {tile // 9 for tile in faces if tile < 27}
            suit_fan = 0 if len(suits) != 1 else 2 if max(faces) < 27 else 1
            for _, concealed in shapes:
                all_pungs = all(meld.kind != 'chi' for meld in (*melds, *concealed))
                flower_count = fixed_flowers + sum(_set_flowers(meld) for meld in concealed)
                fan = len(fixed) + four_flowers + suit_fan + all_pungs
                share = (flower_count + 1) << fan
                if best is None or share > best.share:
                    names = fixed.copy()
                    if four_flowers:
                        names.append('四花')
                    if suit_fan:
                        names.append('清一色' if suit_fan == 2 else '混一色')
                    if all_pungs:
                        names.append('大对子')
                    best = Score(share, (*names, f'花数x{flower_count}', f'番数x{fan}'))
        return best

    def next_dealer(self, dealer, winners, provider):
        return winners[0] if winners else dealer

    @staticmethod
    def settle_hand(game, winners, offer):
        winner, = winners
        share = winner.score.share
        if offer is None:
            payments = tuple(share if seat != winner.actor else 0 for seat in range(game.seat_count))
        else:
            amount = share * (game.seat_count - 1) if offer.kind == 'gang_bu' else share
            payments = tuple(amount if seat == offer.actor else 0 for seat in range(game.seat_count))
        return [(winner.score, payments)]


PINGHU = PinghuRules('pinghu-flower-faan', replace(FOUR_HANGZHOU.OPTIONS, show_winds=True, seven_pairs=False,
                                                   host_deals_first=False), fixed_dealer=1, flip_wildcard=True)


class PinghuGame(FlowerWall, PrintedWildcardGame):
    def _can_take_win(self, actor, offer):
        return offer.kind in ('discard', 'gang_bu')

    def _score(self, actor, hand, melds, ctx):
        return self.rules.faan(hand, melds, ctx, wildcard=self.wildcard_tile,
                               flowers=self.flowers[actor], flower_draw=bool(self.draw_flowers))
