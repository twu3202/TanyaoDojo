"""Linhai white wildcards, win-time chong tiles and reciprocal three-feed liability."""

from dataclasses import replace
import random

from .core import TILES
from .hangzhou import FOUR_HANGZHOU, WHITE, HangzhouGame, HangzhouRules, split_wild
from .honors import DRAGONS, TILE_COUNT, WINDS
from .honors_bonus import Score
from .wildcard_shapes import can_form_standard, wildcard_decompositions


_TILES = tuple(range(TILE_COUNT))
_CHONG_RANKS = ({1, 5, 9}, {2, 6}, {3, 7}, {4, 8})
_CHONG_HONORS = tuple(frozenset(map(TILES.index, names)) for names in ('E', 'SC', 'WF', 'NP'))


class LinhaiRules(HangzhouRules):
    WALL_RESERVE = 14
    BONUS_MODE = 'linhai_chong'
    MULTIPLE_WINS = True

    def next_dealer(self, dealer, winners, provider):
        if dealer in winners:
            return dealer
        return winners[0] if winners else (dealer + 1) % self.OPTIONS.seat_count

    @staticmethod
    def can_win(hand, melds, ctx):
        if ctx.kind not in ('zimo', 'ron', 'rob_kong') or ctx.tile is None or not hand[ctx.tile]:
            return False
        if (ctx.kind != 'zimo' and ctx.tile == WHITE) or sum(hand) != 14 - 3 * len(melds):
            return False
        return can_form_standard(*split_wild(hand))

    def faan(self, hand, melds, ctx):
        if not self.can_win(hand, melds, ctx):
            return None
        natural, wild = split_wild(hand)
        open_faces = {tile for meld in melds for tile in meld.tiles}
        valued = DRAGONS | {ctx.seat_wind}
        best = None
        best_rank = (-1, False)
        for filled, _, shapes in wildcard_decompositions(natural, wild, len(melds), _TILES):
            used = open_faces | {tile for tile, count in enumerate(filled) if count}
            suits = {tile // 9 for tile in used if tile < 27}
            flush = 0 if len(suits) != 1 else 3 if max(used) < 27 else 1
            for pair, concealed in shapes:
                sets = (*melds, *concealed)
                valued_count = sum(m.tile in valued for m in sets if m.kind != 'chi')
                wind_sets = {m.tile for m in sets if m.tile in WINDS}
                four_winds = pair in WINDS and len(wind_sets) == 3 and WINDS == {pair} | wind_sets
                raw_tai = flush + valued_count + 3 * four_winds
                rank = (raw_tai, four_winds)
                if rank > best_rank:
                    best_rank = rank
                    tai = min(3, raw_tai)
                    names = []
                    if flush:
                        names.append('清一色' if flush == 3 else '混一色')
                    if four_winds:
                        names.append('四风聚齐')
                    if valued_count:
                        names.append('门风或字牌刻杠')
                    best = Score(1 << tai, (*names, f'台数x{tai}'))
        return best

    @staticmethod
    def liability(melds, meld):
        if meld.kind in ('chi', 'peng') and sum(m.kind in ('chi', 'peng', 'gang_bu') and m.provider == meld.provider
                                               for m in melds) == 3:
            return ('three_feeds',)
        return ()

    @staticmethod
    def settle_hand(game, winners, offer):
        game.horse_tiles = [game.wall.pop() for _ in range(6)]
        outcomes = []
        for winner in winners:
            actor = winner.actor
            wind = (actor - game.dealer) % game.seat_count
            hits = sum(tile % 9 + 1 in _CHONG_RANKS[wind] if tile < 27 else tile in _CHONG_HONORS[wind]
                       for tile in game.horse_tiles)
            total = winner.score.share + hits
            game.horse_details.append((actor, 6, hits, hits, total))
            if offer is not None:
                payments = tuple(total if seat == offer.actor else 0 for seat in range(game.seat_count))
            else:
                liabilities = game._liabilities_for(actor)
                if liabilities:
                    _, payer = liabilities[0]
                    payments = tuple((game.seat_count - 1) * total if seat == payer else 0
                                     for seat in range(game.seat_count))
                else:
                    payments = tuple(total if seat != actor else 0 for seat in range(game.seat_count))
            outcomes.append((replace(winner.score, share=total), payments))
        return outcomes


LINHAI = LinhaiRules('linhai-six-chong', replace(FOUR_HANGZHOU.OPTIONS, seven_pairs=False, show_winds=True,
                                                host_deals_first=False, pass_win_all=True,
                                                reciprocal_liability=True), fixed_dealer=1)


class LinhaiGame(HangzhouGame):
    def __init__(self, seed, dealer, round_wind, *, rules, dealer_streak=0):
        self.kong_dice = random.Random(f'linhai-kong:{seed}')
        super().__init__(seed, dealer, round_wind, rules=rules, dealer_streak=dealer_streak)

    def _replacement_index(self):
        roll = self.kong_dice.randint(1, 6) + self.kong_dice.randint(1, 6)
        return self.rules.WALL_RESERVE + (roll - 2) % self.remaining

    def _has_float_wait(self, hand, meld_count):
        return False

    def _can_take_win(self, actor, offer):
        return offer.kind in ('discard', 'gang_bu')

    def _score(self, actor, hand, melds, ctx):
        return self.rules.faan(hand, melds, ctx)
