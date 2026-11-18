"""Wuhu eight-suit-tile rooms with documented four-wind and red-dragon profiles."""

from collections import Counter
from dataclasses import dataclass, replace

from .honors import COPIES, PASS, WINDS, HonorsGame, TableOptions, standard_shapes
from .honors_bonus import Score


SUITED = tuple(range(27))
WIND_OPTIONS = TableOptions(tile_kinds=SUITED + tuple(sorted(WINDS)), chow=False, orphans=False,
                            host_deals_first=True, kong_requires_live_tile=True, show_winds=False)
RED_OPTIONS = replace(WIND_OPTIONS, tile_kinds=SUITED + (33,))


def suit_counts(hand, melds):
    counts = [sum(hand[start:start + 9]) for start in (0, 9, 18)]
    for meld in melds:
        if meld.tile < 27:
            counts[meld.tile // 9] += len(meld.tiles)
    return tuple(counts)


@dataclass(frozen=True)
class WuhuScore(Score):
    dealer_bonus: int = 0


@dataclass(frozen=True)
class WuhuRules:
    VARIANT: str
    red_515: bool = False

    MATCH_HANDS = 8
    MIN_FAAN = 0
    WALL_RESERVE = 0
    PASS_LOCK = False
    SWAP_LOCK = False

    @property
    def OPTIONS(self):
        return RED_OPTIONS if self.red_515 else WIND_OPTIONS

    def tile_copies(self, tile):
        return 1 if tile in WINDS else COPIES[tile]

    @staticmethod
    def _shapes(hand, melds, ctx):
        if ctx.kind in ('ron', 'zimo') and max(suit_counts(hand, melds)) >= 8:
            yield from standard_shapes(hand, len(melds))

    def can_win(self, hand, melds, ctx):
        return any(self._shapes(hand, melds, ctx))

    def _inside_wait(self, hand, melds, tile):
        exposed = Counter(face for meld in melds for face in meld.tiles)
        rest = list(hand)
        rest[tile] -= 1
        for other in self.OPTIONS.tile_kinds:
            if other == tile or rest[other] + exposed[other] >= self.tile_copies(other):
                continue
            rest[other] += 1
            valid = max(suit_counts(rest, melds)) >= 8 and any(standard_shapes(rest, len(melds)))
            rest[other] -= 1
            if valid:
                return False
        return True

    def faan(self, hand, melds, ctx):
        shapes = tuple(self._shapes(hand, melds, ctx))
        if not shapes:
            return None
        counts = suit_counts(hand, melds)
        branch = max(counts)
        drawn = ctx.kind == 'zimo'
        outside = Counter(face for meld in melds for face in meld.tiles)
        honors = any(hand[27:]) or any(tile >= 27 for tile in outside)
        kong_points = sum(4 if m.kind == 'gang_an' else 2 if m.kind.startswith('gang_') else 0 for m in melds)
        big = []
        if branch == sum(counts) and not honors:
            big.append('清一色')
        kong_suits = Counter(m.tile // 9 for m in melds if m.tile < 27 and m.kind.startswith('gang_'))
        if any(count >= 8 and any(other != suit and kongs == 2 for other, kongs in kong_suits.items())
               for suit, count in enumerate(counts)):
            big.append('双八支')
        if ctx.opening == '天胡' and drawn:
            big.append('天胡')
        middle = any(m.kind == 'chi' and m.tile + 1 == ctx.tile for _, sets in shapes for m in sets)
        inside_wait = middle and self._inside_wait(hand, melds, ctx.tile)
        scores = []
        for _, sets in shapes:
            complete = (*sets, *melds)
            chows = {m.tile for m in complete if m.kind == 'chi'}
            pungs = {m.tile for m in complete if m.kind in ('triplet', 'peng')}
            roots = sum(hand[tile] + outside[tile] == 4 and any(start <= tile <= start + 2 for start in chows)
                        for tile in pungs)
            mouths = []
            if not chows:
                mouths.append('对对胡')
            if any({start, start + 3, start + 6} <= chows for start in (0, 9, 18)):
                mouths.append('通天')
            if roots == 1:
                mouths.append('四核')
            if branch == sum(counts) and honors:
                mouths.append('混一色')
            if drawn and ctx.kong_chain:
                mouths.append('杠上开花')
            awards = [*big, *(['双四核'] if roots >= 2 else [])]
            inside = inside_wait and ctx.tile - 1 in chows
            scores.append(self._points(branch, ctx.tile, drawn, inside, kong_points, mouths, awards))
        return max(scores, key=lambda score: (score.share, score.dealer_bonus, len(score.patterns)))

    def _points(self, branch, tile, drawn, inside, kongs, mouths, awards):
        base = (6 if drawn else 2) if self.red_515 else (8 if drawn else 4)
        base += 2 * inside
        if awards or len(mouths) > 2:
            points = 50 + 20 * (len(mouths) + drawn) if self.red_515 else 100
            return WuhuScore(points, ('交', *awards, *mouths), dealer_bonus=base)
        branches = branch + (tile % 9 + 1 if inside else 0)
        branches = branches if drawn else branches // 2
        if self.red_515:
            bonus = (10 if drawn else 5) * len(mouths)
        else:
            bonus = (40 if drawn else 20) * len(mouths) - 20 * ('杠上开花' in mouths and drawn)
        detail = '压档' if inside else '平胡'
        return WuhuScore(base + branches + bonus + kongs, (detail, f'八支x{branch}', *mouths), dealer_bonus=base)

    def liability(self, melds, last):
        if not self.red_515 and last.kind == 'peng' and sum(
            m.kind in ('peng', 'gang_bu') and m.provider == last.provider for m in melds
        ) == 3:
            return ('three_feeds',)
        return ()

    @staticmethod
    def next_dealer(dealer, winners, provider):
        return winners[0] if winners else dealer

    def settle_hand(self, game, winners, offer):
        winner, = winners
        actor, score = winner.actor, winner.score
        shares = tuple(score.share + score.dealer_bonus * (game.dealer in (actor, seat)) if seat != actor else 0
                       for seat in range(game.seat_count))
        if offer is None or self.red_515:
            return [(score, shares)]
        payer = next((seat for _, seat in game.players[actor].liabilities), offer.actor)
        if payer != offer.actor:
            score = WuhuScore(score.share, (*score.patterns, '三碰包牌'), dealer_bonus=score.dealer_bonus)
        return [(score, tuple(amount if seat == payer else 0 for seat, amount in enumerate(shares)))]


WUHU = WuhuRules('wuhu-eight-branches')
WUHU_515 = WuhuRules('wuhu-red-515', red_515=True)


class WuhuGame(HonorsGame):
    def _tile_supply(self):
        return Counter({tile: self.rules.tile_copies(tile) for tile in self.options.tile_kinds})

    def _response_actions(self):
        if self.winners:
            return self.responses[0], [PASS]
        return super()._response_actions()
