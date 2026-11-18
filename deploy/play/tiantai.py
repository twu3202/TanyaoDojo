"""Three-player Tiantai with flowers and a 300-hu limit."""

from functools import lru_cache
from itertools import combinations_with_replacement

from .flower_wall import FlowerWall
from .hangzhou import WHITE, split_wild
from .honors import DRAGONS, PASS, RON, TERMINALS, WINDS, HonorsGame, TableOptions, standard_shapes
from .honors_bonus import Score
from .wildcard_shapes import can_form_standard


VARIANT = 'tiantai-three-hu'
MATCH_HANDS = 8
MIN_FAAN = 0
PASS_LOCK = False
SWAP_LOCK = False
WALL_RESERVE = 15
HU_CAP = 300
OPTIONS = TableOptions(tile_kinds=(*range(18), *range(27, 34)), seat_count=3, chow=False, orphans=False,
                       concealed_kong_faces=0, pass_win_all=True, pass_pon_all=True, pass_win_clear_by_call=True,
                       kong_requires_live_tile=True)
_SEASONS = frozenset(range(34, 38))
_PLANTS = frozenset(range(38, 42))
_HONORS = WINDS | DRAGONS
_WILD_ASSIGNABLE = tuple(tile for tile in OPTIONS.tile_kinds if tile != WHITE)
_WILD_EXCLUDED = frozenset(range(34)) - set(_WILD_ASSIGNABLE)


def _winning_placements(pair, concealed, tile):
    if pair == tile:
        yield '单钓', None
    for index, group in enumerate(concealed):
        if tile not in group.tiles:
            continue
        if group.kind == 'triplet':
            yield None, index
        elif tile == group.tile + 1:
            yield '嵌档', None
        elif (group.tile % 9 == 0 and tile == group.tile + 2) or (group.tile % 9 == 6 and tile == group.tile):
            yield '靠柄', None
        else:
            yield None, None


def _set_hu(meld, seat_wind, open_triplet=False):
    if meld.kind == 'chi':
        return 0
    high = meld.tile in TERMINALS or meld.tile in _HONORS
    if meld.kind == 'gang_an':
        return 32 if high else 16
    if meld.kind.startswith('gang_'):
        return 16 if high else 8
    if meld.kind == 'peng' or open_triplet:
        return 4 if meld.tile in TERMINALS or meld.tile == seat_wind or meld.tile in DRAGONS else 2
    return 8 if high else 4


def _flower_tai(flowers, seat_wind):
    group = seat_wind - 27
    matches = sum(tile in (34 + group, 38 + group) for tile in flowers)
    sets = sum(quartet.issubset(flowers) for quartet in (_SEASONS, _PLANTS))
    return ('位花',) * matches + ('四花',) * sets


def faan(hand, melds, ctx, *, flowers=(), flower_replacement=False):
    if ctx.kind not in ('zimo', 'ron', 'rob_kong'):
        return None
    self_draw = ctx.kind == 'zimo'
    if self_draw and len(flowers) == 8:
        return Score(HU_CAP, ('八花',))
    if ctx.tile is None or not hand[ctx.tile]:
        return None
    shapes = tuple(standard_shapes(tuple(hand), len(melds)))
    if not shapes:
        return None
    if ctx.opening == '天胡':
        return Score(HU_CAP, ('天胡',))
    if ctx.opening == '地胡' or ctx.first_nondealer_draw:
        return Score(HU_CAP // 2, ('地胡',))
    return _rank_hand(hand, melds, ctx, shapes, (ctx.tile,), flowers, flower_replacement)[1]


def _rank_hand(hand, melds, ctx, shapes, winning_tiles, flowers, flower_replacement, extra_tai=()):
    self_draw = ctx.kind == 'zimo'
    used = {t for t, count in enumerate(hand) if count}
    used.update(t for meld in melds for t in meld.tiles)
    suits = {t // 9 for t in used if t < 27}
    valued = DRAGONS | {ctx.seat_wind}
    common = list(_flower_tai(flowers, ctx.seat_wind))
    common.extend(extra_tai)
    if len(suits) == 1:
        common.extend(('混一色',) if used & _HONORS else ('清一色',) * 3)
    if ctx.kind == 'rob_kong':
        common.append('抢杠')
    if self_draw and (ctx.kong_chain or flower_replacement):
        common.append('杠上花')
    if self_draw and ctx.last_live_draw:
        common.append('海底')
    fixed_hu = 20 + 4 * len(flowers) + 2 * self_draw + sum(_set_hu(m, ctx.seat_wind) for m in melds)

    best = None
    for pair, concealed in shapes:
        groups = (*melds, *concealed)
        tai = [*common]
        if all(group.kind != 'chi' for group in groups):
            tai.append('大对子')
        for group in groups:
            if group.kind != 'chi' and group.tile in valued:
                tai.append('门风刻杠' if group.tile == ctx.seat_wind else '字牌刻杠')
        for tile in winning_tiles:
            for wait, open_index in _winning_placements(pair, concealed, tile):
                patterns = tai.copy()
                if not wait and pair not in valued and all(group.kind == 'chi' for group in groups):
                    patterns.append('排胡')
                hu = fixed_hu + 2 * (pair in valued) + 2 * bool(wait)
                hu += sum(_set_hu(group, ctx.seat_wind, open_triplet=not self_draw and index == open_index)
                          for index, group in enumerate(concealed))
                raw = hu << len(patterns)
                labels = (*dict.fromkeys(patterns), *((wait,) if wait else ()), f'底胡x{hu}', f'台数x{len(patterns)}')
                candidate = (raw, Score(min(HU_CAP, (raw + 9) // 10 * 10), labels))
                if best is None or raw > best[0]:
                    best = candidate
    return best


@lru_cache(maxsize=256)
def _wild_decompositions(natural, wild, meld_count):
    decompositions = []
    for assignment in combinations_with_replacement(_WILD_ASSIGNABLE, wild):
        filled = list(natural)
        for tile in assignment:
            filled[tile] += 1
        filled = tuple(filled)
        shapes = tuple(standard_shapes(filled, meld_count))
        if shapes:
            decompositions.append((filled, tuple(dict.fromkeys(assignment)), shapes))
    return tuple(decompositions)


def liability(melds, meld):
    return ()


def next_dealer(dealer, winners, provider):
    return dealer if tuple(winners) == (dealer,) else (dealer + 1) % OPTIONS.seat_count


def settle_hand(game, winners, offer):
    return [(winner.score, tuple(winner.score.share if seat != winner.actor and (offer is None or seat == offer.actor)
                                 else 0 for seat in range(game.seat_count))) for winner in winners]


class TiantaiGame(FlowerWall, HonorsGame):
    def _replacement_index(self):
        return self.rules.WALL_RESERVE

    def _draw(self, actor, kong_chain=0, provider=None):
        super()._draw(actor, kong_chain, provider)
        if self.phase == 'over' and self.reason_code == 'draw' and self.draw_flowers and len(self.flowers[actor]) == 8:
            self._settle([actor], None)

    def _score(self, actor, hand, melds, ctx):
        return self.rules.faan(hand, melds, ctx, flowers=self.flowers[actor], flower_replacement=bool(self.draw_flowers))

    def _response_actions(self):
        actor, actions = super()._response_actions()
        return actor, [action for action in actions if action in (PASS, RON)] if self.winners else actions

    def _call_discard(self, offer):
        actor = self.claim[0]
        super()._call_discard(offer)
        self.players[actor].declined_calls.clear()


class TiantaiWildcardRules:
    VARIANT = 'tiantai-three-white-wild'
    MATCH_HANDS = MATCH_HANDS
    MIN_FAAN = MIN_FAAN
    PASS_LOCK = PASS_LOCK
    SWAP_LOCK = SWAP_LOCK
    WALL_RESERVE = WALL_RESERVE
    OPTIONS = OPTIONS
    WILDCARD = WHITE
    require_burst = False
    liability = staticmethod(liability)
    next_dealer = staticmethod(next_dealer)
    settle_hand = staticmethod(settle_hand)

    @staticmethod
    def can_win(hand, melds, ctx):
        if ctx.kind not in ('zimo', 'ron', 'rob_kong') or (ctx.kind != 'zimo' and ctx.tile == WHITE):
            return False
        if ctx.tile is None or not hand[ctx.tile] or sum(hand) != 14 - 3 * len(melds):
            return False
        natural, wild = split_wild(hand)
        return (ctx.kind == 'zimo' and wild == 4) or can_form_standard(natural, wild, _WILD_EXCLUDED)

    def faan(self, hand, melds, ctx, *, flowers=(), flower_replacement=False):
        if ctx.kind == 'zimo' and len(flowers) == 8:
            return Score(HU_CAP, ('八花',))
        if not self.can_win(hand, melds, ctx):
            return None
        natural, wild = split_wild(hand)
        if ctx.kind == 'zimo' and wild == 4:
            return Score(HU_CAP, ('四财神',))
        best = None
        for filled, assignments, shapes in _wild_decompositions(natural, wild, len(melds)):
            winning_tiles = assignments if ctx.tile == WHITE else (ctx.tile,)
            choice = _rank_hand(filled, melds, ctx, shapes, winning_tiles, flowers, flower_replacement,
                                ('清牌',) if not wild else ())
            if best is None or choice[0] > best[0]:
                best = choice
        return best[1] if best is not None else None


TIANTAI_WILD = TiantaiWildcardRules()


class TiantaiWildcardGame(TiantaiGame):
    def legal_actions(self):
        actor, actions = super().legal_actions()
        return actor, [action for action in actions if action not in (('ankan', WHITE), ('bugang', WHITE))]

    def _response_actions(self):
        if self.offer.tile == WHITE:
            return self.responses[0], [PASS]
        return super()._response_actions()
