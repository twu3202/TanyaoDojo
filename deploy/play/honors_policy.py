"""Public-information tile efficiency for rooms using suited tiles and honors."""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType

from common.suit_table import load_table

from . import hongkong
from .anji import AnjiRules
from .changchun import BAO_WIN, EGG_GROUPS
from .chongji import ChongjiRules
from .dongyang import DongyangRules, white_assignments
from .fujian_sixteen import FujianSixteenRules
from .fujian_travel import QuanzhouRules
from .fuzhou import FuzhouRules
from .gold_opening import GOLD_CLAIMS, split_gold
from .hangzhou import FLOAT, WHITE, split_wild
from .honors import DEFAULT_OPTIONS, DRAGONS, ORPHANS, PASS, RON, TILE_COUNT, ZIMO, Meld, WinContext, chow_choices
from .honors_bonus import HorseRoom
from .tiantai import TiantaiWildcardRules
from .taizhou import TaizhouRules
from .linhai import LinhaiRules
from .longyan import TRAVEL, LongyanRules, split_longyan
from .northeast import READY_CALLS, NortheastRules
from .jinyun import JinyunClassicRules
from .jiaxing import JiaxingRules
from .pinghu import PinghuRules
from .quzhou import QuzhouRules, separated_distance
from .sixteen import SET_COUNT, SixteenRules
from .jiangxi import HonorChow, NanchangRules, ShangraoRules, jing_chows
from .jiangxi_shapes import HONOR_CHOW_ID, honor_chow_routes, spaced_distance, split_jing
from .liaoning_joker import next_face
from .changbai import CHANGBAI
from .chaoyang_town import ROB_FED_KONG
from .dalian_tieling import INSIDE_READY
from .liaoning_joker import LiaoningJokerRules
from .liaoning_split import LiaoyangRules
from .wildcard_shapes import without_tile
from .wuhu import WuhuRules, suit_counts
from .xianyou import XianyouRules
from .zhejiang_flipped import ELEVEN_WINDS, TEN_WINDS, DeqingRules, SongyangRules


@dataclass(frozen=True)
class PolicyView:
    hand: tuple[int, ...]
    melds: tuple[Meld, ...]
    visible: tuple[int, ...]
    actions: tuple[tuple[str, int | None], ...]
    seat_wind: int
    round_wind: int
    offered: int | None
    remaining: int
    rules: ModuleType | HorseRoom = hongkong
    offered_by: int | None = None
    wildcard_indicator: int | None = None
    wildcard_tile: int | None = None
    excluded_tiles: frozenset[int] = frozenset()
    travel_level: int = 0

    @property
    def wildcard(self):
        return self.wildcard_tile if self.wildcard_tile is not None else getattr(self.rules, 'WILDCARD', None)


def policy_view(game, actor, actions):
    visible = Counter(tile for river in game.discards for tile in river)
    indicator_tile = getattr(game, 'indicator_tile', None)
    if indicator_tile is not None:
        visible[indicator_tile] += 1
    lower_indicator = getattr(game, 'lower_indicator_tile', None)
    if lower_indicator is not None:
        visible[lower_indicator] += 1
    for seat, player in enumerate(game.players):
        for meld in player.melds:
            if (seat != actor and meld.kind == 'gang_an'
                    and not getattr(game.rules, 'OPTIONS', DEFAULT_OPTIONS).concealed_kong_faces):
                continue
            visible.update(meld.tiles)
        if seat != actor and player.open_ready_hand is not None:
            visible.update(dict(enumerate(player.open_ready_hand)))
    offered = None
    if game.offer is not None:
        hidden_kong = (game.offer.kind == 'gang_an' and game.offer.actor != actor
                       and not getattr(game.rules, 'OPTIONS', DEFAULT_OPTIONS).concealed_kong_faces)
        if not hidden_kong:
            offered = game.offer.tile
            if game.offer.kind != 'discard':
                visible[offered] += 1
    player = game.players[actor]
    before_play = game.phase in ('piao', 'buy_base')
    flower_kinds = getattr(game, 'flower_kinds', frozenset())
    hand = (0,) * TILE_COUNT if before_play else tuple(
        count if tile not in flower_kinds else 0 for tile, count in enumerate(player.hand)
    )
    seat_wind = 27 if before_play else 27 + (actor - game.dealer) % game.seat_count
    return PolicyView(hand, tuple(player.melds), tuple(visible[i] for i in range(TILE_COUNT)),
                      tuple(actions), seat_wind, game.round_wind,
                      offered, game.remaining, game.rules,
                      None if offered is None else game.offer.actor, getattr(game, 'wildcard_indicator', None),
                      getattr(game, 'wildcard_tile', None), getattr(game, 'excluded_tiles', frozenset()),
                      game.travel_level if getattr(game, 'travel_owner', None) == actor else 0)


@lru_cache(maxsize=1)
def _table():
    return load_table()


@lru_cache(maxsize=8192)
def _suit(counts):
    index = sum(count * 5 ** position for position, count in enumerate(counts))
    return tuple((slot // 2, slot % 2, int(n)) for slot, n in enumerate(_table()[index]) if n >= 0)


@lru_cache(maxsize=32768)
def _combine(first, second, groups=4):
    best = {}
    for sets, pair, parts in first:
        for more_sets, more_pair, more_parts in second:
            key = sets + more_sets, pair + more_pair
            if key[0] <= groups and key[1] <= 1:
                best[key] = max(best.get(key, -1), parts + more_parts)
    return tuple((sets, pair, parts) for (sets, pair), parts in sorted(best.items()))


@lru_cache(maxsize=4096)
def _honors(counts, groups=4):
    profile = ((0, 0, 0),)
    for count in counts:
        if count >= 2:
            choices = ((0, 0, 1), (0, 1, 0))
            if count >= 3:
                choices += ((1, 0, 0),)
            if count == 4 and groups == SET_COUNT:
                choices += ((0, 0, 2), (0, 1, 1))
            profile = _combine(profile, choices, groups)
    return profile


@lru_cache(maxsize=8192)
def _five_suit(counts):
    # Ten tiles are necessary to exceed the training table's four-part limit.
    if sum(counts) < 10:
        return _suit(counts)
    first = next(i for i, n in enumerate(counts) if n)
    moves = [((first,), (0, 0, 0))]
    if counts[first] >= 2:
        pair = (first, first)
        moves.extend(((pair, (0, 1, 0)), (pair, (0, 0, 1))))
    if counts[first] >= 3:
        moves.append(((first,) * 3, (1, 0, 0)))
    for other in range(first + 1, min(first + 3, 9)):
        if counts[other]:
            moves.append(((first, other), (0, 0, 1)))
    if first <= 6 and counts[first + 1] and counts[first + 2]:
        moves.append(((first, first + 1, first + 2), (1, 0, 0)))

    profile = {}
    for consumed, added in moves:
        rest = list(counts)
        for tile in consumed:
            rest[tile] -= 1
        states = _combine(_five_suit(tuple(rest)), (added,), SET_COUNT)
        for sets, pair, parts in states:
            key = sets, pair
            profile[key] = max(profile.get(key, -1), min(parts, SET_COUNT))
    return tuple((sets, pair, parts) for (sets, pair), parts in sorted(profile.items()))


@lru_cache(maxsize=65536)
def standard_distance(hand, meld_count, groups=4):
    """Standard shanten on all 34 kinds, using the project's existing suit table."""
    return min(2 * (groups - total) - min(parts, groups - total) - pair
               for sets, pair, parts in _shape_states(hand, groups) if (total := sets + meld_count) <= groups)


def _shape_states(hand, groups=4):
    shape = _five_suit if groups == SET_COUNT else _suit
    suited = _combine(_combine(shape(hand[:9]), shape(hand[9:18]), groups), shape(hand[18:27]), groups)
    return _combine(suited, _honors(hand[27:], groups), groups)


@lru_cache(maxsize=32768)
def _sets_distance(hand, meld_count):
    return min(2 * needed - min(parts, needed) for sets, pair, parts in _shape_states(hand)
               if not pair and (needed := 4 - sets - meld_count) >= 0)


def _triplet_distance(hand, needed):
    groups = sorted(range(TILE_COUNT), key=lambda tile: hand[tile], reverse=True)
    base = sum(min(hand[tile], 3) for tile in groups[:needed])
    outside = groups[needed]
    best = base + min(2, hand[outside])
    for pair in groups[:needed]:
        best = max(best, base - min(3, hand[pair]) + min(3, hand[outside]) + min(2, hand[pair]))
    return 3 * needed + 1 - best


def _wild_distance(natural, wild, meld_count, require_burst, allow_pairs):
    if require_burst and wild:
        # Reserve one wildcard for the drawn tile instead of completing a natural pair.
        wild -= 1
        distance = _sets_distance(natural, meld_count) - wild
    else:
        distance = standard_distance(natural, meld_count) - wild
    if not meld_count and allow_pairs:
        singles = sum(count % 2 for count in natural)
        matched = min(wild, singles)
        pairs = sum(count // 2 for count in natural) + matched + (wild - matched) // 2
        distance = min(distance, 6 - pairs)
    return distance


def _distance(hand, melds, view):
    if isinstance(view.rules, (NanchangRules, ShangraoRules)):
        nanchang = isinstance(view.rules, NanchangRules)
        if nanchang:
            natural, wild = split_jing(hand, (view.wildcard, next_face(view.wildcard)))
        else:
            natural, wild = split_wild(hand, view.wildcard)
        distance = _wild_distance(natural, wild, len(melds), False, True)
        if nanchang:
            for count, honors in honor_chow_routes(natural[27:]):
                if count and count + len(melds) <= 4:
                    distance = min(distance, standard_distance(natural[:27] + honors, count + len(melds)) - wild)
        if not melds:
            distance = min(distance, spaced_distance(natural, wild))
        return max(-1 if sum(hand) % 3 == 2 else 0, distance)
    if isinstance(view.rules, LongyanRules):
        natural, wild = split_longyan(hand, view.wildcard)
        distance = _wild_distance(natural, wild, len(melds), False, True)
        if view.rules.half_self_draw and not melds:
            orphans = 13 - sum(bool(hand[t]) for t in ORPHANS) - any(hand[t] > 1 for t in ORPHANS)
            distance = min(distance, orphans)
        return max(0, distance)
    if isinstance(view.rules, SixteenRules):
        if isinstance(view.rules, FuzhouRules):
            natural, wild = split_gold(hand, view.wildcard)
            return max(0, standard_distance(natural, len(melds), SET_COUNT) - wild)
        if isinstance(view.rules, (FujianSixteenRules, XianyouRules)):
            natural, wild = view.rules.split(hand, view.wildcard)
            return max(0, standard_distance(natural, len(melds), SET_COUNT) - wild)
        return standard_distance(hand, len(melds), SET_COUNT)
    if isinstance(view.rules, WuhuRules):
        return max(standard_distance(hand, len(melds)), 7 - max(suit_counts(hand, melds)))
    wildcard = view.wildcard
    if wildcard is not None:
        natural, wild = split_wild(hand, wildcard)
        white_flexible = isinstance(view.rules, (DongyangRules, QuzhouRules)) and wildcard != WHITE
        choices = white_assignments(natural, wildcard) if white_flexible else (natural,)
        allow_pairs = view.rules.OPTIONS.seven_pairs
        distance = min(_wild_distance(choice, wild, len(melds), view.rules.require_burst, allow_pairs)
                       for choice in choices)
        if isinstance(view.rules, JiaxingRules):
            called_suits = {meld.tile // 9 for meld in melds if meld.tile < 27}
            routes = [standard_distance(tuple(count if t >= 27 or t // 9 == suit else 0
                                              for t, count in enumerate(natural)), len(melds)) - wild
                      for suit in range(3) if called_suits <= {suit}]
            if all(meld.kind != 'chi' for meld in melds):
                routes.append(_triplet_distance(natural, 4 - len(melds)) - wild)
            distance = min(routes, default=9)
        if isinstance(view.rules, QuzhouRules):
            routes = [distance] if view.rules.lata else []
            if not view.rules.lata:
                called_suits = {meld.tile // 9 for meld in melds if meld.tile < 27}
                for choice in choices:
                    routes.extend(standard_distance(tuple(count if tile >= 27 or tile // 9 == suit else 0
                                                          for tile, count in enumerate(choice)), len(melds)) - wild
                                  for suit in range(3) if called_suits <= {suit})
                    routes.append(_triplet_distance(choice, 4 - len(melds)) - wild)
                    if not melds:
                        singles = min(wild, sum(count % 2 for count in choice))
                        routes.append(6 - sum(count // 2 for count in choice) - singles - (wild - singles) // 2)
            if not melds:
                routes.extend(separated_distance(choice, wild) for choice in choices)
            distance = min(routes)
        if isinstance(view.rules, JinyunClassicRules) and not melds:
            missing = sum(not natural[tile] for tile in ORPHANS)
            distance = min(distance, missing - any(natural[tile] >= 2 for tile in ORPHANS) - wild)
        return max(-1 if sum(hand) % 3 == 2 else 0, distance)
    if not view.rules.MIN_FAAN:
        best = standard_distance(hand, len(melds))
        options = getattr(view.rules, 'OPTIONS', DEFAULT_OPTIONS)
        if not melds:
            if options.orphans:
                orphans = 13 - sum(bool(hand[tile]) for tile in ORPHANS) - any(hand[tile] > 1 for tile in ORPHANS)
                best = min(best, orphans)
            if options.seven_pairs:
                best = min(best, 6 - sum(count // 2 for count in hand))
        return best
    open_melds = tuple(meld for meld in melds if meld.kind != 'gang_an')
    valued = {meld.tile for meld in melds if meld.kind != 'chi'}
    value_faan = len(valued & DRAGONS) + (view.seat_wind in valued) + (view.round_wind in valued)
    best = standard_distance(hand, len(melds)) if not open_melds or value_faan >= 3 else 8
    if all(meld.kind != 'chi' for meld in melds):
        best = min(best, _triplet_distance(hand, 4 - len(melds)))
    called_suits = {meld.tile // 9 for meld in melds if meld.tile < 27}
    if len(called_suits) <= 1:
        suits = called_suits or {suit for suit in range(3) if sum(hand[9*suit:9*suit+9]) >= 3}
        for suit in suits:
            suited_hand = tuple(count if tile >= 27 or tile // 9 == suit else 0 for tile, count in enumerate(hand))
            best = min(best, standard_distance(suited_hand, len(melds)))
    if not melds:
        orphans = 13 - sum(bool(hand[tile]) for tile in ORPHANS) - any(hand[tile] > 1 for tile in ORPHANS)
        best = min(best, orphans)
    return best


def _live_tiles(view):
    available = set(getattr(view.rules, 'OPTIONS', DEFAULT_OPTIONS).tile_kinds) - view.excluded_tiles
    if isinstance(view.rules, WuhuRules):
        return tuple(max(0, view.rules.tile_copies(tile) - view.hand[tile] - view.visible[tile])
                     if tile in available else 0 for tile in range(TILE_COUNT))
    return tuple(max(0, 4 - view.hand[tile] - view.visible[tile]) if tile in available else 0
                 for tile in range(TILE_COUNT))


def _ready_waits(hand, melds, view, live):
    waits = 0
    augmented = list(hand)
    params = {}
    if isinstance(view.rules, SongyangRules):
        params['indicator'] = view.wildcard_indicator
    elif isinstance(view.rules, (FuzhouRules, LongyanRules, FujianSixteenRules, XianyouRules)):
        params['gold'] = view.wildcard
    elif isinstance(view.rules, AnjiRules):
        params.update(wildcard=view.wildcard, excluded=view.excluded_tiles)
    elif isinstance(view.rules, (ChongjiRules, DeqingRules, DongyangRules, JiaxingRules, LiaoningJokerRules,
                                  LiaoyangRules, NanchangRules, PinghuRules, QuzhouRules, ShangraoRules)):
        params['wildcard'] = view.wildcard
    if isinstance(view.rules, (JiaxingRules, JinyunClassicRules, LinhaiRules, PinghuRules, TaizhouRules,
                              TiantaiWildcardRules, WuhuRules, SixteenRules, LongyanRules)):
        def can_win(hand, melds, ctx):
            return view.rules.can_win(hand, melds, ctx, **params)
    else:
        def can_win(hand, melds, ctx):
            return view.rules.faan(hand, melds, ctx, **params) is not None
    for tile, count in enumerate(live):
        if not count:
            continue
        augmented[tile] += 1
        ron = WinContext('ron', tile, view.seat_wind, view.round_wind)
        if can_win(augmented, melds, ron):
            waits += 2 * count
        elif can_win(augmented, melds, WinContext('zimo', tile, view.seat_wind, view.round_wind)):
            waits += count
        augmented[tile] -= 1
    return waits


def _rank(hand, melds, view, live):
    distance = _distance(hand, melds, view)
    if distance != 0:
        return distance, 0
    waits = _ready_waits(hand, melds, view, live)
    return (0, -waits) if waits else (1, 0)


def _discard_options(hand, melds, view, live, candidates):
    ranked = []
    for tile in candidates:
        rest = list(hand)
        rest[tile] -= 1
        rank = _rank(tuple(rest), melds, view, live)
        ranked.append((rank, tile, tuple(rest)))
    return ranked


def _ukeire(distance, rest, melds, live, view):
    # Once a route has been chosen, compare ordinary shape improvements among tied discards.
    if distance == 0:
        return 0
    if isinstance(view.rules, (WuhuRules, SixteenRules)) or getattr(view.rules, 'OPTIONS', DEFAULT_OPTIONS).seven_pairs:
        improve = lambda hand: _distance(hand, melds, view)
    else:
        improve = lambda hand: standard_distance(hand, len(melds))
    before = improve(rest)
    if before <= 0:
        return 0
    available = list(rest)
    effective = 0
    for tile, count in enumerate(live):
        if not count:
            continue
        available[tile] += 1
        if improve(tuple(available)) < before:
            effective += count
        available[tile] -= 1
    return effective


def heuristic(view):
    actions = view.actions
    if isinstance(view.rules, (SixteenRules, LongyanRules)):
        special = next((action for action in actions if action[0] in (*GOLD_CLAIMS, 'flower_hu')), None)
        if special is not None and (special[0] != 'three_gold' or RON not in actions):
            return special
    if isinstance(view.rules, LongyanRules) and view.remaining >= 12:
        travels = [action for action in actions if action[0] == TRAVEL]
        if travels and (ZIMO not in actions or view.travel_level or view.hand[view.wildcard] >= 2 or view.remaining >= 36):
            return next((action for action in travels if action[1] != view.wildcard), travels[0])
    if isinstance(view.rules, XianyouRules) and not view.melds and view.remaining >= 28:
        travel = next((action for action in actions if action[0] == TRAVEL), None)
        if travel is not None:
            return travel
    if isinstance(view.rules, QuanzhouRules) and (view.travel_level or not view.melds and view.remaining >= 32):
        travel = next((action for action in actions if action[0] == TRAVEL), None)
        if travel is not None:
            return travel
    if actions[0][0] == 'liability_choice':
        worthwhile = _distance(view.hand, view.melds, view) <= 1 and view.remaining >= 5
        return 'liability_choice', int(worthwhile)
    if actions[0][0] == 'buy_base':
        return 'buy_base', 1
    if actions[0][0] == 'piao':
        return 'piao', 2 if view.rules is CHANGBAI else 1
    if actions[-1][0] == ROB_FED_KONG:
        return actions[-1]
    if actions[0][0] == 'exchange':
        def rank_exchange(action):
            remaining = list(view.hand)
            for tile in action[1]:
                remaining[tile] -= 1
            return _distance(tuple(remaining), (), view), -sum(count // 2 for count in remaining)
        return min(actions, key=rank_exchange)
    if RON in actions:
        return RON
    if isinstance(view.rules, NortheastRules):
        if BAO_WIN in actions:
            return BAO_WIN
        if ZIMO in actions:
            return ZIMO
        declarations = [a for a in actions if a[0] in ('declare_ready', INSIDE_READY)]
        if declarations:
            live = _live_tiles(view)

            def declaration_rank(action):
                hand = without_tile(view.hand, action[1])
                waits = view.rules.waits(hand, view.melds)
                inside = action[0] == INSIDE_READY
                if inside:
                    waits = view.rules.inside_waits(hand, view.melds, waits)
                return sum(live[tile] for tile in waits), inside

            return max(declarations, key=declaration_rank)
        call = next((a for a in actions if a[0] in READY_CALLS or a[0] in EGG_GROUPS
                     or a[0] in ('extend_egg', 'rob_egg_pon')), None)
        if call is not None:
            return call
    winds = next((action for action in actions if action[0] in (TEN_WINDS, ELEVEN_WINDS)), None)
    if winds is not None:
        return winds
    float_action = FLOAT[0], view.wildcard
    if float_action in actions and view.remaining >= 45:
        return float_action
    if ZIMO in actions:
        return ZIMO
    if actions == (PASS,):
        return PASS
    live = _live_tiles(view)
    discards = {tile: kind for kind, tile in actions if kind in ('discard', FLOAT[0])}
    if discards:
        ranked = _discard_options(view.hand, view.melds, view, live, discards)
        best = min(rank for rank, _, _ in ranked)
        for kind, tile in actions:
            if kind not in ('ankan', 'bugang') or view.remaining <= 1:
                continue
            hand = list(view.hand)
            if kind == 'ankan':
                new_melds = view.melds + (Meld('gang_an', tile),)
                hand[tile] -= 4
            else:
                new_melds = tuple(Meld('gang_bu', tile, meld.provider)
                                  if meld.kind == 'peng' and meld.tile == tile else meld for meld in view.melds)
                hand[tile] -= 1
            if _rank(tuple(hand), new_melds, view, live) <= best:
                return kind, tile
        choices = [(tile, rest) for rank, tile, rest in ranked if rank == best]
        wildcard = view.wildcard
        tile, _ = max(choices, key=lambda item: (_ukeire(best[0], item[1], view.melds, live, view),
                                                item[0] != wildcard,
                                                -view.hand[item[0]], item[0]))
        if best[0] == 0 and view.remaining > 0:
            for declaration in ('declare_dark', 'declare_ready'):
                if (declaration, tile) in actions:
                    return declaration, tile
        return discards[tile], tile

    best = _rank(view.hand, view.melds, view, live)
    preferred = PASS
    offered = view.offered
    for kind, argument in actions:
        if kind == 'pass':
            continue
        if kind == 'chi' and isinstance(view.rules, NanchangRules) and argument >= HONOR_CHOW_ID:
            take, block = next((take, block) for start, take, block in jing_chows(view.hand, offered)
                               if start == argument)
            meld = HonorChow('chi', argument)
        elif kind == 'chi':
            options = getattr(view.rules, 'OPTIONS', DEFAULT_OPTIONS)
            swap_rule = getattr(view.rules, 'chow_swap_lock', None)
            swap_lock = view.rules.SWAP_LOCK if swap_rule is None else swap_rule(view.melds, view.offered_by)
            choices = chow_choices(view.hand, offered, swap_lock=swap_lock,
                                   lock_extensions=options.chow_lock_extensions,
                                   allow_locked_discard=options.allow_locked_chow_discard)
            take, block = next((take, block) for start, take, block in choices
                               if start == argument)
            meld = Meld('chi', argument)
        else:
            take = (offered,) * (2 if kind == 'peng' else 3)
            block = frozenset()
            meld = Meld('peng' if kind == 'peng' else 'gang_ming', offered)
        hand = list(view.hand)
        for tile in take:
            hand[tile] -= 1
        if kind == 'peng' and getattr(view.rules, 'PON_SWAP_LOCK', False):
            if any(count and tile != offered for tile, count in enumerate(hand)):
                block = frozenset((offered,))
        melds = view.melds + (meld,)
        if kind == 'zhigang':
            rank = _rank(tuple(hand), melds, view, live)
        else:
            ranked = _discard_options(hand, melds, view, live,
                                      (tile for tile, count in enumerate(hand) if count and tile not in block))
            rank = min(row[0] for row in ranked)
        opens_hand = (isinstance(view.rules, NortheastRules) and view.rules.OPEN_FOR_STANDARD
                      and view.rules.closed(view.melds) and preferred == PASS)
        if rank < best or (opens_hand and rank == best):
            best = rank
            preferred = kind, argument
    return preferred
