"""Public-information tile efficiency for rooms using suited tiles and honors."""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType

from common.suit_table import load_table

from . import hongkong
from .hangzhou import FLOAT, WHITE, split_wild
from .honors import DEFAULT_OPTIONS, DRAGONS, ORPHANS, PASS, RON, TILE_COUNT, ZIMO, Meld, WinContext, chow_choices
from .honors_bonus import HorseRoom
from .zhejiang_flipped import TEN_WINDS, SongyangRules


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


def policy_view(game, actor, actions):
    visible = Counter(tile for river in game.discards for tile in river)
    if getattr(game, 'wildcard_indicator', None) is not None:
        visible[WHITE] += 1
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
    setting_piao = game.phase == 'piao'
    hand = (0,) * TILE_COUNT if setting_piao else tuple(player.hand)
    seat_wind = 27 if setting_piao else 27 + (actor - game.dealer) % game.seat_count
    return PolicyView(hand, tuple(player.melds), tuple(visible[i] for i in range(TILE_COUNT)),
                      tuple(actions), seat_wind, game.round_wind,
                      offered, game.remaining, game.rules,
                      None if offered is None else game.offer.actor, getattr(game, 'wildcard_indicator', None))


@lru_cache(maxsize=1)
def _table():
    return load_table()


@lru_cache(maxsize=8192)
def _suit(counts):
    index = sum(count * 5 ** position for position, count in enumerate(counts))
    return tuple((slot // 2, slot % 2, int(n)) for slot, n in enumerate(_table()[index]) if n >= 0)


@lru_cache(maxsize=32768)
def _combine(first, second):
    best = {}
    for sets, pair, parts in first:
        for more_sets, more_pair, more_parts in second:
            key = sets + more_sets, pair + more_pair
            if key[0] <= 4 and key[1] <= 1:
                best[key] = max(best.get(key, -1), parts + more_parts)
    return tuple((sets, pair, parts) for (sets, pair), parts in sorted(best.items()))


@lru_cache(maxsize=4096)
def _honors(counts):
    profile = ((0, 0, 0),)
    for count in counts:
        if count >= 2:
            choices = ((0, 0, 1), (0, 1, 0))
            if count >= 3:
                choices += ((1, 0, 0),)
            profile = _combine(profile, choices)
    return profile


@lru_cache(maxsize=65536)
def standard_distance(hand, meld_count):
    """Standard shanten on all 34 kinds, using the project's existing suit table."""
    return min(8 - 2 * total - min(parts, 4 - total) - pair
               for sets, pair, parts in _shape_states(hand) if (total := sets + meld_count) <= 4)


def _shape_states(hand):
    suited = _combine(_combine(_suit(hand[:9]), _suit(hand[9:18])), _suit(hand[18:27]))
    return _combine(suited, _honors(hand[27:]))


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


def _distance(hand, melds, view):
    if getattr(view.rules, 'WILDCARD', None) == WHITE:
        natural, wild = split_wild(hand)
        if view.rules.require_burst and wild:
            # Reserve one wildcard for the drawn tile instead of completing a natural pair.
            wild -= 1
            distance = _sets_distance(natural, len(melds)) - wild
        else:
            distance = standard_distance(natural, len(melds)) - wild
        if not melds:
            singles = sum(count % 2 for count in natural)
            matched = min(wild, singles)
            pairs = sum(count // 2 for count in natural) + matched + (wild - matched) // 2
            distance = min(distance, 6 - pairs)
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
    available = getattr(view.rules, 'OPTIONS', DEFAULT_OPTIONS).tile_kinds
    return tuple(max(0, 4 - view.hand[tile] - view.visible[tile]) if tile in available else 0
                 for tile in range(TILE_COUNT))


def _ready_waits(hand, melds, view, live):
    waits = 0
    augmented = list(hand)
    params = {'indicator': view.wildcard_indicator} if isinstance(view.rules, SongyangRules) else {}
    for tile, count in enumerate(live):
        if not count:
            continue
        augmented[tile] += 1
        ron = WinContext('ron', tile, view.seat_wind, view.round_wind)
        if view.rules.faan(augmented, melds, ron, **params) is not None:
            waits += 2 * count
        elif view.rules.faan(augmented, melds, WinContext('zimo', tile, view.seat_wind, view.round_wind), **params) is not None:
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
    if getattr(view.rules, 'OPTIONS', DEFAULT_OPTIONS).seven_pairs:
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
    if actions[0][0] == 'piao':
        return 'piao', 1
    if actions[0][0] == 'exchange':
        def rank_exchange(action):
            remaining = list(view.hand)
            for tile in action[1]:
                remaining[tile] -= 1
            return _distance(tuple(remaining), (), view), -sum(count // 2 for count in remaining)
        return min(actions, key=rank_exchange)
    if RON in actions:
        return RON
    ten_winds = next((action for action in actions if action[0] == TEN_WINDS), None)
    if ten_winds is not None:
        return ten_winds
    if FLOAT in actions and view.remaining >= 45:
        return FLOAT
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
        wildcard = getattr(view.rules, 'WILDCARD', None)
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
        if kind == 'chi':
            options = getattr(view.rules, 'OPTIONS', DEFAULT_OPTIONS)
            swap_rule = getattr(view.rules, 'chow_swap_lock', None)
            swap_lock = view.rules.SWAP_LOCK if swap_rule is None else swap_rule(view.melds, view.offered_by)
            choices = chow_choices(view.hand, offered, swap_lock=swap_lock,
                                   lock_extensions=options.chow_lock_extensions)
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
        melds = view.melds + (meld,)
        if kind == 'zhigang':
            rank = _rank(tuple(hand), melds, view, live)
        else:
            ranked = _discard_options(hand, melds, view, live,
                                      (tile for tile, count in enumerate(hand) if count and tile not in block))
            rank = min(row[0] for row in ranked)
        if rank < best:
            best = rank
            preferred = kind, argument
    return preferred
