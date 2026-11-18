"""Public-information tile efficiency for a three-faan Hong Kong room."""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from common.suit_table import load_table

from . import hongkong
from .honors import DRAGONS, ORPHANS, PASS, RON, TILE_COUNT, ZIMO, Meld, WinContext, chow_choices


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


def policy_view(game, actor, actions):
    visible = Counter(tile for river in game.discards for tile in river)
    for player in game.players:
        for meld in player.melds:
            visible.update(meld.tiles)
    if game.offer is not None and game.offer.kind != 'discard':
        visible[game.offer.tile] += 1
    player = game.players[actor]
    return PolicyView(tuple(player.hand), tuple(player.melds), tuple(visible[i] for i in range(TILE_COUNT)),
                      tuple(actions), 27 + (actor - game.dealer) % 4, game.round_wind,
                      None if game.offer is None else game.offer.tile, len(game.wall))


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
    suited = _combine(_combine(_suit(hand[:9]), _suit(hand[9:18])), _suit(hand[18:27]))
    states = _combine(suited, _honors(hand[27:]))
    return min(8 - 2 * total - min(parts, 4 - total) - pair
               for sets, pair, parts in states if (total := sets + meld_count) <= 4)


def _triplet_distance(hand, needed):
    groups = sorted(range(TILE_COUNT), key=lambda tile: hand[tile], reverse=True)
    base = sum(min(hand[tile], 3) for tile in groups[:needed])
    outside = groups[needed]
    best = base + min(2, hand[outside])
    for pair in groups[:needed]:
        best = max(best, base - min(3, hand[pair]) + min(3, hand[outside]) + min(2, hand[pair]))
    return 3 * needed + 1 - best


def _distance(hand, melds, view):
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
    return tuple(max(0, 4 - held - shown) for held, shown in zip(view.hand, view.visible))


def _ready_waits(hand, melds, view, live):
    waits = 0
    augmented = list(hand)
    for tile, count in enumerate(live):
        if not count:
            continue
        augmented[tile] += 1
        ron = WinContext('ron', tile, view.seat_wind, view.round_wind)
        if hongkong.faan(augmented, melds, ron) is not None:
            waits += 2 * count
        elif hongkong.faan(augmented, melds, WinContext('zimo', tile, view.seat_wind, view.round_wind)) is not None:
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


def _ukeire(distance, rest, melds, live):
    # Once a route has been chosen, compare ordinary shape improvements among tied discards.
    if distance == 0:
        return 0
    before = standard_distance(rest, len(melds))
    if before <= 0:
        return 0
    available = list(rest)
    effective = 0
    for tile, count in enumerate(live):
        if not count:
            continue
        available[tile] += 1
        if standard_distance(tuple(available), len(melds)) < before:
            effective += count
        available[tile] -= 1
    return effective


def heuristic(view):
    actions = view.actions
    if RON in actions:
        return RON
    if ZIMO in actions:
        return ZIMO
    if actions == (PASS,):
        return PASS
    live = _live_tiles(view)
    discards = [tile for kind, tile in actions if kind == 'discard']
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
        tile, _ = max(choices, key=lambda item: (_ukeire(best[0], item[1], view.melds, live),
                                                -view.hand[item[0]], item[0]))
        return 'discard', tile

    best = _rank(view.hand, view.melds, view, live)
    preferred = PASS
    offered = view.offered
    for kind, argument in actions:
        if kind == 'pass':
            continue
        if kind == 'chi':
            take, block = next((take, block) for start, take, block in chow_choices(view.hand, offered)
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
