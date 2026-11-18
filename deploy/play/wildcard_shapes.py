"""Standard Mahjong sets and pairs with a known number of wildcards."""

from functools import lru_cache
from itertools import combinations_with_replacement

from .honors import standard_shapes


def without_tile(counts, tile, count=1):
    return counts[:tile] + (counts[tile] - count,) + counts[tile + 1:]


@lru_cache(maxsize=256)
def wildcard_decompositions(natural, wild, meld_count, assignable):
    """Return complete shapes along with which ranks the jokers can occupy."""
    results = []
    for assignment in combinations_with_replacement(assignable, wild):
        filled = list(natural)
        for tile in assignment:
            filled[tile] += 1
        filled = tuple(filled)
        shapes = tuple(standard_shapes(filled, meld_count))
        if shapes:
            results.append((filled, tuple(dict.fromkeys(assignment)), shapes))
    return tuple(results)


@lru_cache(maxsize=65536)
def can_form_sets(counts, wild, excluded=frozenset()):
    """Every natural tile must be used, and honors never form sequences."""
    if (sum(counts) + wild) % 3:
        return False
    first = next((tile for tile, count in enumerate(counts) if count), None)
    if first is None:
        return True
    if first in excluded:
        return False
    triplet = min(3, counts[first])
    if 3 - triplet <= wild and can_form_sets(without_tile(counts, first, triplet), wild - 3 + triplet, excluded):
        return True
    if first >= 27:
        return False
    suit = first // 9 * 9
    for start in range(max(suit, first - min(2, wild)), min(first, suit + 6) + 1):
        if excluded and excluded.intersection(range(start, start + 3)):
            continue
        rest = list(counts)
        missing = 0
        for tile in range(start, start + 3):
            if rest[tile]:
                rest[tile] -= 1
            else:
                missing += 1
        if missing <= wild and can_form_sets(tuple(rest), wild - missing, excluded):
            return True
    return False


@lru_cache(maxsize=32768)
def can_form_standard(counts, wild, excluded=frozenset()):
    if any(counts[tile] for tile in excluded):
        return False
    if excluded:
        return _with_pair(counts, wild, lambda hand, jokers: can_form_sets(hand, jokers, excluded))
    return _with_pair(counts, wild, can_form_sets)


def can_form_pungs(counts, wild):
    return sum(-count % 3 for count in counts) <= wild and (sum(counts) + wild) % 3 == 0


@lru_cache(maxsize=16384)
def can_form_all_pungs(counts, wild):
    return _with_pair(counts, wild, can_form_pungs)


def _with_pair(counts, wild, sets_fit):
    if (sum(counts) + wild) % 3 != 2:
        return False
    if wild >= 2 and sets_fit(counts, wild - 2):
        return True
    for tile, count in enumerate(counts):
        if count:
            pair = min(2, count)
            if 2 - pair <= wild and sets_fit(without_tile(counts, tile, pair), wild - 2 + pair):
                return True
    return False


def can_form_pairs(counts, wild):
    return (sum(counts) + wild) % 2 == 0 and sum(count % 2 for count in counts) <= wild
