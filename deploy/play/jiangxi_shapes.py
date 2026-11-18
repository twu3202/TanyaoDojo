"""Jiangxi honor sequences and irregular spaced hands with jokers."""

from functools import lru_cache
from itertools import combinations

from .wildcard_shapes import _with_pair, can_form_sets, without_tile


HONOR_CHOWS = (*combinations(range(27, 31), 3), (31, 32, 33))
HONOR_CHOW_ID = 34
SPACED_MASKS = tuple(
    sum(1 << rank for rank in ranks)
    for size in range(4) for ranks in combinations(range(9), size)
    if all(right - left >= 3 for left, right in zip(ranks, ranks[1:]))
)


def split_jing(hand, faces, *, keep_honors=False):
    natural = list(hand)
    wild = 0
    for face in faces:
        kept = int(keep_honors and face >= 27 and natural[face] > 0)
        wild += natural[face] - kept
        natural[face] = kept
    return tuple(natural), wild


@lru_cache(maxsize=65536)
def honor_sets(natural, wild):
    """Partition honors first, then let the shared solver handle numbered suits."""
    if (sum(natural) + wild) % 3:
        return False
    face = next((t for t in range(27, 34) if natural[t]), None)
    if face is None:
        return can_form_sets(natural, wild)

    used = min(3, natural[face])
    if 3 - used <= wild and honor_sets(without_tile(natural, face, used), wild - (3 - used)):
        return True
    for group in HONOR_CHOWS:
        if face not in group:
            continue
        missing = sum(not natural[t] for t in group)
        if missing <= wild:
            rest = list(natural)
            for t in group:
                rest[t] -= bool(rest[t])
            if honor_sets(tuple(rest), wild - missing):
                return True
    return False


@lru_cache(maxsize=65536)
def honor_standard(natural, wild):
    return _with_pair(natural, wild, honor_sets)


@lru_cache(maxsize=8192)
def honor_chow_routes(honors):
    """Enumerate ways to take complete natural honor chows for a distance estimate."""
    routes = {(0, honors)}
    for group in HONOR_CHOWS:
        if all(honors[t - 27] for t in group):
            rest = list(honors)
            for t in group:
                rest[t - 27] -= 1
            routes.update((count + 1, row) for count, row in honor_chow_routes(tuple(rest)))
    return tuple(sorted(routes))


@lru_cache(maxsize=512)
def spaced_capacity(mask):
    return max((candidate.bit_count() for candidate in SPACED_MASKS if candidate & mask == mask), default=-34)


@lru_cache(maxsize=512)
def spaced_kept(mask):
    return max((candidate & mask).bit_count() for candidate in SPACED_MASKS)


def numbered_masks(hand):
    return tuple(sum(1 << rank for rank in range(9) if hand[suit + rank]) for suit in (0, 9, 18))


def can_form_spaced(natural, wild):
    return (sum(natural) + wild == 14 and max(natural) <= 1
            and 7 + sum(spaced_capacity(mask) for mask in numbered_masks(natural)) >= 14)


def spaced_distance(natural, wild):
    kept = sum(spaced_kept(mask) for mask in numbered_masks(natural)) + sum(bool(n) for n in natural[27:])
    return 13 - kept - wild
