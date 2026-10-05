"""Suit / dragon symmetry augmentation for offline training (2026-10-05, patches/symmetry_augmentation).

A riichi game stays a legal game with the same scores when its tiles are relabelled by
  - any permutation of the three suits (6), and
  - a cyclic rotation of the three dragons, haku -> hatsu -> chun -> haku (3).
That is 18 exact symmetries. Dragon transpositions are not among them: the dora indicator relation is directed
(indicator P makes F dora, F makes C, C makes P), so swapping two dragons changes which tile an honour indicator
makes dora, and the replayed game would score dora on the wrong tile. Number reversal (1 <-> 9) fails for the same
reason. The one rule the 18 still break is ryuuiisou (souzu + hatsu only), a yakuman too rare to matter.

The relabelling is done on the raw mjai text, before libriichi parses it, so the observation, the action labels and
the masks are all re-encoded by libriichi itself; nothing downstream needs to know about the permutation.
"""
import itertools
import re

SUITS = 'mps'
DRAGONS = 'PFC'

# every tile string of a mjai log, quoted: "1m".."9s", the red fives "5mr" "5pr" "5sr", and the dragons
_TILE_RE = re.compile(r'"(?:[1-46-9][mps]|5[mps]r?|[PFC])"')


def _tables():
    tables = []
    for sp in itertools.permutations(SUITS):
        smap = dict(zip(SUITS, sp))
        for k in range(3):
            t = {}
            for s in SUITS:
                for n in '123456789':
                    t[f'"{n}{s}"'] = f'"{n}{smap[s]}"'
                t[f'"5{s}r"'] = f'"5{smap[s]}r"'
            for i, d in enumerate(DRAGONS):
                t[f'"{d}"'] = f'"{DRAGONS[(i + k) % 3]}"'
            tables.append(t)
    return tables


TABLES = _tables()          # index 0 is the identity
N_ELEMENTS = len(TABLES)    # 18


def relabel(raw_log, g):
    """Apply symmetry g (0..17) to a raw mjai log (one JSON event per line)."""
    if g == 0:
        return raw_log
    t = TABLES[g]
    return _TILE_RE.sub(lambda m: t[m.group(0)], raw_log)
