"""Quzhou's separated hands and the Duanzi, ordinary Lata and banked Lata tables."""

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import permutations

from .dongyang import isolated_shape, white_assignments
from .hangzhou import FOUR_HANGZHOU, WHITE, HangzhouRules, split_wild
from .honors_bonus import Score
from .jiaxing import PrintedWildcardGame
from .wildcard_shapes import can_form_all_pungs, can_form_pairs, can_form_standard


_SUIT_RESIDUES = tuple(permutations(range(3)))
_DUANZI_POINTS = {
    '混一色': (2, 4), '清一色': (3, 7), '大碰胡': (3, 7), '混碰': (4, 10), '清碰': (5, 13),
    '七对': (3, 7), '混七对': (4, 10), '清七对': (5, 13), '十三百搭': (2, 4), '真百搭': (3, 7),
    '烂百搭': (2, 4), '全风字': (6, 16), '全风字大碰胡': (11, 31), '全风字七对': (11, 31),
    '三财倒': (1, 0), '七风倒': (1, 0),
}


def separated_hand(natural, wild):
    """Return the number of honors in a best strict 147/258/369 arrangement."""
    if max(natural) > 1:
        return 0
    classes = [{rank % 3 for rank in range(9) if natural[9 * suit + rank]} for suit in range(3)]
    if not any(all(group <= {value} for group, value in zip(classes, arrangement, strict=True))
               for arrangement in _SUIT_RESIDUES):
        return 0
    honors = sum(natural[27:])
    maximum = min(7, honors + wild)
    return maximum if maximum >= 5 else 0


@lru_cache(maxsize=16384)
def separated_distance(natural, wild):
    honors = sum(bool(count) for count in natural[27:])
    fits = [[sum(bool(natural[suit * 9 + rank]) for rank in range(residue, 9, 3)) for residue in range(3)]
            for suit in range(3)]
    return 13 - honors - max(sum(fits[suit][value] for suit, value in enumerate(p)) for p in _SUIT_RESIDUES) - wild


@dataclass(frozen=True)
class QuzhouRules(HangzhouRules):
    lata: bool = False
    BANK: int | None = None

    @property
    def WALL_RESERVE(self):
        return 34 if self.lata else 0

    @property
    def MATCH_HANDS(self):
        return None if self.BANK is not None else 8

    def next_dealer(self, dealer, winners, provider):
        if self.lata:
            return dealer if not winners or dealer in winners else (dealer + 1) % self.OPTIONS.seat_count
        return winners[0] if winners else dealer

    def faan(self, hand, melds, ctx, *, wildcard):
        if ctx.kind not in ('zimo', 'ron', 'rob_kong') or ctx.tile is None or sum(hand) != 14 - 3 * len(melds):
            return None
        self_draw = ctx.kind == 'zimo'
        if not hand[ctx.tile] or (not self_draw and ctx.tile == wildcard):
            return None
        natural, wild = split_wild(hand, wildcard)
        candidates = set()
        if self_draw:
            if wild == 3:
                candidates.add('三财倒')
            if not melds and all(hand[rank] for rank in range(27, 34)):
                candidates.add('七风倒')
        open_faces = {tile for meld in melds for tile in meld.tiles}
        choices = white_assignments(natural, wildcard) if wildcard != WHITE else (natural,)
        pungs_allowed = all(meld.kind != 'chi' for meld in melds)
        standard = False
        for counts in choices:
            faces = open_faces | {tile for tile, count in enumerate(counts) if count}
            suits = {tile // 9 for tile in faces if tile < 27}
            honors = max(faces) >= 27
            mixed = len(suits) == 1 and honors
            pure = len(suits) == 1 and not honors
            if not suits:
                candidates.add('全风字')
            if not melds:
                if can_form_pairs(counts, wild):
                    candidates.add('全风字七对' if not suits else '混七对' if mixed else '清七对' if pure else '七对')
                spaced = separated_hand(counts, wild)
                if spaced:
                    candidates.add('真百搭' if spaced == 7 else '十三百搭')
                elif self.lata and isolated_shape(counts, wild):
                    candidates.add('烂百搭')
            if not can_form_standard(counts, wild):
                continue
            standard = True
            if pure or mixed:
                candidates.add('清一色' if pure else '混一色')
            if pungs_allowed and can_form_all_pungs(counts, wild):
                candidates.add('全风字大碰胡' if not suits else '混碰' if mixed else '清碰' if pure else '大碰胡')

        picked = max(candidates, key=lambda n: (_DUANZI_POINTS[n][not self_draw], n), default=None)
        if not picked and not (self.lata and standard):
            return None
        if not self.lata:
            share = _DUANZI_POINTS[picked][not self_draw]
            return Score(share, (picked,))
        if self.BANK is not None:
            return Score(1, (picked or '平胡',))
        if ctx.kind == 'rob_kong':
            return Score(30, ('抢杠胡',))
        if ctx.opening == '地胡' and not self_draw:
            return Score(10, ('地胡',))
        if self_draw and ctx.opening == '天胡':
            return Score(10, ('天胡',))
        if self_draw and ctx.kong_chain:
            return Score(10, ('杠上开花',))
        if len(melds) == 4:
            return Score(10, ('单钓',))
        return Score(4 if picked else 2, (picked or '平胡',))

    def settle_hand(self, game, winners, offer):
        winner, = winners
        actor = winner.actor
        share = winner.score.share
        if self.BANK is not None:
            dues = self._banked_dues(game, actor, offer)
            game.busted = [seat for seat, due in enumerate(dues) if due > game.bank[seat]]
            payments = tuple(min(due, balance) for due, balance in zip(dues, game.bank, strict=True))
        elif offer is None or (self.lata and winner.score.patterns[0] in ('地胡', '单钓')):
            payments = tuple(share if seat != actor else 0 for seat in range(game.seat_count))
        elif self.lata and offer.kind == 'discard':
            payments = tuple(0 if seat == actor else share if seat == offer.actor else share - 1
                             for seat in range(game.seat_count))
        else:
            due = 2 * share if offer.kind == 'gang_bu' and not self.lata else share
            payments = tuple(due if seat == offer.actor else 0 for seat in range(game.seat_count))
        return [(winner.score, payments)]

    @staticmethod
    def _banked_dues(game, winner, offer):
        dues = [0] * game.seat_count
        if offer is None:
            for seat in range(game.seat_count):
                if seat != winner:
                    dues[seat] = 8 if game.dealer in (winner, seat) else 4
        elif winner == game.dealer:
            dues = [0 if seat == winner else 8 if seat == offer.actor else 4 for seat in range(game.seat_count)]
        else:
            dues = [0 if seat == winner else 1 for seat in range(game.seat_count)]
            dues[game.dealer] += 1
            dues[offer.actor] += 2 if offer.actor == game.dealer else 1
        return dues


_OPTIONS = replace(FOUR_HANGZHOU.OPTIONS, host_deals_first=False, show_winds=True, drawn_added_kong_only=True)
QUZHOU_DUANZI = QuzhouRules('quzhou-duanzi', replace(_OPTIONS, chow=False), fixed_dealer=1, flip_wildcard=True)
QUZHOU_LATA = QuzhouRules('quzhou-lata', _OPTIONS, fixed_dealer=1, flip_wildcard=True, lata=True)
QUZHOU_BANKED = QuzhouRules('quzhou-lata-banked', _OPTIONS, fixed_dealer=1, flip_wildcard=True, lata=True, BANK=30)


class QuzhouGame(PrintedWildcardGame):
    def __init__(self, seed, dealer, round_wind, *, rules, dealer_streak=0, bank=None):
        self.bank = tuple(bank) if bank is not None else (rules.BANK,) * rules.OPTIONS.seat_count
        self.busted = []
        super().__init__(seed, dealer, round_wind, rules=rules, dealer_streak=dealer_streak)

    def _can_take_win(self, actor, offer):
        return offer.kind in ('discard', 'gang_bu')

    def _score(self, actor, hand, melds, ctx):
        return self.rules.faan(hand, melds, ctx, wildcard=self.wildcard_tile)
