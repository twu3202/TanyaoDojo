from types import SimpleNamespace
import unittest

from deploy.play.honors import Offer, WinContext, ZIMO
from deploy.play.honors_bonus import Score
from deploy.play.honors_game import HonorsSession
from deploy.play.quzhou import QUZHOU_BANKED, QUZHOU_DUANZI, QUZHOU_LATA, QuzhouGame, separated_hand
from deploy.play.tests.test_hongkong import counts
from deploy.play.tests.test_jinyun import set_hands


class QuzhouTests(unittest.TestCase):
    def test_strict_residues_permute_and_loose_does_not_need_distinct_residues(self):
        five_honors = [27, 28, 29, 30, 31]
        strict = counts([1, 4, 7, 9, 12, 15, 20, 23, 26, *five_honors])
        seven = counts([1, 4, 7, 9, 12, 20, 23, *range(27, 34)])
        loose = counts([0, 3, 6, 9, 12, 15, 18, 21, 24, *five_honors])
        self.assertEqual(separated_hand(strict, 0), 5)
        self.assertEqual(separated_hand(seven, 0), 7)
        self.assertEqual(separated_hand(loose, 0), 0)
        ctx = WinContext('ron', 31, 27, 27)
        self.assertEqual(QUZHOU_DUANZI.faan(strict, (), ctx, wildcard=2), Score(4, ('十三百搭',)))
        self.assertEqual(QUZHOU_DUANZI.faan(seven, (), ctx, wildcard=2), Score(7, ('真百搭',)))
        self.assertIsNone(QUZHOU_DUANZI.faan(loose, (), ctx, wildcard=2))
        self.assertEqual(QUZHOU_LATA.faan(loose, (), ctx, wildcard=2), Score(4, ('烂百搭',)))

    def test_white_substitution_and_plain_hand_gate(self):
        pure = counts([0, 31, 2, 3, 4, 5, 6, 7, 8, 0, 31, 2, 4, 4])
        ctx = WinContext('zimo', 4, 27, 27)
        self.assertEqual(QUZHOU_DUANZI.faan(pure, (), ctx, wildcard=1), Score(3, ('清一色',)))
        mixed_suits = counts([0, 1, 2, 9, 10, 11, 18, 19, 20, 21, 22, 23, 4, 4])
        self.assertIsNone(QUZHOU_DUANZI.faan(mixed_suits, (), ctx, wildcard=32))
        self.assertEqual(QUZHOU_LATA.faan(mixed_suits, (), ctx, wildcard=32), Score(2, ('平胡',)))

    def test_variant_supplier_and_other_players_pay_different_bills(self):
        winner = SimpleNamespace(actor=1, score=Score(4, ('十三百搭',)))
        for profile, kind, expected in (
            (QUZHOU_DUANZI, 'discard', (4, 0, 0, 0)),
            (QUZHOU_DUANZI, 'gang_bu', (8, 0, 0, 0)),
            (QUZHOU_LATA, 'discard', (4, 0, 3, 3)),
            (QUZHOU_BANKED, 'discard', (4, 0, 1, 1)),
        ):
            game = QuzhouGame(9, 0, 27, rules=profile)
            self.assertEqual(profile.settle_hand(game, [winner], Offer(0, 11, kind))[0][1], expected)

    def test_banked_game_finishes_only_on_an_overdrawn_payment(self):
        self.assertEqual(HonorsSession(variant=QUZHOU_BANKED.VARIANT).totals, [30] * 4)
        for funds, finished, totals in (
            ([30, 2, 40, 48], 'finished', [48, 0, 32, 40]),
            ([30, 8, 36, 46], 'round_end', [54, 0, 28, 38]),
        ):
            with self.subTest(funds=funds):
                session = HonorsSession(variant=QUZHOU_BANKED.VARIANT, seed=7, pacing=0)
                session.hand_number = 5
                session.totals = funds
                game = QuzhouGame(9, 0, 27, rules=QUZHOU_BANKED, bank=funds)
                session.game = game
                game.indicator_tile = game.wildcard_tile = game.wildcard_indicator = 33
                set_hands(game, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 4, 4], [], [], []], taken=(33,))
                game.drawn_tile = 4
                game.step(ZIMO)
                session.play_round()
                self.assertEqual(session.status, finished)
                self.assertEqual(session.result['scores'], totals)
                self.assertEqual(game.physical_counts(), game.tile_supply)
                if finished == 'finished':
                    self.assertEqual(session.result['busted'], [1])
                    self.assertEqual(session.result['match_hands'], 5)
                    self.assertEqual(session.result['match_winners'], [0])
                else:
                    self.assertNotIn('busted', session.result)
