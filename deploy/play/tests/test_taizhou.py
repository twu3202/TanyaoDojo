from collections import Counter
from types import SimpleNamespace
import unittest

from deploy.play.hangzhou import WHITE
from deploy.play.honors import PASS, RON, Meld, Offer, WinContext
from deploy.play.honors_bonus import Score
from deploy.play.taizhou import TAIZHOU, TaizhouGame, _one_from_ready, pair_bills
from deploy.play.tests.test_hongkong import counts


class TaizhouTests(unittest.TestCase):
    def test_pairwise_differences_caps_and_liability(self):
        self.assertEqual(pair_bills([34, 18, 41, 7], 0), [
            [0, 0, 7, 0], [16, 0, 11.5, 0], [0, 0, 0, 0], [27, 5.5, 17, 0],
        ])
        self.assertEqual(pair_bills([100, 100, 40, 0], 0), [
            [0, 0, 0, 0], [0, 0, 0, 0], [100, 100, 0, 0], [100, 100, 20, 0],
        ])
        self.assertEqual(pair_bills([0, 42, 20, 8], 0, 0), [
            [0, 70, 26, 8], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
        ])

    def test_winner_scores_wait_pung_and_wildcard_restoration(self):
        hand = counts([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27, 32, 32])
        inside = TAIZHOU.faan(hand, (), WinContext('ron', 1, 27, 27))
        self.assertEqual(inside.share, 88)
        self.assertIn('嵌档', inside.patterns)
        self.assertIn('无得', inside.patterns)
        pung = TAIZHOU.faan(hand, (), WinContext('ron', 27, 27, 27))
        self.assertEqual(pung.share, 64)
        hand = counts([2, 3, WHITE, 3, WHITE, 5, 18, 19, 20, 27, 27, 27, 32, 32])
        restored = TAIZHOU.faan(hand, (), WinContext('zimo', WHITE, 27, 27), indicator=4)
        self.assertEqual(restored.share, 96)
        self.assertIn('得还原', restored.patterns)
        self.assertIsNone(TAIZHOU.faan(hand, (), WinContext('ron', WHITE, 27, 27), indicator=4))
        hand = counts([0, WHITE, 2, 0, WHITE, 2, 18, 19, 20, 27, 27, 27, 32, 32])
        best = TAIZHOU.faan(hand, (), WinContext('zimo', WHITE, 27, 27), indicator=1)
        self.assertEqual(best.share, 100)
        self.assertNotIn('得还原', best.patterns)

    def test_loser_scores_printed_faces_and_excludes_robbed_tile(self):
        game = TaizhouGame(4, 0, 27, rules=TAIZHOU)
        game.wildcard_indicator = 4
        game.players[1].hand = counts([4, 4, WHITE, 32, 32, 32, 33, 33])
        self.assertEqual(game.loser_points(1, None), 28)
        self.assertEqual(game.loser_points(1, Offer(1, 32, 'gang_bu')), 8)
        game.players[1].hand = counts([WHITE, WHITE, WHITE, 29, 29, 29])
        game.wildcard_indicator = WHITE
        self.assertEqual(game.loser_points(1, None), 16)

    def test_fresh_discard_pays_all_pair_bills_and_advances_dealer(self):
        game = TaizhouGame(4, 0, 27, rules=TAIZHOU)
        game.wildcard_indicator = 8
        game.indicator_tile = WHITE
        hands = [[1], [0, 2, 9, 10, 11, 18, 19, 20, 28, 28, 28, 32, 32], [], []]
        wall = list((game.tile_supply - Counter([WHITE] + sum(hands, []))).elements())
        for seat, tiles in enumerate(hands):
            while len(tiles) < 13 + (seat == 0):
                tiles.append(wall.pop())
            game.players[seat].hand = counts(tiles)
        game.wall = wall[:30]
        game.discards[2] = wall[30:]
        game.played_faces.update(game.discards[2])
        self.assertNotIn(1, game.played_faces)
        self.assertEqual(game.physical_counts(), game.tile_supply)
        game.step(('discard', 1))
        self.assertIn(RON, game.legal_actions()[1])
        game.step(RON)
        while game.phase != 'over':
            self.assertEqual(game.legal_actions()[1], [PASS])
            game.step(PASS)
        self.assertEqual(game.physical_counts(), game.tile_supply)
        self.assertEqual(game.winners, [1])
        self.assertEqual(game.next_dealer, 1)
        result = game.taizhou_settlement
        self.assertEqual((result['liable'], result['liability_reason']), (0, 'late_fresh_tile'))
        self.assertEqual(result['values'][1], 88)
        self.assertTrue(all(points > 0 for points in result['bills'][0][1:]))
        self.assertEqual(game.scores(), [-sum(result['bills'][0]), *result['bills'][0][1:]])

    def test_structural_one_away_and_pure_suit_supplier(self):
        melds = (Meld('chi', 0, 1), Meld('chi', 3, 1), Meld('chi', 6, 1))
        self.assertTrue(_one_from_ready(tuple(counts([9, 10, 18, 27])), melds))
        self.assertFalse(_one_from_ready(tuple(counts([9, 10, 27, 27])), melds))
        game = TaizhouGame(4, 0, 27, rules=TAIZHOU)
        game.players[2].melds = list(melds)
        winner = SimpleNamespace(actor=2, score=Score(100, ('清一色',)))
        self.assertEqual(game.liable_seat(winner, None), (1, 'pure_suit_feeds'))
        self.assertEqual(game.liable_seat(winner, Offer(3, 4, 'gang_bu')), (3, 'pure_suit_feeds'))

    def test_fresh_dragon_and_suited_white_substitute_liability(self):
        game = TaizhouGame(4, 0, 27, rules=TAIZHOU)
        game.wildcard_indicator = 8
        game.fresh_discard = True
        game.players[0].melds = [Meld('chi', t, 3) for t in (0, 3, 6)]
        game.players[0].hand = counts([9, 10, 18, 32])
        winner = SimpleNamespace(actor=1, score=Score(42, ()))
        for tile in (32, 8):
            with self.subTest(tile=tile):
                self.assertEqual(game.liable_seat(winner, Offer(0, tile, 'discard')), (0, 'fresh_dragon'))
        game.fresh_discard = False
        self.assertEqual(game.liable_seat(winner, Offer(0, 8, 'discard')), (None, None))

    def test_draw_keeps_wall_and_skips_all_settlement(self):
        game = TaizhouGame(6, 2, 27, rules=TAIZHOU)
        reserve = game.wall[:16]
        while game.phase != 'over':
            _, actions = game.legal_actions()
            game.step(PASS if PASS in actions else next(action for action in actions if action[0] == 'discard'))
        self.assertEqual(game.wall, reserve)
        self.assertEqual(game.physical_counts(), game.tile_supply)
        self.assertEqual(game.scores(), [0, 0, 0, 0])
        self.assertEqual(game.next_dealer, 3)
        self.assertIsNone(game.taizhou_settlement)
