import random
import unittest

from deploy.play import guangzhou
from deploy.play.honors import COPIES, ORPHANS, PASS, RON, ZIMO, HonorsGame, Meld, WinContext
from deploy.play.honors_bonus import LIUZHOU
from deploy.play.tests.test_hongkong import SEQUENCES, counts, fixture


class LiuzhouExplodingFish(unittest.TestCase):
    def test_three_feeds_are_direct_and_add_to_an_outside_provider(self):
        score = LIUZHOU.faan(counts(SEQUENCES), (), WinContext('ron', 22, 27, 27))
        self.assertEqual(LIUZHOU.horse_hit(2, 0, 0), 3)
        self.assertEqual(LIUZHOU.horse_hit(28, 0, 0), 5)
        score = LIUZHOU.with_horses(score, 5)
        partners = [('three_feeds', 1)]
        self.assertEqual(score.share, 6)
        self.assertEqual(LIUZHOU.settlement(score, 'ron', 0, 1, partners), (0, 12, 0, 0))
        self.assertEqual(LIUZHOU.settlement(score, 'ron', 0, 2, partners), (0, 6, 6, 0))
        self.assertEqual(LIUZHOU.settlement(score, 'zimo', 0, None, partners), (0, 18, 6, 6))
        self.assertEqual(LIUZHOU.next_dealer(0, [2, 3], 1), 2)
        self.assertEqual(LIUZHOU.next_dealer(0, [2, 3, 0], 1), 1)

    def test_stack_big_patterns_and_direct_kong_provider_beats_feeds(self):
        big = [0]*3 + [2]*3 + [5]*3 + [7]*3 + [8]*2
        score = LIUZHOU.faan(counts(big), (), WinContext('ron', 8, 27, 27))
        self.assertEqual(score.share, 6)
        score = LIUZHOU.faan(counts(big), (), WinContext('zimo', 8, 27, 27))
        self.assertEqual(score.share, 8)  # Four for big patterns, doubled for closed self-draw.
        dealer = [9,9,9,10,11,12,20,21,22,23,24,25,27,27]
        game = fixture({0: [9, *range(13, 26)], 1: dealer[:-1]}, rules=LIUZHOU, drawn=9)
        game.wall.remove(27)
        game.wall.insert(1, 27)
        fish = game.wall[0]
        for action in (('discard', 9), ('zhigang', None), PASS, PASS, ZIMO):
            game.step(action)
        win = game.win_details[0]
        self.assertEqual(game.horse_tiles, [fish])
        self.assertEqual(win.payments, (3 * win.points, 0, 0, 0))
        self.assertEqual(game.kong_deltas, [-1, 1, 0, 0])
        self.assertEqual(game.scores(), [-win.payments[0]-1, win.payments[0]+1, 0, 0])
        self.assertEqual(game.physical_counts(), COPIES)

    def test_added_kong_charges_original_pong_provider_and_chow_blocks_only_the_tile(self):
        handle = [9, 0, 1, 2, 3, 4, 5, 6, 14, 15, 16]
        game = fixture({0: handle}, {0: [Meld('peng', 9, 2)]}, rules=LIUZHOU, drawn=9)
        for move in (('bugang', 9), PASS, PASS, PASS):
            game.step(move)
        self.assertEqual(game.kong_deltas, [1, 0, -1, 0])
        self.assertEqual(game.players[0].melds, [Meld('gang_bu', 9, 2)])
        self.assertEqual(game.physical_counts(), COPIES)
        dealer = [0, *range(10, 23)]
        hands = {0: dealer, 1: [0, 1, 2, 3, *range(15, 24)]}
        game = fixture(hands, rules=LIUZHOU)
        for move in (('discard', 0), ('chi', 0), PASS, PASS):
            game.step(move)
        self.assertNotIn(('discard', 0), game.legal_actions()[1])
        self.assertIn(('discard', 3), game.legal_actions()[1])


class GuangzhouThreeFaan(unittest.TestCase):
    def test_opening_haidi_kong_and_uncapped_shapes(self):
        hand = counts(SEQUENCES)
        def score(**context):
            return guangzhou.faan(hand, (), WinContext('zimo', 22, 27, 27, **context)).fan
        self.assertEqual(score(), 1)
        self.assertEqual(score(last_live_draw=True), 3)
        self.assertEqual(score(kong_chain=1, kong_provider=0), 3)
        self.assertEqual(score(opening='天胡'), 6)
        self.assertEqual(score(first_nondealer_draw=True), 6)
        flush = counts([0]*3 + [1]*3 + [4]*3 + [6]*3 + [8]*2)
        self.assertEqual(guangzhou.faan(flush, (), WinContext('ron', 8, 27, 27)).fan, 3)
        orphans = counts([*ORPHANS, 27])
        self.assertEqual(guangzhou.faan(orphans, (), WinContext('rob_concealed', 27, 27, 27)).fan, 6)
        self.assertIsNone(guangzhou.faan(hand, (), WinContext('rob_concealed', 22, 27, 27)))
        four_winds = counts([27]*3 + [28]*3 + [29]*3 + [30]*3 + [31]*2)
        self.assertEqual(guangzhou.faan(four_winds, (), WinContext('zimo', 31, 27, 27, kong_chain=1)).fan, 8)

    def test_triple_ron_promotes_all_three_scores_and_only_bills_discarder(self):
        dealer = [22, 5, 6, 7, 8, 20, 21, 25, 26, 27, 29, 30, 31, 32]
        game = fixture({0: dealer, 1: SEQUENCES[:-1], 2: SEQUENCES[:-1], 3: SEQUENCES[:-1]},
                       rules=guangzhou)
        for move in (('discard', 22), RON, RON, RON):
            game.step(move)
        self.assertEqual([win.fan for win in game.win_details], [3, 3, 3])
        self.assertEqual(game.scores(), [-24, 8, 8, 8])
        self.assertEqual(game.physical_counts(), COPIES)

    def test_only_supplier_first_twelve_open_obligation_is_recorded(self):
        melds = {
            1: [Meld('peng', tile, 3) for tile in (0, 1, 2)],
            2: [Meld('peng', tile, 3) for tile in (3, 4, 10)],
        }
        hands = {0: [8, 9, *range(12,24)], 1: [8, 8, 7, 6], 2: [9, 9, 16, 17]}
        game = fixture(hands, melds, rules=guangzhou, drawn=8)
        for move in (('discard', 8), ('peng', None), PASS, PASS):
            game.step(move)
        self.assertEqual(game.players[1].liabilities, [('twelve_open', 0)])
        game.step(('discard', 7))
        for _ in range(3):
            game.step(PASS)
        while game.cur != 0 or game.offer is not None:
            game.step(PASS if game.offer is not None else ('discard', game.drawn_tile))
        for move in (('discard', 9), PASS, ('peng', None), PASS):
            game.step(move)
        self.assertEqual(game.players[2].liabilities, [])
        self.assertEqual(game.physical_counts(), COPIES)

        score = guangzhou.Score(3, ('杠上花',), frozenset({'twelve_open'}), kong_provider=3)
        self.assertEqual(guangzhou.settlement(score, 'zimo', 1, None, [('twelve_open', 0)]), (0, 0, 0, 24))
        self.assertEqual(guangzhou.settlement(score, 'rob_kong', 1, 2, [('twelve_open', 0)]), (0, 0, 24, 0))

    def test_successful_pon_clears_only_the_receivers_win_restriction(self):
        hands = {0: [9, *range(13, 26)], 1: [9,9,0,1,2,3,4,5,6,7,8,11,12]}
        game = fixture(hands, rules=guangzhou)
        game.players[1].declined_wins.add(22)
        game.players[2].declined_wins.add(20)
        for move in (('discard', 9), ('peng', None), PASS, PASS):
            game.step(move)
        self.assertEqual(game.players[1].declined_wins, set())
        self.assertEqual(game.players[2].declined_wins, {20})

    def test_both_rooms_random_play_conserves_tiles_and_points(self):
        rng = random.Random(923)
        for rules in (LIUZHOU, guangzhou):
            for seed in range(150):
                game = HonorsGame(seed, seed % 4, 27, rules=rules)
                while game.phase != 'over':
                    game.step(rng.choice(game.legal_actions()[1]))
                    self.assertEqual(game.physical_counts(), COPIES)
                    self.assertEqual(sum(game.scores()), 0)
                net = game.kong_deltas.copy()
                for win in game.win_details:
                    for payer, points in enumerate(win.payments):
                        net[payer] -= points
                        net[win.actor] += points
                self.assertEqual(game.scores(), net)
