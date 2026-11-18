from dataclasses import replace
import unittest

from deploy.play.suited import COPIES, PASS, RON, Profile, SuitedGame
from deploy.play.sichuan_game import distance
from deploy.play.suited_regional import WinContext, lishui_payment, shaanxi_payment, wake_award, wenxi_payment
from deploy.play.tests.test_suited import CALLER, READY, SPARE, counts, fixture


def win_context(**changes):
    return replace(WinContext(1, 2, 0, 'ron', None, False, None, 25, 1), **changes)


class RegionalPayments(unittest.TestCase):
    def test_shaanxi_doubles_only_the_dealer_side_of_each_payment(self):
        self.assertEqual(shaanxi_payment([], [], win_context())[2], (0, 0, 1, 0))
        self.assertEqual(shaanxi_payment([], [], win_context(provider=0))[2], (2, 0, 0, 0))
        self.assertEqual(shaanxi_payment([], [], win_context(provider=None, kind='zimo'))[2], (4, 0, 2, 2))
        self.assertEqual(shaanxi_payment([], [], win_context(actor=0, provider=None, kind='zimo'))[2], (0, 4, 4, 4))

    def test_lishui_triple_win_and_last_four_combine(self):
        hand = counts(READY + [0])
        self.assertEqual(lishui_payment(hand, [], win_context(tiles_left=4))[2], (0, 0, 1, 0))
        self.assertEqual(lishui_payment(hand, [], win_context(winner_count=3))[2], (0, 0, 2, 0))
        self.assertEqual(lishui_payment(hand, [], win_context(tiles_left=3))[2], (0, 0, 6, 0))
        patterns, _, shares = lishui_payment(hand, [], win_context(tiles_left=3, winner_count=3))
        self.assertEqual(patterns, ('平胡', '尾牌包三', '一炮三响'))
        self.assertEqual(shares, (0, 0, 12, 0))
        self.assertEqual(wenxi_payment(hand, [], win_context(tiles_left=3, winner_count=3))[2], (0, 0, 1, 0))

    def test_qingtian_pair_shapes_and_kong_win_bonuses(self):
        hand = counts([0] * 3 + [3] * 3 + [9] * 3 + [11] * 3 + [16] * 2)
        self.assertEqual(lishui_payment(hand, [], win_context())[:2], (('豪华大对',), 18))
        self.assertEqual(wenxi_payment(hand, [], win_context())[:2], (('豪华大对',), 3))
        hand = counts([0,1,2,3,4,5,6] * 2)
        self.assertEqual(lishui_payment(hand, [], win_context())[:2], (('清七对',), 18))
        self.assertEqual(wenxi_payment(hand, [], win_context())[:2], (('清七对',), 4))
        hand = counts(READY + [0])
        context = win_context(kind='zimo', provider=None, after_kong=True, kong_provider=0, tiles_left=0)
        self.assertEqual(lishui_payment(hand, [], context)[2], (24, 0, 0, 0))
        self.assertEqual(wenxi_payment(hand, [], context)[2], (6, 0, 0, 0))
        self.assertEqual(lishui_payment(hand, [], win_context(kind='rob_kong'))[2], (0, 0, 6, 0))

    def test_wake_wraps_ranks_extends_only_for_kongs_and_shares_the_prefix(self):
        wall, indicators = [25, 1, 0], []
        self.assertEqual(wake_award(counts([1, 1, 2, 3, 4]), [('gang_an', 8)], wall, indicators), (2, 7))
        self.assertEqual(wake_award(counts([1, 2]), [], wall, indicators), (1, 1))
        self.assertEqual(indicators, [0, 1])
        self.assertEqual(wall, [25])
        self.assertEqual(wake_award(counts([1, 2]), [], [], []), (0, 0))


class RegionalRules(unittest.TestCase):
    def test_shaanxi_nearest_wins_without_asking_later_seats(self):
        game = fixture([SPARE + [0], CALLER, READY, READY], profile=Profile.SHAANXI)
        for action in (('discard', 0), ('peng', None), RON):
            game.step(action)
        self.assertEqual(game.winners, [2])
        self.assertEqual(game.scores(), [-2, 0, 2, 0])
        self.assertEqual(game.next_dealer, 2)
        self.assertEqual(game.phase, 'over')
        self.assertEqual(game.physical_counts(), COPIES)

    def test_shaanxi_pairs_option_excludes_only_nonstandard_hands(self):
        game = SuitedGame(0, profile=Profile.SHAANXI)
        self.assertFalse(game._can_win(counts([0, 2, 4, 6, 9, 11, 13] * 2), []))
        self.assertTrue(game._can_win(counts([0, 1, 2, 3, 4, 5, 6] * 2), []))
        hand = counts([0, 2, 4, 6, 9, 11] * 2 + [13])
        self.assertEqual(distance(hand, 0, None), 0)
        self.assertGreater(distance(hand, 0, None, allow_seven_pairs=False), 0)

    def test_shaanxi_same_tile_but_guilin_all_tiles_are_locked_before_draw(self):
        ready = list(range(9)) + [9, 9, 18, 18]
        hands = [[9] + [19]*4 + [20]*4 + [21]*4 + [22],
                 [18] + [23]*4 + [24]*4 + [25]*4, ready, [10]*4 + [11]*4 + [12]*4 + [13]]
        for profile, other_wait_allowed in ((Profile.SHAANXI, True), (Profile.GUILIN, False)):
            with self.subTest(profile=profile):
                game = fixture(hands, profile=profile)
                game.step(('discard', 9))
                game.step(PASS)
                self.assertIn(RON, game.legal_actions()[1])
                game.step(PASS)
                game.step(PASS)
                self.assertTrue(game._discard_win_locked(game.players[2], 9))
                game.step(('discard', 18))
                self.assertEqual(RON in game.legal_actions()[1], other_wait_allowed)
                game.step(('peng', None))
                game.step(PASS)
                game.step(PASS)
                expected = {9} if profile is Profile.GUILIN else set()
                self.assertEqual(game.players[2].declined_tiles, expected)

    def test_direct_kong_replacement_win_is_charged_to_original_provider(self):
        hands = [SPARE + [0], [0, 0, 0, *range(1, 9), 9, 9],
                 [22]*4 + [23]*4 + [24]*4 + [25], [10]*4 + [11]*4 + [12]*4 + [13]]
        for profile, expected in ((Profile.LISHUI, 15), (Profile.WENXI, 7)):
            with self.subTest(profile=profile):
                game = fixture(hands, profile=profile)
                game.wall.remove(3)
                game.wall.insert(0, 3)
                for action in (('discard', 0), ('zhigang', None), PASS, PASS, ('zimo', None)):
                    game.step(action)
                self.assertEqual(game.scores(), [-expected, expected, 0, 0])
                self.assertEqual(game.physical_counts(), COPIES)

    def test_only_qingtian_rooms_refund_kongs_at_exhaustion(self):
        hands = [SPARE + [0], [0, 0, 0, 6, 6, 7, 7, 8, 8, 15, 15, 16, 16], READY, READY]
        for profile, amount, refund in ((Profile.LISHUI, 3, True), (Profile.WENXI, 1, True), (Profile.GUILIN, 3, False)):
            with self.subTest(profile=profile):
                game = fixture(hands, profile=profile)
                for action in (('discard', 0), ('zhigang', None), PASS, PASS):
                    game.step(action)
                self.assertEqual(game.scores(), [-amount, amount, 0, 0])
                for _ in range(500):
                    if game.phase == 'over':
                        break
                    game.step(game.legal_actions()[1][0])
                self.assertEqual(game.reason_code, 'draw')
                expected = [0, 0, 0, 0] if refund else [-amount, amount, 0, 0]
                self.assertEqual(game.scores(), expected)
                self.assertEqual(game.physical_counts(), COPIES)

    def test_guilin_two_winners_use_one_physical_wake_indicator(self):
        game = fixture([SPARE + [0], CALLER, READY, READY], profile=Profile.GUILIN)
        game.wall.remove(1)
        game.wall.append(1)
        for action in (('discard', 0), PASS, RON, RON):
            game.step(action)
        self.assertEqual(game.wake_indicators, [1])
        self.assertEqual(game.wake_results, {2: (1, 2), 3: (1, 2)})
        self.assertEqual(game.scores(), [-6, 0, 3, 3])
        self.assertEqual(game.physical_counts(), COPIES)
