from collections import Counter
import unittest

from deploy.play.chongji import CHONGJI, CHONGJI_SEVEN, ChongjiGame, seven_gap_pairs
from deploy.play.honors import PASS, RON, ZIMO, Meld, Offer, WinContext
from deploy.play.honors_policy import heuristic, policy_view
from deploy.play.tests.test_hangzhou import play_passes
from deploy.play.tests.test_hongkong import counts
from deploy.play.zhejiang_flipped import ELEVEN_WINDS


def fixed_game(rules=CHONGJI, streak=0):
    game = ChongjiGame(6, 0, 27, rules=rules, dealer_streak=streak)
    game.wildcard_indicator = game.indicator_tile = 7
    game.wildcard_tile = 8
    return game


WAIT = [0, 1, 9, 10, 11, 18, 19, 20, 27, 27, 27, 31, 31]


def set_hands(game, starts):
    used = Counter([game.wildcard_indicator])
    for start, player in zip(starts, game.players, strict=True):
        used.update(start)
        for meld in player.melds:
            used.update(meld.tiles)
    wall = list((game.tile_supply - used).elements())
    for seat, (start, player) in enumerate(zip(starts, game.players, strict=True)):
        filler = [wall.pop() for _ in range(13 + (seat == 0) - len(start) - 3 * len(player.melds))]
        player.hand = counts([*start, *filler])
    game.wall = wall


class ChongjiTests(unittest.TestCase):
    def test_odd_discard_share_rounds_up_and_next_dealer_rotates(self):
        game = fixed_game(streak=1)
        set_hands(game, [[2], [], WAIT, []])
        self.assertEqual(game.physical_counts(), game.tile_supply)
        game.opening = False
        game._discard(0, 2)
        game.step(PASS)
        self.assertIn(RON, game.legal_actions()[1])
        game.step(RON)
        play_passes(game)
        self.assertEqual(game.dealer_stake, 21)
        self.assertEqual(game.win_details[0].payments, (23, 12, 0, 12))
        self.assertEqual(game.scores(), [-23, -12, 47, -12])
        self.assertEqual(game.next_dealer, 1)
        self.assertEqual(CHONGJI.next_dealer(0, (0,), None), 0)
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_first_discard_earth_collects_three_full_shares(self):
        game = fixed_game()
        set_hands(game, [[2], WAIT, [], []])
        game._discard(0, 2)
        game.step(RON)
        play_passes(game)
        win, = game.win_details
        self.assertIn('地胡', win.patterns)
        self.assertEqual(win.payments, (23, 0, 23, 23))
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_robbery_cancels_new_kong_but_keeps_previous_outside_bill(self):
        game = fixed_game(CHONGJI_SEVEN)
        original = Meld('peng', 2, 2)
        earlier = Meld('gang_ming', 29, 2)
        game.players[0].melds = [original]
        game.players[3].melds = [earlier]
        set_hands(game, [[2], WAIT, [], []])
        game._pay_kong(3, earlier)
        self.assertEqual(game.kong_deltas, [-10, -10, -20, 40])
        game.drawn_tile = 2
        game.step(('bugang', 2))
        self.assertIn(RON, game.legal_actions()[1])
        game.step(RON)
        play_passes(game)
        self.assertEqual(game.win_details[0].payments, (39, 0, 0, 0))
        self.assertEqual(game.kong_deltas, [-10, -10, -20, 40])
        self.assertEqual(game.scores(), [-49, 29, -20, 40])
        self.assertEqual(game.players[0].melds, [original])
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_full_payments_and_eleven_wind_declaration(self):
        game = fixed_game()
        game.players[0].hand = counts([0, 27, 32])
        game.ten_winds[0] = 10
        game.drawn_tile = 32
        action = (ELEVEN_WINDS, 27)
        self.assertIn(action, game.legal_actions()[1])
        self.assertNotIn((ELEVEN_WINDS, 0), game.legal_actions()[1])
        self.assertEqual(heuristic(policy_view(game, *game.legal_actions())), action)
        game.step(action)
        self.assertEqual(game.reason_code, 'eleven_winds_win')
        self.assertEqual(game.scores(), [300, -100, -100, -100])
        self.assertEqual(game.discards[0], [27])

        hand = counts([0, 5, 9, 13, 24])
        melds = tuple(Meld('gang_an', tile) for tile in (27, 28, 29))
        score = CHONGJI.faan(hand, melds, WinContext('zimo', 24, 27, 27), wildcard=8)
        self.assertEqual((score.share, score.patterns), (100, ('三杠全缴',)))

    def test_seven_gaps_can_win_a_hand_with_no_standard_shape(self):
        hand = counts([0, 2, 0, 2, 1, 3, 9, 11, 12, 14, 18, 20, 22, 24])
        ctx = WinContext('zimo', 24, 27, 27)
        self.assertIsNone(CHONGJI.faan(hand, (), ctx, wildcard=8))
        self.assertEqual(CHONGJI_SEVEN.faan(hand, (), ctx, wildcard=8).patterns, ('无财神', '七嵌'))
        hand[24] -= 1
        self.assertFalse(seven_gap_pairs(tuple(hand), 0))
        self.assertTrue(seven_gap_pairs(tuple(hand), 1))
        hand[27] += 1
        self.assertFalse(seven_gap_pairs(tuple(hand), 0))

    def test_declined_self_draw_blocks_discard_until_next_draw(self):
        game = fixed_game()
        game.opening = False
        game.players[0].hand = counts([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27, 8, 6])
        game.drawn_tile = 6
        self.assertIn(ZIMO, game.legal_actions()[1])
        game.step(('discard', 6))
        self.assertIn(0, game.skipped_draw_win)
        self.assertFalse(game._can_take_win(0, Offer(1, 2, 'discard')))
        self.assertTrue(game.discarded_for_burst[0])
        game._draw(0)
        self.assertTrue(game._can_take_win(0, Offer(1, 2, 'discard')))
        score, _, _ = game._win(0)
        self.assertIn('抛龙', score.patterns)

    def test_kong_paired_shape_and_call_discard_fallback(self):
        melds = (Meld('gang_an', 33),)
        hand = counts([0, 0, 2, 2, 9, 9, 13, 13, 20, 20, 26])
        score = CHONGJI.faan(hand, melds, WinContext('zimo', 26, 27, 27), wildcard=8)
        self.assertIn('一杠一达', score.patterns)

        game = fixed_game()
        game.players[1].hand = counts([0, 1, 2])
        self.assertEqual(list(game._chows(game.players[1], 0, 0)), [(0, (1, 2), frozenset())])
        game.players[1].hand[3] += 1
        self.assertIn((0, (1, 2), frozenset({0})), list(game._chows(game.players[1], 0, 0)))
