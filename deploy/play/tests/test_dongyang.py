from collections import Counter
import unittest

from deploy.play.dongyang import DONGYANG, DongyangGame, white_assignments
from deploy.play.hangzhou import WHITE
from deploy.play.honors import PASS, RON, Meld, Offer, WinContext
from deploy.play.honors_policy import _live_tiles, heuristic, policy_view
from deploy.play.tests.test_hangzhou import play_passes
from deploy.play.tests.test_hongkong import counts


def dongyang_game(wildcard):
    game = DongyangGame(6, 0, 27, rules=DONGYANG)
    game.wildcard_indicator = game.indicator_tile = game.wildcard_tile = wildcard
    return game


class DongyangTests(unittest.TestCase):
    def test_white_dragon_has_two_concealed_uses(self):
        melds = (Meld('chi', 9), Meld('chi', 19), Meld('peng', 28))
        borrowed = counts([0, 1, WHITE, 27, 27])
        ctx = WinContext('zimo', 27, 27, 27)
        self.assertIsNotNone(DONGYANG.faan(borrowed, melds, ctx, wildcard=2))
        self.assertIsNone(DONGYANG.faan(borrowed, melds, ctx, wildcard=4))
        printed = counts([0, 1, 2, WHITE, WHITE])
        natural_ctx = WinContext('zimo', WHITE, 27, 27)
        self.assertIsNotNone(DONGYANG.faan(printed, melds, natural_ctx, wildcard=4))
        choices = white_assignments(tuple(printed), 4)
        self.assertEqual([(hand[WHITE], hand[4]) for hand in choices], [(2, 0), (1, 1), (0, 2)])

    def test_calls_use_printed_faces_and_wildcard_indicator_is_public(self):
        game = dongyang_game(2)
        game.players[1].hand = counts([1, 2, WHITE, WHITE])
        game._open_offer(Offer(0, 0, 'discard'))
        self.assertEqual(game.legal_actions()[1], [PASS])
        live = _live_tiles(policy_view(game, *game.legal_actions()))
        self.assertEqual(live[2], 2)
        self.assertEqual(live[WHITE], 2)
        game._open_offer(Offer(0, WHITE, 'discard'))
        self.assertIn(('peng', None), game.legal_actions()[1])

    def test_plain_discard_has_no_win_after_first_discard(self):
        hand = counts([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27, WHITE, WHITE])
        ron = WinContext('ron', 2, 28, 27)
        self.assertIsNone(DONGYANG.faan(hand, (), ron, wildcard=8))
        earth = WinContext('ron', 2, 28, 27, opening='地胡')
        score = DONGYANG.faan(hand, (), earth, wildcard=8)
        self.assertEqual((score.share, score.patterns), (10, ('地胡',)))

    def test_loose_and_strict_honor_shapes_and_wind_joker_exception(self):
        strict = [0, 3, 6, 9, 12, 18, 21, *range(27, 34)]
        ctx = WinContext('zimo', 0, 27, 27)
        score = DONGYANG.faan(counts(strict), (), ctx, wildcard=8)
        self.assertEqual((score.share, score.patterns), (4, ('真十三不搭', '七风')))
        strict[strict.index(27)] = WHITE
        score = DONGYANG.faan(counts(strict), (), ctx, wildcard=27)
        self.assertEqual((score.share, score.patterns), (4, ('真十三不搭', '七风')))
        strict[strict.index(WHITE)] = 27
        score = DONGYANG.faan(counts(strict), (), ctx, wildcard=27)
        self.assertEqual((score.share, score.patterns), (2, ('真十三不搭',)))
        non_strict = [0, 4, 7, 9, 12, 15, 18, 21, 24, *range(27, 32)]
        score = DONGYANG.faan(counts(non_strict), (), ctx, wildcard=8)
        self.assertEqual((score.share, score.patterns), (1, ('十三不搭',)))

    def test_discard_half_shares_settle_and_claimed_tile_is_counted_once(self):
        game = dongyang_game(8)
        game.opening = False
        offered = 27
        waiting = [0, 4, 7, 9, 12, 15, 18, 21, 24, 28, 29, 30, WHITE]
        wall = list((game.tile_supply - Counter([offered, *waiting, game.wildcard_indicator])).elements())
        for seat, start in enumerate(([offered], waiting, [], [])):
            filler = [wall.pop() for _ in range(13 + (seat == 0) - len(start))]
            game.players[seat].hand = counts([*start, *filler])
        game.wall = wall
        self.assertEqual(game.physical_counts(), game.tile_supply)
        game._discard(0, offered)
        self.assertIn(RON, game.legal_actions()[1])
        game.step(RON)
        play_passes(game)
        self.assertEqual(game.scores(), [-1, 2, -0.5, -0.5])
        self.assertEqual(game.win_details[0].payments, (1, 0, 0.5, 0.5))
        self.assertEqual(game.next_dealer, 1)
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_white_borrowing_can_enable_float_and_direct_kong_burst(self):
        game = dongyang_game(2)
        game.players[0].hand = counts([0, 1, WHITE, 9, 10, 11, 18, 19, 20, 27, 27, 27, 2, 2])
        game.drawn_tile = 2
        move = ('float_wildcard', 2)
        self.assertIn(move, game.legal_actions()[1])
        self.assertEqual(heuristic(policy_view(game, *game.legal_actions())), move)
        game.step(move)
        self.assertEqual(game.float_cycle_owner, 0)

        hand = counts([0, 1, 2, 9, 10, 11, 18, 19, 20, 4, 24])
        melds = (Meld('gang_ming', 29, 3),)
        ctx = WinContext('zimo', 24, 27, 27, kong_chain=1, kong_provider=3)
        self.assertEqual(DONGYANG.faan(hand, melds, ctx, wildcard=4).share, 4)

    def test_three_jokers_and_dealer_rotation(self):
        hand = counts([2, 2, 2, 0, 4, 7, 8, 11, 13, 16, 20, 24, 28, 32])
        score = DONGYANG.faan(hand, (), WinContext('zimo', 2, 27, 27), wildcard=2)
        self.assertEqual((score.share, score.patterns), (1, ('三财神',)))
        self.assertEqual(DONGYANG.next_dealer(3, (3,), None), 0)
        self.assertEqual(DONGYANG.next_dealer(3, (), None), 0)
        self.assertEqual(DONGYANG.next_dealer(3, (1,), None), 1)
