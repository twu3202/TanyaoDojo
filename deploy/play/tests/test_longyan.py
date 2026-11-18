from collections import Counter
from types import SimpleNamespace
import unittest

from deploy.play.honors import PASS, RON, ZIMO, Meld, Offer, WinContext
from deploy.play.honors_bonus import Score
from deploy.play.longyan import LONGYAN, LONGYAN_HALF, TRAVEL, WHITE, LongyanGame
from deploy.play.tests.test_hongkong import counts
from deploy.play.tests.test_jinyun import pass_offer


def table(rules, hands, draws=(), *, gold=5):
    game = LongyanGame(7, 0, 27, rules=rules)
    game.indicator_tile = game.wildcard_tile = gold
    game.flowers = [[] for _ in game.players]
    game.phase = 'action'
    game.cur = game.dealer
    game.follow_deltas[:] = [0] * 4
    game.flower_kong_deltas[:] = [0] * 4
    taken = Counter([gold, *draws, *(tile for hand in hands for tile in hand)])
    wall = list((game.tile_supply - taken).elements())
    supply = [tile for tile in wall if tile < 34]
    for i, specified in enumerate(hands):
        fill = 13 + (i == game.dealer) - len(specified)
        extra = [supply.pop() for _ in range(fill)]
        for tile in extra:
            wall.remove(tile)
        game.players[i].hand = counts([*specified, *extra])
        game.players[i].score = 0
    game.wall = [*wall, *reversed(draws)]
    game.drawn_tile = hands[0][-1]
    return game


class LongyanTests(unittest.TestCase):
    def test_white_supplies_original_face_inside_hand_and_gold_ron_is_rejected(self):
        hand = counts([0, 1, WHITE, 3, 4, 5, 9, 10, 11, 18, 19, 20, 27, 27])
        self.assertTrue(LONGYAN.can_win(hand, (), WinContext('ron', WHITE, 27, 27), gold=2))
        self.assertFalse(LONGYAN_HALF.can_win(hand, (), WinContext('ron', WHITE, 27, 27), gold=2))
        hand[WHITE] -= 1
        hand[2] += 1
        self.assertFalse(LONGYAN.can_win(hand, (), WinContext('ron', 2, 27, 27), gold=2))

    def test_passing_direct_kong_for_pon_only_locks_half_self_draw(self):
        for rules in (LONGYAN, LONGYAN_HALF):
            with self.subTest(rules=rules):
                game = table(rules, [[23], [23, 23, 23], [], []], (18,))
                game.step(('discard', 23))
                self.assertIn(('zhigang', None), game.legal_actions()[1])
                game.step(('peng', None))
                pass_offer(game)
                # The remaining fourth tile may be upgraded after a later ordinary draw.
                game.step(next(a for a in game.legal_actions()[1] if a[1] != 23))
                game.offer = None
                game._draw(1)
                self.assertEqual(('bugang', 23) in game.legal_actions()[1], rules is LONGYAN)

    def test_three_real_draws_advance_travel_and_restrict_the_other_seats(self):
        initial = [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27, 5, 33]
        draws = [3, 4, 6, 5, 7, 8, 12, 5, 13, 14, 15, 16]
        game = table(LONGYAN, [initial, [], [], []], draws)
        for level, discard in ((1, 33), (2, 5), (3, 5)):
            self.assertIn((TRAVEL, discard), game.legal_actions()[1])
            game.step((TRAVEL, discard))
            self.assertEqual(game.travel_level, level)
            while game.offer is not None:
                if level >= 2:
                    self.assertEqual(game.legal_actions()[1], [PASS])
                game.step(PASS)
            for seat in range(1, 4):
                self.assertEqual(game.cur, seat)
                if level == 3:
                    self.assertNotIn(ZIMO, game.legal_actions()[1])
                game.step(('discard', game.drawn_tile))
                pass_offer(game)
        self.assertEqual(game.cur, 0)
        game.step(ZIMO)
        (win,) = game.win_details
        self.assertEqual((win.points, win.patterns, win.payments), (16, ('三游',), (0, 32, 32, 32)))
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_follow_first_two_matching_rounds_then_stops(self):
        game = table(LONGYAN, [[0, 27, 1], [0, 27, 2], [0, 27, 2], [0, 27, 2]])
        for tiles, expected in (((0,)*4, [-6, 2, 2, 2]), ((27,)*4, [-12, 4, 4, 4]),
                                ((1, 2, 2, 2), [-12, 4, 4, 4])):
            for seat, tile in enumerate(tiles):
                self.assertEqual(game.cur, seat)
                game.step(('discard', tile))
                pass_offer(game)
            self.assertEqual(game.follow_deltas, expected)
        self.assertFalse(game.following)
        self.assertEqual(game.follow_rounds, 2)
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_flower_cycles_and_role_weighted_kongs(self):
        for rules, expected in ((LONGYAN, [-4, 8, -2, -2]), (LONGYAN_HALF, [-2, 4, -1, -1])):
            game = table(rules, [[0], [], [], []])
            for flower in range(34, 39):
                game.wall.remove(flower)
                game._reveal_flower(1, flower)
            self.assertEqual(game.flower_kong_deltas, expected)
            game._pay_kong(0, Meld('gang_an', 5))
            self.assertEqual(game.kong_deltas, [12, -4, -4, -4])
            self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_different_payers_for_robbery_and_new_chapter_discard(self):
        game = LongyanGame(1, 0, 27, rules=LONGYAN)
        for actor, supplier, normal, half in ((1, 2, (4, 0, 2, 2), (0, 0, 8, 0)),
                                             (1, 0, (4, 0, 2, 2), (12, 0, 0, 0)),
                                             (0, 3, (0, 4, 4, 4), (0, 0, 0, 12))):
            winner = SimpleNamespace(actor=actor, kind='rob_kong', score=Score(2, ('抢杠胡',)))
            offer = Offer(supplier, 4, 'gang_bu')
            self.assertEqual(LONGYAN.settle_hand(game, [winner], offer)[0][1], normal)
            self.assertEqual(LONGYAN_HALF.settle_hand(game, [winner], offer)[0][1], half)
        winner = SimpleNamespace(actor=1, kind='ron', score=Score(1, ('平胡',)))
        self.assertEqual(LONGYAN.settle_hand(game, [winner], Offer(2, 4, 'discard'))[0][1], (2, 0, 2, 1))
