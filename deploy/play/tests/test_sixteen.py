from collections import Counter
import unittest

from deploy.play.fuzhou import FUZHOU, HONOR_FLOWERS, FuzhouGame
from deploy.play.honors import PASS, RON, ZIMO, Meld, WinContext, standard_shapes
from deploy.play.honors_game import HonorsSession
from deploy.play.honors_policy import heuristic, policy_view, standard_distance
from deploy.play.taiwan import TAIWAN, TaiwanGame
from deploy.play.tests.test_hongkong import counts
from deploy.play.tests.test_jinyun import pass_offer


class SixteenTests(unittest.TestCase):
    def test_five_concealed_sets_in_one_suit_or_all_honors(self):
        suited = counts([0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 8, 8])
        honors = counts([t for t in range(27, 32) for _ in range(3)] + [32, 32])
        for hand, wait in ((suited, 8), (honors, 32)):
            self.assertTrue(any(standard_shapes(hand, 0, groups=5)))
            self.assertEqual(list(standard_shapes(hand, 0)), [])
            self.assertEqual(standard_distance(tuple(hand), 0, 5), -1)
            hand[wait] -= 1
            self.assertEqual(standard_distance(tuple(hand), 0, 5), 0)

    def test_fuzhou_deal_and_replacement_flower_order(self):
        game = FuzhouGame(0, 0, 27, rules=FUZHOU)
        self.assertEqual([sum(p.hand) for p in game.players], [17, 16, 16, 16])
        self.assertTrue(all(not any(p.hand[t] for t in HONOR_FLOWERS) for p in game.players))
        self.assertEqual(game.flowers, [[31, 29, 29], [32, 31, 35, 34, 28, 33, 30, 27, 41],
                                        [30, 32, 28], [32, 27, 38, 32, 37, 31, 40]])
        self.assertEqual(game.wildcard_tile, 12)
        self.assertEqual(game.remaining, 38)
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_gold_claim_non_dealer_starts_with_sixteen_and_does_not_see_dealer_draw(self):
        game = FuzhouGame(1, 1, 27, rules=FUZHOU)
        while game.cur != 0:
            game.step(PASS)
        self.assertEqual(game.legal_actions(), (0, [PASS, ('claim_gold', None)]))
        session = HonorsSession(variant=FUZHOU.VARIANT)
        session.game = game
        view = session.view()
        self.assertIsNone(view['drawn'])
        self.assertTrue(view['gold_indicator'])
        self.assertTrue(all(not p['hand'] for p in view['players'][1:]))
        score = 30 + 2 * (len(game.flowers[0]) + game.players[0].hand[game.wildcard_tile] + 1)
        game.step(('claim_gold', None))
        self.assertEqual(game.reason_code, 'gold_claim')
        self.assertEqual(sum(game.revealed[0]), 17)
        self.assertEqual(game.scores(), [3 * score, -score, -score, -score])
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_dealer_claim_swaps_a_tile_and_first_three_gold_has_priority(self):
        game = FuzhouGame(8, 0, 27, rules=FUZHOU)
        actor, actions = game.legal_actions()
        self.assertEqual((actor, actions), (0, [PASS, ('claim_gold', 3)]))
        self.assertEqual(heuristic(policy_view(game, actor, actions)), ('claim_gold', 3))
        game.step(('claim_gold', 3))
        self.assertEqual(game.discards[0], [3])
        self.assertEqual(sum(game.revealed[0]), 17)
        self.assertEqual(game.physical_counts(), game.tile_supply)

        game = FuzhouGame(53, 1, 27, rules=FUZHOU)
        self.assertEqual(game.legal_actions(), (0, [PASS, ('three_gold', None)]))
        game.step(('three_gold', None))
        self.assertEqual(game.reason_code, 'three_gold_win')
        self.assertEqual(sum(game.revealed[0]), 16)
        self.assertIn('三金倒', game.win_details[0].patterns)
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_idle_gold_requires_self_draw_and_score_uses_max_award(self):
        complete = counts([0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 18, 19, 20, 8, 22])
        self.assertFalse(FUZHOU.can_win(complete, (), WinContext('ron', 22, 28, 27), gold=8))
        self.assertTrue(FUZHOU.can_win(complete, (), WinContext('zimo', 22, 28, 27), gold=8))
        partial = counts([0, 1, 2, 3, 4, 5, 9, 10, 11, 18, 19, 20, 8, 8])
        score = FUZHOU.score(partial, (Meld('gang_an', 12),), gold=8, flowers=4,
                             dealer_streak=2, kind='zimo')
        self.assertEqual(score.share, 80)
        self.assertEqual(score.patterns, ('金雀', '花牌x4', '金牌x2'))


class TaiwanTests(unittest.TestCase):
    def test_both_flower_wins_pay_with_per_seat_dealer_tai(self):
        for provider, payments in ((None, (16, 0, 11, 11)), (2, (0, 0, 11, 0))):
            with self.subTest(provider=provider):
                game = TaiwanGame(14, 0, 27, rules=TAIWAN, dealer_streak=2)
                self.assertEqual([sum(p.hand) for p in game.players], [17, 16, 16, 16])
                game.wall = [t for t in game.wall if t < 34]
                game.flowers = [[], list(range(34, 42)), [], []]
                if provider is not None:
                    game.flowers[provider].append(game.flowers[1].pop())
                game._offer_flower_win()
                self.assertEqual(game.legal_actions(), (1, [PASS, ('flower_hu', None)]))
                game.step(('flower_hu', None))
                self.assertEqual(game.win_details[0].payments, payments)
                self.assertEqual(game.win_details[0].provider, provider)
                self.assertEqual(game.next_dealer, 1)
                self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_passed_win_clears_only_after_nonwinning_draw_and_discard(self):
        game = TaiwanGame(2, 0, 27, rules=TAIWAN)
        ready = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 18, 19, 20, 28]
        hands = [[28], ready, [], []]
        reserved = (28, 29)
        use = Counter([t for tiles in hands + game.flowers for t in tiles] + list(reserved))
        wall = list((game.tile_supply - use).elements())
        playable = [t for t in wall if t < 34]
        for actor, tiles in enumerate(hands):
            fill = 16 + (actor == game.dealer) - len(tiles)
            for _ in range(fill):
                tile = playable.pop()
                tiles.append(tile)
                wall.remove(tile)
            game.players[actor].hand = counts(tiles)
        game.wall = wall + [29, 28]
        game._discard(0, 28)
        self.assertIn(RON, game.legal_actions()[1])
        game.step(PASS)
        pass_offer(game)
        self.assertEqual(game.cur, 1)
        self.assertEqual(game.drawn_tile, 28)
        self.assertNotIn(ZIMO, game.legal_actions()[1])
        game.step(('discard', 28))
        self.assertIn(1, game.passed_win)
        # Directly advance to the next ordinary draw for this seat.
        game.offer = None
        game._draw(1)
        self.assertEqual(game.drawn_tile, 29)
        self.assertIn(1, game.passed_win)
        game.step(('discard', 29))
        self.assertNotIn(1, game.passed_win)
        self.assertEqual(game.physical_counts(), game.tile_supply)
