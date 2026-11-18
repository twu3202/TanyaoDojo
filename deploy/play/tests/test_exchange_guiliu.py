import random
import unittest

from deploy.play import golden_guangdong as golden, guiliu
from deploy.play.honors import COPIES, PASS, RON, ZIMO, HonorsGame, Offer, WinContext, WinningHand
from deploy.play.honors_policy import heuristic, policy_view
from deploy.play.tests.test_hongkong import SEQUENCES, counts, fixture


class GoldenIsland(unittest.TestCase):
    def test_exchange_keeps_selections_private_and_swaps_all_hands_together(self):
        game = HonorsGame(51, 2, 27, rules=golden)
        dealt = [player.hand.copy() for player in game.players]
        selected = []
        for seat in range(4):
            actor, actions = game.legal_actions()
            self.assertEqual(actor, seat)
            self.assertEqual(game.phase, 'exchange')
            self.assertEqual([p.hand for p in game.players], dealt)
            view = policy_view(game, actor, actions)
            self.assertEqual(view.hand, tuple(dealt[seat]))
            self.assertEqual(sum(view.visible), 0)
            self.assertEqual({move[0] for move in actions}, {'exchange'})
            move = heuristic(view)
            selected.append(move[1])
            game.step(move)
            self.assertEqual(game.physical_counts(), COPIES)
        for seat, hand in enumerate(dealt):
            expected = hand.copy()
            for tile in selected[seat]:
                expected[tile] -= 1
            for tile in selected[seat - 1]:
                expected[tile] += 1
            self.assertEqual(game.players[seat].hand, expected)
        self.assertEqual(game.legal_actions()[0], 2)
        self.assertEqual(game.phase, 'action')

    def test_official_bilateral_example_pays_eight_eight_six(self):
        hand = [0, 1, 2, 3, 4, 5, 9, 10, 11, 22, 22, 22, 16, 16]
        game = fixture({0: hand}, rules=golden, drawn=16)
        game.phase = 'action'
        horses = [0, 4, 1, 2]
        for tile in horses:
            game.wall.remove(tile)
        game.wall[:0] = horses
        game.step(ZIMO)
        self.assertEqual(game.scores(), [22, -8, -8, -6])
        self.assertEqual(game.bilateral_horses, [{'actor': 0, 'hits': [2, 1, 1, 0]}])
        self.assertEqual(game.horse_tiles, horses)
        self.assertEqual(game.physical_counts(), COPIES)

    def test_points_stack_and_late_wall_closes_calls(self):
        regular = counts(SEQUENCES)
        self.assertIsNone(golden.faan(regular, (), WinContext('ron', 22, 27, 27)))
        self.assertEqual(golden.faan(regular, (), WinContext('rob_kong', 22, 27, 27)).share, 4)
        flush = counts([0, 1, 2, 3, 4, 5, 6] * 2)
        self.assertEqual(golden.faan(flush, (), WinContext('ron', 6, 27, 27)).share, 20)
        game = fixture({0: [9] + list(range(13, 26)), 1: [9,9, *range(11)]}, rules=golden)
        game.phase = 'action'
        game.discards[3].extend(game.wall[7:])
        del game.wall[7:]
        self.assertEqual(game.remaining, 3)
        game.step(('discard', 9))
        self.assertNotIn(('peng', None), game.legal_actions()[1])
        self.assertEqual(game.physical_counts(), COPIES)


class GuiliuWake(unittest.TestCase):
    def test_sixty_point_shape_and_ragged_shape(self):
        hand = counts([0] * 4 + [1,2,3,4,5] * 2)
        score = guiliu.faan(hand, (), WinContext('zimo', 5, 27, 27))
        self.assertEqual(score.share // score.factor, 60)
        self.assertEqual(score.patterns, ('清一色', '龙七对', '门清'))
        hand = counts([*range(27,34), 0, 3, 6, 9, 12, 18, 21])
        self.assertIn('七星烂', guiliu.faan(hand, (), WinContext('ron', 21, 27, 27)).patterns)
        hand[21], hand[19] = 0, 1
        self.assertIsNone(guiliu.faan(hand, (), WinContext('ron', 19, 27, 27)))
        self.assertEqual(guiliu.adjacent(27), (30, 28))
        self.assertEqual(guiliu.adjacent(31), (32, 33))
        self.assertEqual(guiliu.adjacent(0), (1,))

    def test_multiron_shares_prefix_with_only_one_winner_revealing_again(self):
        dealer = [22, 5, 6, 7, 8, 20, 21, 25, 26, 27, 29, 30, 31, 32]
        first = [0,0,0,9,10,11,12,13,14,18,19,20,22]
        game = fixture({0: dealer, 1: first, 2: SEQUENCES[:-1]}, rules=guiliu)
        for tile in (7, 1):
            game.wall.remove(tile)
            game.wall.append(tile)
        for action in (('discard', 22), RON, RON, PASS):
            game.step(action)
        self.assertEqual(game.wake_tiles, [1,7])
        self.assertEqual(game.wake_details, [{'actor':1,'count':2,'hits':3}, {'actor':2,'count':1,'hits':2}])
        self.assertEqual(game.scores(), [-50, 28, 22, 0])
        self.assertEqual(game.physical_counts(), COPIES)

    def test_last_tile_as_indicator_counts_once_and_keeps_physical_copies(self):
        game = fixture({0: SEQUENCES}, rules=guiliu, drawn=22)
        game.discards[3].extend(game.wall)
        game.wall.clear()
        game.step(ZIMO)
        self.assertEqual(game.wake_tiles, [22])
        self.assertEqual(game.wake_draws, [])
        self.assertEqual(game.wake_details[0]['count'], 1)
        self.assertEqual(game.physical_counts(), COPIES)

    def test_linked_winners_only_pay_each_other_if_scores_differ(self):
        game = HonorsGame(8, 0, 27, rules=guiliu)
        game.wall.clear()
        game.related_seats = [(1, 2)]
        offer = Offer(0, 22, 'discard')
        quiet = [0] * 34
        scores = [guiliu.Score(amount, amount, 1, ('平胡',)) for amount in (5, 35)]
        first = WinningHand(1, scores[0], quiet, 'ron')
        second = WinningHand(2, scores[0], quiet, 'ron')
        paired = guiliu.settle_hand(game, (first, second), offer)
        self.assertEqual([bill for _, bill in paired], [(5,0,0,0), (5,0,0,0)])
        second = WinningHand(2, scores[1], quiet, 'ron')
        paired = guiliu.settle_hand(game, (first, second), offer)
        self.assertEqual([bill for _, bill in paired], [(5,0,5,0), (35,35,0,0)])

    def test_random_full_hands_preserve_all_tiles_and_payments(self):
        rng = random.Random(911)
        for rules in (golden, guiliu):
            for seed in range(135):
                game = HonorsGame(seed, seed % 4, 27, rules=rules)
                while game.phase != 'over':
                    game.step(rng.choice(game.legal_actions()[1]))
                    self.assertEqual(game.physical_counts(), COPIES)
                    self.assertEqual(sum(game.scores()), 0)
                net = [kong + echo for kong, echo in zip(game.kong_deltas, game.echo_deltas)]
                for win in game.win_details:
                    for payer, amount in enumerate(win.payments):
                        net[payer] -= amount
                        net[win.actor] += amount
                self.assertEqual(game.scores(), net)
