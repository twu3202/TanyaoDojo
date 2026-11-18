from collections import Counter
from types import SimpleNamespace
import unittest

from deploy.play.core import TILES
from deploy.play.daxi import DAXI, DaxiGame
from deploy.play.honors import PASS, RON, Meld, Offer, WinContext
from deploy.play.honors_bonus import Score
from deploy.play.honors_game import HonorsSession
from deploy.play.honors_policy import heuristic, policy_view
from deploy.play.linhai import LINHAI, WHITE, LinhaiGame
from deploy.play.tests.test_hongkong import counts


class DaxiTests(unittest.TestCase):
    def test_buy_in_hides_hands_until_all_choices_are_complete(self):
        game = DaxiGame(7, 2, 27, rules=DAXI)
        session = HonorsSession(variant=DAXI.VARIANT, seed=7)
        session.game = game
        start = session.view()
        model_input = policy_view(game, *game.legal_actions())
        self.assertEqual(sum(model_input.hand), 0)
        self.assertEqual(heuristic(model_input), ('buy_base', 1))
        self.assertEqual(start['dealer'], 2)
        self.assertTrue(all(p['hand_count'] == 0 and 'bought_base' not in p for p in start['players']))
        for seat, value in ((3, 1), (0, 0), (1, 1)):
            self.assertEqual(game.legal_actions()[0], seat)
            game.step(('buy_base', value))
        self.assertEqual(game.cur, game.dealer)
        self.assertEqual(game.drawn_tile, game.opening_draw)
        self.assertEqual([p['bought_base'] for p in session.view()['players']], [False, True, True, True])
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_same_discard_pays_two_winners_and_each_buy_in(self):
        game = DaxiGame(7, 0, 27, rules=DAXI)
        for choice in (0, 1, 0):
            game.step(('buy_base', choice))
        wait = [0, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27, 32, 32]
        hands = [[1], wait.copy(), wait.copy(), []]
        for old, new in ((27, 28), (32, 33)):
            hands[2] = [new if tile == old else tile for tile in hands[2]]
        wall = list((game.tile_supply - Counter(sum(hands, []))).elements())
        for seat, hand in enumerate(hands):
            while len(hand) < 13 + (seat == 0):
                hand.append(wall.pop())
            game.players[seat].hand = counts(hand)
        game.wall = wall
        game.step(('discard', 1))
        for seat in (1, 2):
            actor, legal = game.legal_actions()
            self.assertEqual(actor, seat)
            self.assertIn(RON, legal)
            game.step(RON)
        game.step(PASS)
        self.assertEqual(game.winners, [1, 2])
        self.assertEqual(game.buy_base_deltas, [-3, 1, 2, 0])
        self.assertEqual(game.scores(), [-5, 2, 3, 0])
        self.assertEqual(game.next_dealer, 1)
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_kong_bills_survive_draw(self):
        game = DaxiGame(7, 0, 27, rules=DAXI)
        game._pay_kong(1, Meld('gang_ming', 2, 3))
        game._pay_kong(1, Meld('gang_bu', 3, 2))
        game._pay_kong(1, Meld('gang_an', 4))
        game._end('draw')
        self.assertEqual(game.kong_deltas, [-1, 5, -2, -2])
        self.assertEqual(game.scores(), game.kong_deltas)
        self.assertEqual(game.next_dealer, 1)


class LinhaiTests(unittest.TestCase):
    def test_wildcard_four_winds_and_cap(self):
        hand = counts([27, 27, 27, 28, 28, 28, 29, 29, WHITE, 30, WHITE, 0, 1, 2])
        score = LINHAI.faan(hand, (), WinContext('zimo', WHITE, 27, 27))
        self.assertEqual(score.share, 8)
        self.assertIn('四风聚齐', score.patterns)
        self.assertIsNone(LINHAI.faan(hand, (), WinContext('ron', WHITE, 27, 27)))

    def test_matching_chong_tiles_count_per_winner_and_supplier(self):
        game = LinhaiGame(7, 0, 27, rules=LINHAI)
        game.wall[-6:] = [TILES.index(tile) for tile in ('1m', '2m', '3m', '5m', 'C', 'P')]
        winners = [SimpleNamespace(actor=actor, score=Score(8, ('台数x3',))) for actor in (0, 1)]
        results = LINHAI.settle_hand(game, winners, Offer(3, 6, 'discard'))
        self.assertEqual([TILES[tile] for tile in game.horse_tiles], ['P', 'C', '5m', '3m', '2m', '1m'])
        self.assertEqual(game.horse_details, [(0, 6, 2, 2, 10), (1, 6, 2, 2, 10)])
        self.assertEqual([payments for _, payments in results], [(0, 0, 0, 10), (0, 0, 0, 10)])

    def test_reciprocal_third_feed_survives_upgrade_and_pays_self_draw(self):
        game = LinhaiGame(7, 0, 27, rules=LINHAI)
        last = Meld('peng', 3, 2)
        melds = (Meld('chi', 0, 2), Meld('gang_bu', 9, 2), last)
        self.assertEqual(LINHAI.liability(melds, last), ('three_feeds',))
        self.assertEqual(LINHAI.liability((*melds[:2], Meld('gang_ming', 3, 2)), Meld('gang_ming', 3, 2)), ())
        game.related_seats.append((1, 2))
        game.wall[-6:] = [TILES.index(tile) for tile in ('3m', '7m', '3p', '7p', 'W', 'F')]
        winner = SimpleNamespace(actor=2, score=Score(2, ('台数x1',)))
        (score, payments), = LINHAI.settle_hand(game, [winner], None)
        self.assertEqual(score.share, 8)
        self.assertEqual(payments, (0, 24, 0, 0))

    def test_last_live_draw_only_replenishes_inside_live_wall(self):
        game = LinhaiGame(7, 0, 27, rules=LINHAI)
        game.wall = game.wall[:15]
        tail = game.wall[:-1].copy()
        next_tile = game.wall[-1]
        self.assertEqual(game._take_tile(0, True), next_tile)
        self.assertEqual(game.wall, tail)
        self.assertEqual(game.remaining, 0)
        self.assertIsNone(game._take_tile(0, True))
