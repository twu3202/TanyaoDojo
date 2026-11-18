from types import SimpleNamespace
import unittest

from deploy.play.bengbu import BENGBU, BENGBU_FLOWER, FLOWER_FACES, BengbuFlowerGame, BengbuGame
from deploy.play.honors import PASS, RON, ZIMO, Meld, Offer, WinContext
from deploy.play.honors_policy import policy_view
from deploy.play.lipu import LIPU, LipuGame
from deploy.play.tests.test_hongkong import counts
from deploy.play.tests.test_jinyun import pass_offer, set_hands


WAIT_A = [3, 4, 9, 10, 11, 18, 19, 20, 28, 28, 28, 32, 32]
WAIT_B = [6, 7, 12, 13, 14, 21, 22, 23, 29, 29, 29, 33, 33]


class BengbuTests(unittest.TestCase):
    def test_opening_kong_defers_flowers_until_first_ordinary_draw(self):
        game = BengbuFlowerGame(1660, 0, 27, rules=BENGBU_FLOWER)
        self.assertEqual(game.flowers[0], [27, 31, 31, 32])
        self.assertEqual(game.flower_started, {0})
        self.assertEqual(game.players[1].hand[33], 1)
        game.step(('discard', 1))
        game.step(('zhigang', None))
        pass_offer(game)
        actor, actions = game.legal_actions()
        self.assertEqual(actor, 1)
        self.assertEqual(game.drawn_tile, 32)
        self.assertEqual(game.flowers[1], [])
        self.assertEqual(game.players[1].hand[32:34], [1, 1])
        self.assertTrue(all(tile not in FLOWER_FACES for _, tile in actions))
        self.assertEqual(policy_view(game, actor, actions).hand[32:34], (0, 0))
        for seat in (1, 2, 3, 0):
            actor, actions = game.legal_actions()
            self.assertEqual(actor, seat)
            game.step(next(action for action in actions if action[0] == 'discard'))
            pass_offer(game)
        self.assertEqual(game.cur, 1)
        self.assertIn(1, game.flower_started)
        self.assertTrue({32, 33} <= set(game.flowers[1]))
        self.assertFalse(any(game.players[1].hand[face] for face in FLOWER_FACES))
        self.assertEqual(sum(game.players[1].hand), 11)
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_classic_robbery_charges_three_shares_and_accepts_only_nearest(self):
        game = BengbuGame(7, 0, 27, rules=BENGBU)
        pong = Meld('peng', 5, 3)
        game.players[0].melds = [pong]
        set_hands(game, [[5], WAIT_A, WAIT_B, []], taken=pong.tiles)
        reserve = game.wall[:14]
        game.drawn_tile = 5
        game.step(('bugang', 5))
        self.assertIn(RON, game.legal_actions()[1])
        game.step(RON)
        self.assertEqual(game.legal_actions(), (2, [PASS]))
        pass_offer(game)
        self.assertEqual(game.winners, [1])
        self.assertEqual(game.win_details[0].payments, (7, 0, 0, 0))
        self.assertEqual(game.wall[:14], reserve)
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_flower_scores_one_robbery_share_and_winner_only_kongs(self):
        hand = counts([3, 4, 5, 9, 10, 11, 18, 19, 20, 28, 28])
        melds = (Meld('gang_bu', 8, 2),)
        rob = WinContext('rob_kong', 5, 28, 27)
        score = BENGBU_FLOWER.faan(hand, melds, rob, flowers=3)
        self.assertEqual(score.share, 5)
        self.assertIn('花牌x3', score.patterns)
        winner = SimpleNamespace(actor=1, score=score)
        game = BengbuFlowerGame(7, 0, 27, rules=BENGBU_FLOWER)
        offer = Offer(2, 5, 'gang_bu')
        self.assertEqual(BENGBU_FLOWER.settle_hand(game, [winner], offer), [(score, (0, 0, 5, 0))])
        draw = WinContext('zimo', 5, 28, 27, kong_chain=1)
        self.assertEqual(BENGBU_FLOWER.faan(hand, melds, draw, flowers=3).share, 7)
        self.assertEqual(BENGBU.faan(hand, melds, draw).share, 6)
        game._pay_kong(1, melds[0])
        self.assertEqual(game.scores(), [0, 0, 0, 0])

    def test_classic_kong_replacement_and_draw_preserve_reserve(self):
        game = BengbuGame(7, 0, 27, rules=BENGBU)
        set_hands(game, [[0, 0, 0, 0], [], [], []], replacement=8)
        reserve = game.wall[:14]
        game.step(('ankan', 0))
        pass_offer(game)
        self.assertEqual(game.drawn_tile, 8)
        while game.phase != 'over':
            _, actions = game.legal_actions()
            game.step(next(action for action in actions if action[0] in ('pass', 'discard')))
        self.assertEqual(game.wall, reserve)
        self.assertEqual(game.reason_code, 'draw')
        self.assertEqual(game.physical_counts(), game.tile_supply)


class LipuTests(unittest.TestCase):
    def test_operator_example_and_multiple_discard_wins(self):
        melds = tuple(Meld('peng', tile, (tile + 1) % 4) for tile in range(4))
        score = LIPU.faan(counts([8, 8]), melds, WinContext('ron', 8, 28, 27))
        self.assertEqual(score.share, 12)
        self.assertEqual(set(score.patterns), {'大吊车', '清一色'})
        game = LipuGame(7, 0, 27, rules=LIPU)
        set_hands(game, [[5], WAIT_A, WAIT_B, []])
        game.step(('discard', 5))
        game.step(RON)
        game.step(RON)
        pass_offer(game)
        self.assertEqual(game.winners, [1, 2])
        self.assertEqual(game.next_dealer, 0)
        self.assertEqual(game.scores(), [-2, 1, 1, 0])
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_third_pon_seals_supplier_but_keeps_earlier_kong_points(self):
        game = LipuGame(7, 0, 27, rules=LIPU)
        set_hands(game, [[0, 9, 18], [0, 0, 9, 9, 18, 18, 3, 4, 5, 6, 7, 8, 22], [], []])
        game._pay_kong(0, Meld('gang_an', 10))
        for tile in (0, 9, 18):
            game._discard(0, tile)
            game.step(('peng', None))
            pass_offer(game)
        self.assertEqual(game.sealed, {0})
        self.assertEqual(game.related_seats, [(1, 0)])
        self.assertEqual(game._liabilities_for(0), [('three_feeds', 1)])
        self.assertEqual(game._liabilities_for(1), [('three_feeds', 0)])
        game._pay_kong(0, Meld('gang_an', 11))
        self.assertEqual(game.kong_deltas, [3, -1, -1, -1])
        hand = counts([*WAIT_A, 5])
        ctx = WinContext('ron', 5, 27, 27)
        self.assertIsNotNone(LIPU.faan(hand, (), ctx))
        self.assertIsNone(game._score(0, hand, (), ctx))

    def test_four_feeds_and_three_meld_liability_payment(self):
        game = LipuGame(7, 0, 27, rules=LIPU)
        set_hands(game, [[0, 3, 6, 9], [1, 2, 4, 5, 7, 8, 10, 11], [], []])
        for start in (0, 3, 6, 9):
            game._discard(0, start)
            game.step(('chi', start))
            pass_offer(game)
            if start == 6:
                self.assertEqual(game.sealed, set())
                self.assertEqual(game.related_seats, [(1, 0)])
        self.assertEqual(game.sealed, {0})
        score = LIPU.faan(counts([*WAIT_A, 5]), (), WinContext('ron', 5, 28, 27))
        winner = SimpleNamespace(actor=1, score=score)
        for offer, payments in ((Offer(2, 5, 'discard'), (1, 0, 1, 0)),
                                (Offer(0, 5, 'discard'), (1, 0, 0, 0)),
                                (None, (3, 0, 0, 0))):
            self.assertEqual(LIPU.settle_hand(game, [winner], offer), [(score, payments)])

    def test_only_last_kong_is_reversed_and_direct_kong_draw_has_one_payer(self):
        for chain in (False, True):
            with self.subTest(chain=chain):
                game = LipuGame(7, 0, 27, rules=LIPU)
                if chain:
                    caller = [0, 0, 0, 9, 9, 9, 9, 3, 4, 5, 19, 20, 20]
                    draw = 20
                else:
                    caller = [0, 0, 0, 3, 4, 5, 9, 10, 11, 18, 19, 20, 28]
                    draw = 28
                set_hands(game, [[0], caller, [], []], replacement=draw)
                if chain:
                    # Keep the two sequential tail draws fixed without changing the tile inventory.
                    index = next(i for i, tile in enumerate(game.wall) if tile == 18)
                    game.wall[1], game.wall[index] = game.wall[index], game.wall[1]
                game.step(('discard', 0))
                game.step(('zhigang', None))
                pass_offer(game)
                if chain:
                    game.step(('ankan', 9))
                    pass_offer(game)
                self.assertIn(ZIMO, game.legal_actions()[1])
                game.step(ZIMO)
                expected = (3, 0, 3, 3) if chain else (7, 0, 0, 0)
                self.assertEqual(game.win_details[0].payments, expected)
                self.assertEqual(game.kong_deltas, [-1, 1, 0, 0] if chain else [0] * 4)
                self.assertEqual(game.kong_refunds, [1, -3, 1, 1] if chain else [1, -1, 0, 0])
                self.assertEqual(game.scores(), [-4, 10, -3, -3] if chain else [-7, 7, 0, 0])
                self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_two_post_kong_winners_reverse_the_same_concealed_kong_once(self):
        game = LipuGame(7, 0, 27, rules=LIPU)
        set_hands(game, [[0, 0, 0, 0, 5], WAIT_A, WAIT_B, []], replacement=24)
        game.step(('ankan', 0))
        pass_offer(game)
        self.assertEqual(game.kong_deltas, [3, -1, -1, -1])
        game.step(('discard', 5))
        game.step(RON)
        game.step(RON)
        pass_offer(game)
        self.assertEqual([win.points for win in game.win_details], [9, 9])
        self.assertEqual(game.kong_refunds, [-3, 1, 1, 1])
        self.assertEqual(game.kong_deltas, [0] * 4)
        self.assertEqual(game.scores(), [-18, 9, 9, 0])
        self.assertEqual(game.physical_counts(), game.tile_supply)
