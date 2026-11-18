import random
import unittest

from deploy.play import jj_three
from deploy.play.honors import Meld, Offer, PASS, RON, ZIMO, WinContext
from deploy.play.honors_game import HonorsSession
from deploy.play.honors_policy import heuristic, policy_view
from deploy.play.kawuxing import SHIYAN, KaWuXingGame
from deploy.play.tests.test_games import wait
from deploy.play.tests.test_hongkong import counts
from deploy.play.tests.test_kawuxing import FIVE, READY_FIVE, REGULAR


JJ_READY = [tile - 9 for tile in READY_FIVE]
JJ_FIVE = FIVE - 9


def jj_game():
    return jj_three.JJThreeGame(3, 0, 27, rules=jj_three)


def offer(game, tile):
    game.discards[0].append(tile)
    game._open_offer(Offer(0, tile, 'discard'))


class ShiyanTests(unittest.TestCase):
    def test_plain_ron_gate_and_kong_events(self):
        game = KaWuXingGame(3, 0, 27, rules=SHIYAN)
        hand = counts(REGULAR + [31])
        self.assertIsNone(SHIYAN.faan(hand, (), WinContext('ron', 31, 27, 27)))
        self.assertEqual(SHIYAN.faan(hand, (), WinContext('zimo', 31, 27, 27)).share, 1)
        self.assertEqual(SHIYAN.faan(hand, (), WinContext('ron', 31, 27, 27, kong_chain=1)).share, 2)
        self.assertEqual(SHIYAN.faan(hand, (), WinContext('rob_kong', 31, 27, 27)).share, 2)
        self.assertIsNone(SHIYAN.faan(hand, (), WinContext('rob_concealed', 31, 27, 27)))
        game.players[0].hand = hand
        game.drawn_tile = 31
        self.assertIn(ZIMO, game.legal_actions()[1])
        self.assertFalse(any(kind == 'declare_ready' for kind, _ in game.legal_actions()[1]))
        game.players[1].hand = counts(REGULAR)
        offer(game, 31)
        self.assertNotIn(RON, game.legal_actions()[1])

    def test_only_winning_five_counts_and_double_win_moves_one_tile(self):
        won = counts(READY_FIVE + [FIVE])
        self.assertEqual(SHIYAN.faan(won, (), WinContext('ron', FIVE, 27, 27)).patterns, ('卡五星',))
        self.assertIsNone(SHIYAN.faan(won, (), WinContext('ron', 12, 27, 27)))
        game = KaWuXingGame(3, 0, 27, rules=SHIYAN)
        game.players[1].hand = counts(READY_FIVE)
        game.players[2].hand = counts(READY_FIVE)
        offer(game, FIVE)
        game.step(RON)
        game.step(RON)
        self.assertEqual(game.scores(), [-4, 2, 2])
        self.assertEqual(game.winners, [1, 2])
        self.assertEqual(game.claimed_tile, FIVE)
        self.assertEqual(game.discards[0], [])
        self.assertEqual(game.next_dealer, 0)

    def test_wall_draw_preserves_kong_points_and_dealer(self):
        game = KaWuXingGame(3, 2, 27, rules=SHIYAN)
        game._pay_kong(1, Meld('gang_an', FIVE))
        game.wall.clear()
        game._draw(0)
        self.assertEqual(game.scores(), [-2, 4, -2])
        self.assertEqual(game.kong_deltas, [-2, 4, -2])
        self.assertEqual((game.reason_code, game.next_dealer), ('draw', 2))


class JJThreeTests(unittest.TestCase):
    def test_actual_chow_and_pon_settle_only_after_response_priority(self):
        for pon, net, chow_deltas, pon_deltas in (
            (False, [-1, 1, 0], [-1, 1, 0], [0, 0, 0]),
            (True, [-2, 0, 2], [0, 0, 0], [-2, 0, 2]),
        ):
            game = jj_game()
            game.players[1].hand = counts([1, 2, 8])
            game.players[2].hand = counts([3, 3, 11])
            offer(game, 3)
            game.step(('chi', 1))
            self.assertEqual(game.scores(), [0, 0, 0])
            game.step(('peng', None) if pon else PASS)
            self.assertEqual(game.scores(), net)
            self.assertEqual(game.side_deltas['chow_deltas'], chow_deltas)
            self.assertEqual(game.side_deltas['pon_deltas'], pon_deltas)

    def test_win_overrides_chow_without_call_points(self):
        game = jj_game()
        game.players[1].hand = counts([3, 5, 8])
        game.players[2].hand = counts(JJ_READY)
        offer(game, JJ_FIVE)
        game.step(('chi', 3))
        game.step(RON)
        self.assertEqual(game.scores(), [-5, 0, 5])
        self.assertEqual(game.side_deltas['chow_deltas'], [0, 0, 0])
        self.assertEqual(game.discards[0], [])

    def test_ready_pays_two_per_opponent_and_keeps_tiles_private(self):
        game = jj_game()
        game.players[0].hand = counts(JJ_READY + [33])
        game.drawn_tile = 33
        game.step(('declare_ready', 33))
        self.assertEqual(game.scores(), [4, -2, -2])
        self.assertEqual(game.side_deltas['closed_ready_deltas'], [4, -2, -2])
        self.assertTrue(game.players[0].closed_ready)
        self.assertIsNone(game.players[0].open_ready_hand)
        session = HonorsSession(variant=jj_three.VARIANT)
        session.game = game
        game.players[1].closed_ready = True
        table = session.view()
        self.assertTrue(table['players'][1]['ready'])
        self.assertEqual(table['players'][1]['hand'], [])
        visible = policy_view(game, 0, [])
        game.players[1].hand.reverse()
        self.assertEqual(policy_view(game, 0, []), visible)
        game.step(PASS)
        game.step(PASS)
        game.wall.append(32)
        game._draw(0)
        self.assertEqual(game.legal_actions()[1], [('discard', 32)])
        game.wall.clear()
        game._end('draw')
        self.assertEqual(game.scores(), [4, -2, -2])

    def test_declaration_discard_may_be_won_and_payment_stands(self):
        game = jj_game()
        game.players[0].hand = counts(JJ_READY + [31])
        game.players[1].hand = counts([tile - 9 if tile < 27 else tile for tile in REGULAR])
        game.drawn_tile = 31
        game.step(('declare_ready', 31))
        game.step(RON)
        game.step(PASS)
        self.assertEqual(game.scores(), [-1, 3, -2])
        self.assertEqual(game.win_details[0].payments, (5, 0, 0))

    def test_kong_amounts_self_draw_and_double_robbery(self):
        game = jj_game()
        game._pay_kong(0, Meld('gang_ming', 1, 2))
        game._pay_kong(1, Meld('gang_bu', 3, 2))
        game._pay_kong(2, Meld('gang_an', 7))
        self.assertEqual(game.kong_deltas, [-1, -3, 4])
        self.assertEqual(game.scores(), [-1, -3, 4])

        game = jj_game()
        game.players[0].hand = counts(JJ_READY + [JJ_FIVE])
        game.drawn_tile = JJ_FIVE
        game.step(ZIMO)
        self.assertEqual(game.scores(), [10, -5, -5])

        game = jj_game()
        game.players[0].hand = counts([JJ_FIVE])
        game.players[0].melds = [Meld('peng', JJ_FIVE, 1)]
        game.drawn_tile = JJ_FIVE
        game.players[1].hand = counts(JJ_READY)
        game.players[2].hand = counts(JJ_READY)
        game.step(('bugang', JJ_FIVE))
        game.step(RON)
        game.step(RON)
        self.assertEqual(game.scores(), [-10, 5, 5])
        self.assertEqual(game.kong_deltas, [0, 0, 0])
        self.assertEqual(game.players[0].melds, [Meld('peng', JJ_FIVE, 1)])
        self.assertEqual(game.claimed_tile, JJ_FIVE)

    def test_full_hand_and_session_continue(self):
        for seed in range(12):
            game = jj_three.JJThreeGame(seed, seed % 3, 27, rules=jj_three)
            rng = random.Random(seed)
            self.assertEqual(set(game.tile_supply), {*range(18), 31, 32, 33})
            while game.phase != 'over':
                actor, actions = game.legal_actions()
                move = heuristic(policy_view(game, actor, actions)) if seed % 2 else rng.choice(actions)
                game.step(move)
                self.assertEqual(game.physical_counts(), game.tile_supply)
                self.assertEqual(sum(game.scores()), 0)
            net = [game.kong_deltas[i] + sum(values[i] for values in game.side_deltas.values()) for i in range(3)]
            for win in game.win_details:
                for payer, points in enumerate(win.payments):
                    net[payer] -= points
                    net[win.actor] += points
            self.assertEqual(net, game.scores())

        for variant in (jj_three.VARIANT, SHIYAN.VARIANT):
            session = HonorsSession(variant=variant, seed=12, pacing=0)
            try:
                session.begin()
                wait(session, {'your_turn'})
                session.auto()
                wait(session, {'round_end'})
                scores = session.snapshot()['result']['scores']
                self.assertEqual(len(scores), 3)
                session.begin()
                wait(session, {'your_turn'})
                self.assertEqual([p['score'] for p in session.snapshot()['table']['players']], scores)
            finally:
                session.close()
