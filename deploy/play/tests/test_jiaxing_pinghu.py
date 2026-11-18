from collections import deque
from types import SimpleNamespace
import unittest

from deploy.play.honors import PASS, Meld, Offer, WinContext
from deploy.play.honors_bonus import Score
from deploy.play.honors_game import HonorsSession
from deploy.play.honors_policy import _live_tiles, policy_view
from deploy.play.jiaxing import JIAXING, JiaxingGame
from deploy.play.pinghu import PINGHU, PinghuGame
from deploy.play.tests.test_hongkong import counts


class JiaxingTests(unittest.TestCase):
    def test_printed_wildcard_is_reserved_and_not_consumed_in_a_chow(self):
        game = JiaxingGame(7, 2, 27, rules=JIAXING)
        self.assertEqual(game.physical_counts(), game.tile_supply)
        self.assertEqual(game.indicator_tile, game.wildcard_tile)
        self.assertEqual(game.physical_tile(31), 31)
        actor, actions = game.legal_actions()
        view = policy_view(game, actor, actions)
        self.assertEqual(_live_tiles(view)[game.wildcard_tile], 3 - view.hand[game.wildcard_tile])
        game.wildcard_tile = 1
        player = game.players[(game.cur + 1) % 4]
        player.hand = counts([0, 1, 3, 4])
        self.assertEqual([start for start, _, _ in game._chows(player, 2, game.cur)], [2])

    def test_tai_gate_and_stacked_yellow_faan(self):
        ordinary = counts([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27, 32, 32])
        ctx = WinContext('zimo', 32, 27, 27)
        self.assertIsNone(JIAXING.faan(ordinary, (), ctx, wildcard=30))
        mixed = counts([0, 1, 2, 3, 4, 5, 6, 7, 8, 27, 27, 27, 32, 32])
        score = JIAXING.faan(mixed, (), ctx, wildcard=30, yellow='黄番')
        self.assertEqual(score.share, 10)
        self.assertEqual(score.patterns[-2:], ('台数x5', '番数x1'))
        self.assertIsNone(JIAXING.faan(mixed, (), WinContext('ron', 32, 27, 27), wildcard=30))

    def test_any_tile_wildcard_wait_cannot_rob_added_kong(self):
        melds = tuple(Meld('peng', tile, 3) for tile in range(27, 31))
        hand = counts([5, 9])
        self.assertIsNotNone(JIAXING.faan(hand, melds, WinContext('zimo', 9, 27, 27), wildcard=5))
        self.assertIsNone(JIAXING.faan(hand, melds, WinContext('rob_kong', 9, 27, 27), wildcard=5))

    def test_third_call_prompts_other_seats_and_folding_forces_draw_discard(self):
        game = JiaxingGame(7, 2, 27, rules=JIAXING)
        game.wildcard_tile = 31
        game.players[1].melds = [Meld('peng', 2, 2), Meld('gang_bu', 6, 2)]
        game.players[1].hand = counts([9, 9, 10, 14, 23, 25, 30])
        game.players[2].hand = counts([9, 10, 11, 16, 17, 19, 24, 25, 26, 27, 28, 29, 30, 31])
        game.step(('discard', 9))
        while game.offer:
            actor, actions = game.legal_actions()
            game.step(('peng', None) if actor == 1 else PASS)
        self.assertEqual(game.related_seats, [(1, 2)])
        self.assertEqual(game.legal_actions()[0], 0)
        session = HonorsSession(variant=JIAXING.VARIANT)
        session.game = game
        self.assertIsNone(session.view()['drawn'])
        game.step(('liability_choice', 0))
        self.assertEqual(game.legal_actions()[0], 3)
        game.step(('liability_choice', 1))
        self.assertEqual((game.phase, game.cur), ('action', 1))
        self.assertTrue(session.view()['players'][0]['folded'])
        game._draw(0)
        self.assertEqual(game.legal_actions(), (0, [('discard', game.drawn_tile)]))

    def test_liability_stacks_and_yellow_persists_until_win(self):
        game = JiaxingGame(7, 2, 27, rules=JIAXING)
        game.related_seats = [(1, 2)]
        game.folded = {0}
        winner = SimpleNamespace(actor=1, score=Score(3, ()))
        self.assertEqual(JIAXING.settle_hand(game, [winner], Offer(2, 5, 'gang_bu'))[0][1], (0, 0, 24, 6))
        self.assertEqual(JIAXING.settle_hand(game, [winner], None)[0][1], (0, 0, 15, 6))
        game.yellow = deque(['老头黄'])
        game._end('draw')
        self.assertEqual(list(game.yellow), ['老头黄', '黄番'])
        next_game = JiaxingGame(8, 2, 27, rules=JIAXING)
        next_game.wildcard_tile = game.wildcard_tile = 27
        next_game.dice = SimpleNamespace(randint=lambda low, high: 1)
        next_game.carry_yellow(game)
        self.assertEqual(list(next_game.yellow), ['老头黄', '黄番', '老头黄', '豹子黄', '百搭连庄'])
        next_game._end('suited_self_draw')
        self.assertEqual(list(next_game.yellow), ['黄番', '老头黄', '豹子黄', '百搭连庄'])


class PinghuTests(unittest.TestCase):
    def test_source_flower_example_and_robber_bill(self):
        game = PinghuGame(7, 0, 27, rules=PINGHU)
        hand = counts([27, 27, 27, 28, 28, 28, 0, 0, 0, 3, 3, 8, 8, 18])
        ctx = WinContext('zimo', 8, 27, 27)
        score = PINGHU.faan(hand, (), ctx, wildcard=18, flowers=(34,))
        self.assertEqual(score.share, 16)
        self.assertEqual(score.patterns[-2:], ('花数x3', '番数x2'))
        rob = PINGHU.faan(hand, (), WinContext('rob_kong', 8, 27, 27), wildcard=18, flowers=(34,))
        self.assertEqual(rob.share, 32)
        winner = SimpleNamespace(actor=1, score=rob)
        self.assertEqual(PINGHU.settle_hand(game, [winner], Offer(2, 8, 'gang_bu'))[0][1], (0, 0, 96, 0))

    def test_flower_chain_and_wall_exhaustion(self):
        game = PinghuGame(7, 0, 27, rules=PINGHU)
        self.assertEqual(sum(game.tile_supply.values()), 144)
        self.assertEqual(game.physical_counts(), game.tile_supply)
        tail = game.wall[:16]
        flowers = game.flowers[0].copy()
        game.wall = [*tail, 34, 3, 35]
        game._draw(0)
        self.assertEqual(game.drawn_tile, 3)
        self.assertEqual(game.flowers[0], [*flowers, 35, 34])
        self.assertEqual(game.draw_flowers, 2)
        self.assertEqual(game.wall, tail)
        self.assertEqual(game.remaining, 0)
