import unittest

from deploy.play.changchun import BAO_WIN, CHICKEN, Egg
from deploy.play.honors import PASS, RON, ZIMO, Meld, WinContext
from deploy.play.honors_game import HonorsSession
from deploy.play.jilin_shuangliao import JILIN, SHUANGLIAO, JilinGame, ShuangliaoGame
from deploy.play.northeast import RED
from deploy.play.tests.test_hongkong import counts
from deploy.play.tests.test_jinyun import pass_offer
from deploy.play.tests.test_northeast_bao import CHANGCHUN_WAIT, table


class JilinShuangliaoTests(unittest.TestCase):
    def test_jilin_caps_seven_pairs_and_requires_readiness_after_opening(self):
        seven = [tile for tile in (0, 1, 4, 9, 11, 18, 22) for _ in range(2)]
        game = table(JilinGame(7, 0, 27), [seven, [], [], []])
        self.assertEqual(JILIN.pattern(tuple(counts(seven)), (), seven[-1]), '七小对')
        game.opening = False
        self.assertNotIn(ZIMO, game.legal_actions()[1])
        game.opening = True
        game.step(ZIMO)
        self.assertEqual(game.win_details[0].points, 16)
        self.assertEqual(game.win_details[0].payments, (0, 16, 16, 16))

    def test_first_ready_seat_opens_before_other_seats_and_private_dice_match_pays(self):
        game = table(JilinGame(3, 0, 27), [[*CHANGCHUN_WAIT, 26], [], [], []], tail=(RED,) * 3)
        game.step(('declare_ready', 26))
        self.assertEqual(game.bao_starter, 0)
        game.declared_waits[1] = (4,)
        self.assertFalse(game._can_open_bao(1))
        # Keep the three red copies in the candidate positions and use a deterministic roll.
        game.bao_rng.seed(1)
        game._draw(0)
        self.assertEqual((game.phase, game.bao_slot), ('bao_peek', 1))
        self.assertEqual(game.bao_view(1), ())
        session = HonorsSession(variant=JILIN.VARIANT)
        session.game = game
        self.assertEqual(session.view()['bao_roll'], 2)
        self.assertEqual(session.option(BAO_WIN)['win_type'], 'tong_bao')
        game.step(BAO_WIN)
        self.assertEqual(game.reason_code, 'tong_bao_win')
        self.assertIn('通宝', game.win_details[0].patterns)
        self.assertEqual(game.win_details[0].payments, (0, 16, 16, 16))
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_jilin_chicken_fills_the_wait_after_bao_seen_on_draw_only(self):
        game = table(JilinGame(7, 0, 27), [[CHICKEN], CHANGCHUN_WAIT, [], []], tail=(5,), draws=(CHICKEN,))
        game.bao_slot = 0
        game.bao_history.add(0)
        game.bao_seen[1] = 0
        game.declared_waits[1] = (RED,)
        game.step(('discard', CHICKEN))
        self.assertNotIn(RON, game.legal_actions()[1])
        pass_offer(game)
        self.assertEqual(game.drawn_tile, CHICKEN)
        game.step(ZIMO)
        self.assertIn('搂宝', game.win_details[0].patterns)
        self.assertEqual(game.revealed[1][CHICKEN], 2)
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_jilin_wind_egg_and_suited_kongs_use_the_capped_table(self):
        game = table(JilinGame(7, 0, 27), [[27, 28, 29], [], [], []])
        action = ('egg_winds', ((27, 28, 29), None))
        self.assertIn(action, game.legal_actions()[1])
        game.step(action)
        egg, = game.players[0].melds
        self.assertIsInstance(egg, Egg)
        self.assertEqual(egg.tiles, (27, 28, 29))
        self.assertEqual(game.kong_deltas, [9, -3, -3, -3])
        self.assertEqual(game.physical_counts(), game.tile_supply)
        game._pay_kong(2, Meld('gang_an', 7))
        self.assertEqual(game.kong_deltas, [-7, -19, 45, -19])

    def test_shuangliao_distinguishes_closed_true_pungs_from_open_123_piao(self):
        base = [9] * 3 + [18] * 3 + [27] * 3 + [4, 4]
        game = table(ShuangliaoGame(7, 0, 27, rules=SHUANGLIAO), [[0, 0, 0, *base], [], [], []])
        game.step(ZIMO)
        self.assertEqual(game.win_details[0].patterns, ('飘中飘', '自摸', '带幺鸡', '站立'))
        self.assertEqual(game.win_details[0].payments, (0, 256, 256, 256))
        self.assertEqual(game.physical_counts(), game.tile_supply)

        game = table(ShuangliaoGame(7, 0, 27, rules=SHUANGLIAO), [base, [], [], []],
                     melds={0: (Meld('chi', 0, 3),)})
        game.step(ZIMO)
        self.assertEqual(game.win_details[0].patterns, ('飘胡', '自摸', '带幺鸡'))
        self.assertEqual(game.win_details[0].payments, (0, 32, 32, 32))

    def test_shuangliao_four_copies_and_kong_draw_double_only_once(self):
        melds = (Meld('peng', 9, 1),)
        tiles = [9, 10, 11, 3, 4, 5, 19, 20, 21, 28, 28]
        game = ShuangliaoGame(7, 0, 27, rules=SHUANGLIAO)
        hand = counts(tiles)
        direct = game._score(0, hand, melds, WinContext('zimo', 28, 27, 27))
        replacement = game._score(0, hand, melds, WinContext('zimo', 28, 27, 27, kong_chain=1))
        self.assertIn('四归一', direct.patterns)
        self.assertIn('杠翻', replacement.patterns)
        self.assertNotIn('四归一', replacement.patterns)
        self.assertEqual(direct.share, replacement.share)

    def test_shuangliao_resolves_only_the_first_discard_winner(self):
        first = [*range(12), RED]
        second = [*range(12, 24), RED]
        game = table(ShuangliaoGame(7, 0, 27, rules=SHUANGLIAO), [[RED], first, second, []])
        game.step(('discard', RED))
        self.assertIn(RON, game.legal_actions()[1])
        game.step(RON)
        self.assertEqual(game.legal_actions(), (2, [PASS]))
        pass_offer(game)
        self.assertEqual(game.winners, [1])
        self.assertEqual(game.physical_counts(), game.tile_supply)
