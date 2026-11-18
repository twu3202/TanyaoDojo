import unittest

from deploy.play.changchun import Egg
from deploy.play.fuxin import FUXIN_DARK, FUXIN_ZAWU, FuxinGame
from deploy.play.honors import PASS, RON, ZIMO, Meld, WinContext
from deploy.play.honors_game import HonorsSession
from deploy.play.honors_policy import policy_view
from deploy.play.northeast import RED
from deploy.play.shenyang import SHENYANG, ShenyangGame
from deploy.play.tests.test_hongkong import counts
from deploy.play.tests.test_jinyun import pass_offer
from deploy.play.tests.test_northeast_bao import CHANGCHUN_WAIT, table


OPEN_WAIT = [0, 1, 2, 9, 10, 11, 18, 19, 20, 8]
FIVE_WAIT = [0, 1, 2, 9, 10, 11, 21, 23, 17, 17]


class ShenyangFuxinTests(unittest.TestCase):
    def test_shenyang_whirlwind_is_closed_but_adds_to_another_seats_pon(self):
        wind = Egg('egg_winds', 27, material=(27, 28, 29, 30))
        dragon = Egg('egg_dragons', 31, material=(31, 32, RED))
        self.assertTrue(SHENYANG.closed((wind, dragon)))
        self.assertFalse(SHENYANG.rack_allowed(counts([*OPEN_WAIT, 8]), (wind,)))
        for face, egg in ((RED, dragon), (27, wind)):
            game = table(ShenyangGame(8, 0, 27), [[*OPEN_WAIT, 8], [], [], []],
                         melds={0: (Meld('peng', face, 1),), 2: (egg,)})
            hand = game.players[0].hand
            melds = tuple(game.players[0].melds)
            raised = game._score(0, hand, melds, WinContext('zimo', 8, 27, 27))
            game.players[2].melds.clear()
            base = game._score(0, hand, melds, WinContext('zimo', 8, 27, 27))
            self.assertIn('旋风碰番', raised.patterns)
            self.assertEqual(raised.fan, base.fan + 1)
            self.assertEqual('箭刻' in base.patterns, face == RED)

    def test_shenyang_final_four_pass_without_discard_or_reveal(self):
        game = table(ShenyangGame(8, 0, 27), [[8], [], [], []])
        game.discards[0].extend(game.wall[:-4])
        del game.wall[:-4]
        game.step(('discard', 8))
        pass_offer(game)
        before = [river.copy() for river in game.discards]
        for seat in (1, 2, 3, 0):
            self.assertEqual(game.legal_actions(), (seat, [PASS]))
            self.assertTrue(game.split_draw)
            game.step(PASS)
        self.assertEqual((game.phase, game.reason_code), ('over', 'draw'))
        self.assertEqual([len(tiles) for tiles in game.split_tiles], [1, 1, 1, 1])
        self.assertEqual(game.discards, before)
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_shenyang_final_draw_can_win_and_discard_supplier_pays_alone(self):
        for split in (False, True):
            tile = 8
            game = table(ShenyangGame(8, 0, 27), [[tile], OPEN_WAIT, [], []],
                         melds={1: (Meld('peng', 27, 3),)}, draws=(tile,))
            if split:
                game.discards[0].extend(game.wall[:-4])
                del game.wall[:-4]
            game.step(('discard', tile))
            if split:
                game.step(PASS)
                pass_offer(game)
                self.assertEqual(game.legal_actions(), (1, [PASS, ZIMO]))
                game.step(ZIMO)
                self.assertIn('海底捞月', game.win_details[0].patterns)
                self.assertTrue(all(game.win_details[0].payments[seat] for seat in (0, 2, 3)))
            else:
                game.step(RON)
                pass_offer(game)
                self.assertEqual(game.win_details[0].payments[1:], (0, 0, 0))
                self.assertGreater(game.win_details[0].payments[0], 0)
            self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_fuxin_middle_five_extra_and_follow_dealer(self):
        game = table(FuxinGame(3, 0, 27, rules=FUXIN_ZAWU), [[31], [31], [31], [31]])
        for actor in range(4):
            self.assertEqual(game.cur, actor)
            game.step(('discard', 31))
            pass_offer(game)
        self.assertEqual(game.echo_deltas, [-15, 5, 5, 5])
        for ready, bill in ((False, 22), (True, 21)):
            game = table(FuxinGame(3, 0, 27, rules=FUXIN_ZAWU), [[22], FIVE_WAIT, [], []],
                         melds={1: (Meld('peng', 27, 3),)})
            game.declared_waits[1] = (22,)
            if ready:
                game.declared_waits[0] = (7,)
            game.step(('discard', 22))
            game.step(RON)
            pass_offer(game)
            self.assertIn('扎五', game.win_details[0].patterns)
            self.assertEqual(game.win_details[0].payments, (bill, 0, 1, 1))
            self.assertEqual(game.physical_counts(), game.tile_supply)
        game = table(FuxinGame(3, 0, 27, rules=FUXIN_ZAWU), [[*FIVE_WAIT, 22], [], [], []],
                     melds={0: (Meld('peng', 27, 3),)})
        game.declared_waits[0] = (22,)
        game.step(ZIMO)
        self.assertEqual(game.win_details[0].payments, (0, 12, 12, 12))

    def test_laoyou_folded_ready_never_reaches_other_snapshot_or_policy(self):
        game = table(FuxinGame(3, 1, 27, rules=FUXIN_DARK), [[], [*CHANGCHUN_WAIT, 26], [], []])
        game.step(('declare_ready', 26))
        self.assertEqual(game.folded_ready_discards[1], 26)
        self.assertIsNone(game.offer)
        self.assertEqual(game.discards[1], [])
        view = policy_view(game, *game.legal_actions())
        self.assertEqual(view.visible[26], 0)
        session = HonorsSession(variant=FUXIN_DARK.VARIANT)
        session.game = game
        self.assertEqual(session.view()['players'][1]['hidden_ready_discard'], {'tile': None})
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_laoyou_value_threshold_and_ready_supplier_change_liability(self):
        low_wait = [0, 1, 3, 4, 5, 9, 10, 11, 18, 19, 20, RED, RED]
        game = table(FuxinGame(3, 0, 27, rules=FUXIN_DARK), [[2], low_wait, [], []], draws=(2,))
        game.declared_waits[1] = (2,)
        game.step(('discard', 2))
        self.assertNotIn(RON, game.legal_actions()[1])
        pass_offer(game)
        game.step(ZIMO)
        self.assertEqual(game.win_details[0].payments, (6, 0, 6, 6))

        wait = [3, 4, 9, 10, 11, 12, 13, 14, 18, 19, 20, RED, RED]
        for ready, pay in ((False, (18, 0, 0, 0)), (True, (6, 0, 6, 6))):
            game = table(FuxinGame(3, 0, 27, rules=FUXIN_DARK), [[5], wait, [], []])
            game.declared_waits[1] = (2, 5)
            if ready:
                game.declared_waits[0] = (7,)
            game.step(('discard', 5))
            game.step(RON)
            pass_offer(game)
            self.assertEqual(game.win_details[0].payments, pay)
            self.assertEqual('一包三' in game.win_details[0].patterns, not ready)
            self.assertEqual(game.physical_counts(), game.tile_supply)
