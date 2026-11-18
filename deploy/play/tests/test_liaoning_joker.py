import unittest

from deploy.play.honors import PASS, RON, ZIMO, Meld, WinContext
from deploy.play.honors_game import HonorsSession
from deploy.play.honors_policy import policy_view
from deploy.play.liaoning_joker import ANSHAN, PANJIN, LiaoningJokerGame, PanjinGame, next_face
from deploy.play.tests.test_hongkong import counts
from deploy.play.tests.test_jinyun import pass_offer
from deploy.play.tests.test_liaoning_split import shorten_wall
from deploy.play.tests.test_northeast_bao import table


ANSHAN_GUN_WIN = [0, 1, 2, 9, 10, 11, 18, 19, 20, 6, 6, 6, 1, 1]
PANJIN_RON = [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27, 8]


def joker_table(rules, hands, *, melds=None, indicator=12, draws=()):
    cls = PanjinGame if rules is PANJIN else LiaoningJokerGame
    game = table(cls(4, 0, 27, rules=rules), hands, melds=melds, tail=(indicator,), draws=draws)
    game.indicator_tile = game.wildcard_indicator = game.wall.pop(0)
    game.wildcard_tile = next_face(game.indicator_tile)
    return game


class LiaoningJokerTests(unittest.TestCase):
    def test_successor_wraps_independently_by_suit_and_honor_group(self):
        self.assertEqual([next_face(t) for t in (8, 17, 26, 30, 33, 32, 31)], [0, 9, 18, 27, 32, 31, 33])

    def test_panjin_joker_can_complete_chow_but_not_pung_or_terminal(self):
        joker = 13
        ctx = WinContext('zimo', joker, 27, 27)
        chow = [0,1,2, 9,10,joker, 18,19,20, 27,27,27, 4,4]
        pung = [0,1,2, 9,10,11, 18,19,20, 27,27,joker, 4,4]
        terminal = [0,joker, 3,4,5, 9,10,11, 18,19,20, 27,27,27]
        self.assertIsNotNone(PANJIN.faan(counts(chow), (), ctx, wildcard=joker))
        self.assertIsNone(PANJIN.faan(counts(pung), (), ctx, wildcard=joker))
        self.assertIsNone(PANJIN.faan(counts(terminal), (), ctx, wildcard=joker))
        self.assertIsNotNone(ANSHAN.faan(counts(terminal), (), ctx, wildcard=joker))

    def test_panjin_four_jokers_form_paid_kong_but_cannot_be_thrown(self):
        joker = 13
        game = joker_table(PANJIN, [[joker] * 4, [], [], []])
        self.assertIn(('ankan', joker), game.legal_actions()[1])
        self.assertNotIn(('discard', joker), game.legal_actions()[1])
        game.step(('ankan', joker))
        pass_offer(game)
        self.assertEqual(game.players[0].melds, [Meld('gang_an', joker)])
        self.assertEqual(game.kong_deltas, [192, -64, -64, -64])
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_panjin_last_discard_can_be_konged_into_first_split_draw(self):
        game = joker_table(PANJIN, [[8], [8] * 3, [], []], draws=(6, 7, 16, 25))
        shorten_wall(game, 4)
        game.step(('discard', 8))
        actor, actions = game.legal_actions()
        self.assertEqual(actor, 1)
        self.assertIn(('zhigang', None), actions)
        self.assertNotIn(('peng', None), actions)
        game.step(('zhigang', None))
        pass_offer(game)
        self.assertEqual((game.cur, game.split_draw, game.drawn_tile), (1, True, 25))
        rivers = [river.copy() for river in game.discards]
        for seat in (1, 2, 3, 0):
            self.assertEqual(game.legal_actions()[0], seat)
            self.assertLessEqual(set(game.legal_actions()[1]), {PASS, ZIMO})
            game.step(PASS)
        self.assertEqual(len(game.wall), 8)
        self.assertEqual(game.discards, rivers)
        self.assertEqual(game.scores(), [-1, 3, -1, -1])
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_panjin_ron_supplier_covers_the_three_actual_bills(self):
        game = joker_table(PANJIN, [[8], PANJIN_RON, [], []])
        game.step(('discard', 8))
        game.step(RON)
        pass_offer(game)
        self.assertEqual(game.win_details[0].payments, (32, 0, 0, 0))
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_anshan_public_gun_and_per_copy_fan_with_all_losers_paying(self):
        game = joker_table(ANSHAN, [ANSHAN_GUN_WIN, [], [], []], indicator=4)
        actor, actions = game.legal_actions()
        self.assertEqual(policy_view(game, actor, actions).visible[4], 1)
        session = HonorsSession(variant=ANSHAN.VARIANT)
        session.game = game
        self.assertEqual(session.view()['open_gun'], '7m')
        self.assertEqual(session.view()['wildcard_indicator'], '5m')
        self.assertEqual(session.view()['wildcard'], '6m')
        game.step(ZIMO)
        self.assertEqual(game.win_details[0].patterns.count('明枪'), 3)
        self.assertEqual(game.win_details[0].payments, (0, 256, 256, 256))
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_anshan_joker_cannot_supply_a_wholly_absent_suit(self):
        joker = 13
        missing = [1,2,3, 9,10,11, joker,joker,joker, 27,27,27, 4,4]
        self.assertIsNone(ANSHAN.faan(counts(missing), (), WinContext('zimo', joker, 27, 27), wildcard=joker))

    def test_anshan_discard_from_all_joker_rack_locks_claims_and_never_opens_offer(self):
        joker = 13
        melds = (Meld('peng', 0, 1), Meld('peng', 9, 2), Meld('peng', 18, 3), Meld('peng', 27, 1))
        game = joker_table(ANSHAN, [[joker, joker], [], [], []], melds={0: melds})
        prior = game.players[0].hand.copy()
        self.assertIn(ZIMO, game.legal_actions()[1])
        game.step(('discard', joker))
        self.assertIsNone(game.offer)
        self.assertEqual(game.discards[0], [joker])
        self.assertEqual(game.cur, 1)
        self.assertIsNone(game._score(0, prior, melds, WinContext('zimo', joker, 27, 27)))
        session = HonorsSession(variant=ANSHAN.VARIANT)
        session.game = game
        self.assertTrue(session.view()['players'][0]['joker_win_lock'])
        self.assertEqual(game.physical_counts(), game.tile_supply)
