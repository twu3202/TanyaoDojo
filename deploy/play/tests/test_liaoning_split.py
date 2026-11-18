import unittest

from deploy.play.honors import PASS, RON, ZIMO, Meld, WinContext
from deploy.play.honors_game import HonorsSession
from deploy.play.honors_policy import policy_view
from deploy.play.liaoning_split import JIANCHANG, LIAOYANG, JianchangGame, LiaoyangGame
from deploy.play.tests.test_hongkong import counts
from deploy.play.tests.test_jinyun import pass_offer
from deploy.play.tests.test_northeast_bao import table


EYE = [0, 1, 2, 9, 10, 11, 21, 23, 17, 17, 22]
JOKER = 22


def liaison(hands, *, melds=None, draws=()):
    game = table(LiaoyangGame(1, 0, 27, rules=LIAOYANG), hands, melds=melds, tail=(JOKER,), draws=draws)
    game.indicator_tile = game.wildcard_indicator = game.wildcard_tile = game.wall.pop(0)
    return game


def shorten_wall(game, count):
    game.discards[3].extend(game.wall[game.rules.WALL_RESERVE:-count])
    del game.wall[game.rules.WALL_RESERVE:-count]


class LiaoningSplitTests(unittest.TestCase):
    def test_jianchang_eye_gate_dragon_pung_and_seven_pairs(self):
        ron = WinContext('ron', 22, 27, 27)
        self.assertEqual(JIANCHANG.faan(counts(EYE), (Meld('peng', 27, 3),), ron).patterns, ('眼胡',))
        self.assertIsNone(JIANCHANG.faan(counts(EYE), (Meld('peng', 33, 3),), ron))
        self.assertIsNone(JIANCHANG.faan(counts(EYE), (Meld('gang_an', 27),), ron))
        pairs = [0, 2, 4, 9, 12, 18, 33]
        self.assertEqual(JIANCHANG.faan(counts(pairs * 2), (), ron).fan, 2)

    def test_jianchang_announces_no_discard_in_split_and_pays_draw(self):
        game = table(JianchangGame(7, 0, 27, rules=JIANCHANG), [[8], [], [], []])
        shorten_wall(game, 4)
        game.step(('discard', 8))
        pass_offer(game)
        rivers = [r.copy() for r in game.discards]
        while game.phase != 'over':
            self.assertEqual(game.legal_actions()[1], [PASS])
            game.step(PASS)
        self.assertEqual(len(game.split_passes), 4)
        self.assertEqual(game.discards, rivers)
        self.assertEqual(len(game.wall), 16)
        self.assertEqual(game.scores(), [-3, 1, 1, 1])
        self.assertEqual(game.draw_deltas, [-3, 1, 1, 1])
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_jianchang_fed_dragon_kong_charges_the_supplier_only(self):
        game = table(JianchangGame(7, 0, 27, rules=JIANCHANG), [[33], [33,33,33], [], []])
        game.step(('discard', 33))
        game.step(('zhigang', None))
        pass_offer(game)
        self.assertEqual(game.kong_deltas, [-4, 4, 0, 0])
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_jianchang_split_win_only_after_entering_split(self):
        for remaining, has_split_bonus in ((4, True), (5, False)):
            game = table(JianchangGame(7, 0, 27, rules=JIANCHANG), [[8], EYE[:-1], [], []],
                         melds={1: (Meld('peng', 27, 3),)}, draws=(22,))
            shorten_wall(game, remaining)
            game.step(('discard', 8))
            pass_offer(game)
            self.assertEqual(game.split_draw, has_split_bonus)
            self.assertIn(ZIMO, game.legal_actions()[1])
            game.step(ZIMO)
            self.assertEqual('海捞' in game.win_details[0].patterns, has_split_bonus)
            self.assertEqual(game.win_details[0].payments, (32, 0, 16, 16) if has_split_bonus else (16, 0, 8, 8))
            self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_liaoyang_joker_cannot_make_terminal_or_third_suit(self):
        zimo = WinContext('zimo', JOKER, 27, 27)
        hands = ([1,2,3, 9,10,11, 18,19,20, 27,27,27, 4,JOKER],
                 [0,JOKER, 3,4,5, 9,10,11, 18,19,20, 27,27,27],
                 [1,2,3, 9,10,11, JOKER,JOKER,JOKER, 27,27,27, 4,4])
        self.assertIsNotNone(LIAOYANG.faan(counts(hands[0]), (), zimo, wildcard=JOKER))
        for hand in hands[1:]:
            self.assertIsNone(LIAOYANG.faan(counts(hand), (), zimo, wildcard=JOKER))

    def test_liaoyang_treasure_is_sealed_until_win_and_outside_cap(self):
        game = liaison([[0,1,2, 9,10,11, 18,19,20, 33,33,33, JOKER,JOKER], [], [], []], draws=(JOKER,))
        actor, actions = game.legal_actions()
        self.assertIn(ZIMO, actions)
        self.assertNotIn(('discard', JOKER), actions)
        self.assertEqual(policy_view(game, actor, actions).visible[JOKER], 1)
        session = HonorsSession(variant=LIAOYANG.VARIANT)
        session.game = game
        self.assertEqual(session.view()['wildcard'], '5s')
        self.assertNotIn('treasure', session.view())
        game.step(ZIMO)
        self.assertEqual(game.win_details[0].patterns, ('天胡',))
        self.assertEqual((game.treasure_tile, game.treasure_points), (JOKER, 4))
        self.assertEqual(game.win_details[0].payments, (0, 68, 68, 68))
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_liaoyang_counts_exposed_treasure_faces_and_single_near_ron(self):
        hand = [0,1,2, 9,10,11, 18,19,20, 8]
        game = liaison([[8], hand, hand, []], melds={1: (Meld('peng', 27, 3),),
                       2: (Meld('peng', 28, 3),)}, draws=(27,))
        game.step(('discard', 8))
        self.assertIn(RON, game.legal_actions()[1])
        game.step(RON)
        self.assertEqual(game.legal_actions(), (2, [PASS]))
        pass_offer(game)
        self.assertEqual(game.winners, [1])
        self.assertEqual((game.treasure_tile, game.treasure_points), (27, 4))
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_liaoyang_fourth_pon_cannot_leave_only_nondiscardable_jokers(self):
        melds = (Meld('peng', 27, 0), Meld('peng', 9, 2), Meld('peng', 18, 3))
        game = liaison([[5], [5, 5, JOKER, JOKER], [], []], melds={1: melds})
        game.step(('discard', 5))
        actor, actions = game.legal_actions()
        self.assertEqual(actor, 1)
        self.assertIn(RON, actions)
        self.assertNotIn(('peng', None), actions)
