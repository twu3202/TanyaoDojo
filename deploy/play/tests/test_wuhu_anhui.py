from collections import Counter
from types import SimpleNamespace
import unittest

from deploy.play.hongzhong import ANHUI, RED
from deploy.play.honors import PASS, RON, WINDS, Meld, Offer, WinContext
from deploy.play.honors_policy import _live_tiles, policy_view
from deploy.play.tests import test_hongzhong
from deploy.play.tests.test_hongkong import counts
from deploy.play.tests.test_jinyun import pass_offer, set_hands
from deploy.play.wuhu import WUHU, WUHU_515, WuhuGame, WuhuScore, suit_counts


class WuhuTests(unittest.TestCase):
    def test_tile_supply_gate_and_kong_contributes_four(self):
        classic = WuhuGame(1, 0, 27, rules=WUHU)
        red = WuhuGame(1, 0, 27, rules=WUHU_515)
        self.assertEqual(sum(classic.tile_supply.values()), sum(red.tile_supply.values()))
        self.assertEqual(sum(classic.tile_supply.values()), 112)
        self.assertEqual({t: classic.tile_supply[t] for t in WINDS}, {t: 1 for t in WINDS})
        self.assertEqual({t: red.tile_supply[t] for t in (*WINDS, 33)}, {27: 0, 28: 0, 29: 0, 30: 0, 33: 4})
        seat, actions = classic.legal_actions()
        live = _live_tiles(policy_view(classic, seat, actions))
        self.assertTrue(all(0 <= live[t] <= 1 for t in WINDS))

        hand = counts([0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 18, 18])
        ctx = WinContext('ron', 2, 28, 27)
        self.assertIsNone(WUHU.faan(hand, (), ctx))
        melds = (Meld('gang_an', 0), Meld('gang_ming', 9, 2))
        hand = counts([1, 2, 3, 4, 5, 6, 6, 6])
        self.assertEqual(suit_counts(hand, melds), (12, 4, 0))
        self.assertIsNotNone(WUHU.faan(hand, melds, ctx))
        self.assertIsNone(WUHU.faan(hand, melds, WinContext('rob_kong', 2, 28, 27)))

    def test_unique_inside_wait_and_declared_double_eight(self):
        hand = counts([*range(9), 9, 10, 11, 18, 18])
        score = WUHU.faan(hand, (), WinContext('ron', 1, 28, 27))
        self.assertEqual(score, WuhuScore(31, ('压档', '八支x9', '通天'), dealer_bonus=6))
        score = WUHU.faan(hand, (), WinContext('zimo', 1, 28, 27))
        self.assertEqual(score.share, 61)
        self.assertEqual(score.dealer_bonus, 10)
        melds = (Meld('gang_an', 9), Meld('gang_ming', 10, 3))
        hand = counts([0, 1, 2, 3, 4, 5, 6, 6])
        score = WUHU.faan(hand, melds, WinContext('ron', 6, 28, 27))
        self.assertEqual(score.share, 100)
        self.assertIn('双八支', score.patterns)

    def test_double_root_is_not_a_quad_used_as_a_pair(self):
        hand = counts([0]*4 + [1, 2] + [3]*4 + [4, 5, 18, 18])
        score = WUHU.faan(hand, (), WinContext('ron', 5, 28, 27))
        self.assertEqual(score.share, 100)
        self.assertIn('双四核', score.patterns)
        hand = counts([0]*4 + [1, 1, 2, 2] + [9, 10, 11, 18, 19, 20])
        score = WUHU.faan(hand, (), WinContext('ron', 2, 28, 27))
        self.assertNotIn('四核', score.patterns)
        self.assertNotIn('双四核', score.patterns)

    def test_third_pon_redirects_only_classic_ron(self):
        for rules, payments in ((WUHU, (16, 0, 0, 0)), (WUHU_515, (16, 0, 12, 12))):
            with self.subTest(rules=rules):
                game = WuhuGame(1, 0, 27, rules=rules)
                set_hands(game, [[0, 2, 5], [0, 0, 2, 2, 5, 5, 9, 10, 11, 18, 19, 20, 15], [], []])
                for tile in (0, 2, 5):
                    game._discard(0, tile)
                    game.step(('peng', None))
                    pass_offer(game)
                self.assertEqual(game.players[1].liabilities, [('three_feeds', 0)] if rules is WUHU else [])
                score = WuhuScore(12, ('平胡',), dealer_bonus=4)
                winner = SimpleNamespace(actor=1, score=score)
                outcome, = rules.settle_hand(game, [winner], Offer(2, 1, 'discard'))
                self.assertEqual(outcome[1], payments)
                self.assertEqual('三碰包牌' in outcome[0].patterns, rules is WUHU)
                self.assertEqual(rules.settle_hand(game, [winner], None)[0][1], (16, 0, 12, 12))

    def test_red_515_allows_natural_red_pung_and_charges_every_opponent(self):
        game = WuhuGame(1, 0, 27, rules=WUHU_515)
        ready = [0, 1, 2, 3, 4, 5, 8, 8, 9, 10, 33, 33, 33]
        set_hands(game, [[11], ready, [], []])
        game.step(('discard', 11))
        self.assertIn(RON, game.legal_actions()[1])
        game.step(RON)
        self.assertEqual(game.legal_actions(), (2, [PASS]))
        pass_offer(game)
        self.assertEqual(game.scores(), [-8, 20, -6, -6])
        self.assertEqual(game.physical_counts(), game.tile_supply)


class AnhuiRedDragonTests(unittest.TestCase):
    def test_only_natural_player_can_rob_and_kong_does_not_complete(self):
        game = test_hongzhong.RuleTests().rob_fixture(red_first=True, profile=ANHUI)
        reserve = game.wall[:4]
        game.step(('bugang', 0))
        self.assertEqual(game.legal_actions()[0], 2)
        game.step(('ron', None))
        self.assertEqual(game.winners, [2])
        self.assertEqual(game.bonus, reserve)
        share = 2 + 2 * sum(t == RED or t % 9 in (0, 4, 8) for t in reserve)
        self.assertEqual(game.scores(), [-3*share, 0, 3*share, 0])
        self.assertEqual(game.gang_ledger, [])
        self.assertEqual(game.players[0].melds, [('peng', 0)])
        self.assertEqual(game.next_dealer, 2)

    def test_four_red_is_a_win_but_cannot_be_melded(self):
        game = test_hongzhong.fixture([[0, 2, 4, 6, 8, 9, 11, 13, 15, 17, *([RED]*4)],
                                      [], [], []], profile=ANHUI)
        self.assertNotIn(('ankan', RED), game.legal_actions()[1])
        self.assertIn(('zimo', None), game.legal_actions()[1])
        game.step(('zimo', None))
        self.assertEqual(game.reason_code, 'four_red')
        game = test_hongzhong.fixture([[RED, 0], [RED, RED, RED], [], []], profile=ANHUI)
        game.step(('discard', RED))
        self.assertEqual(game.legal_actions(), (1, [('pass', None)]))

    def test_passed_kong_draws_outside_reserve_and_reserve_survives_draw(self):
        game = test_hongzhong.RuleTests().rob_fixture(profile=ANHUI)
        reserve = game.wall[:4]
        substitute = game.wall[4]
        game.step(('bugang', 0))
        while game.phase == 'rob_kong':
            game.step(('pass', None))
        self.assertEqual(game.drawn[0], substitute)
        self.assertEqual(game.wall[:4], reserve)
        self.assertIn(('gang_bu', 0), game.players[0].melds)
        game.discards[3].extend(game.wall[4:])
        game.wall = game.wall[:4]
        game.step(('discard', 6))
        self.assertEqual(game.reason_code, 'hongzhong_draw')
        self.assertEqual(game.wall, reserve)
        self.assertEqual(game.scores(), [3, -1, -1, -1])
        self.assertEqual(game.physical_counts(), Counter({i: 4 for i in range(28)}))
