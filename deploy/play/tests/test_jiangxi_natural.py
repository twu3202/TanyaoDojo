import unittest

from deploy.play.honors import PASS, RON, ZIMO, Meld, Offer, WinContext
from deploy.play.honors_game import HonorsSession
from deploy.play.honors_policy import heuristic, policy_view
from deploy.play.jiangxi import HonorChow, HonorChowGame
from deploy.play.jiangxi_natural import PINGXIANG, SAME_SUIT_FEED, YICHUN, PingxiangGame
from deploy.play.jiangxi_shapes import HONOR_CHOW_ID, honor_standard
from deploy.play.tests.test_hongkong import counts
from deploy.play.tests.test_jinyun import pass_offer
from deploy.play.tests.test_northeast_bao import table


NORMAL_READY = [0,1,2,9,10,11,18,19,20,31,31,31,4]
WIND_READY = [0,1,2,9,10,11,18,19,20,27,28,29,33]
OTHER_WIND_READY = [3,4,5,12,13,14,21,22,23,27,29,30,33]
PURE_OPEN = (Meld('chi',0,3), Meld('chi',3,3))


class YichunTests(unittest.TestCase):
    def test_two_winners_claim_one_printed_tile_and_discarder_pays_both(self):
        game = table(HonorChowGame(9,0,27,rules=YICHUN), [[33],WIND_READY,OTHER_WIND_READY,[]])
        game.opening = False
        game.step(('discard',33))
        self.assertIn(RON, game.legal_actions()[1])
        game.step(RON)
        self.assertIn(RON, game.legal_actions()[1])
        game.step(RON)
        pass_offer(game)
        self.assertEqual(game.winners, [1,2])
        self.assertEqual(game.scores(), [-4,2,2,0])
        self.assertEqual(game.next_dealer, 0)
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_honor_chow_is_a_printed_call_under_yichun_rules(self):
        g = table(HonorChowGame(9,0,27,rules=YICHUN), [[30],[27,29],[],[]])
        g.step(('discard',30))
        actor, actions = g.legal_actions()
        self.assertEqual(actor, 1)
        chow = ('chi', HONOR_CHOW_ID+2)
        self.assertIn(chow, actions)
        self.assertIn(heuristic(policy_view(g,actor,actions)), actions)
        g.step(chow)
        pass_offer(g)
        self.assertEqual(g.players[1].melds, [HonorChow('chi',HONOR_CHOW_ID+2,0)])
        self.assertEqual(g.physical_counts(), g.tile_supply)

    def test_all_honors_can_win_unstructured_and_doubles_if_set_based(self):
        mixed = tuple(counts([27,27,27,27,28,28,28,28,29,29,29,30,31,33]))
        formed = counts([27,27,27,28,28,28,29,29,29,30,30,30,31,31])
        self.assertFalse(honor_standard(mixed,0))
        ctx = WinContext('zimo',33,27,27)
        self.assertEqual(YICHUN.faan(mixed,(),ctx).patterns, ('字一色',))
        self.assertEqual(YICHUN.faan(mixed,(),ctx).share, 6)
        self.assertEqual(YICHUN.faan(formed,(),ctx).patterns, ('字一色成阙',))
        self.assertEqual(YICHUN.faan(formed,(),ctx).share, 12)

    def test_passing_only_locks_that_discard_face_until_own_turn(self):
        ready = [0,1,2,3,4,9,10,11,18,19,20,31,31]
        g = table(HonorChowGame(9,0,27,rules=YICHUN), [[5],ready,[],[]])
        g.step(('discard',5))
        self.assertIn(RON, g.legal_actions()[1])
        g.step(PASS)
        self.assertIn(5,g.players[1].declined_wins)
        for offered, legal in ((5,False),(2,True)):
            g._open_offer(Offer(0,offered,'discard'))
            self.assertEqual(RON in g.legal_actions()[1], legal)
        g.offer = None
        g._draw(1)
        self.assertFalse(g.players[1].declined_wins)

    def test_last_discard_can_win_but_last_draw_cannot_kong(self):
        rack = [0,0,0,0,8,8,9,10,11,18,19,20,31,32]
        game = table(HonorChowGame(9,0,27,rules=YICHUN), [rack,[],[],[]])
        game.opening = False
        game.discards[2].extend(game.wall)
        game.wall.clear()
        self.assertNotIn(('ankan',0),game.legal_actions()[1])
        game = table(HonorChowGame(9,0,27,rules=YICHUN), [[33],WIND_READY,[],[]])
        game.opening = False
        game.discards[2].extend(game.wall)
        game.wall.clear()
        game.step(('discard',33))
        game.step(RON)
        pass_offer(game)
        self.assertIn('海底炮',game.win_details[0].patterns)
        self.assertEqual(game.win_details[0].payments, (4,0,0,0))
        self.assertEqual(game.physical_counts(),game.tile_supply)


class PingxiangTests(unittest.TestCase):
    def test_true_pair_basic_self_draw_and_opening_exception(self):
        hand = counts([*NORMAL_READY,4])
        zimo = WinContext('zimo',4,27,27)
        ron = WinContext('ron',4,27,27)
        earth = WinContext('ron',4,27,27,opening='地胡')
        self.assertEqual(PINGXIANG.faan(hand,(),zimo).patterns,('平胡','真将'))
        self.assertIsNone(PINGXIANG.faan(hand,(),ron))
        self.assertEqual(PINGXIANG.faan(hand,(),earth).share,32)
        self.assertIsNone(PINGXIANG.faan(counts([*NORMAL_READY[:-1],33,33]),(),zimo))

    def test_chow_requires_pure_suit_but_allows_ordinary_pair(self):
        melds = (Meld('chi',0,3),)
        zoned = counts([3,4,5,6,7,8,6,7,8,0,0])
        mixed = counts([3,4,5,6,7,8,9,10,11,1,1])
        ron = WinContext('ron',0,27,27)
        self.assertEqual(PINGXIANG.faan(zoned,melds,ron).patterns,('清一色',))
        self.assertIsNone(PINGXIANG.faan(mixed,melds,ron))
        self.assertIsNone(PINGXIANG.faan(mixed,melds,WinContext('zimo',1,27,27)))

    def test_pungs_need_true_pair_and_highest_scoring_decomposition(self):
        ctx = WinContext('ron',4,27,27)
        true = counts([0]*3+[1]*3+[2]*3+[3]*3+[4,4])
        false = counts([0]*3+[9]*3+[18]*3+[27]*3+[33,33])
        self.assertEqual(PINGXIANG.faan(true,(),ctx).share,16)
        self.assertIsNone(PINGXIANG.faan(false,(),ctx))
        self.assertEqual(PINGXIANG.faan(counts([0]*4+[9]*4+[18]*4+[33,33]),(),ctx).share,32)

    def test_third_same_suit_chow_records_actual_feeder_and_wraps_self_draw(self):
        game = table(PingxiangGame(9,3,27), [[1,1,3,4,18,7,8],[],[],[6]],
                     melds={0:PURE_OPEN},draws=(0,2,4,5))
        game.step(('discard',6))
        game.step(('chi',6))
        pass_offer(game)
        self.assertEqual(game.players[0].liabilities,[(SAME_SUIT_FEED,3)])
        session = HonorsSession(variant=PINGXIANG.VARIANT)
        session.game = game
        self.assertEqual(session.view()['suit_feed_liabilities'],[[0,3]])
        for discard in (18,0,2,4):
            game.step(('discard',discard))
            pass_offer(game)
        game.step(ZIMO)
        win = game.win_details[0]
        self.assertEqual(win.patterns,('清一色','真将','包牌'))
        self.assertEqual(win.payments,(0,0,0,24))
        self.assertEqual(game.physical_counts(),game.tile_supply)

    def test_robbed_added_kong_is_billed_to_kong_owner(self):
        game = table(PingxiangGame(3,0,27), [[6],[3,4,5,3,4,5,7,8,0,0],[],[]],
                     melds={0:(Meld('peng',6,3),),1:(Meld('chi',0,0),)})
        player = game.players[1]
        player.liabilities = [(SAME_SUIT_FEED,3)]
        game.step(('bugang',6))
        self.assertIn(RON,game.legal_actions()[1])
        game.step(RON)
        pass_offer(game)
        self.assertEqual(game.win_details[0].payments,(4,0,0,0))
        self.assertIn('抢杠胡',game.win_details[0].patterns)
        self.assertEqual(game.kong_deltas,[0]*4)
        self.assertEqual(game.physical_counts(),game.tile_supply)

    def test_draw_checks_dealer_ready_and_cancels_completed_kongs(self):
        for last, ready in ((4,True),(33,False)):
            rack = [*NORMAL_READY[:-1],last,32]
            game = table(PingxiangGame(9,0,27), [rack,[],[],[]], melds=None)
            game._pay_kong(2,Meld('gang_bu',21,1))
            self.assertEqual(game.kong_deltas,[0,-1,1,0])
            game.discards[2].extend(game.wall)
            game.wall.clear()
            game.step(('discard',32))
            pass_offer(game)
            self.assertEqual(game.reason_code,'draw')
            self.assertEqual(game.dealer_ready_deltas,[3,-1,-1,-1] if ready else [0]*4)
            self.assertEqual(game.scores(),game.dealer_ready_deltas)
            self.assertEqual(game.kong_deltas,[0]*4)
            self.assertEqual(game.next_dealer,1)
            self.assertEqual(game.physical_counts(),game.tile_supply)

    def test_four_matching_opening_discards_are_charged_to_dealer(self):
        game = table(PingxiangGame(9,0,27), [[32],[32],[32],[32]])
        game.uninterrupted = False
        for _ in range(4):
            game.step(('discard',32))
            pass_offer(game)
        self.assertEqual(game.echo_deltas,[-3,1,1,1])
        self.assertEqual(game.scores(),game.echo_deltas)
        self.assertEqual(game.physical_counts(),game.tile_supply)
