import unittest

from deploy.play.changbai import CHANGBAI, ChangbaiGame, public_fan
from deploy.play.chaoyang_town import CHAOYANG_TOWN, ROB_FED_KONG, ChaoyangGame
from deploy.play.changchun import Egg
from deploy.play.honors import PASS, RON, ZIMO, Meld, WinContext
from deploy.play.honors_game import HonorsSession
from deploy.play.honors_policy import policy_view
from deploy.play.tests.test_hongkong import counts
from deploy.play.tests.test_jinyun import pass_offer
from deploy.play.tests.test_northeast_bao import CHANGCHUN_WAIT, table


def changbai():
    game = ChangbaiGame(7,0,27)
    for stake in (6,2,4,0):
        game.step(('piao',stake))
    return game


def reverse_chow(second_hand):
    game = table(ChaoyangGame(7,0,27), [[4],[1,2,3,4],second_hand,[]])
    game.step(('discard',4))
    game.step(('chi',2))
    pass_offer(game)
    return game


class ChangbaiChaoyangTests(unittest.TestCase):
    def test_changbai_does_not_deal_or_reveal_until_all_piao_choices(self):
        game = ChangbaiGame(7,0,27)
        session = HonorsSession(variant=CHANGBAI.VARIANT)
        session.game = game
        for actor, stake in enumerate((6,2,4,0)):
            seat, actions = game.legal_actions()
            self.assertEqual(actor, seat)
            self.assertEqual([a[1] for a in actions],[0,2,4,6])
            self.assertEqual(sum(policy_view(game,seat,actions).hand),0)
            self.assertTrue(all(p['hand_count']==0 and 'piao' not in p for p in session.view()['players']))
            self.assertEqual(len(game.wall),136)
            game.step(('piao',stake))
        self.assertEqual([sum(p.hand) for p in game.players],[14,13,13,13])
        self.assertEqual([p['piao'] for p in session.view()['players']],[6,2,4,0])
        self.assertEqual(game.physical_counts(),game.tile_supply)

    def test_changbai_two_ready_winners_each_receive_their_own_piao(self):
        other = [28 if t==27 else t for t in CHANGCHUN_WAIT]
        game = table(changbai(),[[33],CHANGCHUN_WAIT,other,[]])
        self.assertNotIn(RON,game.legal_actions()[1])
        game.declared_waits[1] = game.declared_waits[2] = (33,)
        game.step(('discard',33))
        game.step(RON)
        game.step(RON)
        pass_offer(game)
        self.assertEqual(game.winners,[1,2])
        self.assertEqual(game.piao_deltas,[-18,8,10,0])
        self.assertEqual(game.next_dealer,1)
        self.assertEqual(game.physical_counts(),game.tile_supply)

    def test_changbai_public_four_in_one_and_duplicate_chow_raise_losing_bill(self):
        shown = (Meld('chi',0,3),)*4
        ready = [3,4,5,9,10,11,18,19,20,27,27,27,33]
        game = table(changbai(),[[33],ready,[],[]],melds={0:shown})
        game.declared_waits[1] = (33,)
        game.step(('discard',33))
        game.step(RON)
        pass_offer(game)
        win = game.win_details[0]
        self.assertEqual(public_fan(shown),5)
        self.assertEqual(win.payments,(win.points*64,0,0,0))
        self.assertEqual(game.piao_deltas,[-8,8,0,0])

    def test_changbai_hidden_four_is_winning_only(self):
        melds = (Meld('chi',0,3),)
        hand = [0,0,0,9,10,11,18,19,20,27,27]
        score = CHANGBAI.faan(counts(hand),melds,WinContext('zimo',27,27,27))
        self.assertIn('暗四归一',score.patterns)
        self.assertEqual(public_fan(melds),0)

    def test_changbai_kong_releases_tail_and_draw_pays_dealer_penalty(self):
        for kong in (False,True):
            game = table(changbai(),[[27]*4,[],[],[]])
            if kong:
                game.step(('ankan',27))
                pass_offer(game)
            reserve = game._replacement_index()
            self.assertEqual(reserve,0 if kong else 14)
            game.discards[3].extend(game.wall[reserve:-4])
            del game.wall[reserve:-4]
            move = next(a for a in game.legal_actions()[1] if a[0]=='discard')
            game.step(move)
            pass_offer(game)
            while game.phase!='over':
                self.assertTrue(game.split_draw)
                self.assertLessEqual(set(game.legal_actions()[1]),{PASS,ZIMO})
                game.step(PASS)
            self.assertEqual(len(game.wall),reserve)
            self.assertEqual(game.draw_deltas,[-6,2,2,2])
            self.assertEqual(game.scores(),[a+b for a,b in zip(game.draw_deltas,game.kong_deltas)])
            self.assertEqual(game.physical_counts(),game.tile_supply)

    def test_chaoyang_opposite_end_feed_covers_three_winner_bills(self):
        waiting = [0,2,9,10,11,18,19,20,27,27,27,33,33]
        game = reverse_chow(waiting)
        self.assertEqual(game.reverse_end,1)
        self.assertNotIn(('discard',4),game.legal_actions()[1])
        game.step(('discard',1))
        game.step(RON)
        pass_offer(game)
        self.assertEqual(game.winners,[2])
        self.assertEqual(game.win_details[0].payments,(0,32,0,0))
        self.assertEqual(game.physical_counts(),game.tile_supply)

    def test_chaoyang_opposite_end_kong_covers_all_kong_bills(self):
        game = reverse_chow([1]*3)
        game.step(('discard',1))
        game.step(('zhigang',None))
        pass_offer(game)
        if game.phase==ROB_FED_KONG:
            game.step(PASS)
        self.assertEqual(game.kong_deltas,[0,-4,4,0])
        self.assertEqual(game.physical_counts(),game.tile_supply)

    def test_chaoyang_shape_offers_missing_egg_and_ding(self):
        game = ChaoyangGame(7,0,27)
        egg = (Meld('peng',27,1),)
        hand = [0,9,18,3,4,5,21,22,23,33,33]
        self.assertIn('蛋夹',game._score(0,counts(hand),egg,WinContext('ron',18,27,27)).patterns)
        ding = (Meld('chi',21,3),)
        hand = [0,1,2,9,10,11,19,19,19,33,33]
        self.assertIn('丁胡',game._score(0,counts(hand),ding,WinContext('ron',19,27,27)).patterns)

    def test_chaoyang_supplier_can_reclaim_tile_without_paying_kong(self):
        hand = [0,2,9,10,11,18,19,20,27,27,27,33,33,1]
        game = table(ChaoyangGame(7,0,27),[hand,[1]*3,[],[]])
        game.step(('discard',1))
        game.step(('zhigang',None))
        pass_offer(game)
        self.assertEqual(game.legal_actions(),(0,[PASS,(ROB_FED_KONG,1)]))
        game.step((ROB_FED_KONG,1))
        self.assertEqual(game.win_details[0].provider,1)
        self.assertEqual(game.reason_code,'rob_kong_win')
        self.assertEqual(game.kong_deltas,[0]*4)
        self.assertEqual(game.physical_counts(),game.tile_supply)

    def test_chaoyang_robbed_extension_leaves_egg_intact_and_unpaid(self):
        egg = Egg('egg_dragons',31,material=(31,32,33))
        game = table(ChaoyangGame(7,0,27),[[33],[33,33],[],[]],melds={0:(egg,)})
        game.step(('extend_egg',(0,33)))
        game.step(('rob_egg_pon',None))
        pass_offer(game)
        self.assertEqual(game.players[0].melds,[egg])
        self.assertEqual(game.players[1].melds,[Meld('peng',33,0)])
        self.assertEqual(game.kong_deltas,[0]*4)
        self.assertEqual(game.physical_counts(),game.tile_supply)
