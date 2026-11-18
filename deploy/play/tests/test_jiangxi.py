import unittest

from deploy.play.honors import PASS, RON, ZIMO, Meld, Offer, WinContext
from deploy.play.honors_game import HonorsSession
from deploy.play.honors_policy import heuristic, policy_view
from deploy.play.jiangxi import NANCHANG, SHANGRAO, HonorChow, NanchangGame, ShangraoGame, deal_bonus
from deploy.play.jiangxi_shapes import HONOR_CHOW_ID, can_form_spaced
from deploy.play.tests.test_hongkong import counts
from deploy.play.tests.test_jinyun import pass_offer
from deploy.play.tests.test_northeast_bao import table


ORDINARY = [0,1,2,9,10,11,18,19,20,27,27,27,8,8]
SEVEN_STARS = [0,4,8,10,13,16,20,*range(27,34)]
BAO_MELDS = (Meld('chi',0,3),Meld('chi',9,3),Meld('peng',27,1),Meld('peng',28,2))


def with_jing(hands, *, upper=27, lower=5, melds=None):
    game = table(NanchangGame(7,0,27),hands,melds=melds,tail=(upper,lower))
    game.indicator_tile = game.wildcard_tile = upper
    game.lower_indicator_tile = lower
    game.lower_deltas = [0]*4
    for p in game.players:
        p.score = 0
    return game


class JiangxiTests(unittest.TestCase):
    def test_nanchang_noncontiguous_wind_chow_can_be_called_and_displayed(self):
        game = with_jing([[30],[27,29],[],[]],upper=10)
        game.step(('discard',30))
        actor, actions = game.legal_actions()
        self.assertEqual(actor,1)
        option = ('chi',HONOR_CHOW_ID + 2)
        self.assertIn(option,actions)
        session = HonorsSession(variant=NANCHANG.VARIANT)
        session.game = game
        self.assertEqual(session.option(option)['tiles'],['E','W','N'])
        self.assertIn(heuristic(policy_view(game,actor,actions)),actions)
        game.step(option)
        pass_offer(game)
        self.assertEqual(game.players[1].melds,[HonorChow('chi',HONOR_CHOW_ID+2,0)])
        self.assertEqual(game.physical_counts(),game.tile_supply)

    def test_nanchang_honor_runs_are_not_a_shangrao_shape(self):
        hand = counts([0,1,2,9,10,11,18,19,20,27,29,30,33,33])
        ctx = WinContext('zimo',33,27,27)
        self.assertIsNotNone(NANCHANG.faan(hand,(),ctx,wildcard=4))
        self.assertIsNone(SHANGRAO.faan(hand,(),ctx,wildcard=5))

    def test_nanchang_printed_jing_can_score_german_and_remain_awards(self):
        hand = counts(ORDINARY)
        ctx = WinContext('zimo',8,27,27)
        score = NANCHANG.faan(hand,(),ctx,wildcard=10)
        self.assertEqual(score.patterns,('平胡','自摸','德国'))
        self.assertEqual((score.base,score.extra),(4,5))
        self.assertEqual(NanchangGame.jing_credit(hand,(),10),3)

    def test_nanchang_german_promotes_at_settlement_when_no_player_has_upper_jing(self):
        game = with_jing([ORDINARY,[],[],[]],upper=13)
        kept = tuple(game.wall[:34])
        for p in game.players[1:]:
            for face in (13,14):
                replacement = next(i for i in range(34,len(game.wall)) if game.wall[i] not in (13,14))
                while p.hand[face]:
                    t = game.wall[replacement]
                    p.hand[t] += 1
                    p.hand[face] -= 1
                    game.wall[replacement] = face
                    if p.hand[face]:
                        replacement = next(i for i in range(34,len(game.wall)) if game.wall[i] not in (13,14))
        game.opening = False
        game.step(ZIMO)
        self.assertIn('德中德',game.win_details[0].patterns)
        self.assertEqual(game.win_details[0].payments,(0,21,21,21))
        self.assertEqual(tuple(game.wall[:34]),kept)
        self.assertEqual(game.physical_counts(),game.tile_supply)

    def test_non_uniform_rank_spacing_and_seven_actual_honors(self):
        ctx = WinContext('zimo',20,27,27)
        self.assertTrue(can_form_spaced(tuple(counts(SEVEN_STARS)),0))
        hand = counts(SEVEN_STARS)
        self.assertIn('七星十三烂',NANCHANG.faan(hand,(),ctx,wildcard=31).patterns)
        hand[33] -= 1
        hand[6] += 1
        score = NANCHANG.faan(hand,(),ctx,wildcard=6)
        self.assertIn('十三烂',score.patterns)
        self.assertNotIn('七星十三烂',score.patterns)

    def test_lower_jing_is_an_independent_opening_ledger(self):
        game = NanchangGame(7,0,27)
        expected = [game.jing_credit(p.hand,p.melds,game.lower_indicator_tile) for p in game.players]
        self.assertNotEqual(expected,[0]*4)
        self.assertEqual(game.scores(),game.lower_deltas)
        self.assertEqual(sum(game.lower_deltas),0)
        self.assertEqual(game.jing_deltas,[0]*4)
        session = HonorsSession(variant=NANCHANG.VARIANT)
        session.game = game
        self.assertEqual(len(session.view()['lower_jing']),2)
        self.assertEqual(game.physical_counts(),game.tile_supply)

    def test_discarded_jing_sponsors_only_extra_threshold_reward(self):
        game = with_jing([[27],[27,28,28,29],[],[]])
        # Exercise the crossing on a controlled physical rack without fill-in tiles of either jing face.
        for face, limit in ((27,1),(28,2)):
            while game.players[1].hand[face] > limit:
                replacement = next(i for i in range(34,len(game.wall)) if game.wall[i] not in (27,28))
                replacement_face = game.wall[replacement]
                game.players[1].hand[face] -= 1
                game.players[1].hand[replacement_face] += 1
                game.wall[replacement] = face
        game.step(('discard',27))
        game.step(('chi',HONOR_CHOW_ID))
        pass_offer(game)
        self.assertEqual(game.chong_sponsors[1],0)
        deal_bonus(game,(0,5,0,0),game.jing_deltas,game.chong_sponsors)
        self.assertEqual(game.jing_deltas,[-40,60,-10,-10])
        self.assertEqual(game.physical_counts(),game.tile_supply)

    def test_nanchang_pass_value_includes_provider_dealer(self):
        game = with_jing([[4],[],ORDINARY[:-1],[]],upper=13)
        low, high = Offer(1,8,'discard'), Offer(0,8,'discard')
        score = game._win(2,low)[0]
        self.assertLess(game._pass_win_value(2,low,score),game._pass_win_value(2,high,score))
        game.players[2].declined_win_share=game._pass_win_value(2,low,score)
        game.offer=low
        game.responses.append(2)
        self.assertNotIn(RON,game.legal_actions()[1])
        game.offer=high
        self.assertIn(RON,game.legal_actions()[1])

    def test_shangrao_single_bao_eyes_accept_arbitrary_draws(self):
        for face in (8,15,33):
            game = table(ShangraoGame(4,0,27),[[5,face],[],[],[]],melds={0:BAO_MELDS},tail=(4,))
            game.indicator_tile, game.wildcard_tile = 4,5
            game.step(ZIMO)
            self.assertIn('宝吊头',game.win_details[0].patterns)
            self.assertEqual(game.win_details[0].payments,(0,2,2,2))
            self.assertEqual(game.physical_counts(),game.tile_supply)

    def test_shangrao_discard_and_added_kong_do_not_offer_ron(self):
        game = table(ShangraoGame(4,0,27),[[8],ORDINARY[:-1],[],[]],tail=(4,))
        game.indicator_tile, game.wildcard_tile = 4,5
        game.step(('discard',8))
        self.assertNotIn(RON,game.legal_actions()[1])
        self.assertIsNone(SHANGRAO.faan(counts(ORDINARY),(),WinContext('rob_kong',8,27,27),wildcard=5))
        pass_offer(game)
        self.assertEqual(game.physical_counts(),game.tile_supply)
