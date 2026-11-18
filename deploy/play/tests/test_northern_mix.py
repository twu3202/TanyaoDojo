import unittest
from unittest.mock import patch

from deploy.play.honors import PASS, RON, ZIMO, Meld, WinContext
from deploy.play.honors_game import HonorsSession
from deploy.play.honors_policy import heuristic, policy_view
from deploy.play.northern_mix import BEIJING, RAISE_PHASE, TIANJIN, BeijingGame, TianjinGame
from deploy.play.tests.test_hongkong import counts
from deploy.play.tests.test_jiangxi_natural import NORMAL_READY
from deploy.play.tests.test_jinyun import pass_offer
from deploy.play.tests.test_northeast_bao import table


PAIR = [28,28]
DRAGON = list(range(9,18))
NO_HUN = [0,1,2,14,15,16,17,17,17,19,20,21,*PAIR]


def raised_game(seed=4, dealer=0, stakes=(1,2,4)):
    game = TianjinGame(seed,dealer,27)
    for level in stakes:
        game.step((RAISE_PHASE,level))
    return game


def tianjin_fixture(hands, *, dealer=0, indicator=12, melds=None, draws=()):
    game = table(raised_game(dealer=dealer),hands,melds=melds,draws=draws,tail=(indicator,))
    game.indicator_tile = game.wildcard_indicator = indicator
    game.wildcard_tile = TIANJIN.mixed_faces(indicator)[1]
    return game


def beijing_fixture(hands, *, dealer=0, wildcard=10, melds=None, previous_draw=False):
    game = table(BeijingGame(4,dealer,27,previous_draw=previous_draw),hands,melds=melds)
    game.wildcard_tile = wildcard
    return game


class TianjinTests(unittest.TestCase):
    def test_raises_are_blind_and_only_between_each_guest_and_dealer(self):
        game = TianjinGame(4,3,27)
        session = HonorsSession(variant=TIANJIN.VARIANT)
        session.game = game
        for seat, level in enumerate((4,1,2)):
            actor, actions = game.legal_actions()
            self.assertEqual(actor,seat)
            view = policy_view(game,actor,actions)
            self.assertEqual(heuristic(view),(RAISE_PHASE,1))
            self.assertEqual(sum(view.hand)+sum(view.visible),0)
            self.assertIsNone(view.wildcard_indicator)
            wire = session.view()
            self.assertNotIn('hun_faces',wire)
            self.assertNotIn('flip_roll',wire)
            self.assertTrue(all(not p['hand'] and not p['hand_count'] for p in wire['players']))
            self.assertTrue(all('dealer_raise' not in p for p in wire['players']))
            game.step((RAISE_PHASE,level))
        self.assertEqual(session.view()['players'][0]['dealer_raise'],4)
        self.assertEqual([game.dealer_factor(3,s) for s in range(3)],[8,2,4])
        self.assertEqual(game.dealer_factor(0,1),1)
        self.assertEqual(game.physical_counts(),game.tile_supply)

    def test_dead_basic_hand_and_no_joker_minimum(self):
        ready = [0,1,2,14,15,16,17,17,17,19,20,12,*PAIR]
        ctx = WinContext('zimo',20,27,27)
        self.assertIsNone(TIANJIN.faan(counts(ready),(),ctx,indicator=12))
        self.assertEqual(TIANJIN.faan(counts(NO_HUN),(),ctx,indicator=12).share,2)
        self.assertEqual(TIANJIN.faan(counts(ready),(),WinContext('zimo',20,27,27,kong_chain=1),indicator=12).share,2)
        self.assertIsNone(TIANJIN.faan(counts(NO_HUN),(),WinContext('ron',20,27,27),indicator=12))

    def test_single_and_double_hun_require_pre_draw_groups(self):
        single = [0,1,2,14,15,16,17,17,17,19,20,21,12,31]
        double = [0,1,2,14,15,16,17,17,17,*PAIR,12,13,31]
        ctx = WinContext('zimo',31,27,27)
        self.assertEqual(TIANJIN.faan(counts(single),(),ctx,indicator=12).patterns,('混吊',))
        self.assertEqual(TIANJIN.faan(counts(double),(),ctx,indicator=12).patterns,('双混吊',))

    def test_catch_five_and_dragon_share_one_valid_decomposition(self):
        hand = counts([*DRAGON,3,4,5,*PAIR])
        score = TIANJIN.faan(hand,(),WinContext('zimo',4,27,27),indicator=20)
        self.assertEqual((score.share,score.patterns),(14,('龙','捉五','素提溜')))
        self.assertEqual(TIANJIN.faan(hand,(),WinContext('zimo',5,27,27),indicator=20).share,8)
        native = TIANJIN.faan(hand,(),WinContext('zimo',4,27,27),indicator=12)
        self.assertEqual((native.share,native.patterns),(14,('本混龙','捉五')))
        hand = counts([*range(9),17,17,17,*PAIR])
        self.assertEqual(TIANJIN.faan(hand,(),WinContext('zimo',4,27,27),indicator=20).share,14)

    def test_discarding_either_hun_locks_and_cannot_be_called(self):
        for face in (12,13):
            game = tianjin_fixture([[face],[face,face],[],[]])
            game.step(('discard',face))
            self.assertEqual(game.sealed,{0})
            for _ in range(3):
                self.assertEqual(game.legal_actions()[1],[PASS])
                game.step(PASS)
            self.assertIsNone(game._score(0,counts(NO_HUN),(),WinContext('zimo',20,27,27)))

    def test_identical_jokers_make_a_golden_kong_with_pairwise_stakes(self):
        hand = [13]*4 + [0,1,2,14,15,16,17,17,17,28]
        g = tianjin_fixture([hand,[],[],[]],draws=())
        self.assertIn(('ankan',13),g.legal_actions()[1])
        g.step(('ankan',13))
        pass_offer(g)
        self.assertEqual(g.kong_deltas,[112,-16,-32,-64])
        self.assertEqual(g.physical_counts(),g.tile_supply)


class BeijingTests(unittest.TestCase):
    def test_exposed_and_three_joker_hands_require_self_draw(self):
        natural = counts([*NORMAL_READY,4])
        ron = WinContext('ron',4,27,27)
        self.assertIsNotNone(BEIJING.faan(natural,(),ron,wildcard=6))
        chow = (Meld('chi',0,3),)
        hand = counts([*NORMAL_READY[3:],4])
        self.assertIsNone(BEIJING.faan(hand,chow,ron,wildcard=6))
        self.assertIsNotNone(BEIJING.faan(hand,chow,WinContext('zimo',4,27,27),wildcard=6))
        mixed = counts([0,1,2,9,10,11,18,19,20,27,27,6,6,6])
        self.assertIsNone(BEIJING.faan(mixed,(),ron,wildcard=6))
        self.assertIsNotNone(BEIJING.faan(mixed,(),WinContext('zimo',6,27,27),wildcard=6))

    def test_four_jokers_force_a_win_without_dealer_doubling(self):
        hand = [10]*4 + [0,2,4,6,8,9,12,15,28,33]
        g = beijing_fixture([hand,[],[],[]])
        self.assertEqual(g.legal_actions()[1],[ZIMO])
        g.step(ZIMO)
        self.assertEqual(g.win_details[0].patterns,('混杠',))
        self.assertEqual(g.win_details[0].payments,(0,20,20,20))

    def test_joker_cannot_be_discarded_or_used_to_chow(self):
        g = beijing_fixture([[11],[9,10],[],[]])
        self.assertNotIn(('discard',10),g.legal_actions()[1])
        g.step(('discard',11))
        self.assertNotIn(('chi',9),g.legal_actions()[1])

    def test_upstairs_discard_win_consolidates_all_three_bills(self):
        g = beijing_fixture([[4],NORMAL_READY,[],[]],wildcard=6,previous_draw=True)
        g.opening = False
        g.step(('discard',4))
        g.step(RON)
        pass_offer(g)
        win = g.win_details[0]
        self.assertEqual(win.points,8)
        self.assertIn('上楼',win.patterns)
        self.assertEqual(win.payments,(32,0,0,0))
        self.assertEqual(g.physical_counts(),g.tile_supply)

    def test_rob_concealed_requires_actual_drawn_rank_inside_the_straight(self):
        inside = [t for t in DRAGON if t!=13] + [18,19,20,27,27]
        outside = [*range(9),12,14,27,27]
        for hand,legal in ((inside,True),(outside,False)):
            g = beijing_fixture([[13]*4,hand,[],[]],wildcard=25)
            g.step(('ankan',13))
            self.assertEqual(RON in g.legal_actions()[1],legal)
            if legal:
                g.step(RON)
                pass_offer(g)
                self.assertEqual(g.reason_code,'beijing_dragon_rob')
                self.assertIn('一条龙',g.win_details[0].patterns)
                self.assertEqual(g.kong_deltas,[0]*4)
                self.assertEqual(g.physical_counts(),g.tile_supply)

    def test_burned_hand_next_goes_upstairs_and_only_fourth_dealer_hand_doubles(self):
        g = beijing_fixture([[31],[31],[31],[31]])
        for _ in range(4):
            g.step(('discard',31))
            pass_offer(g)
        self.assertEqual(g.reason_code,'beijing_burn_draw')
        self.assertEqual(g.next_dealer,g.dealer)
        session = HonorsSession(variant=BEIJING.VARIANT)
        session.game = g
        session.status = 'round_end'
        with patch.object(session,'start_worker'):
            session.begin()
        self.assertTrue(session.game.upstairs)
        self.assertTrue(BeijingGame(4,0,27,dealer_streak=3).upstairs)
        self.assertFalse(BeijingGame(4,0,27,dealer_streak=4).upstairs)

    def test_only_the_winner_collects_completed_kongs_at_settlement(self):
        groups = {0:(Meld('gang_an',31),),1:(Meld('gang_ming',8,2),)}
        rack = [0,1,2,9,10,11,18,19,20,27,27]
        g = beijing_fixture([rack,[],[],[]],wildcard=6,melds=groups,previous_draw=True)
        g.opening = False
        self.assertEqual(g.kong_deltas,[0]*4)
        g.step(ZIMO)
        self.assertEqual(g.kong_deltas,[12,-4,-4,-4])
        self.assertEqual(g.physical_counts(),g.tile_supply)
