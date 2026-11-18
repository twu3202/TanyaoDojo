import unittest
from unittest.mock import patch

from deploy.play.changchun import BAO_WIN, Egg
from deploy.play.dalian_tieling import DALIAN, DALIAN_PEEK, INSIDE_READY, TIELING, DalianGame, TielingGame
from deploy.play.honors import PASS, RON, ZIMO, Meld, WinContext
from deploy.play.honors_game import HonorsSession
from deploy.play.honors_policy import policy_view
from deploy.play.tests.test_jinyun import pass_offer
from deploy.play.tests.test_northeast_bao import CHANGCHUN_WAIT, table


DALIAN_WAIT = [0, 1, 2, 9, 10, 11, 18, 18, 18, 19]
DALIAN_PUNG = Meld('peng', 27, 3)


def pass_cycle(game, tiles=None):
    for seat in (1, 2, 3):
        actor, options = game.legal_actions()
        assert actor == seat
        move = next(a for a in options if a[0] == 'discard' and (tiles is None or a[1] == tiles[seat - 1]))
        game.step(move)
        pass_offer(game)


class DalianTielingTests(unittest.TestCase):
    def test_dalian_inside_ready_locks_only_edge_wait_and_can_win_without_a_draw(self):
        game = table(DalianGame(2,0,27), [[*DALIAN_WAIT,26],[],[],[]],
                     melds={0:(DALIAN_PUNG,)}, tail=(20,))
        self.assertIn(('declare_ready', 26), game.legal_actions()[1])
        game.step((INSIDE_READY, 26))
        self.assertEqual(game.declared_waits[0], (20,))
        self.assertEqual(game.bao_view(0), ())
        pass_offer(game)
        pass_cycle(game)
        self.assertEqual(game.legal_actions(), (0, [PASS, BAO_WIN]))
        self.assertIsNone(game.drawn_tile)
        self.assertEqual(game.bao_view(0), ((None,20),))
        self.assertTrue(all(not game.bao_view(seat) for seat in (1,2,3)))
        session = HonorsSession(variant=DALIAN.VARIANT)
        session.game = game
        self.assertEqual(session.view()['bao_tiles'], [{'owner':None,'tile':'3s'}])
        reserve = game.wall[:14]
        game.step(BAO_WIN)
        self.assertEqual(game.reason_code, DALIAN_PEEK)
        self.assertEqual(game.win_details[0].patterns, ('自摸', '夹胡', '冲宝'))
        self.assertEqual(game.win_details[0].payments, (0,128,128,128))
        self.assertEqual(game.wall[:14], reserve)
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_dalian_third_actual_discard_rotates_the_reserved_bao(self):
        game = table(DalianGame(2,0,27), [[*DALIAN_WAIT,26],[20],[20],[20]],
                     melds={0:(DALIAN_PUNG,)}, tail=(20,12,22))
        game.bao_slot = game.bao_seen[0] = 0
        game.step(('declare_ready', 26))
        self.assertEqual(game.declared_waits[0], (19,20))
        pass_offer(game)
        pass_cycle(game, (20,20,20))
        self.assertEqual(game.discarded_faces[20], 3)
        self.assertEqual(game.bao_slot, 2)
        self.assertEqual(game.bao_view(0), ((None,22),))
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_dalian_visible_pung_does_not_rotate_bao_without_three_discards(self):
        game = table(DalianGame(2,0,27), [[*DALIAN_WAIT,26],[],[],[]],
                     melds={0:(DALIAN_PUNG,),1:(Meld('peng',20,2),)}, tail=(20,))
        game.bao_slot = game.bao_seen[0] = 0
        game.discarded_faces[20] = 1
        actor, moves = game.legal_actions()
        self.assertEqual(policy_view(game,actor,moves).visible[20], 3)
        self.assertEqual(game.bao_view(0), ((None,20),))
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_dalian_missed_peek_then_actual_draw_of_bao_fills_a_locked_wait(self):
        game = table(DalianGame(2,0,27), [[*DALIAN_WAIT,26],[],[],[]],
                     melds={0:(DALIAN_PUNG,)}, tail=(22,), draws=(3,4,5,22))
        game.step((INSIDE_READY, 26))
        pass_offer(game)
        pass_cycle(game)
        self.assertEqual(game.drawn_tile, 22)
        self.assertIn(ZIMO, game.legal_actions()[1])
        game.step(ZIMO)
        self.assertEqual(game.win_details[0].patterns, ('自摸', '夹胡', '摸宝'))
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_dalian_ordinary_ron_charges_just_the_supplier(self):
        game = table(DalianGame(2,0,27), [[19],DALIAN_WAIT,[],[]], melds={1:(DALIAN_PUNG,)})
        game.step(('discard', 19))
        game.step(RON)
        pass_offer(game)
        self.assertEqual(game.win_details[0].patterns, ('点炮',))
        self.assertEqual(game.win_details[0].payments, (16,0,0,0))
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_tieling_unready_closed_hand_cannot_make_a_concealed_kong(self):
        closed = table(TielingGame(7,0,27), [[27] * 4,[],[],[]])
        opened = table(TielingGame(7,0,27), [[27] * 4,[],[],[]], melds={0:(Meld('peng',33,3),)})
        self.assertNotIn(('ankan',27), closed.legal_actions()[1])
        self.assertIn(('ankan',27), opened.legal_actions()[1])

    def test_tieling_closed_ready_keeps_fan_after_wait_preserving_kong(self):
        baseline = table(TielingGame(7,0,27), [[*CHANGCHUN_WAIT,33],[],[],[]])
        self.assertNotIn(ZIMO, baseline.legal_actions()[1])
        game = table(TielingGame(7,0,27), [[*CHANGCHUN_WAIT,26],[],[],[]], tail=(33,), draws=(3,4,5,27))
        game.step(('declare_ready', 26))
        pass_offer(game)
        pass_cycle(game)
        self.assertEqual(game.drawn_tile, 27)
        self.assertEqual(game.declared_waits[0], (33,))
        self.assertIn(('ankan',27), game.legal_actions()[1])
        game.step(('ankan',27))
        pass_offer(game)
        self.assertEqual(game.drawn_tile, 33)
        game.step(ZIMO)
        self.assertTrue({'闷听','杠上开花','暗杠'} <= set(game.win_details[0].patterns))
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_tieling_longmao_carries_into_next_hand_through_session(self):
        session = HonorsSession(seed=7, variant=TIELING.VARIANT)
        session.game = TielingGame(7,0,27)
        for prior, expected in (('draw', True),('suited_self_draw', False)):
            session.game.reason_code = prior
            session.status = 'round_end'
            with patch.object(session, 'start_worker'):
                session.begin()
            self.assertEqual(session.view()['longmao'], expected)

    def test_tieling_matching_four_first_discards_trigger_one_fan_without_transfers(self):
        game = table(TielingGame(7,0,27), [[8],[8],[8],[8]])
        game.step(('discard',8))
        pass_offer(game)
        for seat in (1, 2, 3):
            self.assertFalse(game.long_hair)
            game.step(('discard',8))
            pass_offer(game)
        self.assertTrue(game.long_hair)
        self.assertEqual(game.scores(), [0,0,0,0])
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_tieling_other_seats_dragon_egg_adds_two_fan_to_an_arrow_pung(self):
        dragon_egg = Egg('egg_dragons', 31, material=(31,32,33))
        game = table(TielingGame(7,0,27,previous_draw=True),
                     [[0,1,2,9,10,11,18,19,20,8,8],[],[],[]],
                     melds={0:(Meld('peng',33,1),),2:(dragon_egg,)})
        game.split_draw = True
        game.step(ZIMO)
        win = game.win_details[0]
        self.assertTrue({'喜杠加番','箭刻','海底捞月','长毛'} <= set(win.patterns))
        self.assertEqual(win.fan, 7)
        self.assertEqual(game.physical_counts(), game.tile_supply)
