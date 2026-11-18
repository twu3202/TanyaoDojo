from collections import Counter
from random import Random
import unittest

from deploy.play.changchun import BAO_WIN, CHANGCHUN, ChangchunGame, Egg
from deploy.play.honors import RON, ZIMO, Meld
from deploy.play.honors_game import HonorsSession
from deploy.play.honors_policy import heuristic, policy_view
from deploy.play.northeast import HARBIN, RED, HarbinGame
from deploy.play.tests.test_hongkong import counts
from deploy.play.tests.test_jinyun import pass_offer


HARBIN_WAIT = [0, 1, 2, 12, 12, 12, 19, 21, RED, RED]
CHANGCHUN_WAIT = [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27, RED]


def table(game, hands, *, melds=None, tail=(), draws=(), seed=1):
    melds = melds or {}
    used = Counter([*tail, *draws, *(tile for hand in hands for tile in hand),
                    *(tile for sets in melds.values() for meld in sets for tile in meld.tiles)])
    assert used <= game.tile_supply
    leftovers = list((game.tile_supply - used).elements())
    Random(seed).shuffle(leftovers)
    for actor, (player, tiles) in enumerate(zip(game.players, hands, strict=True)):
        player.melds = list(melds.get(actor, ()))
        need = 13 + (actor == game.dealer) - 3 * len(player.melds) - len(tiles)
        player.hand = counts([*tiles, *(leftovers.pop() for _ in range(need))])
    game.wall = [*tail, *leftovers, *reversed(draws)]
    game.drawn_tile = hands[game.dealer][-1]
    if isinstance(game, ChangchunGame):
        game.egg_racks = [p.hand.copy() for p in game.players]
    return game


class NortheastBaoTests(unittest.TestCase):
    def test_harbin_can_ready_chow_from_two_seats_away_over_a_direct_kong(self):
        game = table(HarbinGame(1, 0, 27), [[9], [9, 9, 9], [*HARBIN_WAIT, 10, 11, 26], []])
        game.step(('discard', 9))
        self.assertIn(('zhigang', None), game.legal_actions()[1])
        game.step(('zhigang', None))
        self.assertIn(('ready_chi', 9), game.legal_actions()[1])
        game.step(('ready_chi', 9))
        pass_offer(game)
        actor, actions = game.legal_actions()
        self.assertEqual(actor, 2)
        self.assertTrue(all(a[0] == 'declare_ready' for a in actions))
        self.assertIn(('declare_ready', 26), actions)
        game.step(('declare_ready', 26))
        self.assertEqual(game.declared_waits[actor], (20,))
        self.assertEqual(game.players[actor].melds, [Meld('chi', 9, 0)])
        self.assertEqual(game.bao_owners, [2])
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_harbin_view_and_own_middle_bao_are_scoped_and_paid(self):
        game = table(HarbinGame(1, 0, 27), [[*HARBIN_WAIT, 26], [], [], []],
                     melds={0: (Meld('chi', 9, 3),)}, tail=(20, 15, 14, 13), draws=(20,))
        self.assertNotIn(ZIMO, game.legal_actions()[1])
        session = HonorsSession(variant=HARBIN.VARIANT)
        session.game = game
        self.assertEqual(session.view()['bao_tiles'], [])
        game.step(('declare_ready', 26))
        self.assertEqual(session.view()['bao_tiles'], [{'owner': 0, 'tile': '3s'}])
        self.assertTrue(session.view()['players'][0]['river'][-1]['ready'])
        self.assertEqual(game.bao_view(1), ())
        # Turn to the ready seat with the next actual drawn tile below the tail.
        game.offer = None
        game._draw(0)
        self.assertNotIn(('discard', 0), game.legal_actions()[1])
        self.assertIn(ZIMO, game.legal_actions()[1])
        game.step(ZIMO)
        self.assertEqual(game.win_details[0].patterns, ('大宝中宝', '卡当'))
        self.assertEqual(game.win_details[0].payments, (0, 16, 16, 16))
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_harbin_other_slot_matches_only_on_a_draw_and_stays_fixed(self):
        game = table(HarbinGame(1, 0, 27), [[*HARBIN_WAIT, 26], [], [], []],
                     melds={0: (Meld('chi', 9, 3),)}, tail=(20, 15, 14, 13), draws=(20,))
        game.bao_owners.append(2)
        game.declared_waits[2] = (6,)
        self.assertEqual(game.bao_view(0), ())
        game.step(('declare_ready', 26))
        self.assertEqual(game.bao_view(0), ((2, 20), (0, 15)))
        self.assertEqual(game.bao_view(2), game.bao_view(0))
        self.assertEqual(game.bao_view(1), ())
        game.offer = None
        game._draw(0)
        game.step(ZIMO)
        self.assertEqual(game.win_details[0].patterns[0], '小宝中宝')
        self.assertEqual(game.win_details[0].payments, (0, 8, 8, 8))
        self.assertEqual(game.wall[:4], [20, 15, 14, 13])

    def test_changchun_private_match_precedes_draw_and_pays_nonselfdraw_fan(self):
        game = table(ChangchunGame(1, 0, 27), [[*CHANGCHUN_WAIT, 26], [], [], []], tail=(RED,))
        session = HonorsSession(variant=CHANGCHUN.VARIANT)
        session.game = game
        game.bao_slot = 0
        game.bao_history.add(0)
        game.step(('declare_ready', 26))
        self.assertEqual(session.view()['bao_tiles'], [])
        game.offer = None
        wall = game.wall.copy()
        game._draw(0)
        self.assertEqual((game.phase, game.drawn_tile), ('bao_peek', None))
        self.assertEqual(session.view()['bao_tiles'], [{'owner': None, 'tile': 'C'}])
        self.assertEqual(game.bao_view(1), ())
        self.assertEqual(game.wall, wall)
        self.assertEqual(heuristic(policy_view(game, *game.legal_actions())), BAO_WIN)
        game.step(BAO_WIN)
        self.assertEqual(game.reason_code, 'bao_peek_win')
        self.assertEqual(game.win_details[0].payments, (0, 32, 32, 32))
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_changchun_rotates_only_when_three_copies_are_public(self):
        game = table(ChangchunGame(1, 0, 27), [[*CHANGCHUN_WAIT, 25], [], [], []], tail=(26, RED))
        game.bao_slot = 0
        game.bao_history.add(0)
        game.bao_seen[0] = 0
        game.declared_waits[0] = (RED,)
        game._refresh_bao()
        self.assertEqual(game.bao_slot, 0)
        game.discards[2] = [26, 26, 26]
        for player in game.players:
            player.hand[26] = 0
        game.wall[1:] = [tile for tile in game.wall[1:] if tile != 26]
        game._refresh_bao()
        self.assertNotEqual(game.bao_slot, 0)
        self.assertEqual(game.wall[0], 26)
        self.assertEqual(game.bao_view(0), ())
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_changchun_opponent_peek_does_not_publish_the_private_phase(self):
        game = table(ChangchunGame(1, 0, 27), [[24], CHANGCHUN_WAIT, [], []], tail=(RED,))
        game.bao_slot = 0
        game.bao_history.add(0)
        game.declared_waits[1] = (RED,)
        game._draw(1)
        self.assertEqual(game.phase, 'bao_peek')
        session = HonorsSession(variant=CHANGCHUN.VARIANT)
        session.game = game
        self.assertEqual((session.view()['phase'], session.view()['bao_tiles']), ('action', []))

    def test_changchun_bao_on_discard_keeps_its_face_but_a_draw_can_fill_the_wait(self):
        game = table(ChangchunGame(1, 0, 27), [[4], CHANGCHUN_WAIT, [], []], tail=(4,), draws=(4,))
        game.bao_slot = 0
        game.bao_history.add(0)
        game.bao_seen[1] = 0
        game.declared_waits[1] = (RED,)
        game.step(('discard', 4))
        self.assertNotIn(RON, game.legal_actions()[1])
        pass_offer(game)
        self.assertEqual((game.cur, game.drawn_tile), (1, 4))
        game.step(ZIMO)
        self.assertIn('摸宝', game.win_details[0].patterns)
        self.assertEqual(game.revealed[1][4], 1)
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_changchun_opens_a_chicken_dragon_egg_and_can_rob_an_extension(self):
        game = table(ChangchunGame(1, 0, 27), [[31, 32, 18], [], [], []])
        self.assertIn(('egg_dragons', RED), game.legal_actions()[1])
        game.step(('egg_dragons', RED))
        egg, = game.players[0].melds
        self.assertEqual((egg.tiles, egg.substitutions), ((31, 32, 18), 1))
        self.assertEqual(sum(game.players[0].hand), 11)
        self.assertEqual(game.kong_deltas, [3, -1, -1, -1])
        self.assertEqual(game.physical_counts(), game.tile_supply)

        egg = Egg('egg_winds', 27, material=(27, 28, 29, 30))
        game = table(ChangchunGame(1, 0, 27), [[27], [27, 27], [], []], melds={0: (egg,)})
        game.egg_racks[0] = None
        game.step(('extend_egg', (0, 27)))
        self.assertIn(('rob_egg_pon', None), game.legal_actions()[1])
        game.step(('rob_egg_pon', None))
        pass_offer(game)
        self.assertEqual(game.players[0].melds, [egg])
        self.assertEqual(game.players[1].melds, [Meld('peng', 27, 0)])
        self.assertEqual(game.kong_deltas, [0, 0, 0, 0])
        self.assertEqual(game.physical_counts(), game.tile_supply)
