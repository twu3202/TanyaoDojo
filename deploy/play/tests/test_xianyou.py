from collections import Counter
import unittest

from deploy.play.honors import PASS, RON, ZIMO, Meld, WinContext
from deploy.play.honors_game import HonorsSession
from deploy.play.honors_policy import heuristic, policy_view
from deploy.play.longyan import TRAVEL
from deploy.play.tests.test_hongkong import counts
from deploy.play.tests.test_jinyun import pass_offer
from deploy.play.xianyou import XIANYOU_FLOWER, XIANYOU_ROOMS, XIANYOU_SELF_DRAW, XIANYOU_SUBSTITUTE, WHITE, XianyouGame


REGULAR = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 18, 19, 20, 8, 8]


def table(room, hands, draws=(), *, gold=22):
    game = XianyouGame(9, 0, 27, rules=room)
    game.phase = 'action'
    game.cur = game.dealer
    game.wildcard_tile = gold
    game.indicator_tile = None if room.FIXED_GOLD is not None else (gold // 9) * 9 + (gold - 1) % 9
    game.flowers = [[] for _ in game.players]
    taken = Counter([*draws, *(tile for hand in hands for tile in hand)])
    if game.indicator_tile is not None:
        taken[game.indicator_tile] += 1
    assert taken <= game.tile_supply
    game.wall = list((game.tile_supply - taken).elements())
    available = [tile for tile in game.wall if tile not in game.flower_kinds and tile != gold]
    for actor, (player, given) in enumerate(zip(game.players, hands, strict=True)):
        extras = [available.pop() for _ in range(16 + (actor == game.dealer) - len(given))]
        for tile in extras:
            game.wall.remove(tile)
        player.hand = counts([*given, *extras])
    game.wall.extend(reversed(draws))
    game.drawn_tile = hands[0][-1]
    return game


class XianyouTests(unittest.TestCase):
    def test_distinct_tile_supplies_and_public_views(self):
        for rules, total in zip(XIANYOU_ROOMS, (120, 120, 112), strict=True):
            game = XianyouGame(5, 1, 27, rules=rules)
            session = HonorsSession(variant=rules.VARIANT)
            session.game = game
            view = session.view()
            self.assertEqual([sum(p.hand) for p in game.players], [16, 17, 16, 16])
            self.assertEqual(sum(game.tile_supply.values()), total)
            self.assertEqual(game.physical_counts(), game.tile_supply)
            self.assertTrue(all(not p['hand'] for p in view['players'][1:]))
            self.assertIsNone(view['drawn'])
            self.assertEqual(sum(p.hand[WHITE] for p in game.players) == 0, rules.WHITE_FLOWER)
            if rules.FIXED_GOLD is None:
                self.assertNotEqual(view['wildcard_indicator'], view['wildcard'])
                self.assertEqual(view.get('white_substitute'), view['wildcard'] if rules.WHITE_PROXY else None)
            else:
                self.assertTrue(view['fixed_gold'])
            self.assertEqual(view.get('white_flower', False), rules.WHITE_FLOWER)

    def test_white_flower_draw_is_replaced_and_never_reaches_the_hand(self):
        game = table(XIANYOU_FLOWER, [[20], [], [], []], (WHITE,))
        tail = game._replacement_index()
        tile = game.wall.pop(game.wall.index(1))
        game.wall.insert(tail, tile)
        game.step(('discard', 20))
        pass_offer(game)
        self.assertEqual(game.cur, 1)
        self.assertEqual(game.drawn_tile, 1)
        self.assertEqual(game.flowers[1], [WHITE])
        self.assertEqual(game.players[1].hand[WHITE], 0)
        self.assertEqual(sum(game.players[1].hand), 17)
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_substitution_hand_and_kong_payments(self):
        substitute = counts([0, 1, WHITE, *REGULAR[3:]])
        self.assertTrue(XIANYOU_SUBSTITUTE.can_win(substitute, (), WinContext('zimo', WHITE, 28, 27), gold=2))
        self.assertFalse(XIANYOU_FLOWER.can_win(substitute, (), WinContext('zimo', WHITE, 28, 27), gold=2))
        tiles = counts(REGULAR[3:])
        kong = (Meld('gang_an', 0),)
        ctx = WinContext('zimo', 8, 28, 27)
        score = XIANYOU_FLOWER.score(tiles, kong, ctx, 22, (WHITE,))
        self.assertEqual((score.share, score.patterns[0]), (16, '白胡'))
        self.assertEqual(XIANYOU_SUBSTITUTE.score(tiles, kong, ctx, 22, ()).share, 4)
        for rules, fee in ((XIANYOU_SUBSTITUTE, 3), (XIANYOU_FLOWER, 0), (XIANYOU_SELF_DRAW, 0)):
            game = XianyouGame(2, 0, 27, rules=rules)
            game._pay_kong(1, Meld('gang_ming', 22, 3))
            self.assertEqual(game.kong_deltas, [0, fee, 0, -fee])

    def test_four_gold_deal_and_borrowed_indicator_claim(self):
        game = XianyouGame(5, 0, 27, rules=XIANYOU_FLOWER)
        actor, options = game.legal_actions()
        self.assertEqual((actor, game.indicator_tile, game.wildcard_tile), (2, 8, 0))
        self.assertIn(('four_gold', None), options)
        self.assertEqual(heuristic(policy_view(game, actor, options)), ('four_gold', None))
        game.step(('four_gold', None))
        self.assertEqual(game.win_details[0].points, (2 + 4 + len(game.flowers[2])) * 10)
        self.assertEqual(game.physical_counts(), game.tile_supply)

        game = table(XIANYOU_SUBSTITUTE, [[26], REGULAR[:-1], [], []])
        game._queue_gold_choices([(1, None)])
        game.step(('claim_gold', None))
        self.assertEqual(game.reason_code, 'gold_claim')
        self.assertEqual(game.win_details[0].payments, (2, 0, 2, 2))
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_three_gold_travel_survives_the_others_turns_and_wins_a_real_draw(self):
        waiting = [*REGULAR[:12], 18, 22, 22, 22]
        game = table(XIANYOU_FLOWER, [[*waiting, 26], [], [], []], (15, 16, 17, 23))
        self.assertIn((TRAVEL, 26), game.legal_actions()[1])
        game.step((TRAVEL, 26))
        self.assertEqual(game.travel_level, 3)
        pass_offer(game)
        for seat in range(1, 4):
            self.assertEqual(game.cur, seat)
            game.step(('discard', game.drawn_tile))
            if game.responses[0] == 0:
                self.assertEqual(game.legal_actions()[1], [PASS])
            pass_offer(game)
        self.assertEqual((game.cur, game.drawn_tile), (0, 23))
        game.step(ZIMO)
        self.assertEqual(game.win_details[0].patterns[0], '三金游')
        self.assertEqual(game.win_details[0].payments, (0, 30, 30, 30))
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_heuristic_travels_early_and_takes_the_win_late(self):
        waiting = [*REGULAR[:-2], 22, 26]
        game = table(XIANYOU_SUBSTITUTE, [waiting, [], [], []])
        actor, actions = game.legal_actions()
        self.assertIn(ZIMO, actions)
        self.assertEqual(heuristic(policy_view(game, actor, actions)), (TRAVEL, 26))
        game.wall = game.wall[:game.rules.WALL_RESERVE + 27]
        self.assertEqual(heuristic(policy_view(game, *game.legal_actions())), ZIMO)

    def test_self_draw_turns_off_ron_but_accepts_fixed_gold_after_first_discard(self):
        ready = [tile for tile in REGULAR if tile != 2]
        game = table(XIANYOU_SELF_DRAW, [[2], ready, [WHITE] * 4, []], gold=WHITE)
        self.assertIsNone(game.indicator_tile)
        game.step(('discard', 2))
        self.assertNotIn(RON, game.legal_actions()[1])
        game.step(PASS)
        self.assertIn(('four_gold', None), game.legal_actions()[1])
        game.step(('four_gold', None))
        self.assertEqual(game.win_details[0].payments, (10, 10, 0, 10))
        self.assertEqual(game.discards[0], [2])
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_two_gold_pair_uses_the_ordinary_self_draw_rate(self):
        rack = counts([*REGULAR[:-2], 22, 22])
        score = XIANYOU_SUBSTITUTE.score(rack, (), WinContext('zimo', 22, 28, 27), 22, ())
        self.assertEqual((score.share, score.patterns[0]), (2, '二金倒'))
