from collections import Counter
from itertools import combinations_with_replacement
from random import Random
import unittest

from deploy.play.fujian_travel import NINGDE_RANDOM_GOLD, QUANZHOU, FujianTravelGame, quanzhou_closed_points
from deploy.play.honors import PASS, Meld, WinContext, standard_shapes
from deploy.play.honors_game import HonorsSession
from deploy.play.longyan import TRAVEL
from deploy.play.tests.test_hongkong import counts
from deploy.play.tests.test_jinyun import pass_offer


REGULAR = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 18, 19, 20, 8, 8]


def travel_fixture():
    gold = 22
    hand = [*REGULAR[:-2], gold, 8]
    draws = [15, 16, 17, gold, 24, 25, 26, gold]
    game = FujianTravelGame(7, 0, 27, rules=QUANZHOU, dealer_streak=2)
    game.indicator_tile = game.wildcard_tile = gold
    game.flowers = [[] for _ in game.players]
    game.phase = 'action'
    game.cur = game.dealer
    game.opening = False
    supply = game.tile_supply - Counter([gold, *hand, *draws])
    extras = [tile for tile in supply.elements() if tile < 34]
    Random(2).shuffle(extras)
    for actor, player in enumerate(game.players):
        tiles = hand if actor == 0 else extras[16 * (actor - 1):16 * actor]
        if actor != 0:
            supply.subtract(tiles)
        player.hand = counts(tiles)
    game.wall = [*supply.elements(), *reversed(draws)]
    game.drawn_tile = hand[-1]
    return game


class FujianTravelTests(unittest.TestCase):
    def test_quanzhou_pungs_take_the_highest_complete_decomposition(self):
        hand = counts([0] * 3 + [1] * 3 + [2] * 3 + [27] * 3 + [29] * 3 + [17, 17])
        interpretations = list(standard_shapes(hand, 0, groups=5))
        self.assertGreater(len(interpretations), 1)
        self.assertEqual(quanzhou_closed_points(tuple(hand), 0), 7)
        hand[29] -= 1
        self.assertEqual(quanzhou_closed_points(tuple(hand), 1), 7)

    def test_weighted_search_matches_exhaustive_gold_assignment(self):
        rng = Random(181)
        for wild in range(3):
            for index in range(12):
                complete = REGULAR.copy() if index % 2 else [0] * 3 + [1] * 3 + [2] * 3 + [27] * 3 + [29] * 3 + [17, 17]
                rng.shuffle(complete)
                if index % 3 == 0:
                    complete[-1] = rng.randrange(34)
                natural = complete[wild:]
                expected = max((sum(1 + (m.tile >= 27) for m in melds if m.kind != 'chi')
                                for assignment in combinations_with_replacement(range(34), wild)
                                for _, melds in standard_shapes(counts([*natural, *assignment]), 0, groups=5)), default=None)
                self.assertEqual(quanzhou_closed_points(tuple(counts(natural)), wild), expected)

    def test_quanzhou_kong_contributes_to_both_subtotals_and_dealer_bill_only(self):
        game = FujianTravelGame(7, 0, 27, rules=QUANZHOU, dealer_streak=2)
        game.wildcard_tile = 22
        hand = counts(REGULAR[3:])
        kong = (Meld('gang_an', 27),)
        score = QUANZHOU.score(game, hand, kong, WinContext('ron', 8, 28, 27), ())
        self.assertEqual(score.patterns[:3], ('平胡', '刻分x4', '杠分x3'))
        self.assertEqual(score.share, 8)
        self.assertEqual(QUANZHOU.payments(game, 1, 0, score), (10, 0, 0, 0))
        self.assertEqual(QUANZHOU.payments(game, 1, 2, score), (0, 0, 8, 0))
        self.assertEqual(QUANZHOU.next_dealer(game.dealer, [2], None), 2)

    def test_third_travel_settles_immediately_after_the_third_physical_discard(self):
        game = travel_fixture()
        for level, tile in ((1, 8), (2, 22), (3, 22)):
            self.assertIn((TRAVEL, tile), game.legal_actions()[1])
            game.step((TRAVEL, tile))
            self.assertEqual(game.travel_level, level)
            if level < 3:
                pass_offer(game)
                for actor in range(1, 4):
                    self.assertEqual(game.cur, actor)
                    game.step(('discard', game.drawn_tile))
                    if level == 2:
                        self.assertEqual(game.legal_actions()[1], [PASS])
                    pass_offer(game)
        self.assertEqual((game.phase, game.reason_code), ('over', 'third_travel_win'))
        self.assertEqual(game.discards[0], [8, 22, 22])
        self.assertEqual(sum(game.revealed[0]), 16)
        self.assertEqual(game.win_details[0].payments, (0, 36, 36, 36))
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_eighth_flower_wins_if_it_consumes_the_final_live_tile(self):
        game = travel_fixture()
        for flower in range(34, 41):
            game.wall.remove(flower)
            game.flowers[1].append(flower)
        game.wall.remove(41)
        reserve = game.rules.WALL_RESERVE
        game.discards[0].extend(game.wall[reserve:])
        game.wall = [*game.wall[:reserve], 41]
        game._draw(1)
        self.assertEqual(game.winners, [1])
        self.assertEqual(game.reason_code, 'flower_hu')
        self.assertEqual(game.win_details[0].patterns[0], '八花')
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_ningde_water_awards_count_subtotal_and_highest_special(self):
        game = FujianTravelGame(1, 1, 27, rules=NINGDE_RANDOM_GOLD, dealer_streak=2)
        game.wildcard_tile = 22
        hand = counts(REGULAR)
        ron, zimo = WinContext('ron', 8, 28, 27), WinContext('zimo', 8, 28, 27)
        self.assertEqual(NINGDE_RANDOM_GOLD.score(game, hand, (), ron, ()).share, 32)
        score = NINGDE_RANDOM_GOLD.score(game, hand, (), zimo, (34,))
        self.assertEqual((score.share, score.patterns), (21, ('一水胡', '水数x1', '金牌x0')))
        self.assertEqual(NINGDE_RANDOM_GOLD.score(game, hand, (), zimo, (34, 35)).share, 8)
        one_suit = counts([0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 6, 7, 8, 8, 8])
        self.assertEqual(NINGDE_RANDOM_GOLD.score(game, one_suit, (), ron, ()).share, 242)

    def test_ningde_public_indicator_and_white_dragon_interpretation(self):
        game = FujianTravelGame(1, 1, 27, rules=NINGDE_RANDOM_GOLD, dealer_streak=2)
        session = HonorsSession(variant=game.rules.VARIANT)
        session.game = game
        view = session.view()
        self.assertEqual([sum(p.hand) for p in game.players], [16, 17, 16, 16])
        self.assertEqual(sum(game.tile_supply.values()), 144)
        self.assertEqual(game.physical_counts(), game.tile_supply)
        self.assertEqual(view['dealer_streak'], 2)
        self.assertTrue(view['gold_indicator'])
        self.assertIsNone(view['drawn'])
        self.assertTrue(all(not p['hand'] for p in view['players'][1:]))
        hand = counts([0, 1, 31, *REGULAR[3:]])
        ctx = WinContext('ron', 31, 28, 27)
        self.assertTrue(NINGDE_RANDOM_GOLD.can_win(hand, (), ctx, gold=2))
        self.assertFalse(QUANZHOU.can_win(hand, (), ctx, gold=2))
