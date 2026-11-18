from collections import Counter
import random
import unittest

from deploy.play.anji import ANJI, ANJI_TEA_HOUSE, EAST, FLOWER_TILES, NORTH, AnjiGame, AnjiTeaHouseGame
from deploy.play.hangzhou import WHITE
from deploy.play.honors import RON, Meld, Offer, WinContext
from deploy.play.honors_policy import _live_tiles, heuristic, policy_view
from deploy.play.tests.test_hongkong import counts


class AnjiTests(unittest.TestCase):
    def test_all_indicator_faces_leave_fifteen_flowers(self):
        seen = set()
        for seed in range(900):
            game = AnjiGame(seed, seed % 4, 27, rules=ANJI)
            indicator = game.wildcard_indicator
            if indicator in seen:
                continue
            seen.add(indicator)
            expect = Counter({tile: 1 for tile in FLOWER_TILES})
            expect[WHITE] = 4
            if indicator == WHITE or indicator in FLOWER_TILES:
                expect[NORTH] = 4
                self.assertEqual(game.wildcard_tile, EAST)
            else:
                expect[indicator] = 4
                self.assertNotEqual(game.wildcard_tile, indicator)
            expect[indicator] -= 1
            actual = Counter(t for t in game.wall if t in game.flower_kinds)
            actual.update(tile for flowers in game.flowers for tile in flowers)
            self.assertEqual((sum(actual.values()), actual), (15, expect))
            self.assertEqual(sum(game.physical_counts().values()), 144)
            self.assertEqual(game.physical_counts(), game.tile_supply)
            self.assertTrue(all(not any(player.hand[tile] for tile in game.excluded_tiles) for player in game.players))
            if len(seen) == 42:
                break
        self.assertEqual(seen, set(range(42)))

    def test_chain_replacement_keeps_flowers_out_of_the_hand(self):
        game = AnjiGame(6, 0, EAST, rules=ANJI)
        safe = next(tile for tile in range(27) if tile not in game.excluded_tiles)
        game.wall = [35, safe, 34]
        before = game.flowers[1].copy()
        game._draw(1)
        self.assertEqual(game.flowers[1], [*before, 34, 35])
        self.assertEqual(game.drawn_tile, safe)
        self.assertEqual(game.remaining, 0)
        self.assertEqual(game.draw_flowers, 2)

    def test_wall_of_remaining_flowers_can_end_in_draw(self):
        game = AnjiGame(6, 0, EAST, rules=ANJI)
        game.wall = [WHITE, 34, 35]
        start = len(game.flowers[1])
        game._draw(1)
        self.assertEqual(len(game.flowers[1]) - start, 3)
        self.assertEqual(game.reason_code, 'draw')
        self.assertEqual(game.scores(), [0] * 4)

    def test_joker_cannot_complete_a_sequence_through_excluded_rank(self):
        hand = counts([0, 1, 4, 27, 27])
        melds = (Meld('peng', 10), Meld('chi', 18), Meld('peng', 30))
        ctx = WinContext('zimo', 27, 27, 27)
        self.assertIsNotNone(ANJI.faan(hand, melds, ctx, wildcard=4))
        self.assertIsNone(ANJI.faan(hand, melds, ctx, wildcard=4, excluded=frozenset({2, WHITE})))

    def test_public_view_and_no_flower_score(self):
        game = AnjiGame(6, 0, EAST, rules=ANJI)
        actor, actions = game.legal_actions()
        view = policy_view(game, actor, actions)
        self.assertEqual(view.excluded_tiles, game.excluded_tiles)
        self.assertTrue(all(_live_tiles(view)[tile] == 0 for tile in view.excluded_tiles))
        hand = counts([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27, 29, 29])
        ctx = WinContext('zimo', 29, 27, 27)
        self.assertEqual(ANJI.faan(hand, (), ctx, wildcard=6).share, 10)
        self.assertEqual(ANJI.faan(hand, (), ctx, wildcard=6, flowers=1).share, 2)

    def test_three_wildcards_need_a_float_or_kong_type(self):
        hand = counts([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 6, 6, 6, 23])
        ctx = WinContext('zimo', 23, 27, 27)
        self.assertIsNone(ANJI.faan(hand, (), ctx, wildcard=6))
        self.assertEqual(ANJI.faan(hand, (), ctx, wildcard=6, flower_draw=True).patterns, ('杠暴',))
        score = ANJI.faan(hand, (), ctx, wildcard=6, floats=1, kongs_before_float=1)
        self.assertEqual((score.share, score.patterns), (25, ('四财神', '财飘', '杠飘')))

    def test_added_kong_after_call_replaces_without_an_offer(self):
        game = AnjiGame(6, 0, EAST, rules=ANJI)
        tile = next(t for t in range(27) if t not in {*game.excluded_tiles, game.wildcard_tile})
        game.opening = False
        game.uninterrupted = False
        game.drawn_tile = None
        game.players[0].hand = counts([tile])
        game.players[0].melds = [Meld('peng', tile, 1)]
        action = ('bugang', tile)
        self.assertIn(action, game.legal_actions()[1])
        game.step(action)
        self.assertIsNone(game.offer)
        self.assertEqual(game.kong_chain, 1)
        self.assertEqual(game.players[0].melds[0].kind, 'gang_bu')

    def test_full_hands_conserve_physical_tiles(self):
        for rules, engine in ((ANJI, AnjiGame), (ANJI_TEA_HOUSE, AnjiTeaHouseGame)):
            for seed in range(16):
                game = engine(seed, seed % 4, EAST, rules=rules)
                rng = random.Random(seed)
                for _ in range(500):
                    if game.phase == 'over':
                        break
                    actor, moves = game.legal_actions()
                    move = heuristic(policy_view(game, actor, moves)) if seed % 2 else rng.choice(moves)
                    game.step(move)
                    self.assertEqual(game.physical_counts(), game.tile_supply)
                    self.assertEqual(sum(game.scores()), 0)
                self.assertEqual(game.phase, 'over')


class AnjiTeaHouseTests(unittest.TestCase):
    def test_only_physical_flowers_and_all_indicator_faces(self):
        seen = set()
        for seed in range(900):
            game = AnjiTeaHouseGame(seed, seed % 4, EAST, rules=ANJI_TEA_HOUSE)
            indicator = game.wildcard_indicator
            if indicator in seen:
                continue
            seen.add(indicator)
            self.assertEqual(game.flower_kinds, set(FLOWER_TILES))
            self.assertFalse(game.excluded_tiles)
            if indicator == WHITE or indicator in FLOWER_TILES:
                self.assertEqual(game.wildcard_tile, EAST)
                self.assertEqual(game.physical_counts()[WHITE], 4)
                self.assertEqual(game.physical_counts()[NORTH], 4)
            replaceable = sum(len(flowers) for flowers in game.flowers) + sum(t in FLOWER_TILES for t in game.wall)
            self.assertEqual(replaceable, 8 - (indicator in FLOWER_TILES))
            self.assertEqual(game.physical_counts(), game.tile_supply)
            if len(seen) == 42:
                break
        self.assertEqual(seen, set(range(42)))

    def test_first_draw_after_other_calls_and_no_earth_on_discard(self):
        game = AnjiTeaHouseGame(6, 0, EAST, rules=ANJI_TEA_HOUSE)
        waiting = counts([0, 1, 9, 10, 11, 18, 19, 20, 27, 27, 27, WHITE, WHITE])
        game.wildcard_tile = 6
        game.players[1].hand = waiting.copy()
        game._open_offer(Offer(0, 2, 'discard', opening='地胡'))
        self.assertNotIn(RON, game.legal_actions()[1])
        game.uninterrupted = False
        game.players[1].draws = 1
        game.flowers[1] = [34, 35]
        waiting[2] += 1
        ctx = WinContext('zimo', 2, 28, EAST)
        score = game._score(1, waiting, (), ctx)
        self.assertEqual((score.share, score.patterns), (12, ('地胡', '花牌x2')))

    def test_pon_upgrade_waits_until_own_draw(self):
        game = AnjiTeaHouseGame(6, 0, EAST, rules=ANJI_TEA_HOUSE)
        tile = next(t for t in range(27) if t != game.wildcard_tile)
        game.opening = False
        game.drawn_tile = None
        game.players[0].hand = counts([tile])
        game.players[0].melds = [Meld('peng', tile, 1)]
        action = ('bugang', tile)
        self.assertNotIn(action, game.legal_actions()[1])
        game._draw(0)
        self.assertIn(action, game.legal_actions()[1])
        game.step(action)
        self.assertIsNone(game.offer)
        self.assertEqual(game.kong_chain, 1)

    def test_flower_bonus_natural_pairs_and_isolated_honors(self):
        ctx = WinContext('zimo', WHITE, 27, EAST)
        pairs = counts([0, 0, 2, 2, 9, 9, 11, 11, 18, 18, 20, 20, WHITE, WHITE])
        score = ANJI_TEA_HOUSE.faan(pairs, (), ctx, wildcard=6, flowers=3)
        self.assertEqual((score.share, score.patterns), (8, ('七对子', '花牌x3')))
        loose = counts([0, 3, 6, 9, 12, 18, 21, *range(27, 34)])
        score = ANJI_TEA_HOUSE.faan(loose, (), ctx, wildcard=8, flowers=1)
        self.assertEqual((score.share, score.patterns), (3, ('十三不搭', '花牌x1')))

    def test_two_flower_replacements_during_a_float_add_separately(self):
        hand = counts([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 6, 6, 6, 23])
        ctx = WinContext('zimo', 23, 27, EAST)
        score = ANJI_TEA_HOUSE.faan(hand, (), ctx, wildcard=6, floats=1, kongs_before_float=1,
                                   flower_draw=2, flowers=3)
        self.assertEqual((score.share, score.patterns), (38, ('四财神', '财飘', '飘杠', '花牌x3')))
