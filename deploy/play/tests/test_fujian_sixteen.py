from collections import Counter
import unittest

from deploy.play.fujian_sixteen import (
    NANPING_HONORS, NANPING_RED, NANPING_SUITED, NINGDE, RED, WHITE, FujianSixteenGame,
)
from deploy.play.honors import PASS, RON, ZIMO, Meld, WinContext
from deploy.play.honors_game import HonorsSession
from deploy.play.tests.test_hongkong import counts
from deploy.play.tests.test_jinyun import pass_offer


REGULAR = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 18, 19, 20, 8, 8]


def table(rules, hands, melds=None, *, opening=False):
    game = FujianSixteenGame(7, 0, 27, rules=rules)
    game.cur = game.dealer
    game.phase = 'action'
    game.opening = opening
    game.flowers = [[] for _ in game.players]
    taken = Counter(t for hand in hands for t in hand)
    if game.indicator_tile is not None:
        taken[game.indicator_tile] += 1
    for i, player in enumerate(game.players):
        player.melds = (melds or {}).get(i, [])
        taken.update(tile for meld in player.melds for tile in meld.tiles)
    assert taken <= game.tile_supply
    game.wall = list((game.tile_supply - taken).elements())
    available = [t for t in game.wall if t < 34 and t != game.wildcard_tile]
    for i, (hand, player) in enumerate(zip(hands, game.players, strict=True)):
        extras = [available.pop() for _ in range(16 + (i == game.dealer) - len(hand) - 3 * len(player.melds))]
        for tile in extras:
            game.wall.remove(tile)
        player.hand = counts([*hand, *extras])
    game.drawn_tile = hands[0][-1]
    return game


class FujianSixteenTests(unittest.TestCase):
    def test_tile_supplies_and_public_gold_without_exposing_another_draw(self):
        cases = ((NANPING_HONORS, 144, None), (NANPING_SUITED, 116, None),
                 (NANPING_RED, 120, RED), (NINGDE, 112, WHITE))
        for rules, total, gold in cases:
            with self.subTest(rules=rules):
                game = FujianSixteenGame(3, 1, 27, rules=rules)
                session = HonorsSession(variant=rules.VARIANT)
                session.game = game
                view = session.view()
                self.assertEqual(sum(game.tile_supply.values()), total)
                self.assertEqual([sum(p.hand) for p in game.players], [16, 17, 16, 16])
                self.assertTrue(all(not p['hand'] for p in view['players'][1:]))
                self.assertIsNone(view['drawn'])
                self.assertEqual(game.physical_counts(), game.tile_supply)
                if gold is None:
                    self.assertEqual(game.indicator_tile, game.wildcard_tile)
                    self.assertTrue(view['gold_indicator'])
                else:
                    self.assertIsNone(game.indicator_tile)
                    self.assertEqual(game.wildcard_tile, gold)
                    self.assertTrue(view['fixed_gold'])

    def test_nanping_public_score_table_and_direct_kong(self):
        hand = counts([6, 7, WHITE, *REGULAR[3:-2], 26, 26])
        ctx = WinContext('ron', WHITE, 28, 27)
        self.assertTrue(NANPING_HONORS.can_win(hand, (), ctx, gold=8))
        self.assertFalse(NANPING_SUITED.can_win(hand, (), ctx, gold=8))
        plain = counts(REGULAR)
        for rules, fee in ((NANPING_HONORS, 1), (NANPING_SUITED, 3)):
            game = FujianSixteenGame(2, 0, 27, rules=rules, dealer_streak=2)
            game.wildcard_tile = 23
            score = rules.score(game, plain, (), ctx, (34, 35, 36, 37))
            self.assertEqual(score.share, 9 if rules is NANPING_HONORS else 1)
            game._pay_kong(1, Meld('gang_ming', 22, 3))
            self.assertEqual(game.kong_deltas, [0, fee, 0, -fee])

    def test_fixed_gold_seven_pairs_and_pung_and_gold_kong_award(self):
        paired = counts([1, 1, 4, 4, 8, 8, 10, 10, 12, 12, 15, 15, 19, 19])
        pung = (Meld('peng', 0, 0),)
        ctx = WinContext('ron', 1, 28, 27)
        self.assertTrue(NANPING_RED.can_win(paired, pung, ctx, gold=RED))
        self.assertTrue(NINGDE.can_win(paired, pung, ctx, gold=WHITE))
        self.assertFalse(NANPING_HONORS.can_win(paired, pung, ctx, gold=RED))
        game = FujianSixteenGame(2, 0, 27, rules=NANPING_RED)
        normal = counts([*REGULAR[:12], 8, 8])
        gold_kong = (Meld('gang_an', RED),)
        self.assertTrue(NANPING_RED.can_win(normal, gold_kong, ctx, gold=RED))
        score = NANPING_RED.score(game, normal, gold_kong, ctx, ())
        self.assertEqual(score.share, 81)
        self.assertIn('金杠', score.patterns)
        self.assertEqual(NANPING_RED.score(game, normal, gold_kong, WinContext('zimo', 2, 28, 27), ()).share, 102)

    def test_ningde_loser_difference_is_pairwise_and_draw_has_no_transfers(self):
        hands = [REGULAR, [WHITE, WHITE], [WHITE], []]
        melds = {2: [Meld('gang_an', 15)], 3: [Meld('gang_ming', 22, 1)]}
        game = table(NINGDE, hands, melds)
        game.step(ZIMO)
        self.assertEqual(game.win_details[0].payments, (0, 5, 5, 5))
        self.assertEqual(game.balance_deltas, [0, 0, 3, -3])
        self.assertEqual(game.scores(), [15, -5, -2, -8])
        self.assertEqual(game.physical_counts(), game.tile_supply)
        game = table(NINGDE, hands, melds)
        game._end('draw')
        self.assertEqual(game.balance_deltas, [0] * 4)
        self.assertEqual(game.scores(), [0] * 4)

    def test_ningde_first_discard_win_precedes_an_accepted_three_gold(self):
        ready = [tile for tile in REGULAR if tile != 2]
        game = table(NINGDE, [[2], [WHITE] * 3, ready, []], opening=True)
        game.step(('discard', 2))
        self.assertIn(('three_gold', None), game.legal_actions()[1])
        game.step(('three_gold', None))
        self.assertIn(RON, game.legal_actions()[1])
        game.step(RON)
        pass_offer(game)
        self.assertEqual(game.winners, [2])
        self.assertEqual(game.win_details[0].patterns[0], '地胡')
        self.assertEqual(game.physical_counts(), game.tile_supply)
        game = table(NINGDE, [[2], [WHITE] * 3, ready, []], opening=True)
        game.step(('discard', 2))
        game.step(('three_gold', None))
        game.step(PASS)
        pass_offer(game)
        self.assertEqual(game.winners, [1])
        self.assertEqual(game.reason_code, 'three_gold_win')
        self.assertEqual(game.discards[0], [2])
        self.assertEqual(game.physical_counts(), game.tile_supply)
