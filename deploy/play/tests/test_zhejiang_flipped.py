import random
import unittest

from deploy.play.hangzhou import WHITE
from deploy.play.honors import Meld, Offer, ORPHANS, PASS, RON, WinContext
from deploy.play.honors_policy import heuristic, policy_view
from deploy.play.tests.test_hangzhou import play_passes
from deploy.play.tests.test_hongkong import counts
from deploy.play.zhejiang_flipped import (SONGYANG, SUICHANG, TEN_WINDS, XIAOSHAN, SongyangGame,
                                          SuichangGame, XiaoshanGame)


class XiaoshanTests(unittest.TestCase):
    def test_wildcard_gate_and_kong_kind(self):
        hand = counts([1, 2, WHITE, 9, 10, 11, 18, 19, 20, 27, 27, 27, 33, 33])
        ctx = WinContext('zimo', 33, 27, 27)
        self.assertIsNone(XIAOSHAN.faan(hand, (), ctx))
        replacement = WinContext('zimo', 33, 27, 27, kong_chain=1)
        self.assertEqual(XIAOSHAN.faan(hand, (), replacement, last_kong='gang_ming').share, 2)
        concealed = XIAOSHAN.faan(hand, (), replacement, last_kong='gang_an')
        self.assertEqual((concealed.share, concealed.patterns), (4, ('豪杠开',)))

    def test_honor_groups_and_unstructured_honors_do_not_stack(self):
        ctx = WinContext('zimo', 33, 27, 27, kong_chain=1)
        triplets = counts([27] * 3 + [28] * 3 + [29] * 3 + [30] * 3 + [33, 33])
        self.assertEqual(XIAOSHAN.faan(triplets, (), ctx).patterns, ('正风',))
        self.assertEqual(XIAOSHAN.faan(triplets, (), ctx).share, 128)
        unstructured = counts([27] * 4 + [28] * 4 + [29, 30, 32, 32, 33, 33])
        self.assertEqual(XIAOSHAN.faan(unstructured, (), ctx).patterns, ('乱风',))
        self.assertEqual(XIAOSHAN.faan(unstructured, (), ctx).share, 64)

    def test_ten_winds_discard_ends_hand_without_requiring_a_win_shape(self):
        game = XiaoshanGame(1, 0, 27, rules=XIAOSHAN)
        game.wildcard_indicator = 4
        game.players[0].hand = counts(list(range(12)) + [27, 33])
        game.drawn_tile = 33
        game.ten_winds[0] = 9
        actor, actions = game.legal_actions()
        self.assertIn((TEN_WINDS, 4), actions)  # physical White Dragon is eligible
        self.assertIn((TEN_WINDS, 33), actions)
        self.assertNotIn((TEN_WINDS, 0), actions)
        self.assertEqual(heuristic(policy_view(game, actor, actions))[0], TEN_WINDS)
        game.step((TEN_WINDS, 33))
        self.assertEqual(game.discards[0], [33])
        self.assertEqual(sum(game.players[0].hand), 13)
        self.assertEqual(game.reason_code, 'ten_winds_win')
        self.assertEqual(game.win_details[0].patterns, ('十风',))
        self.assertEqual(game.scores(), [48, -16, -16, -16])

    def test_dealer_factor_bills_kongs_then_caps_each_payer(self):
        game = XiaoshanGame(2, 0, 27, rules=XIAOSHAN, dealer_streak=5)
        game.players[1].melds.append(Meld('gang_an', 29))
        self.assertEqual(game.dealer_multiplier, 8)
        self.assertEqual(game.payments(1, 4), (33, 0, 5, 5))
        self.assertEqual(game.payments(1, 64), (128, 0, 65, 65))

    def test_second_chow_changes_swap_discards_and_pon_pass_is_local(self):
        game = XiaoshanGame(3, 0, 27, rules=XIAOSHAN)
        caller = game.players[1]
        caller.hand = counts([0, 0, 1, 2, 3, 13, 17, 21])
        self.assertEqual(next(game._chows(caller, 0, 0))[2], frozenset())
        caller.melds.append(Meld('peng', 20, 0))
        self.assertEqual(next(game._chows(caller, 0, 0))[2], {0, 3})
        game._open_offer(Offer(0, 0, 'discard'))
        self.assertIn(('peng', None), game.legal_actions()[1])
        game.step(PASS)
        self.assertIn(0, caller.declined_calls)
        game._draw(1)
        self.assertNotIn(0, caller.declined_calls)


class SongyangTests(unittest.TestCase):
    def test_natural_and_wild_orphans_and_seven_winds_use_physical_face(self):
        indicator = 13
        natural_orphans = [indicator if t == WHITE else t for t in ORPHANS] + [0]
        ctx = WinContext('zimo', 0, 27, 27)
        score = SONGYANG.faan(counts(natural_orphans), (), ctx, indicator=indicator)
        self.assertEqual((score.share, score.patterns), (4, ('正规十三幺',)))
        natural_orphans[natural_orphans.index(27)] = WHITE
        score = SONGYANG.faan(counts(natural_orphans), (), ctx, indicator=indicator)
        self.assertEqual((score.share, score.patterns), (2, ('不正规十三幺',)))
        honors = [27, 28, 29, 30, indicator, 32, 33]
        winds = honors + [0, 3, 6, 9, 12, 18, 21]
        self.assertEqual(SONGYANG.faan(counts(winds), (), ctx, indicator=indicator).patterns, ('正规七风',))
        winds[1] = WHITE
        self.assertEqual(SONGYANG.faan(counts(winds), (), ctx, indicator=indicator).patterns, ('不正规七风',))
        winds[winds.index(3)] = 1
        self.assertIsNone(SONGYANG.faan(counts(winds), (), ctx, indicator=indicator))

    def test_added_kong_robbery_charges_three_shares_to_supplier(self):
        game = SongyangGame(6, 0, 27, rules=SONGYANG)
        game.players[1].hand = counts([0, 1, 9, 10, 11, 18, 19, 20, 27, 27, 27, 33, 33])
        game.players[3].hand = counts([2])
        game.players[3].melds = [Meld('peng', 2, 2)]
        game._open_offer(Offer(3, 2, 'discard'))
        game.step(PASS)
        self.assertNotIn(RON, game.legal_actions()[1])
        game._open_offer(Offer(3, 2, 'gang_bu'))
        game.step(PASS)
        self.assertIn(RON, game.legal_actions()[1])
        game.step(RON)
        play_passes(game)
        self.assertEqual(game.reason_code, 'rob_kong_win')
        self.assertEqual(game.scores(), [0, 16, 0, -16])
        self.assertEqual(game.players[3].melds, [Meld('peng', 2, 2)])

    def test_others_still_pay_when_three_feed_liability_applies(self):
        game = SongyangGame(7, 0, 27, rules=SONGYANG)
        for tile in (12, 15, 33):
            game.players[2].hand = counts([tile, tile])
            game.discards[1].append(tile)
            game._open_offer(Offer(1, tile, 'discard'))
            game.step(('peng', None))
            play_passes(game)
        self.assertEqual(game.related_seats, [(2, 1)])
        self.assertEqual(game.payments(2, 1), (2, 4, 0, 1))
        self.assertEqual(game.payments(1, 1), (2, 0, 8, 1))

    def test_twenty_tile_reserve_survives_replacement_and_wall_exhaustion(self):
        game = SongyangGame(14, 0, 27, rules=SONGYANG)
        self.assertEqual(game.remaining, 62)
        game.wall = game.wall[:21]
        reserve = game.wall[:20]
        game._draw(1, kong_chain=1)
        self.assertEqual(game.wall, reserve)
        self.assertEqual(game.remaining, 0)
        game.step(('discard', game.drawn_tile))
        play_passes(game)
        self.assertEqual(game.reason_code, 'draw')
        self.assertEqual(game.scores(), [0] * 4)


class SuichangTests(unittest.TestCase):
    def test_discard_win_charges_other_losers_and_stops_after_first_claim(self):
        game = SuichangGame(6, 0, 27, rules=SUICHANG)
        waiting = counts([0, 1, 9, 10, 11, 18, 19, 20, 27, 27, 27, 33, 33])
        game.players[0].hand = waiting.copy()
        game.players[1].hand = waiting.copy()
        game.discards[3].append(2)
        game._open_offer(Offer(3, 2, 'discard'))
        self.assertIn(RON, game.legal_actions()[1])
        game.step(RON)
        self.assertNotIn(RON, game.legal_actions()[1])
        play_passes(game)
        self.assertEqual(game.reason_code, 'suited_discard_win')
        self.assertEqual(game.scores(), [4, -1, -1, -2])
        self.assertEqual(game.next_dealer, 0)

    def test_added_kong_robbery_charges_its_supplier_and_keeps_reserve(self):
        game = SuichangGame(6, 0, 27, rules=SUICHANG)
        game.players[1].hand = counts([0, 1, 9, 10, 11, 18, 19, 20, 27, 27, 27, 33, 33])
        game.players[0].hand = counts([2])
        original = Meld('peng', 2, 2)
        game.players[0].melds = [original]
        before = game.remaining
        game._open_offer(Offer(0, 2, 'gang_bu'))
        game.step(RON)
        play_passes(game)
        self.assertEqual(game.scores(), [-8, 8, 0, 0])
        self.assertEqual(game.players[0].melds, [original])
        self.assertEqual(game.remaining, before)
        self.assertEqual(game.payments(1, 3), (6, 0, 6, 6))

    def test_completed_kong_releases_two_reserve_tiles_and_draws_one(self):
        game = SuichangGame(14, 0, 27, rules=SUICHANG)
        self.assertEqual(game.remaining, 48)
        game.players[0].hand = counts([7] * 4)
        game.wall = game.wall[:35]
        expected = game.wall[32]
        game._finish_own_kong(Offer(0, 7, 'gang_an'))
        self.assertEqual(game.drawn_tile, expected)
        self.assertEqual(game.remaining, 2)
        self.assertEqual(game._replacement_index(), 32)

    def test_seven_pairs_and_honor_tiers(self):
        pairs = counts([0] * 4 + [9] * 4 + [18] * 4 + [27] * 2)
        score = SUICHANG.faan(pairs, (), WinContext('zimo', 27, 27, 27))
        self.assertEqual((score.share, score.patterns), (16, ('三豪华七对',)))
        irregular = counts([27] * 4 + [28] * 4 + [29, 30, 32, 32, 33, 33])
        score = SUICHANG.faan(irregular, (), WinContext('ron', 33, 27, 27))
        self.assertEqual((score.share, score.patterns), (4, ('全字',)))


class SharedFlipTests(unittest.TestCase):
    def test_complete_hands_keep_all_physical_tiles_and_points(self):
        for rules, engine in ((XIAOSHAN, XiaoshanGame), (SONGYANG, SongyangGame), (SUICHANG, SuichangGame)):
            for seed in range(16):
                rng = random.Random(seed)
                game = engine(seed, seed % 4, 27, rules=rules, dealer_streak=seed % 3)
                for _ in range(500):
                    if game.phase == 'over':
                        break
                    actor, moves = game.legal_actions()
                    action = heuristic(policy_view(game, actor, moves)) if seed % 2 else rng.choice(moves)
                    game.step(action)
                    self.assertEqual(game.physical_counts(), game.tile_supply)
                    self.assertEqual(sum(game.scores()), 0)
                self.assertEqual(game.phase, 'over')
