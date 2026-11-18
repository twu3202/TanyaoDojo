import random
import unittest

from deploy.play import hongkong, macau
from deploy.play.errors import ClientError
from deploy.play.honors import COPIES, ORPHANS, PASS, RON, ZIMO, HonorsGame, Meld, WinContext
from deploy.play.honors_game import HonorsSession
from deploy.play.honors_policy import heuristic, policy_view
from deploy.play.tests.test_hongkong import SEQUENCES, counts, fixture


class MacauScoring(unittest.TestCase):
    def test_zero_faan_discard_and_faan_schedule(self):
        hand = counts([9, 10, 11, 15, 16, 17, 24, 24, 24, 22, 22])
        melds = (Meld('chi', 0, 1),)
        ron = WinContext('ron', 22, 28, 27)
        self.assertIsNone(hongkong.faan(hand, melds, ron))
        score = macau.faan(hand, melds, ron)
        self.assertEqual(score.fan, 0)
        self.assertEqual(macau.settlement(score, 'ron', 1, 0, []), (5, 0, 0, 0))
        for fan in range(1, 7):
            score = macau.Score(fan, ())
            self.assertEqual(macau.settlement(score, 'zimo', 1, None, []), (10*fan, 0, 10*fan, 10*fan))
            self.assertEqual(macau.settlement(score, 'ron', 1, 0, []), (20*fan, 0, 0, 0))
        self.assertEqual(macau.Score(12, ()).fan, 6)

    def test_fixed_tiers_and_nine_gates_uses_pre_win_wait(self):
        hand = counts(sorted(ORPHANS) + [27])
        orphans = macau.faan(hand, (), WinContext('ron', 27, 28, 27))
        self.assertEqual((orphans.tier, orphans.fan), ('A', None))
        self.assertEqual(macau.settlement(orphans, 'ron', 1, 0, []), (500, 0, 0, 0))
        hand = counts([0, 0, 0, *range(1, 8), 8, 8, 8, 4])
        nine_gates = macau.faan(hand, (), WinContext('ron', 4, 28, 27))
        self.assertEqual((nine_gates.tier, nine_gates.patterns), ('B', ('九莲宝灯',)))
        self.assertEqual(macau.settlement(nine_gates, 'zimo', 1, None, []), (500, 0, 500, 500))
        self.assertIsNone(macau.faan(hand, (), WinContext('ron', 1, 28, 27)).tier)

    def test_opening_names_and_pure_straight(self):
        hand = counts(SEQUENCES)
        first_ron = WinContext('ron', 22, 28, 27, opening='地胡')
        first_draw = WinContext('zimo', 22, 28, 27, first_nondealer_draw=True)
        self.assertEqual(macau.faan(hand, (), first_ron).patterns, ('人和',))
        self.assertEqual(macau.faan(hand, (), first_draw).patterns, ('地胡',))
        pure_straight = counts([*range(9), 9, 10, 11, 22, 22])
        score = macau.faan(pure_straight, (), WinContext('ron', 22, 28, 27))
        self.assertEqual(score.fan, 3)
        self.assertIn('一条龙', score.patterns)

    def test_liability_depends_on_final_pattern_and_only_self_draw(self):
        sets = [Meld('chi', 0, 1), Meld('chi', 1, 2), Meld('chi', 3, 2), Meld('chi', 6, 3)]
        self.assertEqual(macau.liability(sets, sets[-1]), ('清一色',))
        liability = [('清一色', 3)]
        context = WinContext('zimo', 4, 27, 27)
        flush = macau.faan(counts([4, 4]), tuple(sets), context)
        mixed = macau.faan(counts([13, 13]), tuple(sets), WinContext('zimo', 13, 27, 27))
        self.assertEqual(macau.settlement(flush, 'zimo', 0, None, liability), (0, 0, 0, 180))
        self.assertEqual(mixed.patterns, ('平胡', '一条龙', '自摸'))
        self.assertEqual(macau.settlement(mixed, 'zimo', 0, None, liability), (0, 30, 30, 30))
        self.assertEqual(macau.settlement(flush, 'ron', 0, 1, liability), (0, 120, 0, 0))
        dragons = [Meld('peng', tile, 1) for tile in (31, 32, 33)]
        winds = [Meld('peng', tile, 2) for tile in (27, 28, 29, 30)]
        kongs = [Meld('gang_an', tile) for tile in (0, 1, 2)] + [Meld('gang_ming', 3, 2)]
        self.assertEqual(macau.liability(dragons, dragons[-1]), ('大三元',))
        self.assertEqual(macau.liability(winds, winds[-1]), ('大四喜',))
        self.assertEqual(macau.liability(kongs, kongs[-1]), ('四杠',))


class MacauIntegration(unittest.TestCase):
    def test_reserve_is_untouched_at_the_end_and_no_last_draw_kong(self):
        game = fixture({0: [0]*4 + [3, 4, 5, 6, 7, 8, 13, 18, 22, 24]}, rules=macau)
        game.discards[3].extend(game.wall[14:])
        del game.wall[14:]
        self.assertEqual(game.remaining, 0)
        self.assertNotIn(('ankan', 0), game.legal_actions()[1])
        game.step(('discard', 0))
        for _ in range(3):
            self.assertLessEqual(set(game.legal_actions()[1]), {PASS, RON})
            game.step(PASS)
        self.assertEqual(game.reason_code, 'draw')
        self.assertEqual(game.next_dealer, 1)
        self.assertEqual(len(game.wall), 14)
        self.assertEqual(game.physical_counts(), COPIES)

    def test_robbery_is_charged_as_a_discard(self):
        game = fixture({0: [0, 3, 4, 5, 6, 7, 8, 13, 18, 22, 24], 1: SEQUENCES[1:]},
                       {0: [Meld('peng', 0, 3)]}, rules=macau)
        for action in (('bugang', 0), RON, PASS, PASS):
            game.step(action)
        self.assertEqual(game.scores(), [-60, 60, 0, 0])
        self.assertEqual(game.win_details[0].fan, 3)
        self.assertEqual(game.physical_counts(), COPIES)

    def test_current_profile_allows_chow_swap(self):
        game = fixture({0: [0] + [18]*4 + [19]*4 + [20]*4 + [21],
                        1: [0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 27, 28, 33]}, rules=macau)
        for action in (('discard', 0), ('chi', 0), PASS, PASS):
            game.step(action)
        self.assertIn(('discard', 3), game.legal_actions()[1])
        self.assertIn(('discard', 0), game.legal_actions()[1])
        self.assertNotIn(ZIMO, game.legal_actions()[1])

    def test_random_games_preserve_all_tiles_and_points(self):
        for seed in range(240):
            rng = random.Random(seed)
            game = HonorsGame(seed, seed % 4, 27 + seed % 4, rules=macau)
            for _ in range(560):
                if game.phase == 'over':
                    break
                _, actions = game.legal_actions()
                game.step(rng.choice(actions))
                self.assertEqual(game.physical_counts(), COPIES)
                self.assertEqual(sum(game.scores()), 0)
            self.assertEqual(game.phase, 'over')
            self.assertGreaterEqual(len(game.wall), 14)

    def test_sixteen_hands_rotate_and_end_match(self):
        session = HonorsSession(seed=14, pacing=0, variant=macau.VARIANT)
        self.addCleanup(session.close)
        starting_dealer = session.dealer
        for number in range(1, 17):
            session.begin()
            self.assertEqual(session.game.dealer, (starting_dealer + number - 1) % 4)
            self.assertEqual(session.game.round_wind, 27 + (number - 1) // 4)
            session.auto()
            with session.cv:
                self.assertTrue(session.cv.wait_for(lambda: session.status in ('round_end', 'finished', 'error'),
                                                    timeout=8))
            snapshot = session.snapshot()
            self.assertEqual(snapshot['status'], 'finished' if number == 16 else 'round_end', snapshot['error'])
            self.assertEqual(sum(snapshot['result']['scores']), 0)
            self.assertEqual(snapshot['result']['next'], number < 16)
        result = snapshot['result']
        self.assertEqual(result['match_hands'], 16)
        self.assertEqual(result['match_winners'], [i for i, score in enumerate(result['scores'])
                                                  if score == max(result['scores'])])
        with self.assertRaises(ClientError):
            session.begin()

    def test_public_information_policy(self):
        game = HonorsGame(6, 0, 27, rules=macau)
        actor, actions = game.legal_actions()
        public = policy_view(game, actor, actions)
        game.wall.reverse()
        game.players[1].hand, game.players[2].hand = game.players[2].hand, game.players[1].hand
        self.assertEqual(policy_view(game, actor, actions), public)
        self.assertIn(heuristic(public), actions)
