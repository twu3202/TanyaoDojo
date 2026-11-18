import random
import unittest

from deploy.play.honors import COPIES, ORPHANS, PASS, RON, HonorsGame, Meld, WinContext
from deploy.play.honors_bonus import HEZHOU, HUNDRED, SHANGSI
from deploy.play.honors_game import HonorsSession
from deploy.play.honors_policy import _live_tiles, heuristic, policy_view
from deploy.play.tests.test_hongkong import SEQUENCES, counts, fixture


class HorseTileScoring(unittest.TestCase):
    def test_hundred_uses_the_published_fixed_scores_and_legal_stacking(self):
        cases = (
            ([9]*3 + [10]*3 + [11]*3 + [12]*3 + [13]*2, 12),
            ([9]*3 + [10]*3 + [11]*3 + [27]*3 + [28]*2, 8),
            ([9,10,11,12,13,14,15] * 2, 16),
            ([9]*3 + [17]*3 + [18]*3 + [27]*3 + [31]*2, 12),
            ([27]*3 + [28]*3 + [31]*3 + [33]*3 + [32]*2, 16),
        )
        for tiles, score in cases:
            self.assertEqual(HUNDRED.faan(counts(tiles), (), WinContext('zimo', tiles[-1], 27, 27)).share, score)
            self.assertIsNone(HUNDRED.faan(counts(tiles), (), WinContext('ron', tiles[-1], 27, 27)))

    def test_hezhou_premiums_and_three_feed_payments(self):
        hand = counts([9]*4 + [10, 11, 12, 13, 14] * 2)
        score = HEZHOU.faan(hand, (), WinContext('zimo', 14, 27, 27, kong_chain=1))
        self.assertEqual(score.share, 32)
        self.assertEqual(score.patterns, ('龙七对', '清一色', '门清', '杠上花'))
        orphans = counts(list(ORPHANS) + [27])
        self.assertEqual(HEZHOU.faan(orphans, (), WinContext('zimo', 27, 27, 27)).share, 10)
        self.assertIsNone(HEZHOU.faan(orphans, (), WinContext('rob_concealed', 27, 27, 27)))
        score = HEZHOU.faan(counts(SEQUENCES), (), WinContext('ron', 22, 27, 27, kong_chain=1))
        self.assertEqual(score.patterns, ('杠上炮',))
        self.assertEqual(score.share, 2)
        increased = HEZHOU.with_horses(score, 2)
        partners = [('three_feeds', 0), ('three_feeds', 3)]
        self.assertEqual(HEZHOU.settlement(increased, 'ron', 1, 0, partners), (18, 0, 0, 0))
        self.assertEqual(HEZHOU.settlement(increased, 'ron', 1, 2, partners), (0, 0, 6, 0))
        self.assertEqual(HEZHOU.settlement(increased, 'zimo', 1, None, partners), (0, 0, 0, 18))


class HorseRoomRules(unittest.TestCase):
    def test_shangsi_multiron_uses_one_set_of_horses_with_separate_hits(self):
        dealer = [22, 5, 6, 7, 8, 20, 21, 25, 26, 27, 29, 30, 31, 32]
        game = fixture({0: dealer, 1: SEQUENCES[:-1], 2: SEQUENCES[:-1]}, rules=SHANGSI, drawn=22)
        horses = [1, 2, 5, 4]
        for tile in horses:
            game.wall.remove(tile)
        game.wall[:0] = horses
        for move in (('discard', 22), RON, RON, PASS):
            game.step(move)
        self.assertEqual(game.horse_tiles, horses)
        self.assertEqual(game.horse_details, [(1, 4, 2, 3, 2), (2, 4, 1, 2, 1)])
        self.assertEqual(game.scores(), [-5, 3, 2, 0])
        self.assertEqual(game.next_dealer, 0)
        self.assertEqual(game.physical_counts(), COPIES)

    def test_shangsi_opening_echo_and_draw_refund_have_different_ledgers(self):
        hands = {seat: [0] + list(range(1 + 8*seat, 1 + 8*seat + 8)) +
                 list(range(2 + 8*seat, 2 + 8*seat + 4)) for seat in range(4)}
        hands[0].append(8)
        game = fixture(hands, rules=SHANGSI)
        for _ in range(4):
            game.step(('discard', 0))
            for _ in range(3):
                game.step(PASS)
        self.assertEqual(game.echo_deltas, [-3, 1, 1, 1])
        self.assertEqual(game.scores(), [-3, 1, 1, 1])
        self.assertEqual(game.physical_counts(), COPIES)

        dealer = [9] * 4 + list(range(10, 20))
        game = fixture({0: dealer}, rules=SHANGSI, drawn=9)
        before = game.wall[:10]
        for move in (('ankan', 9), PASS, PASS, PASS):
            game.step(move)
        self.assertEqual(game.wall[:10], before)
        self.assertEqual(game.kong_deltas, [6, -2, -2, -2])
        self.exhaust(game)
        self.assertEqual(game.kong_deltas, [0, 0, 0, 0])
        self.assertEqual(game.scores(), [0, 0, 0, 0])
        self.assertEqual(game.horse_tiles, [])
        self.assertEqual(game.wall, before)

    def test_hundred_robbery_uses_original_reserve_and_no_kong_points(self):
        hand = [9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 23, 25, 25]
        exposed = {0: [Meld('peng', 22, 3)]}
        dealer = [22, 13, 16, 17, 28, 28, 29, 29, 30, 31, 32]
        game = fixture({0: dealer, 1: hand, 2: hand}, exposed, rules=HUNDRED, drawn=22)
        reserves = game.wall[:4]
        self.assertEqual(len(game.tile_supply), 25)
        self.assertEqual(sum(game.tile_supply.values()), 100)
        self.assertFalse(any(_live_tiles(policy_view(game, 0, game.legal_actions()[1]))[:9]))
        for move in (('bugang', 22), RON, RON, PASS):
            game.step(move)
        self.assertEqual(game.horse_tiles, reserves)
        self.assertEqual(game.kong_deltas, [0] * 4)
        self.assertEqual(game.players[0].melds, exposed[0])
        self.assertEqual(game.claimed_tile, 22)
        self.assertEqual(game.next_dealer, 0)
        self.assertEqual([win.payments[0] for win in game.win_details],
                         [3 * win.points for win in game.win_details])
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_hundred_only_self_draws_on_ordinary_tiles_and_keeps_kong_points(self):
        hand = [9, 10, 11, 12, 13, 14, 18, 19, 20, 24, 25, 26, 22]
        game = fixture({0: [22, *range(9, 22)], 1: hand}, rules=HUNDRED, drawn=22)
        game.step(('discard', 22))
        self.assertNotIn(RON, game.legal_actions()[1])

        game = fixture({0: [9] * 4 + list(range(10, 20))}, rules=HUNDRED, drawn=9)
        for move in (('ankan', 9), PASS, PASS, PASS):
            game.step(move)
        session = HonorsSession(variant=HUNDRED.VARIANT)
        session.game = game
        self.assertEqual(session.view()['players'][0]['melds'][0]['tiles'], ['1p', None, None, None])
        self.exhaust(game)
        self.assertEqual(game.scores(), [6, -2, -2, -2])
        self.assertEqual(game.kong_deltas, [6, -2, -2, -2])

    def test_hezhou_third_claim_records_both_directions_and_one_provider(self):
        game = fixture({0: [29, *range(13)], 1: [29, 29, 14, 15, 16, 17, 17]},
                       {1: [Meld('peng', 27, 0), Meld('peng', 28, 0)]}, rules=HEZHOU, drawn=29)
        for move in (('discard', 29), ('peng', None), PASS, PASS):
            game.step(move)
        self.assertEqual(game._liabilities_for(1), [('three_feeds', 0)])
        self.assertEqual(game._liabilities_for(0), [('three_feeds', 1)])
        self.assertEqual(game._liabilities_for(2), [])
        self.assertEqual(game.physical_counts(), COPIES)

    def test_hezhou_discard_after_kong_applies_the_extra_share(self):
        dealer = [25]*4 + [5, 6, 7, 8, 18, 19, 20, 27, 28, 31]
        game = fixture({0: dealer, 1: SEQUENCES[:-1]}, rules=HEZHOU, drawn=25)
        game.wall.remove(22)
        game.wall.insert(HEZHOU.WALL_RESERVE, 22)
        for move in (('ankan', 25), PASS, PASS, PASS, ('discard', 22), RON, PASS, PASS):
            game.step(move)
        self.assertEqual(game.win_details[0].patterns, ('杠上炮',))
        self.assertEqual(game.win_details[0].points, 2 * (game.horse_details[0][2] + 1))
        self.assertEqual(game.physical_counts(), COPIES)

    def test_hezhou_pass_hu_locks_other_tiles_and_recovers_on_draw(self):
        ready = [1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 17, 17]
        game = fixture({0: [0, *range(13, 26)], 1: [6, *range(20, 32)], 2: ready}, rules=HEZHOU)
        game.step(('discard', 0))
        game.step(PASS)
        self.assertIn(RON, game.legal_actions()[1])
        game.step(PASS)
        game.step(PASS)
        game.step(('discard', 6))
        self.assertNotIn(RON, game.legal_actions()[1])
        for _ in range(3):
            game.step(PASS)
        self.assertEqual(game.cur, 2)
        self.assertFalse(game.players[2].declined_wins)

    def test_random_play_preserves_tiles_and_points_and_ai_does_not_see_reserve(self):
        for room in (SHANGSI, HUNDRED, HEZHOU):
            rng = random.Random(93)
            for seed in range(150):
                game = HonorsGame(seed, 0, 27, rules=room)
                actor, acts = game.legal_actions()
                if seed == 0:
                    original = policy_view(game, actor, acts)
                    game.wall[0], game.wall[-1] = game.wall[-1], game.wall[0]
                    self.assertEqual(policy_view(game, actor, acts), original)
                while game.phase != 'over':
                    actor, acts = game.legal_actions()
                    choice = heuristic(policy_view(game, actor, acts)) if seed < 15 else rng.choice(acts)
                    game.step(choice)
                    self.assertEqual(game.physical_counts(), game.tile_supply)
                    self.assertEqual(sum(game.scores()), 0)
                net = [kong + echo for kong, echo in zip(game.kong_deltas, game.echo_deltas)]
                for win in game.win_details:
                    for payer, points in enumerate(win.payments):
                        net[payer] -= points
                        net[win.actor] += points
                self.assertEqual(game.scores(), net)

    @staticmethod
    def exhaust(game):
        while game.phase != 'over':
            game.step(game.legal_actions()[1][0])
        assert game.reason_code == 'draw'
