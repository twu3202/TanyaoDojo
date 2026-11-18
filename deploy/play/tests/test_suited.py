import random
import time
import unittest
from collections import Counter

from deploy.play.suited import COPIES, PASS, RON, Profile, SuitedGame, taishun_points, wenzhou_fan, xuanhua_points
from deploy.play.suited_game import SuitedSession, policy_view
from deploy.play.sichuan_game import heuristic


def counts(tiles):
    return [tiles.count(tile) for tile in range(27)]


def fixture(hands, *, melds=None, profile=Profile.WENZHOU, drawn=0):
    game = SuitedGame(2, 0, profile=profile)
    used = Counter()
    for seat, tiles in enumerate(hands):
        player = game.players[seat]
        player.hand = counts(tiles)
        player.melds = list((melds or {}).get(seat, []))
        used.update(tiles)
        for kind, tile in player.melds:
            used[tile] += 3 if kind == 'peng' else 4
    assert all(count <= 4 for count in used.values())
    game.wall = list((COPIES - used).elements())
    game.drawn_tile = drawn
    game.opening = False
    return game


READY = [1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 17, 17]
SPARE = [18] * 4 + [19] * 4 + [20] * 4 + [21]
CALLER = [0, 0, 6, 6, 6, 7, 7, 8, 8, 15, 15, 16, 16]


class SuitedScoring(unittest.TestCase):
    def test_max_base_and_event_fan(self):
        examples = [
            (READY + [0], [], (1, '平胡')),
            ([0] * 3 + [9] * 3 + [11] * 3 + [13] * 3 + [16] * 2, [], (2, '对对胡')),
            ([0, 1, 2, 2, 3, 4, 4, 5, 6, 5, 6, 7, 8, 8], [], (3, '清一色')),
            ([0] * 3 + [2] * 3 + [4] * 3 + [6] * 3 + [8] * 2, [], (4, '清对')),
            ([0, 1, 3, 5, 9, 11, 13] * 2, [], (3, '七对')),
            ([0] * 4 + [1, 3, 5, 9, 11] * 2, [], (5, '龙七对')),
            ([0, 1, 2, 3, 4, 5, 6] * 2, [], (5, '清七对')),
            ([0] * 4 + [1, 2, 3, 5, 7] * 2, [], (6, '清龙七对')),
            ([7, 7], [('peng', i) for i in (0, 2, 9, 11)], (3, '金钩钩')),
            ([9, 9], [('peng', i) for i in (0, 2, 4, 6)], (5, '清金钩钩')),
        ]
        for tiles, melds, (expected, name) in examples:
            with self.subTest(name=name):
                self.assertEqual(wenzhou_fan(counts(tiles), melds, 'ron'), (expected, (name,)))
        hand = counts(READY + [0])
        self.assertEqual(wenzhou_fan(hand, [], 'zimo', after_kong=True), (2, ('平胡', '杠上花')))
        self.assertEqual(wenzhou_fan(hand, [], 'ron', after_kong=True), (2, ('平胡', '杠上炮')))
        self.assertEqual(wenzhou_fan(hand, [], 'zimo', opening='天胡'), (6, ('天胡',)))

    def test_heaven_and_earth_payments(self):
        game = fixture([READY + [0], CALLER, SPARE, [22] * 4 + [23] * 4 + [24] * 4 + [25]])
        game.opening = True
        game.step(('zimo', None))
        self.assertEqual(game.win_details[0].patterns, ('天胡',))
        self.assertEqual(game.scores(), [96, -32, -32, -32])
        game = fixture([SPARE + [0], CALLER, READY, READY])
        game.opening = True
        game.step(('discard', 0)); game.step(PASS); game.step(RON); game.step(PASS)
        self.assertEqual(game.win_details[0].patterns, ('地胡',))
        self.assertEqual(game.scores(), [-32, 0, 32, 0])

    def test_ron_has_priority_over_pon_and_settles_multiple_winners(self):
        game = fixture([SPARE + [0], CALLER, READY, READY])
        game.step(('discard', 0)); game.step(('peng', None)); game.step(RON); game.step(RON)
        self.assertEqual(game.winners, [2, 3])
        self.assertEqual(game.next_dealer, 0)
        self.assertEqual(game.scores(), [-2, 0, 1, 1])
        self.assertEqual(game.players[1].melds, [])
        self.assertEqual(game.physical_counts(), COPIES)

    def test_pass_water_locks_the_suit_until_own_draw(self):
        game = fixture([SPARE + [0], CALLER, READY, READY])
        game.step(('discard', 0)); game.step(PASS); game.step(PASS); game.step(PASS)
        self.assertEqual(game.cur, 1)
        game.step(('discard', 6))
        self.assertEqual(game.legal_actions(), (2, [PASS]))
        game.step(PASS); game.step(PASS); game.step(PASS)
        self.assertEqual(game.cur, 2)
        self.assertEqual(game.players[2].declined_tiles, set())
        self.assertEqual(game.players[3].declined_tiles, {0})

    def test_robbed_added_kong_transfers_the_tile_once(self):
        game = fixture([[0, 6, 6, 7, 7, 8, 8, 15, 15, 16, 16], READY, READY, SPARE],
                       melds={0: [('peng', 0)]})
        game.step(('bugang', 0)); game.step(RON); game.step(RON); game.step(PASS)
        self.assertEqual(game.phase, 'over')
        self.assertEqual(game.players[0].melds, [('peng', 0)])
        self.assertEqual(game.scores(), [-4, 2, 2, 0])
        self.assertEqual(game.next_dealer, 0)
        self.assertEqual(game.physical_counts(), COPIES)

    def test_last_discard_still_allows_wins_and_no_kong(self):
        game = fixture([SPARE + [0], CALLER, READY, READY])
        game.discards[0].extend(game.wall)
        game.wall.clear()
        game.step(('discard', 0))
        self.assertEqual(game.legal_actions(), (1, [PASS]))
        game.step(PASS); game.step(RON); game.step(PASS)
        self.assertEqual(game.scores(), [-1, 0, 1, 0])
        self.assertEqual(game.physical_counts(), COPIES)

    def test_changsha_pays_flat_kong_bonus_and_changes_dealer(self):
        wenzhou = fixture([[1, 2, 3, 4, 5, 6, 7, 8, 17, 17, 9], CALLER, SPARE,
                           [22] * 4 + [23] * 4 + [24] * 4 + [25]], melds={0: [('gang_an', 26)]})
        changsha = fixture([[1, 2, 3, 4, 5, 6, 7, 8, 17, 17, 9], CALLER, SPARE,
                            [22] * 4 + [23] * 4 + [24] * 4 + [25]], melds={0: [('gang_an', 26)]},
                           profile=Profile.CHANGSHA)
        for game in (wenzhou, changsha):
            # Three sequences and a pair remain after the concealed kong.
            game.players[0].hand[9] -= 1
            game.players[0].hand[0] += 1
            game.wall.remove(0)
            game.wall.append(9)
            game.step(('zimo', None))
            self.assertEqual(game.physical_counts(), COPIES)
        self.assertEqual(changsha.scores(), [6, -2, -2, -2])
        self.assertNotEqual(changsha.next_dealer, 0)
        self.assertEqual(wenzhou.next_dealer, 0)


class TaishunAndXuanhuaRules(unittest.TestCase):
    def test_taishun_point_schedule_and_kong_events(self):
        hands = [
            (READY + [0], '平胡', 1),
            ([0] * 3 + [9] * 3 + [11] * 3 + [13] * 3 + [16] * 2, '对对胡', 2),
            ([0] * 3 + [2] * 3 + [4] * 3 + [6] * 3 + [8] * 2, '清一色', 3),
            ([0, 1, 3, 5, 9, 11, 13] * 2, '七对', 3),
            ([0] * 4 + [1, 3, 5, 9, 11] * 2, '龙七对', 6),
        ]
        for tiles, pattern, base in hands:
            with self.subTest(pattern=pattern):
                hand = counts(tiles)
                self.assertEqual(taishun_points(hand, [], 'ron'), ((pattern,), base))
                self.assertEqual(taishun_points(hand, [], 'zimo'), ((pattern, '自摸'), 2 * base))
                self.assertEqual(taishun_points(hand, [], 'ron', after_kong=True), (('杠上炮',), 3))
                self.assertEqual(taishun_points(hand, [], 'zimo', after_kong=True), ((pattern, '杠上花'), 4 * base))

    def test_xuanhua_closed_bonus_including_concealed_kongs(self):
        for melds in ([], [('gang_an', 0)]):
            self.assertEqual(xuanhua_points(melds, 'ron'), (('平胡', '门清'), 2))
            self.assertEqual(xuanhua_points(melds, 'zimo'), (('平胡', '门清', '自摸'), 4))
        for kind in ('peng', 'gang_ming', 'gang_bu'):
            self.assertEqual(xuanhua_points([(kind, 0)], 'ron'), (('平胡',), 1))
            self.assertEqual(xuanhua_points([(kind, 0)], 'rob_kong'), (('平胡', '抢杠'), 2))

    def test_table_host_heaven_and_first_discard(self):
        for profile, expected_ron in ((Profile.TAISHUN, 6), (Profile.XUANHUA, 2)):
            with self.subTest(profile=profile):
                self.assertTrue(all(SuitedGame(seed, profile=profile).dealer == 0 for seed in range(4)))
                self.assertEqual(SuitedGame(5, 3, profile=profile).dealer, 3)
                game = fixture([READY + [0], CALLER, SPARE, [22] * 4 + [23] * 4 + [24] * 4 + [25]],
                               profile=profile)
                game.opening = True
                game.step(('zimo', None))
                self.assertEqual(game.scores(), [18, -6, -6, -6])
                self.assertEqual(game.win_details[0].patterns, ('天胡',))
                game = fixture([SPARE + [0], CALLER, READY, READY], profile=profile)
                game.opening = True
                for action in (('discard', 0), PASS, RON, RON):
                    game.step(action)
                self.assertEqual(game.scores(), [-2 * expected_ron, 0, expected_ron, expected_ron])
                self.assertEqual(game.next_dealer, 0)

    def test_taishun_later_discard_has_no_earth_bonus(self):
        game = fixture([SPARE + [0], CALLER, READY, READY], profile=Profile.TAISHUN)
        game.opening = True
        game.discard_count = 1
        for action in (('discard', 0), PASS, RON, PASS):
            game.step(action)
        self.assertEqual(game.scores(), [-1, 0, 1, 0])
        self.assertEqual(game.win_details[0].patterns, ('平胡',))

    def test_cross_suit_pass_lock_releases_on_correct_action(self):
        ready = list(range(9)) + [9, 9, 18, 18]
        hands = [[9] + [19] * 4 + [20] * 4 + [21] * 4 + [22],
                 [18] + [23] * 4 + [24] * 4 + [25] * 4,
                 ready, [10] * 4 + [11] * 4 + [12] * 4 + [13]]
        for profile in (Profile.TAISHUN, Profile.XUANHUA):
            for acquire in (PASS, ('peng', None)):
                with self.subTest(profile=profile, acquire=acquire):
                    game = fixture(hands, profile=profile)
                    game.step(('discard', 9)); game.step(PASS)
                    self.assertIn(RON, game.legal_actions()[1])
                    game.step(PASS); game.step(PASS)
                    game.step(('discard', 18))
                    self.assertEqual(game.legal_actions(), (2, [PASS, ('peng', None)]))
                    game.step(acquire); game.step(PASS); game.step(PASS)
                    self.assertEqual(game.cur, 2)
                    expected = {9} if profile is Profile.XUANHUA else set()
                    self.assertEqual(game.players[2].declined_tiles, expected)
                    game.step(('discard', 8))
                    self.assertEqual(game.players[2].declined_tiles, set())

    def test_taishun_robbery_provider_covers_three_shares(self):
        game = self.robbery(Profile.TAISHUN)
        for action in (('bugang', 0), RON, RON, PASS):
            game.step(action)
        self.assertEqual(game.scores(), [-24, 12, 12, 0])
        self.assertEqual(game.next_dealer, 0)
        self.assertEqual(game.physical_counts(), COPIES)

    def test_xuanhua_kong_robbery_only_first_accepting_seat_wins(self):
        for nearest_action, scores, winner in ((RON, [-4, 12, -4, -4], 1), (PASS, [-4, -4, 12, -4], 2)):
            with self.subTest(nearest_action=nearest_action):
                game = self.robbery(Profile.XUANHUA)
                game.players[1].declined_tiles.add(0)
                game.step(('bugang', 0)); game.step(nearest_action)
                if nearest_action == PASS:
                    game.step(RON)
                self.assertEqual(game.winners, [winner])
                self.assertEqual(game.phase, 'over')
                self.assertEqual(game.scores(), scores)
                self.assertEqual(game.next_dealer, winner)
                self.assertEqual(game.players[0].melds, [('peng', 0)])
                self.assertEqual(game.physical_counts(), COPIES)

    @staticmethod
    def robbery(profile):
        return fixture([[0, 6, 6, 7, 7, 8, 8, 15, 15, 16, 16], READY, READY, SPARE],
                       melds={0: [('peng', 0)]}, profile=profile)

    def test_kong_payments_and_taishun_unpaid_immediate_upgrade(self):
        caller = [0, 0, 0, 6, 6, 7, 7, 8, 8, 15, 15, 16, 16]
        hands = [SPARE + [0], caller, READY, READY]
        for profile, expected in ((Profile.TAISHUN, [-1, 1, 0, 0]), (Profile.XUANHUA, [-1, 3, -1, -1])):
            with self.subTest(profile=profile):
                game = fixture(hands, profile=profile)
                for action in (('discard', 0), ('zhigang', None), PASS, PASS):
                    game.step(action)
                self.assertEqual(game.scores(), expected)
                self.assertEqual(game.players[1].melds, [('gang_ming', 0)])
                self.assertEqual(game.physical_counts(), COPIES)
                while game.phase != 'over':
                    _, acts = game.legal_actions()
                    game.step(acts[0])
                self.assertEqual(game.scores(), expected)
                self.assertEqual(game.next_dealer, 0)

        game = fixture(hands, profile=Profile.TAISHUN)
        for action in (('discard', 0), ('peng', None), PASS, PASS):
            game.step(action)
        self.assertNotIn(('zimo', None), game.legal_actions()[1])
        game.step(('bugang', 0))
        self.assertEqual(game.legal_actions(), (2, [PASS, RON]))
        game.step(PASS); game.step(PASS); game.step(PASS)
        self.assertEqual(game.players[1].melds, [('gang_bu', 0)])
        self.assertEqual(game.physical_counts(), COPIES)
        self.assertEqual(game.scores(), [0, 0, 0, 0])


class SuitedIntegration(unittest.TestCase):
    def test_three_hundred_complete_random_hands_per_profile(self):
        for profile in Profile:
            for seed in range(300):
                game = SuitedGame(seed, profile=profile)
                rng = random.Random(seed + 20261118)
                for _ in range(500):
                    if game.phase == 'over':
                        break
                    _, acts = game.legal_actions()
                    game.step(rng.choice(acts))
                    self.assertEqual(game.physical_counts(), COPIES)
                    self.assertEqual(sum(game.scores()), 0)
                self.assertEqual(game.phase, 'over')

    def test_session_and_public_policy(self):
        for profile in Profile:
            game = SuitedGame(9, 0, profile=profile)
            actor, actions = game.legal_actions()
            before = policy_view(game, actor, actions)
            game.wall.reverse()
            game.players[1].hand.reverse()
            self.assertEqual(before, policy_view(game, actor, actions))
            self.assertIn(heuristic(before), actions)
            session = SuitedSession(seed=10, profile=profile, pacing=0)
            self.addCleanup(session.close)
            session.begin()
            session.auto()
            until = time.monotonic() + 20
            while session.status not in ('round_end', 'error') and time.monotonic() < until:
                time.sleep(.01)
            self.assertEqual(session.status, 'round_end', session.error)
            snap = session.snapshot()
            self.assertEqual(snap['variant'], profile.value)
            self.assertEqual(sum(snap['result']['scores']), 0)
            self.assertTrue(all(not player['hand'] or player['won'] for player in snap['table']['players'][1:]))
            expected_dealer = session.next_dealer
            session.begin()
            self.assertEqual(session.game.dealer, expected_dealer)
