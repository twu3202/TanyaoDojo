import random
import time
import unittest
from collections import Counter

from deploy.play.suited import COPIES, PASS, RON, Profile, SuitedGame, wenzhou_fan
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
        self.assertEqual(game.players[2].declined_suits, set())
        self.assertEqual(game.players[3].declined_suits, {0})

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
