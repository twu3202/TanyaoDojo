from collections import Counter
import random
import time
import unittest

from deploy.play import hongkong
from deploy.play.honors import COPIES, ORPHANS, PASS, RON, ZIMO, HonorsGame, Meld, WinContext, standard_shapes
from deploy.play.honors_game import HongKongSession
from deploy.play.honors_policy import heuristic, policy_view, standard_distance


SEQUENCES = [0, 1, 2, 3, 4, 5, 9, 10, 11, 15, 16, 17, 22, 22]
FLUSH_EAST = list(range(1, 9)) + [27] * 3 + [28] * 2
FLUSH_DRAGON = list(range(1, 9)) + [31] * 3 + [32] * 2


def counts(tiles):
    return [tiles.count(tile) for tile in range(34)]


def fixture(hands, melds=None, *, drawn=0, opening=False):
    game = HonorsGame(0, 0, 27, rules=hongkong)
    used = Counter(tile for tiles in hands.values() for tile in tiles)
    for actor, player in enumerate(game.players):
        player.melds = (melds or {}).get(actor, []).copy()
        for meld in player.melds:
            used.update(meld.tiles)
    assert all(n <= 4 for n in used.values())
    remain = sorted((COPIES - used).elements())
    for actor, player in enumerate(game.players):
        tiles = hands.get(actor, []).copy()
        expected = 13 + (actor == 0) - 3 * len(player.melds)
        if actor not in hands:
            tiles = remain[-expected:]
            del remain[-expected:]
        assert len(tiles) == expected
        player.hand = counts(tiles)
    game.wall = remain
    game.drawn_tile = drawn
    game.opening = opening
    assert game.physical_counts() == COPIES
    return game


class FaanRules(unittest.TestCase):
    def test_three_faan_threshold_and_exposed_chow(self):
        hand = SEQUENCES.copy()
        draw = WinContext('zimo', 22, 28, 27)
        self.assertIsNone(hongkong.faan(counts(hand), (), WinContext('ron', 22, 28, 27)))
        self.assertEqual(hongkong.faan(counts(hand), (), draw), hongkong.Faan(3, ('平胡', '门清', '自摸')))
        self.assertIsNone(hongkong.faan(counts(hand[3:]), (Meld('chi', 0, 1),), draw))
        self.assertEqual(standard_distance(tuple(counts(hand)), 0), -1)
        self.assertEqual(standard_distance(tuple(counts(hand[:-1])), 0), 0)
        pairs = counts([0, 2, 5, 11, 16, 20, 30] * 2)
        self.assertIsNone(hongkong.faan(pairs, (), draw))
        self.assertGreater(standard_distance(tuple(pairs), 0), -1)

    def test_multiple_interpretations_choose_higher_faan(self):
        hand = counts([0]*3 + [1]*3 + [2]*3 + [24]*3 + [22, 22])
        self.assertEqual(len(list(standard_shapes(hand, 0))), 2)
        self.assertEqual(hongkong.faan(hand, (), WinContext('ron', 22, 28, 27)),
                         hongkong.Faan(8, ('坎坎胡',)))

    def test_east_twice_and_small_dragons_table(self):
        hand = counts([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27, 22, 22])
        result = hongkong.faan(hand, (), WinContext('ron', 22, 27, 27))
        self.assertEqual(result, hongkong.Faan(3, ('门风刻', '圈风刻', '门清')))
        dragons = tuple(Meld('peng', tile, 1) for tile in (31, 32))
        hand = counts([0, 1, 2, 9, 10, 11, 33, 33])
        self.assertEqual(hongkong.faan(hand, dragons, WinContext('ron', 33, 28, 27)),
                         hongkong.Faan(4, ('小三元',)))

    def test_payment_schedule_and_liability(self):
        for fan, share in enumerate((16, 32, 48, 64, 96, 128), start=3):
            self.assertEqual(hongkong.payments(fan, 'ron', 1, 2, liable=3), (0, 0, 2*share, 0))
            self.assertEqual(hongkong.payments(fan, 'zimo', 1), (share, 0, share, share))
            self.assertEqual(hongkong.payments(fan, 'zimo', 1, liable=0), (3*share, 0, 0, 0))
            self.assertEqual(hongkong.payments(fan, 'rob_kong', 1, 3, liable=0), (0, 0, 0, 3*share))
        melds = [Meld('peng', 31, 1), Meld('gang_bu', 32, 2), Meld('peng', 33, 0)]
        self.assertEqual(hongkong.liability(melds, melds[-1]), 0)
        melds.append(Meld('chi', 0, 2))
        self.assertEqual(hongkong.liability(melds, melds[-1]), 2)


class HonorsGameRules(unittest.TestCase):
    def test_chow_only_from_preceding_seat_and_kuikae(self):
        game = fixture({0: [0] + [18]*4 + [19]*4 + [20]*4 + [21],
                        1: [0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 27, 28, 33],
                        2: [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]})
        game.step(('discard', 0))
        self.assertIn(('chi', 0), game.legal_actions()[1])
        game.step(('chi', 0))
        self.assertNotIn(('chi', 0), game.legal_actions()[1])
        game.step(PASS)
        game.step(PASS)
        self.assertEqual(game.cur, 1)
        self.assertEqual(game.players[1].melds, [Meld('chi', 0, 0)])
        self.assertNotIn(('discard', 0), game.legal_actions()[1])
        self.assertNotIn(('discard', 3), game.legal_actions()[1])
        self.assertNotIn(ZIMO, game.legal_actions()[1])
        self.assertEqual(game.physical_counts(), COPIES)

    def test_pon_beats_chow_and_locks_repeated_declined_tile(self):
        hands = {0: [0] + [18]*4 + [19]*4 + [20]*4 + [21],
                 1: [0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 27, 28, 33],
                 2: [0, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}
        game = fixture(hands)
        for action in (('discard', 0), ('chi', 0), ('peng', None), PASS):
            game.step(action)
        self.assertEqual(game.cur, 2)
        self.assertEqual(game.players[2].melds, [Meld('peng', 0, 0)])
        self.assertEqual(game.players[1].melds, [])
        self.assertEqual(game.physical_counts(), COPIES)

        game = fixture(hands)
        for action in (('discard', 0), PASS, PASS, PASS, ('discard', 0)):
            game.step(action)
        self.assertNotIn(('peng', None), game.legal_actions()[1])
        game.step(PASS)
        game.step(PASS)
        game.step(PASS)
        self.assertEqual(game.players[2].declined_calls, set())

    def test_multiple_ron_beat_chow_and_each_is_paid(self):
        game = fixture({0: [0] + [18]*4 + [19]*4 + [20]*4 + [21],
                        1: [1, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17, 27, 33],
                        2: FLUSH_EAST, 3: FLUSH_DRAGON})
        for action in (('discard', 0), ('chi', 0), RON, RON):
            game.step(action)
        self.assertEqual(game.winners, [2, 3])
        self.assertEqual(game.scores(), [-192, 0, 96, 96])
        self.assertEqual(game.next_dealer, 1)
        self.assertEqual(game.players[1].melds, [])
        self.assertEqual(game.physical_counts(), COPIES)

    def test_ordinary_robbery_and_concealed_kong_exception(self):
        game = fixture({0: [0, 3, 4, 5, 6, 7, 8, 13, 18, 22, 24], 1: SEQUENCES[1:]},
                       {0: [Meld('peng', 0, 3)]})
        for action in (('bugang', 0), RON, PASS, PASS):
            game.step(action)
        self.assertEqual(game.scores(), [-48, 48, 0, 0])
        self.assertEqual(game.players[0].melds, [Meld('peng', 0, 3)])
        self.assertEqual(game.physical_counts(), COPIES)

        waiting = sorted(ORPHANS - {0}) + [28]
        game = fixture({0: [0]*4 + [3, 4, 5, 6, 7, 8, 13, 18, 22, 24],
                        1: SEQUENCES[1:], 2: waiting})
        game.step(('ankan', 0))
        self.assertEqual(game.legal_actions(), (1, [PASS]))
        for action in (PASS, RON, PASS):
            game.step(action)
        self.assertEqual(game.reason_code, 'rob_concealed_win')
        self.assertEqual(game.scores(), [-384, 0, 384, 0])
        self.assertEqual(game.players[0].hand[0], 3)
        self.assertEqual(game.physical_counts(), COPIES)

    def test_only_initial_dealer_win_or_discard_is_opening_limit(self):
        game = fixture({0: SEQUENCES}, drawn=22, opening=True)
        game.step(ZIMO)
        self.assertEqual(game.scores(), [384, -128, -128, -128])
        self.assertEqual(game.win_details[0].patterns, ('天胡',))
        self.assertEqual(game.next_dealer, 0)
        game = fixture({0: [0] + [18]*4 + [19]*4 + [20]*4 + [21], 1: SEQUENCES[1:]}, opening=True)
        for action in (('discard', 0), RON, PASS, PASS):
            game.step(action)
        self.assertEqual(game.scores(), [-256, 256, 0, 0])
        self.assertEqual(game.win_details[0].patterns, ('地胡',))

    def test_last_draw_allows_kong_but_no_replacement(self):
        game = fixture({0: [0]*4 + [3, 4, 5, 6, 7, 8, 13, 18, 22, 24]})
        game.discards[3].extend(game.wall)
        game.wall.clear()
        for action in (('ankan', 0), PASS, PASS, PASS):
            game.step(action)
        self.assertEqual(game.phase, 'over')
        self.assertEqual(game.reason_code, 'draw')
        self.assertEqual(game.next_dealer, 1)
        self.assertEqual(game.physical_counts(), COPIES)


class HongKongIntegration(unittest.TestCase):
    def test_random_hands_preserve_tiles_and_points_after_each_action(self):
        for seed in range(300):
            rng = random.Random(seed)
            game = HonorsGame(seed, seed % 4, 27 + seed % 4, rules=hongkong)
            for _ in range(600):
                if game.phase == 'over':
                    break
                _, actions = game.legal_actions()
                game.step(rng.choice(actions))
                self.assertEqual(game.physical_counts(), COPIES)
                self.assertEqual(sum(game.scores()), 0)
            self.assertEqual(game.phase, 'over')

    def test_bots_and_session_use_public_information(self):
        game = HonorsGame(11, 0, 27, rules=hongkong)
        actor, actions = game.legal_actions()
        before = policy_view(game, actor, actions)
        game.wall.reverse()
        game.players[1].hand, game.players[2].hand = game.players[2].hand, game.players[1].hand
        self.assertEqual(before, policy_view(game, actor, actions))
        self.assertIn(heuristic(before), actions)
        session = HongKongSession(seed=11, pacing=0)
        self.addCleanup(session.close)
        session.begin()
        session.auto()
        until = time.monotonic() + 10
        while session.status not in ('round_end', 'error') and time.monotonic() < until:
            time.sleep(.01)
        snapshot = session.snapshot()
        self.assertEqual(snapshot['status'], 'round_end', snapshot['error'])
        self.assertEqual(sum(snapshot['result']['scores']), 0)
        self.assertTrue(all(not p['hand'] or p['won'] for p in snapshot['table']['players'][1:]))
        next_dealer = session.game.next_dealer
        session.begin()
        self.assertEqual(session.game.dealer, next_dealer)
        self.assertEqual(session.hand_number, 2)
