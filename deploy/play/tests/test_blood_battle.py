import unittest
from collections import Counter

from deploy.play.blood_battle import (
    EXCHANGE_DIRECTIONS, SICHUAN_EXCHANGE, YUNNAN, BloodBattleGame, blood_fan,
)
from deploy.play.errors import ClientError
from deploy.play.sichuan_game import SichuanSession, public_features
from deploy.play.tests.test_games import wait


def counts(tiles):
    hand = [0] * 27
    for tile in tiles:
        hand[tile] += 1
    return hand


READY = (0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 7)


def offer(game, source, tile, *seats, after_kong=False):
    game.discards[source].append(tile)
    game.pending_discard = (source, tile)
    game.phase = 'response'
    game.response_queue = list(seats)
    game.ron_winners = []
    game.pending_claim = None
    game.discard_was_gang = after_kong


class BloodBattleTests(unittest.TestCase):
    def test_simultaneous_same_suit_exchange_and_opening_dealer(self):
        directions = set()
        for seed in range(18):
            game = BloodBattleGame(seed, SICHUAN_EXCHANGE, dealer=2)
            dealt = [player.hand[:] for player in game.players]
            self.assertEqual(list(map(sum, dealt)), [13, 13, 14, 13])
            with self.assertRaises(ClientError):
                game.step(('exchange', (0, 9, 18)))
            groups = []
            for seat in range(4):
                self.assertEqual(game.legal_actions()[0], seat)
                actions = game.legal_actions()[1]
                self.assertTrue(all(len({t // 9 for t in group}) == 1 for _, group in actions))
                self.assertIsNone(game.exchange_direction)
                self.assertEqual([player.hand for player in game.players], dealt)
                _, group = actions[-1]
                groups.append(group)
                game.step(('exchange', group))
            offsets = {name: offset for offset, name in EXCHANGE_DIRECTIONS.items()}
            offset = offsets[game.exchange_direction]
            directions.add(offset)
            self.assertEqual(game.phase, 'void')
            for seat, player in enumerate(game.players):
                for tile in range(27):
                    self.assertEqual(player.hand[tile], dealt[seat][tile] - groups[seat].count(tile)
                                     + groups[(seat - offset) % 4].count(tile))
                game.step(('void', 0))
            self.assertEqual((game.cur, game.phase), (2, 'action'))
            self.assertGreater(game.players[2].hand[game.drawn_tile], 0)
            self.assertEqual(game.tile_conservation(), 108)
        self.assertEqual(directions, {1, 2, -1})

    def test_yunnan_pig_pays_winner_and_ready_penalty_stacks(self):
        game = BloodBattleGame(0, YUNNAN, dealer=3)
        for player in game.players:
            player.hand = [0] * 27
            player.void = 2
        game.players[0].hand[18] = 1
        game.players[1].hand = counts(READY)
        game.players[2].hu = True
        game.hu_order.append((2, 0, ('平胡',)))
        game.next_dealer = 2
        game._pay(2, 0, 2, gang=True)
        game._pay(3, 1, 2, gang=True)
        game.wall.clear()
        game._begin_turn(0)
        self.assertEqual(game.pig_deltas, [-48, 16, 16, 16])
        self.assertEqual(game.ready_deltas, [-1, 2, 0, -1])
        self.assertEqual(game.refund_deltas, [-2, 0, 2, 0])
        self.assertEqual(game.scores(), [-49, 20, 16, 13])
        self.assertEqual((game.phase, game.next_dealer), ('over', 3))

    def test_three_winners_end_before_exhaustive_payments(self):
        game = BloodBattleGame(0, YUNNAN)
        for seat in range(3):
            game.players[seat].hu = True
            game.hu_order.append((seat, 0, ('平胡',)))
        game.wall.clear()
        game._begin_turn(3)
        self.assertEqual(game.phase, 'over')
        self.assertFalse(game.exhausted)
        self.assertEqual(game.scores(), [0, 0, 0, 0])

    def test_passed_win_and_pon_locks(self):
        game = BloodBattleGame(0, YUNNAN)
        game.players[1].hand = counts(READY)
        game.players[1].void = 2
        offer(game, 0, 7, 1, 3)
        game.step(('pass', None))
        self.assertEqual(game.passed_win[1], 0)
        offer(game, 2, 7, 1, 3)
        self.assertNotIn(('ron', None), game.legal_actions()[1])
        offer(game, 2, 7, 1, 3, after_kong=True)
        self.assertIn(('ron', None), game.legal_actions()[1])

        game.players[1].hand = counts([4, 4])
        offer(game, 0, 4, 1, 3)
        game.step(('pass', None))
        offer(game, 2, 4, 1, 3)
        self.assertNotIn(('peng', None), game.legal_actions()[1])
        game._begin_turn(1)
        self.assertEqual(game.passed_win[1], -1)
        self.assertFalse(game.passed_pon[1])

    def test_locked_robber_does_not_trigger_reference_autorob(self):
        game = BloodBattleGame(0, YUNNAN)
        game.players[0].hand = counts([7])
        game.players[0].melds = [('peng', 7)]
        game.players[1].hand = counts(READY)
        game.passed_win[1] = 1
        game.phase = 'action'
        game.drawn_tile = 7
        game.step(('bugang', 7))
        self.assertEqual(game.players[0].melds, [('gang_bu', 7)])
        self.assertEqual(game.scores(), [3, -1, -1, -1])
        self.assertFalse(game.hu_order)

    def test_sichuan_own_kong_with_void_tiles_and_late_pon(self):
        game = BloodBattleGame(2, SICHUAN_EXCHANGE)
        game.phase = 'action'
        game.players[0].hand = counts([4, 4, 4, 4, 18])
        game.players[0].void = 2
        game.drawn_tile = 4
        self.assertEqual(game.legal_actions()[1], [('discard', 18), ('ankan', 4)])
        game.wall.clear()
        self.assertEqual(game.legal_actions()[1], [('discard', 18)])
        game.players[1].hand = counts([4, 4, 4])
        offer(game, 0, 4, 1, 2, 3)
        self.assertEqual(game.legal_actions()[1], [('pass', None), ('peng', None)])

    def test_sichuan_pon_instead_of_kong_leaves_unpaid_later_kong(self):
        game = BloodBattleGame(2, SICHUAN_EXCHANGE)
        game.players[1].hand = counts([4, 4, 4, 5])
        offer(game, 0, 4, 1)
        game.step(('peng', None))
        self.assertEqual(game.legal_actions()[1], [('discard', 4), ('discard', 5)])
        game._begin_turn(1)
        self.assertIn(('bugang', 4), game.legal_actions()[1])
        game.step(('bugang', 4))
        self.assertEqual(game.players[1].melds, [('gang_bu', 4)])
        self.assertEqual(game.scores(), [0, 0, 0, 0])

    def test_yunnan_multiron_and_visible_information(self):
        game = BloodBattleGame(0, YUNNAN)
        for seat in (1, 2):
            game.players[seat].hand = counts(READY)
            game.players[seat].void = 2
        offer(game, 0, 7, 1, 2, 3)
        for action in (('ron', None), ('ron', None), ('pass', None)):
            game.step(action)
        self.assertEqual(game.next_dealer, 0)
        self.assertEqual(game.winning_discards, {1: 7, 2: 7})
        self.assertEqual(game.shown_claimed_discards, [7])
        self.assertEqual(public_features(game, 3, [])['public'], {7: 1})
        session = SichuanSession(variant=YUNNAN)
        session.game = game
        players = session.view()['players']
        self.assertEqual([players[i]['win_tile'] for i in (1, 2)], ['8m', '8m'])
        self.assertTrue(all(not players[i]['hand'] for i in (1, 2)))

    def test_pattern_stacks(self):
        dragon = counts([0] * 4 + [1] * 4 + [2] * 4 + [3] * 2)
        self.assertEqual(blood_fan(dragon, [], {'zimo': True}, yunnan=True),
                         (7, ('清三龙七对',)))
        self.assertEqual(blood_fan(dragon, [], {'zimo': True}, yunnan=False),
                         (7, ('清龙七对', '门清', '自摸')))
        hand = counts([2, 2])
        melds = [('gang_an', tile) for tile in (3, 4, 5, 6)]
        self.assertEqual(blood_fan(hand, melds, {'gang_shang_hua': True}, yunnan=True),
                         (9, ('清十八罗汉', '杠上花')))

    def test_full_session_tile_and_payment_conservation_and_next_dealer(self):
        for variant in (SICHUAN_EXCHANGE, YUNNAN):
            with self.subTest(variant=variant):
                session = SichuanSession(variant=variant, seed=17, pacing=0)
                try:
                    session.begin()
                    wait(session, {'your_turn'})
                    self.assertEqual(session.game.phase, 'exchange' if variant == SICHUAN_EXCHANGE else 'void')
                    session.auto()
                    wait(session, {'round_end'})
                    result = session.snapshot()['result']
                    payments = Counter()
                    for win in result['win_details']:
                        payments[win['actor']] += sum(win['payments'])
                        for seat, amount in enumerate(win['payments']):
                            payments[seat] -= amount
                    for key in ('kong_deltas', 'pig_deltas', 'ready_deltas', 'refund_deltas'):
                        for seat, amount in enumerate(result[key]):
                            payments[seat] += amount
                    self.assertEqual([payments[seat] for seat in range(4)], result['deltas'])
                    self.assertEqual(sum(result['deltas']), 0)
                    tiles = Counter(session.game.wall)
                    for seat, player in enumerate(session.game.players):
                        tiles.update(session.game.discards[seat])
                        tiles.update(dict(enumerate(player.hand)))
                        for kind, tile in player.melds:
                            tiles[tile] += 4 if kind.startswith('gang') else 3
                    self.assertEqual(tiles, Counter({t: 4 for t in range(27)}))
                    dealer = session.game.next_dealer
                    session.begin()
                    wait(session, {'your_turn'})
                    self.assertEqual(session.snapshot()['table']['dealer'], dealer)
                    self.assertEqual(session.game.dealer, dealer)
                finally:
                    session.close()
