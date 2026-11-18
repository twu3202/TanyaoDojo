import random
import unittest
from collections import Counter

from deploy.play.kaxinwu import KaxinwuGame, OPEN_KONG, PON, VARIANT, ZIMO, _base_score
from deploy.play.suited import PASS, RON, Offer
from deploy.play.suited_game import SuitedSession, policy_view, suited_policy
from deploy.play.tests.test_games import wait
from deploy.play.tests.test_suited import counts


READY = [9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 23, 23]
FIVE_DOTS = 13


class KaxinwuTests(unittest.TestCase):
    def test_middle_five_uses_the_winning_tile_and_luxury_pairs_do_not_add_roots(self):
        hand = tuple(counts(READY + [FIVE_DOTS]))
        self.assertEqual(_base_score(hand, (), FIVE_DOTS), (2, ('卡心五',), 0))
        self.assertEqual(_base_score(hand, (), 10), (1, ('平胡',), 0))
        pairs = counts([9] * 4 + [10] * 4 + [11] * 4 + [12] * 2)
        self.assertEqual(_base_score(tuple(pairs), (), 12), (6, ('三龙七对', '清一色'), 0))

    def test_same_seats_can_win_twice_and_multi_win_restarts_at_provider(self):
        game = KaxinwuGame(1)
        remaining = game.copies.copy()
        remaining.subtract(Counter(READY * 2 + [FIVE_DOTS] * 2))
        leftovers = list(remaining.elements())
        hands = [leftovers[:13] + [FIVE_DOTS], READY, READY, leftovers[13:26]]
        for player, tiles in zip(game.players, hands):
            player.hand = counts(tiles)
        game.drawn_tile = FIVE_DOTS
        game.wall = leftovers[26:] + [FIVE_DOTS]
        self.assertEqual(game.physical_counts(), game.copies)
        for round_wins in (1, 2):
            game.step(('discard', FIVE_DOTS))
            for action in (RON, RON, PASS):
                game.step(action)
            self.assertEqual(game.win_counts, [0, round_wins, round_wins, 0])
            self.assertEqual(game.won_tiles, [FIVE_DOTS] * round_wins)
            self.assertEqual(game.scores(), [-8 * round_wins, 4 * round_wins, 4 * round_wins, 0])
            self.assertEqual((game.phase, game.cur), ('action', 0))
            self.assertEqual(game.physical_counts(), game.copies)
        self.assertEqual(game.players[1].hand, counts(READY))
        self.assertEqual(game.players[2].hand, counts(READY))
        seat, actions = game.legal_actions()
        inputs = policy_view(game, seat, actions)
        self.assertEqual(inputs['public'][FIVE_DOTS], 2)
        game.players[1].hand.reverse()
        self.assertEqual(policy_view(game, seat, actions), inputs)

    def test_locked_hand_discards_only_the_draw(self):
        game = KaxinwuGame(1)
        game.players[0].hand = counts(READY + [FIVE_DOTS])
        game.drawn_tile = FIVE_DOTS
        game.step(ZIMO)
        self.assertEqual(game.win_counts[0], 1)
        self.assertEqual(game.players[0].hand, counts(READY))
        game.wall.append(22)
        game._draw(0)
        discards = [move for move in game.legal_actions()[1] if move[0] == 'discard']
        self.assertEqual(discards, [('discard', 22)])

    def test_mandatory_locked_open_kong_can_narrow_waits_but_not_lose_a_won_rank(self):
        game = KaxinwuGame(1)
        player = game.players[1]
        player.hand = counts([14, 15, 16, 19, 20, 21, 21, 22, 23, 23, 24, 24, 24])
        self.assertEqual(game._wait_tiles(player.hand, []), {20, 22, 23, 25})
        game._offer(Offer(0, 24, 'ron'))
        game.won_ranks[1] = {20, 23}
        self.assertEqual(game.legal_actions()[1], [OPEN_KONG])
        game.won_ranks[1].add(22)
        self.assertEqual(game.legal_actions()[1], [PASS])
        game._offer(Offer(0, 23, 'ron'))
        self.assertIn(RON, game.legal_actions()[1])
        self.assertNotIn(PON, game.legal_actions()[1])

    def test_consecutive_kong_adds_base_kong_payments_and_extra_win_fan(self):
        game = KaxinwuGame(2)
        game.players[0].hand = counts([9] * 4 + [10] * 4 + [11, 12, 13, 14, 15, 17])
        game.drawn_tile = 9
        game.step(('ankan', 9))
        self.assertEqual(game.kong_deltas, [6, -2, -2, -2])
        game.step(('ankan', 10))
        self.assertEqual(game.kong_deltas, [18, -6, -6, -6])
        self.assertEqual(game.kong_chain, 2)
        plain = counts(READY + [FIVE_DOTS])
        self.assertEqual(game._score(plain, [], FIVE_DOTS, 'zimo', 2),
                         (4, ('卡心五', '双杠上花'), 2))
        self.assertEqual(game._score(plain, [], FIVE_DOTS, 'ron', 2),
                         (3, ('卡心五', '杠上炮'), 2))

    def test_robbed_added_kong_keeps_pon_and_resumes_without_kong_payment(self):
        game = KaxinwuGame(4)
        game.players[0].hand = counts([FIVE_DOTS])
        game.players[0].melds = [('peng', FIVE_DOTS)]
        game.players[1].hand = counts(READY)
        game.drawn_tile = FIVE_DOTS
        game.step(('bugang', FIVE_DOTS))
        for action in (RON, PASS, PASS):
            game.step(action)
        self.assertEqual(game.players[0].melds, [('peng', FIVE_DOTS)])
        self.assertEqual(game.players[0].hand[FIVE_DOTS], 0)
        self.assertEqual(game.win_details[0].kind, 'rob_kong')
        self.assertEqual(game.won_tiles, [FIVE_DOTS])
        self.assertEqual(game.kong_deltas, [0] * 4)
        self.assertEqual(game.scores(), [-8, 8, 0, 0])
        self.assertEqual(game.cur, 1)

    def test_passed_win_locks_tile_and_higher_offer_can_win(self):
        game = KaxinwuGame(4)
        game.players[1].hand = counts(READY)
        game._offer(Offer(0, FIVE_DOTS, 'ron'))
        game.step(PASS)
        self.assertEqual(game.passed_wins[1], {FIVE_DOTS: 2})
        self.assertFalse(game._can_take_offer(1, Offer(3, FIVE_DOTS, 'ron')))
        self.assertTrue(game._can_take_offer(1, Offer(3, FIVE_DOTS, 'ron', after_kong=True)))
        game._draw(1)
        self.assertFalse(game.passed_wins[1])

    def test_passed_self_draw_allows_first_equal_offer_and_locks_second(self):
        game = KaxinwuGame(4)
        game.players[2].hand = counts(READY + [FIVE_DOTS])
        game.cur = 2
        game.drawn_tile = FIVE_DOTS
        game.step(('discard', FIVE_DOTS))
        self.assertEqual(game.passed_self_draw[2], 2)
        game._offer(Offer(1, FIVE_DOTS, 'ron'))
        self.assertIn(RON, game.legal_actions()[1])
        game.step(PASS)
        self.assertTrue(game.first_equal_seen[2])
        self.assertNotIn(RON, game._offered_actions(2))
        self.assertTrue(game._can_take_offer(2, Offer(0, FIVE_DOTS, 'rob_kong')))

    def test_draw_settlement_checks_best_past_shape_and_keeps_kong_points(self):
        game = KaxinwuGame(4)
        for player in game.players:
            player.hand = [0] * 27
        game.players[1].hand = counts(READY)
        game.win_counts[2] = 3
        game.best_scored_shape[2] = 3
        game.kong_deltas[:] = [2, 0, 0, -2]
        game._pay(3, 0, 2)
        game.last_drawer = 2
        game.wall.clear()
        game._draw(3)
        self.assertEqual(game.ready_deltas, [-12, 8, 16, -12])
        self.assertEqual(game.scores(), [-10, 8, 16, -14])
        self.assertEqual((game.phase, game.next_dealer), ('over', 2))

    def test_random_and_ai_hands_preserve_tiles_and_reconcile_every_payment(self):
        for seed in range(24):
            game = KaxinwuGame(seed, seed % 4)
            rng = random.Random(seed)
            while game.phase != 'over':
                actor, acts = game.legal_actions()
                action = suited_policy(policy_view(game, actor, acts)) if seed % 2 else rng.choice(acts)
                game.step(action)
                self.assertEqual(game.physical_counts(), game.copies)
                self.assertEqual(sum(game.scores()), 0)
            self.assertFalse(game.wall)
            expected = [kon + ready for kon, ready in zip(game.kong_deltas, game.ready_deltas)]
            for win in game.win_details:
                expected[win.actor] += sum(win.payments)
                for payer, amount in enumerate(win.payments):
                    expected[payer] -= amount
            self.assertEqual(expected, game.scores())

    def test_session_shows_live_win_events_without_concealed_hands_and_runs_next_hand(self):
        session = SuitedSession(profile=VARIANT, seed=5, pacing=0)
        try:
            session.begin()
            wait(session, {'your_turn'})
            self.assertEqual(session.snapshot()['table']['remaining'], 19)
            session.auto()
            wait(session, {'round_end'})
            snap = session.snapshot()
            self.assertEqual(snap['result']['reason_code'], 'repeat_win_end')
            self.assertTrue(all(not p['hand'] for p in snap['table']['players'][1:]))
            self.assertGreater(max(p['win_count'] for p in snap['table']['players']), 1)
            self.assertEqual(sum(m['code'] == 'flow_win' for m in snap['messages']), len(snap['result']['win_details']))
            self.assertTrue(all(win['tile'][1] in 'ps' for win in snap['result']['win_details']))
            session.begin()
            wait(session, {'your_turn'})
            next_table = session.snapshot()['table']
            self.assertEqual(next_table['hand_number'], 2)
            self.assertFalse(next_table['winning_tiles'])
            self.assertEqual([p['score'] for p in next_table['players']], snap['result']['scores'])
        finally:
            session.close()

    def test_manual_session_skips_forced_discard_after_a_win(self):
        session = SuitedSession(profile=VARIANT, pacing=0)
        session.game = KaxinwuGame(1)

        def play_prompt(table, options, preferred):
            if table['players'][0]['win_count']:
                self.assertNotEqual([option['kind'] for option in options], ['discard'])
            return options[preferred]['action']

        session.ask = play_prompt
        session.play_round()
        human_actions = [message['code'] for message in session.messages
                         if message['params'].get('actor') == 0]
        self.assertIn('discard', human_actions[human_actions.index('flow_win') + 1:])
