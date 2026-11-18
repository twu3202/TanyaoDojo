import random
import unittest

from deploy.play.hangzhou import (CHUNAN, FLOAT, FOUR_HANGZHOU, THREE_HANGZHOU, TONGLU, TWO_HANGZHOU,
                                  WHITE, HangzhouGame, score_hand)
from deploy.play.honors import Meld, Offer, PASS, RON, ZIMO, WinContext
from deploy.play.honors_game import HonorsSession
from deploy.play.honors_policy import heuristic, policy_view
from deploy.play.tests.test_games import wait
from deploy.play.tests.test_hongkong import counts
from deploy.play.wildcard_shapes import can_form_sets


FOUR_SETS = [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27]


def play_passes(game):
    while game.offer is not None:
        game.step(PASS)


class HangzhouScoringTests(unittest.TestCase):
    def score(self, tiles, drawn, *, kongs=0, floats=0, before=0, melds=0, kind='zimo'):
        return score_hand(counts(tiles), melds, WinContext(kind, drawn, 27, 27, kongs), floats, before)

    def test_wildcard_sets_respect_suit_boundaries_and_honors(self):
        self.assertTrue(can_form_sets(tuple(counts([1, 2, 32, 32])), 2))
        self.assertFalse(can_form_sets(tuple(counts([8, 9])), 1))
        self.assertFalse(can_form_sets(tuple(counts([27, 28])), 1))

    def test_burst_requires_the_pre_draw_wait(self):
        for drawn in (8, 27, WHITE, 33):
            score = self.score(FOUR_SETS + [WHITE, drawn], drawn)
            self.assertEqual((score.share, score.patterns), (2, ('爆头',)))
        ordinary = FOUR_SETS[1:] + [WHITE, 33, 33]
        self.assertEqual(self.score(ordinary, 33).share, 1)
        self.assertIsNone(self.score(ordinary, 33, kind='ron'))
        self.assertIsNone(self.score(ordinary, 33, kind='rob_kong'))

    def test_kong_float_order_and_pairs(self):
        hand = FOUR_SETS[3:] + [WHITE, 33]
        for kongs, floats, before, label, share in (
            (3, 0, 0, '三连杠暴', 16), (1, 1, 0, '飘杠', 8),
            (0, 1, 1, '杠飘', 8), (0, 3, 0, '三财飘', 16),
        ):
            score = self.score(hand, 33, kongs=kongs, floats=floats, before=before, melds=1)
            self.assertEqual((score.share, score.patterns), (share, (label,)))
        natural = [0] * 4 + [3] * 4 + [9] * 4 + [32, 32]
        score = self.score(natural, 32)
        self.assertEqual((score.share, score.patterns), (32, ('三豪华清七对',)))
        six_pairs = [0] * 4 + [3, 3, 7, 7, 12, 12, 32, 32]
        score = self.score(six_pairs + [WHITE, 27], 27, floats=1)
        self.assertEqual((score.share, score.patterns), (16, ('豪华七对', '七客', '财飘')))


class HangzhouPlayTests(unittest.TestCase):
    def test_two_floats_force_draw_discards_and_pay_fixed_dealer_share(self):
        game = HangzhouGame(23, 0, 27, rules=THREE_HANGZHOU)
        game.players[0].hand = counts(FOUR_SETS + [WHITE, WHITE])
        game.drawn_tile = WHITE
        game.wall = [2, 8, 21, 22, WHITE, 24, 25]
        for float_count in (1, 2):
            self.assertIn(FLOAT, game.legal_actions()[1])
            game.step(FLOAT)
            self.assertEqual(game.float_streak[0], float_count)
            for seat in (1, 2):
                play_passes(game)
                actor, actions = game.legal_actions()
                self.assertEqual(actor, seat)
                self.assertEqual([action for action in actions if action[0] == 'discard'],
                                 [('discard', game.drawn_tile)])
                game.step(('discard', game.drawn_tile))
            play_passes(game)
            self.assertEqual(game.cur, 0)
            self.assertIsNone(game.float_cycle_owner)
        game.step(ZIMO)
        self.assertEqual(game.win_details[0].patterns, ('双财飘',))
        self.assertEqual(game.win_details[0].payments, (0, 64, 64))
        self.assertEqual(game.scores(), [128, -64, -64])

    def test_plain_wildcard_discard_and_post_float_kong_restrictions(self):
        game = HangzhouGame(42, 0, 27, rules=THREE_HANGZHOU)
        game.players[0].hand = counts([WHITE] * 4 + [32] * 4 + list(range(6)))
        game.players[0].melds = [Meld('peng', 32, 1)]
        game.drawn_tile = 32
        actions = game.legal_actions()[1]
        self.assertNotIn(FLOAT, actions)
        self.assertNotIn(('ankan', WHITE), actions)
        self.assertIn(('bugang', 32), actions)
        game.step(('discard', WHITE))
        self.assertIsNone(game.float_cycle_owner)
        self.assertEqual(game.legal_actions()[1], [PASS])
        play_passes(game)
        game.cur, game.drawn_tile = 0, 32
        game.floated.add(0)
        game.players[0].hand[33] = 4
        self.assertNotIn(('bugang', 32), game.legal_actions()[1])
        self.assertIn(('ankan', 33), game.legal_actions()[1])

    def test_four_player_call_flow_and_liability_exception(self):
        for during_float in (True, False):
            with self.subTest(during_float=during_float):
                game = HangzhouGame(12, 0, 27, rules=FOUR_HANGZHOU)
                game.feeds[1, 0] = game.liability_feeds[1, 0] = 2
                game.players[1].hand = counts([1, 2, 32, 32, 33, 33, 9])
                if during_float:
                    game.floated.add(1)
                    game.float_cycle_owner = 1
                game.discards[0].append(0)
                game._open_offer(Offer(0, 0, 'discard'))
                self.assertIn(('chi', 0), game.legal_actions()[1])
                self.assertNotIn(RON, game.legal_actions()[1])
                game.step(('chi', 0))
                play_passes(game)
                self.assertEqual(game.players[1].melds[-1], Meld('chi', 0, 0))
                self.assertEqual(game.feeds[1, 0], 3)
                if during_float:
                    self.assertEqual(game.related_seats, [])
                    self.assertIsNone(game.float_cycle_owner)
                    self.assertEqual(game.payments(1, 2), (4, 0, 2, 2))
                else:
                    self.assertEqual(game.related_seats, [(1, 0)])
                    self.assertEqual(game.payments(1, 2), (8, 0, 0, 0))
                    self.assertEqual(game.payments(0, 2), (0, 24, 0, 0))
                game.feeds[1, 2] = 2
                game._open_offer(Offer(2, 33, 'discard'))
                game.step(PASS)
                game.step(PASS)
                self.assertEqual(game.legal_actions(), (1, [PASS]))

    def test_hidden_opponent_kong_is_not_in_browser_or_policy_input(self):
        game = HangzhouGame(123, 0, 27, rules=FOUR_HANGZHOU)
        game._open_offer(Offer(1, 32, 'gang_an'))
        snapshot = policy_view(game, 0, [PASS])
        self.assertIsNone(snapshot.offered)
        game._open_offer(Offer(1, 33, 'gang_an'))
        self.assertEqual(snapshot, policy_view(game, 0, [PASS]))
        game.offer = None
        game.players[1].melds = [Meld('gang_an', 32)]
        game.players[0].melds = [Meld('gang_an', 28)]
        inputs = policy_view(game, 0, [PASS])
        game.players[1].melds = [Meld('gang_an', 33)]
        self.assertEqual(inputs, policy_view(game, 0, [PASS]))
        session = HonorsSession(variant=FOUR_HANGZHOU.VARIANT)
        session.game = game
        seats = session.view()['players']
        self.assertEqual(seats[1]['melds'][0]['tiles'], [None] * 4)
        self.assertEqual(seats[0]['melds'][0]['tiles'], ['S'] * 4)

    def test_conservation_and_payers_through_full_hands(self):
        for rules in (THREE_HANGZHOU, FOUR_HANGZHOU):
            for seed in range(12):
                seats = rules.OPTIONS.seat_count
                game = HangzhouGame(seed, seed % seats, 27, rules=rules, dealer_streak=seed % 3)
                randomizer = random.Random(seed + 41)
                while game.phase != 'over':
                    actor, actions = game.legal_actions()
                    action = heuristic(policy_view(game, actor, actions)) if seed % 2 else randomizer.choice(actions)
                    game.step(action)
                    self.assertEqual(game.physical_counts(), game.tile_supply)
                if game.win_details:
                    win, = game.win_details
                    self.assertEqual(game.scores(), [sum(win.payments) if seat == win.actor else -win.payments[seat]
                                                     for seat in range(seats)])
                else:
                    self.assertEqual(game.scores(), [0] * seats)

    def test_session_applies_dealer_repeat_and_passes_total_to_next_hand(self):
        session = HonorsSession(variant=FOUR_HANGZHOU.VARIANT, seed=7, pacing=0)
        expected = 2
        try:
            for hand in (1, 2, 3):
                session.begin()
                wait(session, {'your_turn'})
                table = session.snapshot()['table']
                self.assertEqual(table['wildcard'], 'P')
                self.assertEqual(table['dealer_multiplier'], expected)
                self.assertEqual(table['hand_number'], hand)
                session.auto()
                wait(session, {'round_end'})
                self.assertEqual(sum(session.result['scores']), 0)
                expected = min(8, expected * 2) if table['dealer'] == session.dealer else 2
        finally:
            session.close()


class RegionalHangzhouTests(unittest.TestCase):
    def offer(self, game, provider, actor, tile, kind='discard'):
        if kind == 'discard':
            game.discards[provider].append(tile)
        game._open_offer(Offer(provider, tile, kind))
        while game.legal_actions()[0] != actor:
            game.step(PASS)
        return game.legal_actions()[1]

    def test_tonglu_requires_burst_with_a_wildcard(self):
        ctx = WinContext('zimo', 33, 27, 27)
        ordinary = counts(FOUR_SETS[1:] + [WHITE, 33, 33])
        self.assertEqual(FOUR_HANGZHOU.faan(ordinary, (), ctx).share, 1)
        self.assertIsNone(TONGLU.faan(ordinary, (), ctx))
        self.assertEqual(TONGLU.faan(counts(FOUR_SETS + [33, 33]), (), ctx).share, 1)
        self.assertEqual(TONGLU.faan(counts(FOUR_SETS + [WHITE, 33]), (), ctx).share, 2)
        ctx = WinContext('zimo', 6, 27, 27)
        pairs = counts([3, 3, 9, 9, 22, 22, 27, 27, 29, 29, 32, 32, WHITE, 6])
        self.assertEqual(TONGLU.faan(pairs, (), ctx).patterns, ('七客',))

    def test_tonglu_chows_and_pons_use_separate_slots(self):
        game = HangzhouGame(9, 0, 27, rules=TONGLU)
        caller = game.players[1]
        caller.hand = counts([0, 0, 0, 1, 2, 8, 9, 10])
        caller.melds = [Meld('chi', 9, 0), Meld('chi', 18, 0)]
        actions = self.offer(game, 0, 1, 0)
        self.assertNotIn(('chi', 0), actions)
        self.assertIn(('peng', None), actions)
        self.assertIn(('zhigang', None), actions)
        caller.melds = [Meld('gang_bu', 8, 0), Meld('peng', 9, 0)]
        self.assertIn(('chi', 0), game.legal_actions()[1])
        self.assertNotIn(('peng', None), game.legal_actions()[1])
        self.assertNotIn(('zhigang', None), game.legal_actions()[1])
        self.assertEqual(game.related_seats, [])

    def test_tonglu_floater_can_kong_open_and_opponents_cannot(self):
        game = HangzhouGame(7, 0, 27, rules=TONGLU)
        game.float_cycle_owner = 1
        game.floated.add(1)
        game.players[1].hand = counts([0, 0, 0])
        actions = self.offer(game, 0, 1, 0)
        self.assertIn(('zhigang', None), actions)
        game.step(('zhigang', None))
        play_passes(game)
        self.assertEqual(game.players[1].melds, [Meld('gang_ming', 0, 0)])
        self.assertEqual(game.kong_chain, 1)
        self.assertIsNone(game.float_cycle_owner)
        game.players[1].melds.append(Meld('peng', 33, 2))
        game.players[1].hand[33] += 1
        self.assertIn(('bugang', 33), game.legal_actions()[1])
        game.float_cycle_owner = 2
        self.assertNotIn(('bugang', 33), game.legal_actions()[1])

    def test_chunan_guard_uses_only_callers_wildcard_and_preserves_chow(self):
        game = HangzhouGame(8, 0, 27, rules=CHUNAN)
        guard = game.players[3]
        guard.hand = counts([3, 3, 4, 5, 32, 32])
        self.assertNotIn(('peng', None), self.offer(game, 0, 3, 3))
        guard.hand[WHITE] = 1
        self.assertIn(('peng', None), game.legal_actions()[1])
        guard.hand[WHITE] = 0
        self.assertIn(('peng', None), self.offer(game, 2, 3, 3))
        self.assertIn(('chi', 3), game.legal_actions()[1])
        game.feeds[3, 2] = 2
        self.assertNotIn(('chi', 3), game.legal_actions()[1])
        self.assertNotIn(('peng', None), game.legal_actions()[1])

    def test_chunan_only_dealer_links_ron_and_nearest_win_gets_paid(self):
        game = HangzhouGame(8, 0, 27, rules=CHUNAN)
        game.players[1].hand = counts(FOUR_SETS + [33])
        game.players[2].hand = counts(FOUR_SETS + [33])
        self.assertNotIn(RON, self.offer(game, 1, 2, 33))
        self.assertIn(RON, self.offer(game, 0, 1, 33))
        game.step(RON)
        self.assertEqual(game.legal_actions(), (2, [PASS]))
        play_passes(game)
        self.assertEqual(game.winners, [1])
        self.assertEqual(game.win_details[0].payments, (8, 0, 0, 0))
        self.assertEqual(game.scores(), [-8, 8, 0, 0])
        self.assertEqual(game.next_dealer, 1)

    def test_chunan_rob_added_kong_is_one_dealer_link_without_kong_bonus(self):
        game = HangzhouGame(3, 0, 27, rules=CHUNAN)
        game.players[0].hand = counts(FOUR_SETS + [33])
        game.players[1].hand = counts([33])
        game.players[1].melds = [Meld('peng', 33, 2)]
        game.float_cycle_owner = 2
        self.assertNotIn(RON, self.offer(game, 1, 0, 33, 'gang_bu'))
        game.float_cycle_owner = None
        self.assertNotIn(RON, self.offer(game, 1, 0, 33, 'gang_an'))
        self.assertIn(RON, self.offer(game, 1, 0, 33, 'gang_bu'))
        session = HonorsSession(variant=CHUNAN.VARIANT)
        session.game = game
        self.assertEqual(session.option(RON)['win_type'], 'rob_kong')
        game.step(RON)
        self.assertEqual(game.win_details[0].kind, 'rob_kong')
        self.assertEqual(game.win_details[0].payments, (0, 8, 0, 0))
        self.assertEqual(game.players[1].melds, [Meld('peng', 33, 2)])
        self.assertEqual(game.claimed_tile, 33)

    def test_duel_indicator_counts_and_real_white_dragon_can_complete_chow(self):
        game = next(game for seed in range(120) if (game := HangzhouGame(seed, 0, 27, rules=TWO_HANGZHOU)).wildcard_indicator == 1)
        self.assertEqual(game.physical_counts(), game.tile_supply)
        self.assertEqual(game.remaining, 136 - 27 - 1)
        view = policy_view(game, 0, [])
        self.assertEqual(view.visible[WHITE], 1)
        self.assertEqual(view.visible[1], 0)
        game.players[1].hand = counts([1, 2, WHITE])
        self.assertIn(('chi', 0), self.offer(game, 0, 1, 0))
        session = HonorsSession(variant=TWO_HANGZHOU.VARIANT)
        session.game = game
        self.assertEqual(session.option(('chi', 0))['tiles'], ['1m', 'P', '3m'])
        game.step(('chi', 0))
        self.assertEqual(game.players[1].hand[WHITE], 1)
        table = session.view()
        self.assertEqual(len(table['players']), 2)
        self.assertEqual(table['wildcard'], '2m')
        self.assertEqual(table['white_substitute'], '2m')
        self.assertEqual(table['players'][1]['melds'][0]['tiles'], ['1m', 'P', '3m'])
        self.assertEqual(session.option(('discard', WHITE))['tiles'], ['2m'])

    def test_duel_flipped_white_stays_white(self):
        game = next(game for seed in range(140) if (game := HangzhouGame(seed, 0, 27, rules=TWO_HANGZHOU)).wildcard_indicator == WHITE)
        self.assertEqual(game.physical_counts(), game.tile_supply)
        session = HonorsSession(variant=TWO_HANGZHOU.VARIANT)
        session.game = game
        self.assertEqual(session.view()['wildcard'], 'P')
        self.assertNotIn('white_substitute', session.view())
        self.assertEqual(policy_view(game, 0, []).visible[WHITE], 1)
