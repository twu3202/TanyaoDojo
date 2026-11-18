import random
import unittest
from dataclasses import replace

from deploy.play.suited import COPIES, PASS, RON, Offer, Profile, SuitedGame, _waits
from deploy.play.suited_game import SuitedSession, policy_view, suited_policy
from deploy.play.suited_regional import datang_payment, pook_payment
from deploy.play.tests.test_suited import CALLER, READY, SPARE, counts, fixture
from deploy.play.tests.test_suited_regional import win_context


class ReadyRules(unittest.TestCase):
    def test_declaration_and_following_turn_are_locked_to_drawn_tile(self):
        for profile in (Profile.POOK, Profile.DATANG):
            with self.subTest(profile=profile):
                game = fixture([READY + [26], CALLER, SPARE, [22]*4 + [23]*4 + [24]*4 + [25]],
                               drawn=26, profile=profile)
                self.assertIn(('declare_ready', 26), game.legal_actions()[1])
                self.assertNotIn(('declare_ready', 17), game.legal_actions()[1])
                game.step(('declare_ready', 26))
                player = game.players[0]
                self.assertEqual(player.ready_waits, {0, 3, 6})
                self.assertEqual(player.ready_discard, 0)
                session = SuitedSession(profile=profile.value)
                session.game = game
                self.assertEqual(session.view()['players'][0]['river'], [{'tile': '9s', 'ready': True}])
                self.assertTrue(session.view()['players'][0]['ready'])
                self.assertNotIn('ready_waits', session.view()['players'][0])
                while game.cur != 0 or game.offer is not None:
                    _, actions = game.legal_actions()
                    game.step(actions[0])
                discards = [move for move in game.legal_actions()[1] if move[0] == 'discard']
                self.assertEqual(discards, [('discard', game.drawn_tile)])
                self.assertEqual(game.physical_counts(), COPIES)

    def test_only_opened_datang_allows_wait_preserving_direct_kong(self):
        for profile, allowed in ((Profile.POOK, False), (Profile.DATANG, True)):
            with self.subTest(profile=profile):
                hands = [SPARE + [0], [0, 0, 0, 2, 3, 4, 5, 6, 9, 10, 11, 17, 17],
                         [22]*4 + [23]*4 + [24]*4 + [25], [12]*4 + [13]*4 + [14]*4 + [15]]
                game = fixture(hands, profile=profile)
                player = game.players[1]
                player.ready_waits = _waits(tuple(player.hand), ())
                self.assertEqual(player.ready_waits, {1, 4, 7})
                game.step(('discard', 0))
                _, actions = game.legal_actions()
                self.assertNotIn(('peng', None), actions)
                self.assertEqual(('zhigang', None) in actions, allowed)
                if allowed:
                    for move in (('zhigang', None), PASS, PASS):
                        game.step(move)
                    self.assertEqual(game.scores(), [-3, 3, 0, 0])
                self.assertEqual(game.physical_counts(), COPIES)

    def test_declared_kong_must_use_draw_and_preserve_waits(self):
        for base, expected in (
            ([0, 0, 0, 2, 3, 4, 5, 6, 9, 10, 11, 17, 17], True),
            ([0, 0, 0, 1, 2, 3, 4, 5, 9, 10, 11, 17, 17], False),
        ):
            game = fixture([base + [0], SPARE, [22]*4 + [23]*4 + [24]*4 + [25], CALLER[2:] + [21, 25]],
                           profile=Profile.POOK)
            player = game.players[0]
            player.ready_waits = _waits(tuple(counts(base)), ())
            self.assertTrue(player.ready_waits)
            self.assertEqual(('ankan', 0) in game.legal_actions()[1], expected)

    def test_passed_pon_locks_same_tile_until_next_draw(self):
        hands = [SPARE + [0], [6]*4 + [7]*4 + [8]*4 + [0],
                 [0, 0, 12, 13, 14, 15, 16, 17, 22, 23, 24, 25, 26],
                 [1]*4 + [2]*4 + [3]*4 + [4]]
        game = fixture(hands, profile=Profile.POOK)
        game.step(('discard', 0))
        game.step(PASS)
        self.assertIn(('peng', None), game.legal_actions()[1])
        game.step(PASS)
        game.step(PASS)
        game.step(('discard', 0))
        self.assertNotIn(('peng', None), game.legal_actions()[1])
        self.assertEqual(game.players[2].declined_pon_tiles, {0})
        game.step(PASS)
        game.step(PASS)
        game.step(PASS)
        self.assertFalse(game.players[2].declined_pon_tiles)
        self.assertEqual(game.physical_counts(), COPIES)

    def test_passing_pon_does_not_lock_a_direct_kong(self):
        hands = [SPARE + [0], [0, 0, 0, 2, 3, 4, 5, 6, 9, 10, 11, 17, 17],
                 [22]*4 + [23]*4 + [24]*4 + [25], [12]*4 + [13]*4 + [14]*4 + [15]]
        game = fixture(hands, profile=Profile.POOK)
        game.players[1].declined_pon_tiles.add(0)
        game.step(('discard', 0))
        acts = game.legal_actions()[1]
        self.assertNotIn(('peng', None), acts)
        self.assertIn(('zhigang', None), acts)

    def test_higher_payment_can_win_on_same_tile_after_a_pass(self):
        game = fixture([SPARE + [0], CALLER, READY, READY], profile=Profile.POOK)
        game.step(('discard', 0))
        game.step(PASS)
        self.assertIn(RON, game.legal_actions()[1])
        game.step(PASS)
        self.assertEqual(game.players[2].declined_win_points, 1)
        # Compare the same incoming tile under an equal ordinary offer and a higher rob offer.
        self.assertNotIn(RON, self.actions_for(game, Offer(1, 0, 'ron')))
        self.assertIn(RON, self.actions_for(game, Offer(1, 0, 'rob_kong')))

    @staticmethod
    def actions_for(game, offer):
        game._offer(offer)
        return game.legal_actions()[1]

    def test_multi_robbery_cancels_the_kong_and_datang_excludes_cowinners(self):
        for profile, expected in ((Profile.POOK, [-12, 6, 6, 0]), (Profile.DATANG, [-2, 2, 2, -2])):
            with self.subTest(profile=profile):
                game = fixture([[0, 6, 6, 7, 7, 8, 8, 15, 15, 16, 16], READY, READY, SPARE],
                               melds={0: [('peng', 0)]}, profile=profile)
                actor = game.players[0]
                concealed = actor.hand.copy()
                concealed[0] -= 1
                actor.ready_waits = _waits(tuple(concealed), tuple(actor.melds))
                self.assertTrue(actor.ready_waits)
                if profile is Profile.DATANG:
                    for seat in (1, 2):
                        game.players[seat].ready_waits = _waits(tuple(game.players[seat].hand), ())
                for move in (('bugang', 0), RON, RON, PASS):
                    game.step(move)
                self.assertEqual(game.scores(), expected)
                self.assertEqual(game.kong_deltas, [0] * 4)
                self.assertEqual(game.players[0].melds, [('peng', 0)])
                self.assertEqual(game.physical_counts(), COPIES)

    def test_datang_requires_a_prior_declaration_for_both_types_of_win(self):
        hands = [READY + [0], CALLER, READY, SPARE]
        for profile, allowed in ((Profile.POOK, True), (Profile.DATANG, False)):
            game = fixture(hands, profile=profile)
            self.assertEqual(('zimo', None) in game.legal_actions()[1], allowed)
            game.step(('discard', 0))
            game.step(PASS)
            self.assertEqual(RON in game.legal_actions()[1], allowed)

    def test_rounds_and_payment_ledger_across_play(self):
        for profile in (Profile.POOK, Profile.DATANG):
            rng = random.Random(271)
            for seed in range(120):
                game = SuitedGame(seed, profile=profile)
                while game.phase != 'over':
                    actor, acts = game.legal_actions()
                    game.step(suited_policy(policy_view(game, actor, acts)) if seed < 20 else rng.choice(acts))
                    self.assertEqual(game.physical_counts(), COPIES)
                    self.assertEqual(sum(game.scores()), 0)
                expected = game.kong_deltas.copy()
                if game.reason_code == 'draw' and profile is Profile.DATANG:
                    self.assertEqual(game.next_dealer, game.last_drawer)
                    expected = [0] * 4
                for win in game.win_details:
                    for payer, amount in enumerate(win.payments):
                        expected[payer] -= amount
                        expected[win.actor] += amount
                self.assertEqual(game.scores(), expected)


class ReadyPayments(unittest.TestCase):
    def test_pook_uses_one_event_base_and_stacks_eligible_patterns(self):
        dragon_flush = counts([0] * 4 + [1, 2, 3, 4, 5] * 2)
        ctx = win_context(kind='zimo', provider=None, ready=True, after_kong=True)
        self.assertEqual(pook_payment(dragon_flush, [], ctx),
                         (('杠上花', '龙七对', '清一色', '报听'), 11, (11, 0, 11, 11)))
        self.assertEqual(pook_payment(counts([0, 0]), [('peng', tile) for tile in (1, 2, 3, 9)],
                                      win_context(kind='rob_kong')),
                         (('抢杠', '单钓'), 8, (0, 0, 8, 0)))

    def test_datang_uses_real_straight_and_splits_only_from_a_declared_provider(self):
        hand = counts([*range(9), 9, 10, 11, 17, 17])
        ctx = win_context(provider_ready=True, winners=(1, 3))
        self.assertEqual(datang_payment(hand, [], ctx), (('一条龙', '报听分摊'), 3, (3, 0, 3, 0)))
        self.assertEqual(datang_payment(hand, [], replace(ctx, provider_ready=False))[2], (0, 0, 9, 0))
        hand = counts(READY + [0])
        self.assertEqual(datang_payment(hand, [], win_context(kind='zimo', provider=None))[2], (2, 0, 2, 2))
