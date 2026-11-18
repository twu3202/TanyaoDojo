import random
import unittest

from deploy.play import jj_multiplier
from deploy.play.honors import Meld, Offer, PASS, RON, ZIMO, WinningHand
from deploy.play.honors_game import HonorsSession
from deploy.play.honors_policy import heuristic, policy_view
from deploy.play.tests.test_games import wait
from deploy.play.tests.test_hongkong import counts
from deploy.play.tests.test_kawuxing import REGULAR
from deploy.play.xiaogan import XIAOGAN, XiaoganGame


DARK_HAND = [31, 31, 31, 0, 1, 2, 9, 10, 11, 12, 14, 16, 16]
WAIT = 13


def jj_game():
    return jj_multiplier.JJMultiplierGame(3, 0, 27, rules=jj_multiplier)


def xiaogan_game():
    game = XiaoganGame(1, 0, 27, rules=XIAOGAN)
    for value in (3, 2, 0):
        game.step(('piao', value))
    return game


class DarkReadyTests(unittest.TestCase):
    def declare(self, game, action):
        game.players[0].hand = counts(DARK_HAND + [33])
        game.drawn_tile = 33
        self.assertIn((action, 33), game.legal_actions()[1])
        game.step((action, 33))
        game.step(PASS)
        game.step(PASS)

    def test_dark_ready_reveals_only_fixed_triplet_count(self):
        game = jj_game()
        self.declare(game, 'declare_dark')
        self.assertEqual(game.hidden_pungs[0], {31})
        self.assertEqual(game.ready_waits[0], (WAIT,))
        self.assertEqual(game._public_waits(0), (WAIT,))
        self.assertEqual(game.players[0].open_ready_hand[31], 0)
        session = HonorsSession(variant=jj_multiplier.VARIANT)
        session.game = game
        session.game.players[1].open_ready_hand = session.game.players[0].open_ready_hand
        session.game.dark_seats.add(1)
        session.game.hidden_pungs[1].update({31})
        actor = session.view()['players'][1]
        self.assertTrue(actor['dark_ready'])
        self.assertEqual(actor['hidden_ready_tiles'], 3)
        self.assertEqual(len(actor['hand']), 10)
        self.assertNotIn('P', actor['hand'])
        public = policy_view(game, 2, [])
        self.assertEqual(public.visible[31], 0)
        game.hidden_pungs[1] = {32}
        self.assertEqual(policy_view(game, 2, []), public)
        self.assertEqual(game._public_waits(1), (WAIT,))

    def test_public_wait_is_filtered_except_when_all_discards_are_dangerous(self):
        game = jj_game()
        self.declare(game, 'declare_dark')
        game.players[1].hand = counts([WAIT, 17])
        game.cur = 1
        game.drawn_tile = 17
        self.assertEqual(game.legal_actions()[1], [('discard', 17)])
        game.players[1].hand = counts([WAIT, WAIT])
        self.assertEqual(game.legal_actions()[1], [('discard', WAIT)])

    def test_open_and_dark_can_only_kong_a_kept_concealed_pung(self):
        for name in ('declare_ready', 'declare_dark'):
            with self.subTest(name=name):
                game = jj_game()
                self.declare(game, name)
                game.wall.append(31)
                game._draw(0)
                expected = [('discard', 31), ('ankan', 31)] if name == 'declare_dark' else [('discard', 31)]
                self.assertEqual(game.legal_actions()[1], expected)
                if name == 'declare_dark':
                    game.step(('ankan', 31))
                    game.step(PASS)
                    game.step(PASS)
                    self.assertEqual(game.players[0].melds, [Meld('gang_an', 31)])
                    self.assertEqual(game.kong_deltas, [4, -2, -2])
                    self.assertEqual(game.hidden_ready_tiles(0), 0)
                    self.assertEqual(game._public_waits(0), (WAIT,))

    def test_dark_direct_kong_and_late_declaration_rule(self):
        game = jj_game()
        self.declare(game, 'declare_dark')
        game.discards[1].append(31)
        game._open_offer(Offer(1, 31, 'discard'))
        game.step(PASS)
        self.assertIn(('zhigang', None), game.legal_actions()[1])
        game.step(('zhigang', None))
        self.assertEqual(game.players[0].melds, [Meld('gang_ming', 31, 1)])
        self.assertEqual(game.kong_deltas, [1, -1, 0])

        game = jj_game()
        game.players[0].hand = counts(DARK_HAND + [33])
        game.drawn_tile = 33
        game.wall = game.wall[:3]
        self.assertFalse(any(kind.startswith('declare_') for kind, tile in game.legal_actions()[1]))

    def test_reveal_multiplier_per_pair_and_draw_refund_direction(self):
        game = jj_game()
        hand = counts(REGULAR + [31])
        score = jj_multiplier.Score(1, ('平胡',))
        game.players[1].open_ready_hand = tuple(counts(DARK_HAND))
        win = WinningHand(0, score, hand, 'zimo')
        self.assertEqual(jj_multiplier.settle_hand(game, [win], None)[0][1], (0, 4, 3))
        game.players[0].open_ready_hand = tuple(counts(DARK_HAND))
        self.assertEqual(jj_multiplier.settle_hand(game, [win], None)[0][1], (0, 4, 4))
        game.declarations = [1, 0]
        game._end('draw')
        self.assertEqual(game.ready_deltas, [0, -1, 1])

    def test_same_scoring_form_is_blocked_until_new_draw_or_different_event(self):
        game = jj_game()
        game.players[1].hand = counts(DARK_HAND)
        game._open_offer(Offer(0, WAIT, 'discard'))
        self.assertIn(RON, game.legal_actions()[1])
        game.step(PASS)
        game._open_offer(Offer(0, WAIT, 'discard'))
        self.assertNotIn(RON, game.legal_actions()[1])
        game._open_offer(Offer(0, WAIT, 'discard', kong_chain=1))
        self.assertIn(RON, game.legal_actions()[1])

    def test_second_kong_in_chain_doubles_only_its_own_payment(self):
        game = jj_game()
        game._pay_kong(0, Meld('gang_an', 1))
        self.assertEqual(game.kong_deltas, [4, -2, -2])
        game.kong_chain = 1
        game._pay_kong(0, Meld('gang_an', 3))
        self.assertEqual(game.kong_deltas, [12, -6, -6])


class XiaoganTests(unittest.TestCase):
    def test_piao_choices_and_cards_are_private_until_all_three_commit(self):
        session = HonorsSession(variant=XIAOGAN.VARIANT)
        game = XiaoganGame(1, 2, 27, rules=XIAOGAN)
        session.game = game
        original = game.physical_counts()
        for seat, value in enumerate((3, 0, 2)):
            actor, moves = game.legal_actions()
            self.assertEqual(actor, seat)
            self.assertEqual(moves, [('piao', option) for option in range(4)])
            self.assertFalse(any(policy_view(game, seat, moves).hand))
            table = session.view()
            self.assertIsNone(table['dealer'])
            self.assertTrue(all(not p['hand'] and 'piao' not in p for p in table['players']))
            game.step(('piao', value))
        self.assertEqual(game.physical_counts(), original)
        self.assertEqual(game.cur, 2)
        self.assertEqual([p['piao'] for p in session.view()['players']], [3, 0, 2])

    def test_self_draw_piao_accounts_for_both_endpoints(self):
        game = xiaogan_game()
        game.opening = False
        game.players[0].hand = counts(REGULAR + [31])
        game.drawn_tile = 31
        game.step(ZIMO)
        self.assertEqual(game.win_details[0].payments, (0, 1, 1))
        self.assertEqual(game.piao_deltas, [8, -5, -3])
        self.assertEqual(game.horse_deltas, [0, 0, 0])
        self.assertEqual(game.scores(), [10, -6, -4])

    def test_ready_self_draw_flips_rank_horse_and_each_opponent_pays(self):
        for tile, rank in ((14, 6), (32, 10)):
            with self.subTest(tile=tile):
                game = xiaogan_game()
                game.opening = False
                game.players[0].hand = counts(REGULAR + [31])
                game.players[0].open_ready_hand = tuple(counts(REGULAR))
                game.drawn_tile = 31
                game.wall.insert(0, tile)
                game.step(ZIMO)
                self.assertEqual(game.win_details[0].payments, (0, 2, 2))
                self.assertEqual(game.horse_tiles, [tile])
                self.assertEqual(game.horse_deltas, [2 * rank, -rank, -rank])
                self.assertEqual(game.scores(), [12 + 2 * rank, -7 - rank, -5 - rank])

    def test_double_reveal_discard_bills_supplier_twice_and_piao_once(self):
        game = xiaogan_game()
        for seat in (0, 2):
            game.players[seat].hand = counts(REGULAR)
            game.players[seat].open_ready_hand = tuple(counts(REGULAR))
        game.players[1].open_ready_hand = tuple(counts(REGULAR))
        game.discards[1].append(31)
        game._open_offer(Offer(1, 31, 'discard'))
        game.step(RON)
        self.assertEqual(game.winners, [2])
        self.assertEqual(game.win_details[0].payments, (0, 4, 0))
        self.assertEqual(game.piao_deltas, [0, -2, 2])
        self.assertEqual(game.scores(), [0, -6, 6])
        self.assertEqual(game.horse_tiles, [])

    def test_draw_keeps_only_kong_points(self):
        game = xiaogan_game()
        game._pay_kong(1, Meld('gang_ming', 13, 2))
        game.first_ready = 0
        game._end('draw')
        self.assertEqual(game.scores(), [0, 2, -2])
        self.assertEqual(game.piao_deltas, [0, 0, 0])
        self.assertEqual(game.horse_deltas, [0, 0, 0])


class MatchTests(unittest.TestCase):
    def test_conservation_and_payment_ledger(self):
        for rules, cls in ((jj_multiplier, jj_multiplier.JJMultiplierGame), (XIAOGAN, XiaoganGame)):
            for seed in range(16):
                game = cls(seed, seed % 3, 27, rules=rules)
                rng = random.Random(seed)
                while game.phase != 'over':
                    seat, choices = game.legal_actions()
                    move = heuristic(policy_view(game, seat, choices)) if seed % 2 else rng.choice(choices)
                    game.step(move)
                    self.assertEqual(game.physical_counts(), game.tile_supply)
                    self.assertEqual(sum(game.scores()), 0)
                ledgers = [game.kong_deltas, game.ready_deltas]
                if rules is XIAOGAN:
                    ledgers += [game.piao_deltas, game.horse_deltas]
                net = [sum(values[i] for values in ledgers) for i in range(3)]
                for win in game.win_details:
                    for i, share in enumerate(win.payments):
                        net[i] -= share
                        net[win.actor] += share
                self.assertEqual(net, game.scores())

    def test_sessions_continue(self):
        for variant in (jj_multiplier.VARIANT, XIAOGAN.VARIANT):
            session = HonorsSession(variant=variant, seed=7, pacing=0)
            try:
                for hand in (1, 2):
                    session.begin()
                    wait(session, {'your_turn'})
                    start = session.snapshot()
                    self.assertEqual(start['table']['hand_number'], hand)
                    if variant == XIAOGAN.VARIANT:
                        self.assertEqual(start['table']['phase'], 'piao')
                        self.assertEqual([choice['value'] for choice in start['choices']], [0, 1, 2, 3])
                    session.auto()
                    wait(session, {'round_end'})
                    self.assertEqual(sum(session.result['scores']), 0)
            finally:
                session.close()
