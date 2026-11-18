import random
import unittest
from dataclasses import replace

from deploy.play.honors import Meld, Offer, PASS, RON, WinContext
from deploy.play.honors_game import HonorsSession
from deploy.play.honors_policy import heuristic, policy_view
from deploy.play.kawuxing import SUIZHOU, XIANGYANG, KaWuXingGame, ready_waits
from deploy.play.tests.test_games import wait
from deploy.play.tests.test_hongkong import counts


READY_FIVE = [9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 23, 23]
FIVE = 13
REGULAR = [9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 31]


class KaWuXingTests(unittest.TestCase):
    def game(self, rules=SUIZHOU):
        return KaWuXingGame(12, 0, 27, rules=rules)

    def test_three_seats_84_tiles_and_turn_wrap(self):
        game = self.game()
        self.assertEqual(len(game.tile_supply), 21)
        self.assertEqual(sum(game.tile_supply.values()), 84)
        self.assertEqual([sum(seat.hand) for seat in game.players], [14, 13, 13])
        self.assertEqual(game.remaining, 44)
        for seat in (0, 1, 2, 0):
            actor, acts = game.legal_actions()
            self.assertEqual(actor, seat)
            game.step(next(action for action in acts if action[0] == 'discard'))
            self.assertEqual(list(game.responses), [(seat + 1) % 3, (seat + 2) % 3])
            game.step(PASS)
            game.step(PASS)
            self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_open_ready_snapshot_does_not_reveal_subsequent_draw(self):
        game = self.game()
        game.players[0].hand = counts(READY_FIVE + [33])
        game.drawn_tile = 33
        self.assertIn(('declare_ready', 33), game.legal_actions()[1])
        game.step(('declare_ready', 33))
        self.assertEqual(game.first_ready, 0)
        self.assertEqual(game.players[0].open_ready_hand, tuple(counts(READY_FIVE)))
        self.assertIn(FIVE, ready_waits(game.players[0].open_ready_hand, ()))
        game.step(PASS)
        game.step(PASS)
        game.wall.append(32)
        game._draw(0)
        _, moves = game.legal_actions()
        self.assertEqual(moves, [('discard', 32)])
        initial = policy_view(game, 1, [PASS])
        self.assertEqual(initial.visible[32], 0)
        game.players[0].hand[32] = 0
        game.players[0].hand[31] += 1
        self.assertEqual(initial, policy_view(game, 1, [PASS]))
        self.assertEqual(policy_view(game, 0, moves).visible[23], 0)
        session = HonorsSession(variant=SUIZHOU.VARIANT)
        session.game = game
        self.assertEqual(session.view()['players'][0]['open_ready'], True)

    def test_browser_lists_only_revealed_ready_tiles_of_another_seat(self):
        game = self.game()
        game.players[1].hand = counts(REGULAR + [33])
        game.players[1].open_ready_hand = tuple(counts(REGULAR))
        session = HonorsSession(variant=SUIZHOU.VARIANT)
        session.game = game
        seats = session.view()['players']
        self.assertEqual(len(seats), 3)
        self.assertEqual(seats[1]['hand_count'], 14)
        self.assertEqual(len(seats[1]['hand']), 13)
        self.assertNotIn('C', seats[1]['hand'])
        self.assertEqual(seats[2]['hand'], [])

    def test_open_ready_and_dragon_patterns_do_not_bypass_hand_cap(self):
        big = counts([31] * 3 + [32] * 3 + [33] * 3 + [9, 10, 11, 20, 20])
        ctx = WinContext('ron', 20, 27, 27, ready=True)
        a = SUIZHOU.faan(big, (), ctx)
        b = XIANGYANG.faan(big, (), ctx)
        self.assertEqual((a.share, a.patterns), (16, ('大三元', '亮倒')))
        self.assertEqual((b.share, b.patterns), (8, ('大三元', '亮倒', '八倍封顶')))
        small = counts([31] * 3 + [32] * 3 + [33] * 2 + [9, 10, 11, 18, 19, 20])
        self.assertEqual(SUIZHOU.faan(small, (), replace(ctx, ready=False)).share, 4)

    def test_middle_five_and_four_in_one_require_use_of_the_winning_tile(self):
        ctx = WinContext('ron', FIVE, 27, 27)
        dark = counts([FIVE] * 4 + [12, 14, 15, 16, 17, 18, 19, 20, 23, 23])
        score = SUIZHOU.faan(dark, (), ctx)
        self.assertEqual((score.share, score.patterns), (16, ('卡五星', '暗四归一')))
        self.assertEqual(SUIZHOU.faan(dark, (), replace(ctx, tile=23)).share, 1)
        hand = counts([12, 13, 14, 9, 10, 11, 18, 19, 20, 23, 23])
        score = SUIZHOU.faan(hand, (Meld('peng', FIVE, 2),), ctx)
        self.assertEqual((score.share, score.patterns), (8, ('卡五星', '明四归一')))
        dragon_pairs = counts([9] * 4 + [10] * 4 + [11, 11, 12, 12, 13, 13])
        score = SUIZHOU.faan(dragon_pairs, (), ctx)
        self.assertEqual((score.share, score.patterns), (64, ('清一色', '双龙七对')))

    def _discard_to_winner(self, rules, *, supplier_declared):
        game = self.game(rules)
        game.players[1].hand = counts(READY_FIVE)
        game.players[1].open_ready_hand = tuple(game.players[1].hand)
        game.players[0].open_ready_hand = tuple(game.players[0].hand) if supplier_declared else None
        game.discards[0].append(FIVE)
        game._open_offer(Offer(0, FIVE, 'discard'))
        game.step(RON)
        if game.phase != 'over':
            game.step(PASS)
        return game

    def test_open_ready_package_and_double_ready_exception(self):
        for rules, declared, expected in (
            (SUIZHOU, False, 16), (SUIZHOU, True, 16),
            (XIANGYANG, False, 16), (XIANGYANG, True, 8),
        ):
            with self.subTest(rules=rules.VARIANT, declared=declared):
                game = self._discard_to_winner(rules, supplier_declared=declared)
                self.assertEqual(game.win_details[0].payments, (expected, 0, 0))
                self.assertEqual(game.scores(), [-expected, expected, 0])

    def test_suizhou_double_wins_keep_one_tile_and_xiangyang_takes_nearest(self):
        for rules, winners, scores, dealer in (
            (SUIZHOU, [1, 2], [-8, 4, 4], 0), (XIANGYANG, [1], [-4, 4, 0], 1),
        ):
            game = self.game(rules)
            for seat in (1, 2):
                game.players[seat].hand = counts(READY_FIVE)
            game.discards[0].append(FIVE)
            game._open_offer(Offer(0, FIVE, 'discard'))
            game.step(RON)
            if game.phase != 'over':
                game.step(RON)
            self.assertEqual(game.winners, winners)
            self.assertEqual(game.scores(), scores)
            self.assertEqual(game.next_dealer, dealer)
            self.assertEqual(game.claimed_tile, FIVE)
            self.assertEqual(game.discards[0], [])

    def test_higher_offer_after_pass_is_suizhou_only(self):
        for rules, allowed in ((SUIZHOU, True), (XIANGYANG, False)):
            game = self.game(rules)
            # Plain dragon pair, with open-ready doubling available later.
            game.players[1].hand = counts(REGULAR)
            game._open_offer(Offer(0, 31, 'discard'))
            self.assertIn(RON, game.legal_actions()[1])
            game.step(PASS)
            game._open_offer(Offer(0, 31, 'discard'))
            self.assertNotIn(RON, game.legal_actions()[1])
            game._open_offer(Offer(0, 31, 'gang_bu'))
            self.assertEqual(RON in game.legal_actions()[1], allowed)
            game.offer = None
            game._draw(1)
            game.players[1].hand[game.drawn_tile] -= 1
            game._open_offer(Offer(0, 31, 'discard'))
            self.assertIn(RON, game.legal_actions()[1])

    def test_concealed_rob_and_last_tile_kong(self):
        for rules, conceal in ((SUIZHOU, True), (XIANGYANG, False)):
            game = self.game(rules)
            game.players[0].hand = counts([FIVE] * 4 + READY_FIVE[:10])
            game.drawn_tile = FIVE
            game.players[1].hand = counts(READY_FIVE)
            game.step(('ankan', FIVE))
            self.assertEqual(RON in game.legal_actions()[1], conceal)

            game = self.game(rules)
            game.players[0].hand = counts([FIVE] * 4 + READY_FIVE[:10])
            game.drawn_tile = FIVE
            game.wall.clear()
            self.assertNotIn(('ankan', FIVE), game.legal_actions()[1])
            game.players[1].hand = counts([FIVE] * 3)
            game._open_offer(Offer(0, FIVE, 'discard'))
            self.assertNotIn(('zhigang', None), game.legal_actions()[1])
            self.assertIn(('peng', None), game.legal_actions()[1])

    def test_draw_first_ready_bills_both_opponents_and_keeps_kongs(self):
        game = self.game()
        game.first_ready = 1
        game._pay_kong(2, Meld('gang_an', FIVE))
        game._end('draw')
        self.assertEqual(game.kong_deltas, [-2, -2, 4])
        self.assertEqual(game.ready_deltas, [1, -2, 1])
        self.assertEqual(game.scores(), [-1, -4, 5])
        self.assertEqual(game.next_dealer, 0)

    def test_full_hands_and_ledger_invariants(self):
        for rules in (SUIZHOU, XIANGYANG):
            for seed in range(20):
                game = KaWuXingGame(seed, seed % 3, 27, rules=rules)
                rng = random.Random(seed)
                while game.phase != 'over':
                    actor, actions = game.legal_actions()
                    move = heuristic(policy_view(game, actor, actions)) if seed % 2 else rng.choice(actions)
                    game.step(move)
                    self.assertEqual(game.physical_counts(), game.tile_supply)
                    self.assertEqual(sum(game.scores()), 0)
                net = [kong + ready for kong, ready in zip(game.kong_deltas, game.ready_deltas)]
                for win in game.win_details:
                    for payer, share in enumerate(win.payments):
                        net[payer] -= share
                        net[win.actor] += share
                self.assertEqual(game.scores(), net)

    def test_three_seat_session_next_hand(self):
        for rules in (SUIZHOU, XIANGYANG):
            session = HonorsSession(variant=rules.VARIANT, seed=3, pacing=0)
            try:
                session.begin()
                wait(session, {'your_turn'})
                session.auto()
                wait(session, {'round_end'})
                result = session.snapshot()['result']
                self.assertEqual(len(result['scores']), 3)
                session.begin()
                wait(session, {'your_turn'})
                table = session.snapshot()['table']
                self.assertEqual(table['hand_number'], 2)
                self.assertEqual([player['score'] for player in table['players']], result['scores'])
            finally:
                session.close()
