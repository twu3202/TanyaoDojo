from collections import Counter
from types import SimpleNamespace
import unittest

from deploy.play.hangzhou import WHITE
from deploy.play.honors import PASS, RON, ZIMO, Meld, WinContext
from deploy.play.honors_bonus import Score
from deploy.play.jinyun import JINYUN, JINYUN_CANTONESE, JinyunCantoneseGame, JinyunClassicGame
from deploy.play.tests.test_hongkong import counts


def set_hands(game, hands, *, taken=(), replacement=None):
    wall = list((game.tile_supply - Counter(sum(hands, []) + list(taken))).elements())
    if replacement is not None:
        wall.remove(replacement)
    for seat, hand in enumerate(hands):
        meld_count = len(game.players[seat].melds)
        fill = 13 + (seat == game.dealer) - 3 * meld_count - len(hand)
        game.players[seat].hand = counts(hand + [wall.pop() for _ in range(fill)])
    if replacement is not None:
        wall.insert(game.rules.WALL_RESERVE, replacement)
    game.wall = wall


def pass_offer(game):
    while game.offer is not None:
        game.step(PASS)


class JinyunClassicTests(unittest.TestCase):
    def test_opening_follow_three_branches_and_interruption(self):
        for fourth, interrupted, expected in (
            (6, False, [2, 2, -6, 2]), (WHITE, False, [1, 1, -3, 1]),
            (7, False, [1, -3, 1, 1]), (6, True, [0, 0, 0, 0]),
        ):
            with self.subTest(tile=fourth, interrupted=interrupted):
                game = JinyunClassicGame(7, 2, 27, rules=JINYUN)
                set_hands(game, [[6], [fourth], [6], [6]], taken=(WHITE,))
                for seat in (2, 3, 0, 1):
                    self.assertEqual(game.legal_actions()[0], seat)
                    if interrupted and seat == 1:
                        game.uninterrupted = False
                    game.step(('discard', fourth if seat == 1 else 6))
                    pass_offer(game)
                self.assertEqual(game.follow_deltas, expected)
                self.assertEqual(game.scores(), expected)
                self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_specials_add_and_honors_uses_40(self):
        hand = counts([2, 4, 6, 9, 9, 10, 10, 12, 12, 16, 16, WHITE, WHITE, WHITE])
        score = JINYUN.faan(hand, (), WinContext('zimo', WHITE, 27, 27))
        self.assertEqual(score, Score(8, ('财神会', '七对')))
        all_honors = counts([27, 27, 27, 28, 28, 28, 29, 29, 29, 33, 33, 33, 32, WHITE])
        score = JINYUN.faan(all_honors, (), WinContext('zimo', WHITE, 27, 27, opening='天胡'))
        self.assertEqual(score, Score(40, ('全字',)))

    def test_supply_win_doubles_only_the_dealer_pair(self):
        game = JinyunClassicGame(7, 2, 27, rules=JINYUN)
        score = Score(4, ('地胡',))
        for actor, expected in ((0, (0, 4, 8, 4)), (2, (8, 8, 0, 8))):
            winner = SimpleNamespace(actor=actor, score=score)
            self.assertEqual(JINYUN.settle_hand(game, [winner], None), [(score, expected)])

    def test_draw_keeps_kongs_and_rotates_only_if_nondealer_konged(self):
        game = JinyunClassicGame(7, 0, 27, rules=JINYUN)
        meld = Meld('gang_an', 4)
        game.players[1].melds.append(meld)
        game._pay_kong(1, meld)
        game._end('draw')
        self.assertEqual(game.scores(), [-2, 6, -2, -2])
        self.assertEqual(game.next_dealer, 1)
        game = JinyunClassicGame(7, 0, 27, rules=JINYUN)
        game._end('draw')
        self.assertEqual(game.next_dealer, 0)


class JinyunCantoneseTests(unittest.TestCase):
    def test_direct_kong_replacement_charges_original_supplier(self):
        game = JinyunCantoneseGame(7, 0, 27, rules=JINYUN_CANTONESE)
        set_hands(game, [[0], [0, 0, 0, 1, 2, 3, 9, 10, 11, 18, 19, 20, 27], [], []], replacement=27)
        reserve = game.wall[:9]
        game.step(('discard', 0))
        game.step(('zhigang', None))
        pass_offer(game)
        self.assertEqual(game.cur, 1)
        self.assertIn(ZIMO, game.legal_actions()[1])
        game.step(ZIMO)
        (win,) = game.win_details
        self.assertEqual(win.payments, (3, 0, 0, 0))
        self.assertIn('点杠杠开', win.patterns)
        self.assertEqual(game.scores(), [-6, 6, 0, 0])
        self.assertEqual(game.wall[:9], reserve)
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_robbery_cancels_kong_and_ordinary_discard_cannot_win(self):
        wait = [3, 4, 9, 10, 11, 18, 19, 20, 27, 27, 27, 32, 32]
        game = JinyunCantoneseGame(7, 0, 27, rules=JINYUN_CANTONESE)
        pong = Meld('peng', 5, 3)
        game.players[0].melds = [pong]
        set_hands(game, [[5], wait, [], []], taken=pong.tiles)
        game.drawn_tile = 5
        hand = counts([*wait, 5])
        self.assertIsNone(JINYUN_CANTONESE.faan(hand, (), WinContext('ron', 5, 28, 27)))
        game.step(('bugang', 5))
        self.assertIn(RON, game.legal_actions()[1])
        game.step(RON)
        pass_offer(game)
        self.assertEqual(game.winners, [1])
        self.assertEqual(game.players[0].melds, [pong])
        self.assertEqual(game.kong_deltas, [0] * 4)
        self.assertEqual(game.scores(), [-3, 3, 0, 0])
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_draw_refunds_immediate_kong_transfers(self):
        game = JinyunCantoneseGame(7, 0, 27, rules=JINYUN_CANTONESE)
        game._pay_kong(1, Meld('gang_ming', 4, 3))
        game._pay_kong(2, Meld('gang_an', 8))
        self.assertEqual(game.scores(), [-2, 1, 6, -5])
        game._end('draw')
        self.assertEqual(game.kong_refunds, [2, -1, -6, 5])
        self.assertEqual(game.scores(), [0, 0, 0, 0])
