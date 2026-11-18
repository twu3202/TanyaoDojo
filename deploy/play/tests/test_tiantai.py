from collections import Counter
import unittest

from deploy.play import tiantai
from deploy.play.honors import PASS, RON, Meld, Offer, WinContext
from deploy.play.honors_policy import _live_tiles, policy_view
from deploy.play.tests.test_hongkong import counts


WAITING = [0, 1, 3, 4, 5, 9, 10, 11, 12, 13, 14, 31, 31]


def table(starts, dealer_melds=()):
    game = tiantai.TiantaiGame(17, 0, 27, rules=tiantai)
    game.players[0].melds = list(dealer_melds)
    used = Counter(tile for row in starts for tile in row)
    for meld in dealer_melds:
        used.update(meld.tiles)
    wall = list((game.tile_supply - used).elements())
    for seat, (start, player) in enumerate(zip(starts, game.players, strict=True)):
        extra = 13 + (seat == 0) - 3 * len(player.melds) - len(start)
        filler = []
        for _ in range(extra):
            i = next(i for i in reversed(range(len(wall))) if wall[i] < 34)
            filler.append(wall.pop(i))
        player.hand = counts([*start, *filler])
    game.wall = wall
    game.flowers = [[], [], []]
    return game


class TiantaiTests(unittest.TestCase):
    def test_three_seat_supply_and_public_live_tiles(self):
        game = tiantai.TiantaiGame(8, 2, 27, rules=tiantai)
        self.assertEqual(sum(game.tile_supply.values()), 108)
        self.assertEqual(game.physical_counts(), game.tile_supply)
        self.assertEqual([sum(p.hand) for p in game.players], [13, 13, 14])
        view = policy_view(game, *game.legal_actions())
        self.assertTrue(all(count == 0 for count in _live_tiles(view)[18:27]))

    def test_flower_chain_and_kong_draw_keep_the_reserve(self):
        game = tiantai.TiantaiGame(5, 0, 27, rules=tiantai)
        reserve = [15] * 15
        game.wall = [*reserve, 35, 36, 3, 34]
        before = game.flowers[1].copy()
        game._draw(1)
        self.assertEqual(game.flowers[1], [*before, 34, 35, 36])
        self.assertEqual(game.wall, reserve)
        self.assertEqual((game.drawn_tile, game.remaining, game.draw_flowers), (3, 0, 3))
        game.wall.append(2)
        game._draw(1, kong_chain=1)
        self.assertEqual((game.wall, game.drawn_tile, game.draw_flowers), (reserve, 2, 0))

    def test_base_hu_two_sided_wait_and_double_seat_flowers(self):
        hand = counts([0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 16, 16])
        ctx = WinContext('zimo', 0, 27, 27)
        score = tiantai.faan(hand, (), ctx)
        self.assertEqual((score.share, score.patterns), (50, ('排胡', '底胡x22', '台数x1')))
        matched = tiantai.faan(hand, (), ctx, flowers=(34, 38))
        self.assertEqual((matched.share, matched.patterns), (240, ('位花', '排胡', '底胡x30', '台数x3')))
        inside = tiantai.faan(hand, (), WinContext('zimo', 1, 27, 27))
        self.assertEqual((inside.share, inside.patterns), (30, ('嵌档', '底胡x24', '台数x0')))

    def test_ron_completed_pung_and_cap(self):
        hand = counts([1] * 3 + [4] * 3 + [10] * 3 + [13] * 3 + [15, 15])
        ron = tiantai.faan(hand, (), WinContext('ron', 1, 27, 27))
        zimo = tiantai.faan(hand, (), WinContext('zimo', 1, 27, 27))
        self.assertEqual((ron.share, zimo.share), (70, 80))
        self.assertIn('底胡x34', ron.patterns)
        self.assertIn('底胡x38', zimo.patterns)
        capped = tiantai.faan(hand, (), WinContext('zimo', 1, 27, 27), flowers=(34, 35, 36, 37, 38))
        self.assertEqual(capped.share, 300)

    def test_shared_opening_discard_is_charged_twice_and_counted_once(self):
        game = table([[2], WAITING, WAITING])
        self.assertEqual(game.physical_counts(), game.tile_supply)
        game._discard(0, 2)
        game.step(RON)
        self.assertIn(RON, game.legal_actions()[1])
        game.step(RON)
        self.assertEqual(game.winners, [1, 2])
        self.assertEqual([win.patterns for win in game.win_details], [('地胡',), ('地胡',)])
        self.assertEqual(game.scores(), [-300, 150, 150])
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_two_robbers_leave_original_pon_and_same_tile_supply(self):
        pon = Meld('peng', 2, 2)
        game = table([[2], WAITING, WAITING], [pon])
        game.drawn_tile = 2
        game.step(('bugang', 2))
        game.step(RON)
        game.step(RON)
        self.assertEqual(game.players[0].melds, [pon])
        self.assertEqual(game.reason_code, 'rob_kong_win')
        self.assertEqual(game.scores(), [-100, 50, 50])
        self.assertEqual(game.physical_counts(), game.tile_supply)

    def test_final_discard_has_no_calls_and_eighth_flower_can_use_last_live_tile(self):
        game = tiantai.TiantaiGame(5, 0, 27, rules=tiantai)
        game.players[1].hand = counts([2] * 3)
        game.wall = game.wall[:15]
        game._open_offer(Offer(0, 2, 'discard'))
        self.assertEqual(game.legal_actions()[1], [PASS])

        game = tiantai.TiantaiGame(5, 0, 27, rules=tiantai)
        game.flowers[1] = list(range(34, 41))
        game.wall = game.wall[:15] + [41]
        game._draw(1)
        self.assertEqual(game.remaining, 0)
        self.assertEqual(game.win_details[0].patterns, ('八花',))
        self.assertEqual(game.scores(), [-300, 600, -300])
