"""Rule-oracle regressions: a pending pung/kong must not pre-empt a later win.

Fixtures obey the four-copy limit and tile conservation; every seat rotation is tested.
Run with `python -m unittest sichuan.test_response_priority` (JAX tests skip if absent).
"""
import importlib.util
import unittest
from collections import Counter

from sichuan.reference_impl import SichuanGame, is_hu


def priority_game(claim="peng", offset=0, double=False, win_first=False, empty_wall=False,
                  caller_tiles=None, caller_melds=(), next_tile=None):
    g = SichuanGame(0)
    winner = [1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 17, 17]
    caller = [0] * (3 if claim == "zhigang" else 2) + [6, 6, 7, 7, 8, 8, 15, 15, 16, 16]
    if claim == "peng":
        caller.append(6)
    if caller_tiles is not None:
        caller = caller_tiles
    hands = [[18] * 4 + [19] * 4 + [20] * 4 + [21], caller, winner,
             winner if double else [21] * 3 + [22] * 4 + [23] * 4 + [24] * 2]
    used = Counter([0])  # discard awaiting responses
    for _, tile in caller_melds:
        used[tile] += 3
    for i, tiles in enumerate(hands):
        p = g.players[(i + offset) % 4]
        p.hand = [tiles.count(t) for t in range(27)]
        p.void = 1 if i == 0 else 2
        used.update(tiles)
    assert all(n <= 4 for n in used.values())
    g.wall = [t for t in range(27) for _ in range(4 - used[t])]
    if next_tile is not None:
        g.wall.remove(next_tile)
        g.wall.append(next_tile)
    g.players[(offset + 1) % 4].melds = list(caller_melds)
    g.discards = [[] for _ in range(4)]
    g.discards[offset] = [0]
    if empty_wall:
        g.discards[offset] = g.wall + [0]
        g.wall = []
    if win_first:
        a, b = (offset + 1) % 4, (offset + 2) % 4
        g.players[a], g.players[b] = g.players[b], g.players[a]
    g.cur = offset
    g.phase = "response"
    g.pending_discard = (offset, 0)
    g.response_queue = [(offset + k) % 4 for k in (1, 2, 3)]
    g.ron_winners = []
    g.pending_claim = None
    g.void_declared = 4
    g.discard_was_gang = False
    assert g.tile_conservation() == 108
    return g


class ReferencePriorityTests(unittest.TestCase):
    def make_game(self, *args, **kwargs):
        self.g = priority_game(*args, **kwargs)
        return self.g

    def move(self, action):
        self.assertIn(action, self.g.legal_actions()[1])
        self.g.step(action)
        self.assertEqual(self.g.tile_conservation(), 108)
        self.assertEqual(sum(self.g.scores()), 0)

    def check_drawn(self, tile):
        self.assertEqual(self.g.drawn_tile, tile)

    def test_post_pung_forbids_self_draw_and_self_kongs(self):
        cases = [
            ([0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15], (), ("zimo", None)),
            ([0, 0] + [6] * 4 + [7, 7, 8, 8, 15, 15, 16], (), ("ankan", 6)),
            ([0, 0, 6, 7, 8, 9, 10, 11, 15, 16], (("peng", 6),), ("bugang", 6)),
        ]
        for tiles, melds, forbidden in cases:
            with self.subTest(action=forbidden):
                g = self.make_game(caller_tiles=tiles, caller_melds=melds)
                wall = g.wall[:]
                for action in (("peng", None), ("pass", None), ("pass", None)):
                    self.move(action)
                p = g.players[1]
                self.assertEqual(sum(p.hand) % 3, 2)
                self.assertEqual(g.wall, wall)
                self.check_drawn(None)
                if forbidden[0] == "zimo":
                    self.assertTrue(is_hu(p.hand, len(p.melds), p.void))
                self.assertTrue(g.legal_actions()[1])
                self.assertTrue(all(k == "discard" for k, _ in g.legal_actions()[1]))
                with self.assertRaisesRegex(ValueError, "碰后只能弃牌"):
                    g.step(forbidden)
                self.move(g.legal_actions()[1][0])
                self.check_drawn(None)
                while g.phase == "response":
                    self.move(("pass", None))
                self.assertEqual(g.cur, 2)
                self.check_drawn(wall[-1])

    def test_actual_draw_can_self_win(self):
        g = self.make_game(caller_tiles=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 4, 4],
                           next_tile=17)
        for _ in range(3):
            self.move(("pass", None))
        self.check_drawn(17)
        self.move(("zimo", None))
        self.assertTrue(g.players[1].hu)

    def test_later_ron_cancels_claim(self):
        for claim in ("peng", "zhigang"):
            for offset in range(4):
                with self.subTest(claim=claim, offset=offset):
                    g = self.make_game(claim, offset)
                    caller, winner = (offset + 1) % 4, (offset + 2) % 4
                    original = g.players[caller].hand[:]
                    wall = g.wall[:]
                    self.move((claim, None))
                    self.assertEqual(g.phase, "response")
                    self.assertEqual(g.legal_actions()[0], winner)
                    self.assertIn(("ron", None), g.legal_actions()[1])
                    self.assertEqual(g.wall, wall)
                    self.assertEqual(g.gang_ledger, [])
                    self.assertEqual(g.players[caller].hand, original)
                    self.move(("ron", None))
                    self.move(("pass", None))
                    self.assertTrue(g.players[winner].hu)
                    self.assertEqual(g.players[caller].melds, [])
                    self.assertEqual(g.gang_ledger, [])
                    self.assertGreater(g.players[winner].score_delta, 0)
                    self.assertEqual(g.discards[offset], [])

    def test_passes_resolve_claim_once(self):
        for claim in ("peng", "zhigang"):
            for offset in range(4):
                with self.subTest(claim=claim, offset=offset):
                    g = self.make_game(claim, offset)
                    caller = (offset + 1) % 4
                    n_wall = len(g.wall)
                    next_tile = g.wall[-1]
                    self.move((claim, None))
                    self.move(("pass", None))  # a legal win was declined
                    self.assertEqual(g.players[caller].melds, [])
                    self.move(("pass", None))
                    self.assertEqual(g.cur, caller)
                    self.assertEqual(g.phase, "action")
                    self.assertEqual(g.players[caller].melds,
                                     [("peng" if claim == "peng" else "gang_ming", 0)])
                    self.assertEqual(len(g.wall), n_wall - (claim == "zhigang"))
                    self.check_drawn(next_tile if claim == "zhigang" else None)
                    self.assertEqual(len(g.gang_ledger), int(claim == "zhigang"))
                    self.assertEqual(g.discards[offset], [])
                    # A new discard must forget the previous pending claim.
                    tile = next(a[1] for a in g.legal_actions()[1] if a[0] == "discard")
                    self.move(("discard", tile))
                    self.assertIsNone(g.pending_claim)

    def test_two_later_winners(self):
        for offset in range(4):
            with self.subTest(offset=offset):
                g = self.make_game("peng", offset, double=True)
                self.move(("peng", None))
                self.move(("ron", None))
                self.move(("ron", None))
                self.assertTrue(g.players[(offset + 2) % 4].hu)
                self.assertTrue(g.players[(offset + 3) % 4].hu)
                self.assertEqual(g.players[(offset + 1) % 4].melds, [])
                self.assertEqual(len(g.hu_order), 2)

    def test_earlier_win_cancels_later_claim(self):
        for claim in ("peng", "zhigang"):
            with self.subTest(claim=claim):
                g = self.make_game(claim, win_first=True)
                self.move(("ron", None))
                self.move((claim, None))
                self.move(("pass", None))
                self.assertTrue(g.players[1].hu)
                self.assertEqual(g.players[2].melds, [])
                self.assertEqual(g.gang_ledger, [])

    def test_wall_empty_kong_resolves_without_draw(self):
        for offset in range(4):
            with self.subTest(offset=offset):
                g = self.make_game("zhigang", offset, empty_wall=True)
                self.move(("zhigang", None))
                self.move(("pass", None))
                self.move(("pass", None))
                self.assertEqual(g.phase, "over")
                self.assertEqual(g.wall, [])
                self.assertEqual(g.players[(offset + 1) % 4].melds, [("gang_ming", 0)])


HAS_JAX = importlib.util.find_spec("jax") is not None and importlib.util.find_spec("flax") is not None


@unittest.skipUnless(HAS_JAX, "JAX/Flax unavailable")
class JaxPriorityTests(ReferencePriorityTests):
    @classmethod
    def setUpClass(cls):
        import jax
        from sichuan import env_jax as E
        from sichuan.obs import observe
        cls.step_jit = staticmethod(jax.jit(E._step_core))
        cls.observe_jit = staticmethod(jax.jit(observe))

    def make_game(self, *args, **kwargs):
        import jax.numpy as jnp
        from sichuan import env_jax as E
        g = super().make_game(*args, **kwargs)
        st = E._init_from_wall(jnp.repeat(jnp.arange(27, dtype=jnp.int8), 4))
        river = jnp.array([[r.count(t) for t in range(27)] for r in g.discards], dtype=jnp.int8)
        kinds = [[E.MK_PENG] * len(p.melds) + [0] * (4 - len(p.melds)) for p in g.players]
        tiles = [[tile for _, tile in p.melds] + [-1] * (4 - len(p.melds)) for p in g.players]
        self.st = E._refresh(st.replace(
            hand=jnp.array([p.hand for p in g.players], dtype=jnp.int8),
            melds_kind=jnp.array(kinds, dtype=jnp.int8), melds_tile=jnp.array(tiles, dtype=jnp.int8),
            n_melds=jnp.array([len(p.melds) for p in g.players], dtype=jnp.int8),
            void=jnp.array([p.void for p in g.players], dtype=jnp.int8),
            wall=jnp.array(g.wall + [0] * (108 - len(g.wall)), dtype=jnp.int8),
            wall_ix=jnp.int32(108 - len(g.wall)), river=river,
            phase=jnp.int32(E.PH_RESP), cur=jnp.int32(g.cur), void_declared=jnp.int32(4),
            pending_tile=jnp.int32(0), pending_from=jnp.int32(g.cur),
            resp_queue=jnp.array(g.response_queue, dtype=jnp.int32)))
        return g

    def check_drawn(self, tile):
        import numpy as np
        super().check_drawn(tile)
        self.assertEqual(int(self.st.drawn_tile), -1 if tile is None else tile)
        actual = self.observe_jit(self.st)["planes"][:, 4]
        np.testing.assert_array_equal(np.flatnonzero(actual), [] if tile is None else [tile])

    def test_batched_later_ron(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from sichuan import env_jax as E
        states = []
        for offset in range(4):
            self.make_game("peng", offset)
            states.append(self.st)
        st = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
        step = jax.jit(jax.vmap(E._step_core))
        for action in (E.A_PENG, E.A_HU, E.A_PASS):
            st = step(st, jnp.full(4, action, dtype=jnp.int32))
        for offset in range(4):
            self.assertTrue(bool(st.finished[offset, (offset + 2) % 4]))
            self.assertEqual(int(st.n_melds[offset].sum()), 0)
        np.testing.assert_array_equal(st.score.sum(axis=1), 0)

    def move(self, action):
        import numpy as np
        import jax.numpy as jnp
        from sichuan import env_jax as E
        ids = {"peng": E.A_PENG, "zhigang": E.A_ZHIGANG, "ron": E.A_HU, "zimo": E.A_HU, "pass": E.A_PASS}
        aid = action[1] if action[0] == "discard" else ids[action[0]]
        self.assertTrue(bool(self.st.legal_action_mask[aid]))
        super().move(action)
        self.st = self.step_jit(self.st, jnp.int32(aid))
        g, st = self.g, self.st
        self.assertEqual(int(st.phase), {"void": 0, "action": 1, "response": 2, "over": 3}[g.phase])
        if g.phase != "over":
            i, acts = g.legal_actions()
            self.assertEqual(int(st.current_player), i)
            ref_ids = [t if k == "discard" else E.A_GANG + t if k in ("ankan", "bugang") else ids[k]
                       for k, t in acts]
            np.testing.assert_array_equal(np.flatnonzero(st.legal_action_mask), sorted(ref_ids))
        np.testing.assert_array_equal(st.hand, [p.hand for p in g.players])
        np.testing.assert_array_equal(st.score, g.scores())
        np.testing.assert_array_equal(st.finished, [p.hu for p in g.players])
        np.testing.assert_array_equal(st.n_melds, [len(p.melds) for p in g.players])
        np.testing.assert_array_equal(st.river, [[r.count(t) for t in range(27)] for r in g.discards])
        self.assertEqual(int(st.wall_ix), 108 - len(g.wall))
        ledger = np.zeros((4, 4), dtype=int)
        for payer, receiver, amount in g.gang_ledger:
            ledger[payer, receiver] += amount
        np.testing.assert_array_equal(st.gang_ledger, ledger)


if __name__ == "__main__":
    unittest.main()
