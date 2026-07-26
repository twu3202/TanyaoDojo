"""
R4 评测桥·第二件:LeanJaxEngine——libriichi 'mjai-log' 引擎,把 LeanACNet 接进
OneVsThree 竞技场。

架构(见 MAHJAX_MIGRATION.md R4):
  - 合法性:libriichi cans(last_cans)+ tracker 手牌/食替 → 87 维动作掩码;
  - 观测:KyokuTracker 从 events_json 增量重建(与 env obs 差分全等已验证);
  - 安全网:每个出牌 JSON 先过 state.validate_reaction_json,失败降级
    (摸切/过)并计数——fallback_count 必须≈0,否则掩码有 bug;
  - 见逃:can_ron_agari + 我们输出 none → tracker.note_ron_passed()(furiten 语义)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracker import KyokuTracker, make_jax_helpers, t2i, to34, BLACK5

SUITS = "mps"
HONORS = ["E", "S", "W", "N", "P", "F", "C"]
RED_STR = {34: "5mr", 35: "5pr", 36: "5sr"}
FIVES = {4: 34, 13: 35, 22: 36}  # black five type -> red id


def i2t(v: int) -> str:
    if v >= 34:
        return RED_STR[v]
    if v < 27:
        return f"{v % 9 + 1}{SUITS[v // 9]}"
    return HONORS[v - 27]


class LeanJaxEngine:
    def __init__(self, params_path: str, name: str = "LeanJax", channels: int = 128,
                 blocks: int = 6):
        import pickle
        import jax
        import jax.numpy as jnp
        from net_lean import LeanACNet

        self.engine_type = "mjai-log"
        self.name = name
        self.player_ids = None
        with open(params_path, "rb") as f:
            self.params = pickle.load(f)
        net = LeanACNet(channels=channels, blocks=blocks)

        def fwd(params, planes, scalars):
            logits, _ = net.apply(params, {"planes": planes, "scalars": scalars})
            return logits

        self._fwd = jax.jit(fwd)
        self._jnp = jnp
        self.shan_fn, self.waits_fn = make_jax_helpers()
        self.trackers: dict[int, tuple] = {}   # game_idx -> (kyoku_key, tracker, fed)
        self.fallback_count = 0
        self.decision_count = 0
        self._debug_hand = True
        self._hand_mismatch = 0

    def set_player_ids(self, player_ids):
        self.player_ids = list(player_ids)

    def start_game(self, game_idx: int):
        self.trackers.pop(game_idx, None)

    def end_kyoku(self, game_idx: int):
        pass

    def end_game(self, game_idx: int, scores):
        self.trackers.pop(game_idx, None)

    # ------------------------------------------------------------- core
    def _tracker_for(self, gi: int, me: int, events: list) -> KyokuTracker:
        key = json.dumps(events[0], sort_keys=True)
        cur = self.trackers.get(gi)
        if cur is None or cur[0] != key or cur[2] > len(events):
            tr = KyokuTracker(me, self.shan_fn, self.waits_fn)
            self.trackers[gi] = [key, tr, 0]
            cur = self.trackers[gi]
        _, tr, fed = cur
        for ev in events[fed:]:
            tr.feed(ev)
        cur[2] = len(events)
        return tr

    def _build_mask(self, tr: KyokuTracker, st, me: int) -> np.ndarray:
        cans = st.last_cans
        mask = np.zeros(87, bool)
        if cans.can_discard:
            if st.self_riichi_accepted:
                mask[71] = tr.last_draw >= 0
            else:
                cand = tr.hand37 > 0
                if st.self_riichi_declared:            # 立直宣言后:只许保听打
                    for v in range(37):
                        if cand[v]:
                            h = tr.hand37.copy()
                            h[v] -= 1
                            h34 = h[:34].astype(np.int8)
                            for r, b in BLACK5.items():
                                h34[b] += h[r]
                            cand[v] = self.shan_fn(h34) <= 0
                else:
                    for tt in np.flatnonzero(tr.forbidden34):   # 食替禁打
                        cand[tt] = False
                        if tt in FIVES:
                            cand[FIVES[tt]] = False
                if tr.last_draw >= 0 and cand[tr.last_draw]:
                    cand[tr.last_draw] = tr.hand37[tr.last_draw] >= 2
                mask[:37] = cand
                mask[71] = tr.last_draw >= 0
            if cans.can_riichi and not st.self_riichi_declared:
                mask[72] = True
        if cans.can_tsumo_agari:
            mask[73] = True
        if cans.can_ron_agari:
            mask[74] = True
        target_tile = None
        if cans.can_pon or cans.can_daiminkan or cans.can_chi_low or cans.can_chi_mid or cans.can_chi_high:
            tgt = cans.target_actor
            target_tile = tr.rivers[tgt][-1] if tr.rivers[tgt] else None
        if cans.can_pon and target_tile is not None:
            tt = to34(target_tile)
            blacks = tr.hand37[tt]
            mask[75] = blacks >= 2
            if tt in FIVES and tr.hand37[FIVES[tt]] > 0 and blacks >= 1:
                mask[76] = True
        if cans.can_daiminkan:
            mask[77] = True
        if target_tile is not None:
            tt = to34(target_tile)
            for can, base, need in (
                (cans.can_chi_low, 78, (tt + 1, tt + 2)),
                (cans.can_chi_mid, 80, (tt - 1, tt + 1)),
                (cans.can_chi_high, 82, (tt - 2, tt - 1)),
            ):
                if not can:
                    continue
                five = next((n for n in need if n in FIVES), None)
                blacks_ok = all(
                    (tr.hand37[n] > 0) or (n == five and tr.hand37[FIVES[n]] > 0)
                    for n in need
                )
                if not blacks_ok:
                    continue
                plain_ok = all(tr.hand37[n] > 0 for n in need)
                mask[base] = plain_ok
                if five is not None and tr.hand37[FIVES[five]] > 0:
                    mask[base + 1] = True
        if cans.can_kakan:
            for m in tr.melds[me]:
                if m[0] == "pon":
                    tt = m[2]
                    have = tr.hand37[tt] + (tr.hand37[FIVES[tt]] if tt in FIVES else 0)
                    if have > 0:
                        mask[37 + tt] = True
        if cans.can_ankan:
            h34 = tr._hand34()
            for tt in range(34):
                if h34[tt] == 4:
                    mask[37 + tt] = True
        if cans.can_ryukyoku:
            mask[85] = True
        if cans.can_pass:
            mask[84] = True
        return mask

    def _to_mjai(self, a: int, tr: KyokuTracker, st, me: int) -> dict:
        cans = st.last_cans
        if a < 37:
            return {"type": "dahai", "actor": me, "pai": i2t(a), "tsumogiri": False}
        if a == 71:
            return {"type": "dahai", "actor": me, "pai": tr.last_draw_str,
                    "tsumogiri": True}
        if a == 72:
            return {"type": "reach", "actor": me}
        if a == 73:
            return {"type": "hora", "actor": me, "target": me, "pai": tr.last_draw_str}
        if a == 74:
            tgt = cans.target_actor
            return {"type": "hora", "actor": me, "target": int(tgt),
                    "pai": tr.rivers_str[tgt][-1]}
        if a in (75, 76):
            tgt = cans.target_actor
            pai = tr.rivers_str[tgt][-1]
            tt = to34(t2i(pai))
            if a == 76:
                consumed = [RED_STR[FIVES[tt]], i2t(tt)]
            else:
                consumed = [i2t(tt), i2t(tt)]
            return {"type": "pon", "actor": me, "target": int(tgt), "pai": pai,
                    "consumed": consumed}
        if a == 77:
            tgt = cans.target_actor
            pai = tr.rivers_str[tgt][-1]
            tt = to34(t2i(pai))
            n_red = int(tr.hand37[FIVES[tt]]) if tt in FIVES else 0
            n_black = int(tr.hand37[tt])
            consumed = ([RED_STR[FIVES[tt]]] * n_red if n_red else []) + [i2t(tt)] * n_black
            return {"type": "daiminkan", "actor": me, "target": int(tgt), "pai": pai,
                    "consumed": consumed[:3]}
        if 78 <= a <= 83:
            tgt = cans.target_actor
            pai = tr.rivers_str[tgt][-1]
            tt = to34(t2i(pai))
            base = {78: (tt + 1, tt + 2), 79: (tt + 1, tt + 2),
                    80: (tt - 1, tt + 1), 81: (tt - 1, tt + 1),
                    82: (tt - 2, tt - 1), 83: (tt - 2, tt - 1)}[a]
            use_red = a % 2 == 1
            consumed = []
            for n in base:
                if use_red and n in FIVES and tr.hand37[FIVES[n]] > 0:
                    consumed.append(RED_STR[FIVES[n]])
                else:
                    consumed.append(i2t(n))
            return {"type": "chi", "actor": me, "target": int(tgt), "pai": pai,
                    "consumed": consumed}
        if 37 <= a < 71:
            tt = a - 37
            pon = next((m for m in tr.melds[me] if m[0] == "pon" and m[2] == tt), None)
            if pon is not None and cans.can_kakan:
                if tt in FIVES and tr.hand37[FIVES[tt]] > 0:
                    pai = RED_STR[FIVES[tt]]
                else:
                    pai = i2t(tt)
                return {"type": "kakan", "actor": me, "pai": pai,
                        "consumed": list(pon[4][:3])}
            n_red = int(tr.hand37[FIVES[tt]]) if tt in FIVES else 0
            consumed = ([RED_STR[FIVES[tt]]] * n_red if n_red else []) + [i2t(tt)] * (4 - n_red)
            return {"type": "ankan", "actor": me, "consumed": consumed}
        if a == 85:
            return {"type": "ryukyoku", "actor": me}
        return {"type": "none"}

    def react_batch(self, game_states):
        try:
            return self._react_batch(game_states)
        except Exception:
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise

    def _react_batch(self, game_states):
        jnp = self._jnp
        metas = []
        planes, scalars = [], []
        for gs in game_states:
            gi = gs.game_index
            me = self.player_ids[gi]
            events = json.loads(gs.events_json)
            tr = self._tracker_for(gi, me, events)
            if self._debug_hand:
                th = np.frombuffer(bytes(gs.state.tehai), dtype=np.uint8).astype(np.int32)
                mine = tr._hand34().astype(np.int32)
                if tr.last_draw >= 0:
                    pass  # libriichi tehai 含摸牌,tracker 亦含 → 直接比
                if not np.array_equal(mine, th) and self._hand_mismatch < 5:
                    self._hand_mismatch += 1
                    print(f"HAND_MISMATCH game={gi} me={me} diff={(mine-th).tolist()}\n"
                          f"  last_events={[e.get('type')+str(e.get('actor','')) for e in events[-6:]]}",
                          file=sys.stderr, flush=True)
            mask = self._build_mask(tr, gs.state, me)
            obs = tr.build_obs()
            planes.append(obs["planes"])
            scalars.append(obs["scalars"])
            metas.append((gi, me, tr, gs.state, mask))
        logits = np.asarray(
            self._fwd(self.params, jnp.asarray(np.stack(planes)), jnp.asarray(np.stack(scalars)))
        )
        out = []
        for (gi, me, tr, st, mask), lg in zip(metas, logits):
            self.decision_count += 1
            lg = np.where(mask, lg, -1e9)
            a = int(np.argmax(lg))
            if mask[a]:
                resp = self._to_mjai(a, tr, st, me)
            else:  # 掩码全空(不应发生):摸切或过,并计数
                self.fallback_count += 1
                if tr.last_draw_str and st.last_cans.can_discard:
                    resp = {"type": "dahai", "actor": me, "pai": tr.last_draw_str,
                            "tsumogiri": True}
                else:
                    resp = {"type": "none"}
            if resp.get("type") == "none" and st.last_cans.can_ron_agari:
                tr.note_ron_passed()
            out.append(json.dumps(resp))
        return out
