"""川麻参考实现 v0 用例集。直接运行:python test_sichuan.py(无 pytest 依赖)。"""
import sys
from reference_impl import (NUM_TILES, SichuanGame, calc_fan, is_hu, random_playout, suit_of)

PASS = 0


def check(name, cond):
    global PASS
    assert cond, f"FAIL: {name}"
    PASS += 1


def H(*spec):
    """spec: (tile, count)... -> counts[27]"""
    c = [0] * NUM_TILES
    for t, n in spec:
        c[t] += n
    return c


def m(r): return r - 1          # 万 r
def p(r): return 9 + r - 1      # 筒 r
def s(r): return 18 + r - 1     # 条 r


# ---------- 1. 胡牌型 ----------
# 平胡:123万 456万 789万 111筒 + 99筒,缺条
check("平胡", is_hu(H((m(1),1),(m(2),1),(m(3),1),(m(4),1),(m(5),1),(m(6),1),
                     (m(7),1),(m(8),1),(m(9),1),(p(1),3),(p(9),2)), 0, 2))
# 缺门违规:同一手,缺筒
check("缺门违规不可胡", not is_hu(H((m(1),1),(m(2),1),(m(3),1),(m(4),1),(m(5),1),(m(6),1),
                                    (m(7),1),(m(8),1),(m(9),1),(p(1),3),(p(9),2)), 0, 1))
# 七对
check("七对", is_hu(H((m(1),2),(m(3),2),(m(5),2),(p(2),2),(p(4),2),(p(6),2),(p(8),2)), 0, 2))
# 七对带副露不可
check("七对须门清", not is_hu(H((m(1),2),(m(3),2),(m(5),2),(p(2),2),(p(4),2)), 1, 2))
# 差一张
check("未完成不可胡", not is_hu(H((m(1),1),(m(2),1),(m(4),1),(m(5),1),(m(6),1),(m(7),1),
                                  (m(8),1),(m(9),1),(p(1),3),(p(9),2),(m(9),1)), 0, 2))

# ---------- 2. 番种 ----------
fan, names = calc_fan(H((m(1),3),(m(3),3),(m(5),3),(m(7),3),(m(9),2)), [], 2, {})
check("清对=清一色2+对对1=3番", fan == 3 and "清一色" in names and "对对胡" in names)

fan, names = calc_fan(H((m(1),2),(m(2),2),(m(3),2),(m(4),4),(m(5),2),(m(6),2)), [], 2, {})
check("清龙七对触顶4番", fan == 4 and "龙七对" in names and "清一色" in names)
check("龙七对四张不另计根", "根x1" not in names)

fan, names = calc_fan(H((p(1),2)), [("peng", m(1)), ("gang_ming", m(2)), ("peng", s(1)), ("peng", s(3))], 1, {})
check("金钩钓+对对+根", "金钩钓" in names and "对对胡" in names and "根x1" in names)

fan, names = calc_fan(H((m(1),1),(m(2),1),(m(3),1),(m(4),1),(m(5),1),(m(6),1),
                        (m(7),1),(m(8),1),(m(9),1),(p(1),3),(p(9),2)), [], 2,
                      {"zimo": True, "gang_shang_hua": True})
check("自摸+杠上花加计", fan == 2)

fan, _ = calc_fan(H((m(1),3),(m(3),3),(m(5),3),(m(7),3),(m(9),2)), [], 2,
                  {"zimo": True, "hai_di": True})
check("封顶4番", fan == 4)

# ---------- 3. 刮风下雨结算 ----------
g = SichuanGame(seed=1)
for i, pl in enumerate(g.players):
    pl.void = 2
g.phase = "action"; g.cur = 0
g._gang_money(0, "gang_an")
check("暗杠三家各付2", g.scores() == [6, -2, -2, -2])
check("暗杠入账本", len(g.gang_ledger) == 3)
g._gang_money(1, "gang_ming", provider=0)
check("直杠放杠者付2", g.scores() == [4, 0, -2, -2])
g.players[3].hu = True
g._gang_money(2, "gang_bu")
check("补杠只向在场未胡者收", g.scores() == [3, -1, 0, -2])
check("零和", sum(g.scores()) == 0)

# ---------- 4. 流局查叫与退税 ----------
g = SichuanGame(seed=2)
for i, pl in enumerate(g.players):
    pl.void = 2
    pl.hand = [0] * NUM_TILES
# P0 听牌(单钓,最大番=平胡0番→1分);先给它收一笔杠钱
g.players[0].hand = H((m(1),1),(m(2),1),(m(3),1),(m(4),1),(m(5),1),(m(6),1),
                      (m(7),1),(m(8),1),(m(9),1),(p(1),3),(p(9),1))
# P1 未听 + 有杠钱收入(应退)
g.players[1].hand = H((m(1),1),(m(4),1),(m(7),1),(p(2),1),(p(5),1),(p(8),1),(m(2),1),
                      (p(3),1),(p(6),1),(m(5),1),(m(8),1),(p(1),1),(p(4),1))
g.players[2].hand = g.players[1].hand[:]     # P2 未听
g.players[3].hand = g.players[0].hand[:]     # P3 听
g._pay(0, 1, 2, gang=True)                   # P1 曾收 P0 杠钱 2
base_scores = g.scores()
check("查叫前账本", base_scores == [-2, 2, 0, 0])
g.wall = []
g._liuju_settle()
sc = g.scores()
# 查叫:P1,P2 各付 P0,P3 每人 1 分;退税:P1 未听退 2 给 P0
check("查大叫+退税", sc == [-2 + 2 + 2, 2 - 2 - 2, -2, 2] and sum(sc) == 0)
check("流局后 over", g.phase == "over")

# ---------- 5. 一炮多响 ----------
g = SichuanGame(seed=3)
for i, pl in enumerate(g.players):
    pl.void = 2
    pl.hand = [0] * NUM_TILES
ting = H((m(1),1),(m(2),1),(m(3),1),(m(4),1),(m(5),1),(m(6),1),
         (m(7),1),(m(8),1),(m(9),1),(p(1),3),(p(9),1))
g.players[1].hand = ting[:]
g.players[2].hand = ting[:]
g.players[0].hand = H((p(9),2),(m(1),1),(p(2),1),(p(3),1),(p(4),1),(p(5),1),(p(6),1),
                      (p(7),1),(p(8),1),(m(4),1),(m(7),1),(p(1),1))
g.players[3].hand = H((p(2),3),(p(3),3),(p(4),3),(p(5),3),(p(6),1))
g.phase = "action"; g.cur = 0
g.wall = [m(1)] * 56   # 配平至 108(四手 52 + 墙 56),同时保证摸牌不空
g.step(("discard", p(9)))
i1, acts1 = g.legal_actions()
check("响应者1可荣", i1 == 1 and ("ron", None) in acts1)
g.step(("ron", None))
i2, acts2 = g.legal_actions()
check("响应者2可荣", i2 == 2 and ("ron", None) in acts2)
g.step(("ron", None))
i3, acts3 = g.legal_actions()
g.step(("pass", None))
check("双响均成立", g.players[1].hu and g.players[2].hu)
check("放炮者付两家", g.players[0].score_delta == -2 and
      g.players[1].score_delta == 1 and g.players[2].score_delta == 1)
check("守恒(多响)", g.tile_conservation() == 108)

# ---------- 6. 缺门强制打牌 ----------
g = SichuanGame(seed=4)
g.step(("void", 0)); g.step(("void", 0)); g.step(("void", 0)); g.step(("void", 0))
i, acts = g.legal_actions()
pl = g.players[i]
has_void_tile = any(pl.hand[t] for t in range(NUM_TILES) if suit_of(t) == pl.void)
if has_void_tile:
    check("缺门必打:动作全为缺门牌",
          all(k == "discard" and suit_of(a) == pl.void for k, a in acts))
else:
    check("无缺门牌时不受限", any(k == "discard" for k, a in acts))

# ---------- 7. 随机对局不变量(200 局) ----------
overs = {"hu3": 0, "wall": 0}
total_hu = 0
for seed in range(200):
    g = random_playout(seed)
    check(f"对局{seed}终局", g.phase == "over")
    check(f"对局{seed}零和", sum(g.scores()) == 0)
    check(f"对局{seed}守恒", g.tile_conservation() == 108)
    check(f"对局{seed}胡数<=3", len(g.hu_order) <= 3)
    total_hu += len(g.hu_order)
    if len(g.hu_order) >= 3:
        overs["hu3"] += 1
    else:
        overs["wall"] += 1
    for w, fan, names in g.hu_order:
        check(f"对局{seed}番在范围", 0 <= fan <= 4)

print(f"ALL {PASS} CHECKS PASSED  (200局: 三胡终局={overs['hu3']}, 流局={overs['wall']}, 总胡次={total_hu})")
