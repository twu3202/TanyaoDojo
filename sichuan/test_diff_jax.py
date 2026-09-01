"""P1 硬闸门:env_jax vs reference_impl 逐决策点差分。
两边打同一副牌、走同一动作序列;每一步比对 (合法动作集, 手牌, 副露, 分数, phase, 当前玩家)。
"""
import sys, random, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "/mnt/d/Better_mortal")
from sichuan.reference_impl import SichuanGame, NUM_TILES
from sichuan import env_jax as E

step_jit = jax.jit(E._step_core)
init_jit = jax.jit(E._init_from_wall)

# 参考实现动作 → JAX 动作 id
def ref_to_id(a):
    k, arg = a
    return {"discard": lambda: E.A_DISCARD + arg, "ankan": lambda: E.A_GANG + arg,
            "bugang": lambda: E.A_GANG + arg, "zimo": lambda: E.A_HU,
            "ron": lambda: E.A_HU, "peng": lambda: E.A_PENG,
            "zhigang": lambda: E.A_ZHIGANG, "pass": lambda: E.A_PASS,
            "void": lambda: E.A_VOID + arg}[k]()

def ref_melds(p):
    m = {"peng": E.MK_PENG, "gang_ming": E.MK_GANG_MING,
         "gang_an": E.MK_GANG_AN, "gang_bu": E.MK_GANG_BU}
    return sorted((m[k], t) for k, t in p.melds)

def jax_melds(st, i):
    k, t, n = np.asarray(st.melds_kind[i]), np.asarray(st.melds_tile[i]), int(st.n_melds[i])
    return sorted((int(k[j]), int(t[j])) for j in range(n))

def run_one(seed, verbose=False):
    g = SichuanGame(seed)
    # ⚠️ g.wall 在 __init__ 里已被发牌 pop 掉 52 张,必须重建原始牌墙
    full = [t for t in range(NUM_TILES) for _ in range(4)]
    random.Random(seed).shuffle(full)
    assert list(g.wall) == full[:56], "牌墙重建与 reference 不一致"
    st = init_jit(jnp.asarray(full, jnp.int8))
    rng = random.Random(seed ^ 0x5EED)
    for step in range(4000):
        if g.phase == "over":
            return ("ok_over", step, None) if bool(st.terminated) else ("jax未终局", step, None)
        i, acts = g.legal_actions()
        ids = sorted({ref_to_id(a) for a in acts})
        jm = sorted(np.flatnonzero(np.asarray(st.legal_action_mask)).tolist())
        if ids != jm:
            return ("合法动作不符", step, f"phase={g.phase} ref={ids} jax={jm}")
        if int(st.current_player) != i:
            return ("当前玩家不符", step, f"ref={i} jax={int(st.current_player)}")
        a = rng.choice(acts)
        g.step(a)
        st = step_jit(st, jnp.int32(ref_to_id(a)))
        for p in range(4):
            if list(np.asarray(st.hand[p])) != g.players[p].hand:
                return ("手牌不符", step, f"p{p} act={a}")
            if jax_melds(st, p) != ref_melds(g.players[p]):
                return ("副露不符", step, f"p{p} act={a} ref={ref_melds(g.players[p])} jax={jax_melds(st,p)}")
        if list(np.asarray(st.score)) != g.scores():
            return ("分数不符", step, f"act={a} ref={g.scores()} jax={list(np.asarray(st.score))}")
    return ("超步数", 4000, None)

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
bad = {}
for s in range(n):
    r, step, det = run_one(s)
    if not r.startswith("ok"):
        bad.setdefault(r, []).append((s, step, det))
    if (s + 1) % 50 == 0:
        print(f"{s+1}/{n}  失配 {sum(len(v) for v in bad.values())}", flush=True)
print(f"\n=== 差分结果 {n} 局 ===")
if not bad:
    print("零失配 ✓")
else:
    for k, v in bad.items():
        print(f"  {k}: {len(v)} 例   首例 seed={v[0][0]} step={v[0][1]}  {v[0][2]}")
