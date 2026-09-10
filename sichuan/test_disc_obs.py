"""P3 打牌观测(obs.disc=True)的差分测试。

查表本身已经和参考实现对拍过(common/suit_table.py verify),这里要抓的是**向量化**
引进的新错:
  1. 手里没有的牌 `hand - onehot` = -1,5 进制索引对负计数**静默错**(不越界、不报错,
     只是查到另一条),必须 clip + 遮罩;
  2. disc=True 不许改动前 22 个平面 —— 与 `dual_c=0` 同样的规矩:开关关掉要逐字等价,
     开关打开也不能污染已有特征。

用法: python -m sichuan.test_disc_obs [n]
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.suit_table import load_table, make_jax_ops, NUM_TILES
from sichuan import env_jax as E
from sichuan.obs import observe, _discard_shanten, NUM_PLANES, NUM_PLANES_DISC

_, shanten = make_jax_ops(load_table())


def check_discard_shanten(n: int, rng: np.random.Generator) -> int:
    """逐张单独查表 vs 向量化一把查,必须逐位相等。"""
    bad = 0
    for _ in range(n):
        # 造一个合法的 3k+2 手牌(k = 4-n_melds)
        nm = int(rng.integers(0, 5))
        need = 3 * (4 - nm) + 2
        hand = np.zeros(NUM_TILES, np.int32)
        left = need
        while left > 0:
            t = int(rng.integers(0, NUM_TILES))
            if hand[t] < 4:
                hand[t] += 1
                left -= 1
        vd = int(rng.integers(-1, 3))
        h = jnp.asarray(hand)
        got = np.asarray(_discard_shanten(h, jnp.int32(nm), jnp.int32(vd)))
        for t in range(NUM_TILES):
            if hand[t] == 0:
                continue                      # 非法打牌,由 observe 遮罩,值不参与比较
            want = int(shanten(h.at[t].add(-1), jnp.int32(nm), jnp.int32(vd)))
            if int(got[t]) != want:
                bad += 1
                if bad <= 3:
                    print(f"  ✗ hand={hand.tolist()} nm={nm} vd={vd} t={t} "
                          f"向量化={int(got[t])} 逐张={want}")
    return bad


def check_negative_hazard(rng: np.random.Generator) -> int:
    """专门造"手里没有的牌"的情形:clip 之后应等于原手牌的向听,而不是别的数。"""
    bad = 0
    for _ in range(200):
        nm = int(rng.integers(0, 5))
        need = 3 * (4 - nm) + 2
        hand = np.zeros(NUM_TILES, np.int32)
        # 只用前 9 张牌,保证后 18 个位置为 0(即"手里没有")
        left = need
        while left > 0:
            t = int(rng.integers(0, 9))
            if hand[t] < 4:
                hand[t] += 1
                left -= 1
        vd = int(rng.integers(-1, 3))
        h = jnp.asarray(hand)
        got = np.asarray(_discard_shanten(h, jnp.int32(nm), jnp.int32(vd)))
        base = int(shanten(h, jnp.int32(nm), jnp.int32(vd)))
        for t in range(9, NUM_TILES):
            if int(got[t]) != base:
                bad += 1
                if bad <= 3:
                    print(f"  ✗ 负计数泄漏 t={t} 得到={int(got[t])} 应为原手牌向听={base}")
    return bad


def check_plane_compat(n: int) -> int:
    """disc=True 的前 22 个平面必须与 disc=False 逐位相同。"""
    walls = jnp.stack([jax.random.permutation(jax.random.PRNGKey(s), jnp.arange(E.WALL_SIZE) // 4)
                       for s in range(n)])
    st = jax.vmap(E._init_from_wall)(walls)
    a = jax.vmap(lambda s: observe(s, False))(st)["planes"]
    b = jax.vmap(lambda s: observe(s, True))(st)["planes"]
    assert a.shape[-1] == NUM_PLANES, a.shape
    assert b.shape[-1] == NUM_PLANES_DISC, b.shape
    same = bool(jnp.all(a == b[..., :NUM_PLANES]))
    if not same:
        d = int(jnp.sum(jnp.any(a != b[..., :NUM_PLANES], axis=(1, 2))))
        print(f"  ✗ disc=True 污染了已有平面,{d}/{n} 个状态不一致")
        return 1
    # 新平面必须在 [0,1] 内且非全常数(否则等于没加特征)
    new = b[..., NUM_PLANES:]
    assert float(new.min()) >= 0.0 and float(new.max()) <= 1.0, (float(new.min()), float(new.max()))
    if float(new.std()) == 0.0:
        print("  ✗ 新平面是常数,没有携带信息")
        return 1
    print(f"  新平面范围 [{float(new.min()):.3f}, {float(new.max()):.3f}] "
          f"std={float(new.std()):.4f}  听牌位占比={float((new[..., 1] > 0).mean()):.4f}")
    return 0


def check_live_rollout(n_env: int = 256, n_step: int = 120) -> int:
    """中局检查。开局状态测不出东西:非庄家 13 张(3k+1),特征按设计整段填常数。
    真正要确认的是**在真实决策点上它是活的** —— 否则训练三小时烧的是一个常数平面。"""
    walls = jnp.stack([jax.random.permutation(jax.random.PRNGKey(1000 + s),
                                              jnp.arange(E.WALL_SIZE) // 4)
                       for s in range(n_env)])
    st = jax.vmap(E._init_from_wall)(walls)
    key = jax.random.PRNGKey(7)

    @jax.jit
    def step(st, key):
        mask = jax.vmap(E._legal_mask)(st)
        logits = jnp.where(mask, 0.0, -1e9)
        a = jax.random.categorical(key, logits, axis=-1)
        return jax.vmap(E._step_core)(st, a)

    live, tenpai, vals = 0, 0, []
    for i in range(n_step):
        key, sub = jax.random.split(key)
        st = step(st, sub)
        P = jax.vmap(lambda s: observe(s, True))(st)["planes"]
        sh, tp = P[..., NUM_PLANES], P[..., NUM_PLANES + 1]
        act = sh < 1.0                       # 1.0 = 非法/非打牌阶段的填充值
        live += int(act.any(axis=1).sum())
        tenpai += int((tp > 0).any(axis=1).sum())
        if act.any():
            vals.append(np.asarray(sh[act]))
    frac = live / (n_env * n_step)
    tfrac = tenpai / (n_env * n_step)
    v = np.concatenate(vals) * 8.0
    print(f"  活跃决策点 {frac:.1%}   其中存在一手打成听牌 {tfrac:.1%}")
    print(f"  打后向听分布: min={v.min():.0f} 中位={np.median(v):.0f} "
          f"max={v.max():.0f} 唯一值={sorted(set(v.round().astype(int).tolist()))[:8]}")
    if frac < 0.20:
        print(f"  ✗ 只有 {frac:.1%} 的决策点用得上这个特征,信号太稀")
        return 1
    if len(set(v.round().astype(int).tolist())) < 3:
        print("  ✗ 打后向听几乎没有区分度")
        return 1
    return 0


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(20260910)
    fails = 0
    print(f"1) 逐张对拍 ({n} 手)")
    b = check_discard_shanten(n, rng)
    print(f"   {'✓ 全对' if b == 0 else f'✗ {b} 处不一致'}")
    fails += b
    print("2) 负计数泄漏 (200 手 × 18 空位)")
    b = check_negative_hazard(rng)
    print(f"   {'✓ 无泄漏' if b == 0 else f'✗ {b} 处泄漏'}")
    fails += b
    print("3) 与 disc=False 的平面兼容性")
    fails += check_plane_compat(64)
    print("4) 中局 rollout:特征是否真的活着")
    fails += check_live_rollout()
    print()
    print("全部通过" if fails == 0 else f"失败 {fails} 项")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
