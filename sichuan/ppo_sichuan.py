"""
川麻 P2 训练器:自博弈 PPO + 手工向听势能塑形。

与立直线的差别(都是有意为之,不是省事):
  · **不用 oracle critic**——P2 先把最简形态跑通,判据是"3 亿步内超 L1",
    非对称 critic 留到 P3 再上,免得又一次分不清是方法有效还是别的因素;
  · **不用锚**——川麻从零起,没有 BC 基座可锚,也就没有"锚过紧"这类问题;
  · 四座共用同一套参数,四家的转移全部进训练(4 倍数据)。

三条免费仪表(方案要求,出问题时它们比外评早几小时报警):
  · clip_frac —— 被裁剪的样本比例。>0.3 说明步子迈太大,信任域形同虚设;
  · adv_std_osc —— 相邻两次更新的优势标准差之比取 log 后的绝对值。持续 >0.5
    说明优势尺度在震荡,通常是价值函数没跟上;
  · max_ratio / max_kl —— 逐状态的最坏情况,均值好看但尾巴炸掉时只有它们能看出来。

用法:
  PYTHONPATH=~/mahjax:<repo> python sichuan/ppo_sichuan.py num_envs=1024 total_timesteps=3e8
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, NamedTuple

import distrax
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jax import lax
from omegaconf import OmegaConf
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sichuan import env_jax as E
from sichuan.net import SichuanACNet
import functools

from sichuan.obs import observe
from sichuan.shaping import auto_reset_shaped

NEG = -1e9


class Args(BaseModel):
    num_envs: int = 1024
    num_steps: int = 32
    updates_per_jit: int = 4
    update_epochs: int = 1
    minibatch_size: int = 4096
    total_timesteps: int = 300_000_000
    lr: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    dual_c: float = 0.0          # dual-clip 系数(Ye et al. 2019 用 3.0);0 = 关闭,与原版等价
    obs_disc: bool = False       # P3 观测:加“打出每张牌后的向听/是否听牌”两平面;False = 与 P2 逐字等价
    channels: int = 128
    blocks: int = 6
    shaping_w: float = 0.5
    mem_fraction: float = 0.85
    seed: int = 0
    save_path: str = "runs/sichuan_p2.pkl"
    snapshot_every_blocks: int = 64
    log_every_blocks: int = 8
    init_from: str = ""          # 从已有参数续训(WSL 被回收后重启用)
    steps_done: int = 0          # 续训时的已跑步数,只影响日志与剩余量


args = Args(**OmegaConf.to_object(OmegaConf.from_cli()))
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(args.mem_fraction))
print(args, file=sys.stderr)

ENV = E.make()
STEP_FN = auto_reset_shaped(ENV.step, ENV.init, args.shaping_w)
NP = E.NUM_PLAYERS
BATCH = args.num_envs * args.num_steps
assert BATCH % args.minibatch_size == 0
NUM_MB = BATCH // args.minibatch_size
NUM_UPDATES = int(args.total_timesteps // BATCH)
NUM_JIT = max(1, NUM_UPDATES // args.updates_per_jit)


class Tr(NamedTuple):
    is_new: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray          # (T, B, 4)
    log_prob: jnp.ndarray
    obs: Dict[str, jnp.ndarray]
    mask: jnp.ndarray
    cur: jnp.ndarray


# 观测版本在进程启动时定死:disc 进 jit 是静态参数,不能每步传。
_OBSERVE = functools.partial(observe, disc=args.obs_disc)


def rollout(params, net, env_state, key):
    def one(carry, _):
        st, rng = carry
        rng, ka, ke = jax.random.split(rng, 3)
        obs = jax.vmap(_OBSERVE)(st)
        mask = st.legal_action_mask.astype(jnp.bool_)
        logits, value = net.apply(params, obs)
        logits = jnp.where(mask, logits, NEG)
        dist = distrax.Categorical(logits=logits)
        a, lp = dist.sample_and_log_prob(seed=ka)
        done = jnp.asarray(st.terminated | st.truncated, jnp.bool_)
        cur = jnp.asarray(st.current_player, jnp.int32)
        nxt = jax.vmap(STEP_FN)(st, a, jax.random.split(ke, args.num_envs))
        r = jnp.asarray(nxt.rewards, jnp.float32)
        return (nxt, rng), Tr(done, a, value, r, lp, obs, mask, cur)

    (env_state, _), traj = lax.scan(one, (env_state, key), None, length=args.num_steps)
    return env_state, jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj)


def compute_adv(traj: Tr):
    """逐座位链式 GAE:每步只有 current_player 行动,但四家都可能收到奖励。
    与 ppo_qcritic 的做法同构——按玩家各留一份 (gae, next_value, 奖励累加器),
    轮到该玩家再行动时才把攒下的奖励结算成一次 TD。"""
    def per_env(tr: Tr):
        def scan_fn(carry, t):
            gae, nv, racc, has_nv, next_valid, is_new_next = carry
            p = t.cur
            # 边界量取 t+1 的 is_new(不是 t 自己的):reverse scan 里 is_new[t] 为真说明
            # t 是**开局步**,在它上面归零等于把开局当终局。详见 ppo_qcritic 同处注释。
            done = is_new_next
            gae = jnp.where(done, 0.0, gae)
            racc = jnp.where(done, 0.0, racc)
            nv = jnp.where(done, 0.0, nv)
            has_nv = jnp.where(done, False, has_nv)
            racc = racc + t.reward
            pr = racc[p]
            racc = racc.at[p].set(0.0)
            td = pr + args.gamma * nv[p] - t.value
            new_gae = td + args.gamma * args.gae_lambda * gae[p]
            gae = gae.at[p].set(new_gae)
            valid = has_nv[p] | done | next_valid[p]
            adv = jnp.where(valid, new_gae, 0.0)
            tgt = jnp.where(valid, adv + t.value, t.value)
            carry = (gae, nv.at[p].set(t.value), racc,
                     has_nv.at[p].set(True), next_valid.at[p].set(valid) | done,
                     t.is_new)
            return carry, (adv, tgt, valid)

        init = (jnp.zeros(NP), jnp.zeros(NP), jnp.zeros(NP),
                jnp.zeros(NP, bool), jnp.zeros(NP, bool), jnp.bool_(False))
        _, out = lax.scan(scan_fn, init, tr, reverse=True)
        return out

    return jax.vmap(per_env)(traj)


def make_update(net, tx):
    def update(ts: TrainState, traj: Tr, adv, tgt, valid, key):
        flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), traj)
        adv, tgt, valid = [x.reshape(-1) for x in (adv, tgt, valid)]
        w = valid.astype(jnp.float32)
        adv_n = (adv - (adv * w).sum() / jnp.maximum(w.sum(), 1))
        adv_std = jnp.sqrt((adv_n ** 2 * w).sum() / jnp.maximum(w.sum(), 1) + 1e-8)
        adv_n = adv_n / adv_std

        def epoch(carry, _):
            ts, rng = carry
            rng, k = jax.random.split(rng)
            perm = jax.random.permutation(k, BATCH)

            def mb_step(ts, ix):
                b = jax.tree.map(lambda x: x[ix], flat)
                a_, t_, w_ = adv_n[ix], tgt[ix], w[ix]

                def loss_fn(p):
                    logits, v = net.apply(p, b.obs)
                    logits = jnp.where(b.mask, logits, NEG)
                    d = distrax.Categorical(logits=logits)
                    lp = d.log_prob(b.action)
                    ratio = jnp.exp(lp - b.log_prob)
                    l1 = ratio * a_
                    l2 = jnp.clip(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * a_
                    obj = jnp.minimum(l1, l2)
                    # dual-clip(Ye et al. 2019, arXiv:1912.09729):A<0 且 ratio 远大于 1 时,
                    # 标准 min() 会选中**未裁剪**项 l1 —— 它随 ratio 线性变大、梯度无界,
                    # 表现为"对被判为坏的动作施加极端压制",是熵塌缩的直接机械来源。
                    # 本项目实测:1.52M 小网 max_ratio 全程 ≤8.3,5.78M 大网冲到 49.89
                    # (max_kl 10.0),而 clip_frac/adv_osc 等均值指标照旧全绿 —— 容量越大
                    # 这条尾巴越危险。dual_c<=0 关闭该项,行为与原版逐字相同。
                    if args.dual_c > 0.0:
                        obj = jnp.where(a_ < 0, jnp.maximum(obj, args.dual_c * a_), obj)
                    pg = -(obj * w_).sum() / jnp.maximum(w_.sum(), 1)
                    vl = ((v - t_) ** 2 * w_).sum() / jnp.maximum(w_.sum(), 1)
                    ent = (d.entropy() * w_).sum() / jnp.maximum(w_.sum(), 1)
                    loss = pg + args.vf_coef * vl - args.ent_coef * ent
                    clipped = (jnp.abs(ratio - 1.0) > args.clip_eps) & (w_ > 0)
                    kl = ((b.log_prob - lp) * w_)
                    aux = (vl, ent, clipped.sum() / jnp.maximum(w_.sum(), 1),
                           jnp.max(jnp.where(w_ > 0, ratio, 0.0)),
                           jnp.max(jnp.abs(kl)))
                    return loss, aux

                (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
                return ts.apply_gradients(grads=g), (loss,) + aux

            mbs = perm.reshape(NUM_MB, args.minibatch_size)
            ts, out = lax.scan(mb_step, ts, mbs)
            return (ts, rng), out

        (ts, _), out = lax.scan(epoch, (ts, key), None, length=args.update_epochs)
        m = jax.tree.map(lambda x: x.mean(), out)
        mx = jax.tree.map(lambda x: x.max(), out)
        return ts, (m[0], m[1], m[2], m[3], mx[4], mx[5], adv_std)

    return update


def main():
    net = SichuanACNet(channels=args.channels, blocks=args.blocks)
    key = jax.random.PRNGKey(args.seed)
    key, k0 = jax.random.split(key)
    st0 = jax.vmap(ENV.init)(jax.random.split(k0, args.num_envs))
    sample = jax.tree.map(lambda x: x[:2], jax.vmap(_OBSERVE)(st0))
    params = net.init(jax.random.PRNGKey(1), sample)
    if args.init_from:
        with open(args.init_from, "rb") as f:
            params = pickle.load(f)
        print(f"init from {args.init_from} (已跑 {args.steps_done:,} 步)",
              file=sys.stderr, flush=True)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"params={n_params/1e6:.2f}M  batch={BATCH:,}  updates={NUM_UPDATES:,}",
          file=sys.stderr, flush=True)

    tx = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.lr))
    ts = TrainState.create(apply_fn=net.apply, params=params, tx=tx)
    update = make_update(net, tx)

    @jax.jit
    def block(ts, env_state, key):
        def one(carry, _):
            ts, es, rng = carry
            rng, k1, k2 = jax.random.split(rng, 3)
            es, traj = rollout(ts.params, net, es, k1)
            adv, tgt, valid = compute_adv(traj)
            ts, metrics = update(ts, traj, adv, tgt, valid, k2)
            return (ts, es, rng), metrics
        (ts, env_state, key), ms = lax.scan(
            one, (ts, env_state, key), None, length=args.updates_per_jit)
        return ts, env_state, key, jax.tree.map(lambda x: x[-1], ms)

    env_state, steps, t0 = st0, args.steps_done, time.time()
    prev_adv_std = None
    save = Path(args.save_path)
    save.parent.mkdir(parents=True, exist_ok=True)
    for b in range(NUM_JIT):
        ts, env_state, key, m = block(ts, env_state, key)
        steps += BATCH * args.updates_per_jit
        if (b + 1) % args.log_every_blocks == 0 or b == 0:
            loss, vl, ent, clipf, mratio, mkl, astd = [float(x) for x in m]
            osc = 0.0 if prev_adv_std is None else abs(
                __import__("math").log(max(astd, 1e-8) / max(prev_adv_std, 1e-8)))
            prev_adv_std = astd
            sps = (steps - args.steps_done) / (time.time() - t0)
            print(f"[block {b+1}/{NUM_JIT}] steps={steps:,} sps={sps:,.0f} "
                  f"loss={loss:.4f} vloss={vl:.4f} ent={ent:.3f} "
                  f"clip_frac={clipf:.3f} adv_osc={osc:.3f} "
                  f"max_ratio={mratio:.2f} max_kl={mkl:.3f}", flush=True)
            if clipf > 0.3:
                print("  ⚠️ clip_frac>0.3:步子过大,信任域形同虚设", flush=True)
            if osc > 0.5:
                print("  ⚠️ adv_osc>0.5:优势尺度震荡,价值函数没跟上", flush=True)
        if (b + 1) % args.snapshot_every_blocks == 0:
            with open(f"{save}.b{b+1}.pkl", "wb") as f:
                pickle.dump(jax.device_get(ts.params), f)
        with open(save, "wb") as f:
            pickle.dump(jax.device_get(ts.params), f)


if __name__ == "__main__":
    main()
