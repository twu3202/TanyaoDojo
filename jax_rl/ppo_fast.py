"""
ppo_fast —— R2.5:Mahjax PPO 训练循环的吞吐改造版(算法与 examples/ppo_with_reg.py 严格一致)。

三项工程改动(仅性能,不改算法语义):
 1. 加宽减深:环境步延迟对 batch 平坦(实测) → 默认 num_envs 大、num_steps 小,
    同样样本量的 rollout 墙钟时间下降数倍。
 2. 大 jit:把 K 个 update(rollout+GAE+update)整体放进一个 lax.scan,设备端累积指标,
    每 K 个 update 才回主机一次 —— 消灭逐 update 的 Python 分发与 float() 强制同步。
 3. obs/网络可选:observe_type 2D+CNN(吞吐基线)或 dict+transformer(与上游默认一致)。

用法(与上游同为 OmegaConf CLI):
  python ppo_fast.py num_envs=8192 num_steps=32 updates_per_jit=8 total_timesteps=3e7
依赖:mahjax(pip install -e), examples 目录在 PYTHONPATH(复用其网络定义)。
"""
import sys
import time
import os
import pickle
from functools import partial
from typing import Dict, Literal, NamedTuple, Optional

import jax
from jax import lax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from omegaconf import OmegaConf
from pydantic import BaseModel
import distrax

import mahjax
from mahjax.wrappers.auto_reset_wrapper import auto_reset

NEG = -1e9
MAX_REWARD = 320.0


class Args(BaseModel):
    env_name: str = "no_red_mahjong"
    round_mode: Literal["single", "east", "half"] = "single"
    observe_type: Literal["dict", "2D"] = "dict"   # dict 与上游示例网络保证兼容;2D+CNN 需配套网络
    seed: int = 0
    # 尺寸:加宽减深
    num_envs: int = 8192
    num_steps: int = 32
    total_timesteps: float = 3e7
    updates_per_jit: int = 8          # 大 jit:每个 jit 调用内 scan 多少个 update
    update_epochs: int = 4
    minibatch_size: int = 4096
    # PPO 超参(与上游一致)
    gamma: float = 1.0
    gae_lambda: float = 0.95
    lr: float = 3e-4
    ent_coef: float = 0.01
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    # magnet
    mag_coef: float = 0.0             # 轻量验证默认关;正式训练开 0.2 并给 pretrained 路径
    mag_divergence_type: Literal["kl", "l2"] = "kl"
    pretrained_model_path: Optional[str] = None
    # 输出
    save_model: bool = False
    save_path: str = "ppo_fast_params.pkl"
    mem_fraction: float = 0.75


args = Args(**OmegaConf.to_object(OmegaConf.from_cli()))
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(args.mem_fraction))
print(args, file=sys.stderr)

ENV = mahjax.make(args.env_name, round_mode=args.round_mode, observe_type=args.observe_type)
STEP_FN = auto_reset(ENV.step, ENV.init)
NUM_PLAYERS = ENV.num_players
BATCH_SIZE = args.num_envs * args.num_steps
assert BATCH_SIZE % args.minibatch_size == 0, "minibatch_size 必须整除 num_envs*num_steps"
NUM_MINIBATCHES = BATCH_SIZE // args.minibatch_size
NUM_UPDATES = int(args.total_timesteps // BATCH_SIZE)
NUM_JIT_CALLS = max(1, NUM_UPDATES // args.updates_per_jit)


def get_network():
    # 复用上游 examples 的网络定义(Apache 2.0);examples 需在 PYTHONPATH
    from common import get_network_cls
    return get_network_cls(args.env_name)()


class Transition(NamedTuple):
    is_new_episode: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    observation: Dict[str, jnp.ndarray]
    action_mask: jnp.ndarray
    current_player: jnp.ndarray


def masked_mean(x, mask):
    m = mask.astype(jnp.float32)
    return (x * m).sum() / jnp.maximum(m.sum(), 1.0)


def make_train(network: nn.Module, magnet_params):
    use_magnet = args.mag_coef > 0.0

    # ---------- rollout:scan(vmap(step)) ----------
    def rollout(params, env_state, key):
        def one_step(carry, _):
            state, rng = carry
            rng, k_act, k_env = jax.random.split(rng, 3)
            obs = jax.vmap(ENV.observe)(state)
            mask = state.legal_action_mask.astype(jnp.bool_)
            cur = jnp.asarray(state.current_player, dtype=jnp.int32)
            done = jnp.asarray(state.terminated | state.truncated, dtype=jnp.bool_)
            logits, value = network.apply(params, obs)
            logits = jnp.where(mask, logits, NEG)
            dist = distrax.Categorical(logits=logits)
            action, log_prob = dist.sample_and_log_prob(seed=k_act)
            next_state = jax.vmap(STEP_FN)(state, action, jax.random.split(k_env, args.num_envs))
            reward = jnp.asarray(next_state.rewards, dtype=jnp.float32) / MAX_REWARD
            t = Transition(done, action, jnp.squeeze(value, -1) if value.ndim > 1 else value,
                           reward, log_prob, obs, mask, cur)
            return (next_state, rng), t
        (env_state, _), traj = lax.scan(one_step, (env_state, key), None, length=args.num_steps)
        # traj: (T, B, ...) -> (B, T, ...) 与上游 GAE 的 per-env scan 对齐
        traj = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj)
        return env_state, traj

    # ---------- GAE(逐环境反向 scan,逻辑照抄上游) ----------
    def calculate_gae(traj: Transition):
        def single_env(tr: Transition):
            def scan_fn(carry, t):
                gae, next_value, racc, has_nv, is_new, next_valid = carry
                player, reward, value, done = t.current_player, t.reward, t.value, t.is_new_episode
                gae = jnp.where(done, 0, gae)
                racc = jnp.where(done, 0, racc)
                has_nv = jnp.where(done, False, has_nv)
                next_value = jnp.where(done, 0, next_value)
                racc = racc + reward
                pr = racc[player]
                racc = racc.at[player].set(0.0)
                td = pr + args.gamma * next_value[player] - value
                new_gae = td + args.gamma * args.gae_lambda * gae[player]
                gae = gae.at[player].set(new_gae)
                is_valid = has_nv[player] | done | next_valid[player]
                adv = jnp.where(is_valid, new_gae, 0.0)
                tgt = jnp.where(is_valid, adv + value, value)
                new_carry = (gae, next_value.at[player].set(value), racc,
                             has_nv.at[player].set(True), done, next_valid.at[player].set(is_valid) | done)
                out = (jnp.zeros(NUM_PLAYERS).at[player].set(adv),
                       jnp.zeros(NUM_PLAYERS).at[player].set(tgt),
                       jnp.zeros(NUM_PLAYERS, dtype=bool).at[player].set(is_valid))
                return new_carry, out
            init = (jnp.zeros(NUM_PLAYERS), jnp.zeros(NUM_PLAYERS), jnp.zeros(NUM_PLAYERS),
                    jnp.zeros(NUM_PLAYERS, dtype=bool), False, jnp.zeros(NUM_PLAYERS, dtype=bool))
            _, (adv, tgt, vm) = lax.scan(scan_fn, init, tr, reverse=True)
            return adv, tgt, vm
        return jax.vmap(single_env)(traj)

    def process(traj: Transition):
        adv, tgt, vm = calculate_gae(traj)
        flat = jax.tree.map(lambda x: x.reshape((BATCH_SIZE,) + x.shape[2:]), traj)
        adv = adv.reshape((BATCH_SIZE, NUM_PLAYERS))
        tgt = tgt.reshape((BATCH_SIZE, NUM_PLAYERS))
        vm = vm.reshape((BATCH_SIZE, NUM_PLAYERS))
        mu = masked_mean(adv, vm)
        std = jnp.sqrt(masked_mean((adv - mu) ** 2, vm))
        adv = (adv - mu) / (std + 1e-8)
        return flat, adv, tgt, vm

    # ---------- update(逻辑照抄上游) ----------
    def update(train_state, key, batch):
        flat, adv, tgt, vm = batch

        def epoch(carry, _):
            ts, rng = carry
            rng, pk = jax.random.split(rng)
            perm = jax.random.permutation(pk, BATCH_SIZE)
            sh = (jax.tree.map(lambda x: x[perm], flat), adv[perm], tgt[perm], vm[perm])
            mbs = jax.tree.map(lambda x: x.reshape((NUM_MINIBATCHES, args.minibatch_size) + x.shape[1:]), sh)

            def mb_step(ts_in, mb):
                tr, a_mb, t_mb, m_mb = mb

                def loss_fn(params):
                    logits, values = network.apply(params, tr.observation)
                    logits = jnp.where(tr.action_mask, logits, NEG)
                    dists = distrax.Categorical(logits=logits)
                    log_ratio = dists.log_prob(tr.action) - tr.log_prob
                    ratio = jnp.exp(log_ratio)[..., None]
                    ppo_loss = -masked_mean(jnp.minimum(ratio * a_mb,
                                jnp.clip(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * a_mb), m_mb)
                    mag_kl = 0.0
                    if use_magnet:
                        mlogits, _ = network.apply(magnet_params, tr.observation)
                        mdists = distrax.Categorical(logits=jnp.where(tr.action_mask, mlogits, NEG))
                        if args.mag_divergence_type == "kl":
                            vals = dists.kl_divergence(mdists)
                        else:
                            vals = 0.5 * jnp.sum((dists.probs - mdists.probs) ** 2, axis=-1)
                        mag_kl = masked_mean(vals[..., None], m_mb)
                    entropy = masked_mean(dists.entropy()[..., None], m_mb)
                    loss_actor = ppo_loss - args.ent_coef * entropy + args.mag_coef * mag_kl
                    v = values[..., None]
                    v_old = tr.value[..., None]
                    v_clip = v_old + jnp.clip(v - v_old, -args.clip_eps, args.clip_eps)
                    loss_critic = 0.5 * args.vf_coef * masked_mean(
                        jnp.maximum((v - t_mb) ** 2, (v_clip - t_mb) ** 2), m_mb)
                    return loss_actor + loss_critic, {
                        "loss": loss_actor + loss_critic, "entropy": entropy, "mag_kl": mag_kl,
                        "approx_kl": masked_mean((ratio - 1.0) - log_ratio[..., None], m_mb),
                    }
                grads, metrics = jax.grad(loss_fn, has_aux=True)(ts_in.params)
                return ts_in.apply_gradients(grads=grads), metrics
            ts, metrics = lax.scan(mb_step, ts, mbs)
            return (ts, rng), jax.tree.map(jnp.mean, metrics)
        (train_state, _), metrics = lax.scan(epoch, (train_state, key), None, length=args.update_epochs)
        return train_state, jax.tree.map(jnp.mean, metrics)

    # ---------- 大 jit:一次 scan 跑 updates_per_jit 个 update ----------
    # 注:曾用 donate_argnums=(0,1),但 vmap(init) 的输出存在 XLA 缓冲别名(相同常量共享 buffer),
    # 触发 "donate the same buffer twice"。捐赠收益(省一次拷贝)不值得别名风险,弃用。
    @jax.jit
    def train_block(train_state, env_state, key):
        def one_update(carry, _):
            ts, es, rng = carry
            rng, k_roll, k_upd = jax.random.split(rng, 3)
            es, traj = rollout(ts.params, es, k_roll)
            batch = process(traj)
            ts, metrics = update(ts, k_upd, batch)
            extra = {"avg_reward": jnp.mean(traj.reward), "eps_rate": jnp.mean(traj.is_new_episode.astype(jnp.float32))}
            return (ts, es, rng), {**metrics, **extra}
        (train_state, env_state, key), metrics = lax.scan(one_update, (train_state, env_state, key), None,
                                                          length=args.updates_per_jit)
        return train_state, env_state, key, jax.tree.map(jnp.mean, metrics)

    return train_block


def main():
    rng = jax.random.PRNGKey(args.seed)
    rng, k_net, k_reset = jax.random.split(rng, 3)
    network = get_network()
    dummy_obs = jax.vmap(ENV.observe)(jax.vmap(ENV.init)(jax.random.split(jax.random.PRNGKey(0), 2)))
    params = network.init(k_net, dummy_obs)
    magnet_params = params
    if args.pretrained_model_path:
        with open(args.pretrained_model_path, "rb") as f:
            loaded = pickle.load(f)
        params = loaded if isinstance(loaded, dict) else {"params": loaded}
        magnet_params = params
        print(f"loaded pretrained/magnet from {args.pretrained_model_path}", file=sys.stderr)
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=optax.adamw(args.lr, eps=1e-5))
    env_state = jax.vmap(ENV.init)(jax.random.split(k_reset, args.num_envs))

    train_block = make_train(network, magnet_params)

    steps = 0
    t_compile0 = time.time()
    train_state, env_state, rng, metrics = train_block(train_state, env_state, rng)
    jax.block_until_ready(metrics)
    t_compile = time.time() - t_compile0
    steps += BATCH_SIZE * args.updates_per_jit
    print(f"[compile+first block] {t_compile:.1f}s  metrics={jax.tree.map(float, metrics)}", flush=True)

    t0 = time.time()
    for i in range(1, NUM_JIT_CALLS):
        train_state, env_state, rng, metrics = train_block(train_state, env_state, rng)
        steps += BATCH_SIZE * args.updates_per_jit
        if i % max(1, NUM_JIT_CALLS // 8) == 0:
            m = jax.tree.map(float, metrics)
            el = time.time() - t0
            print(f"[block {i}/{NUM_JIT_CALLS-1}] steps={steps:,} sps={(steps - BATCH_SIZE*args.updates_per_jit)/el:,.0f} "
                  f"loss={m['loss']:.4f} ent={m['entropy']:.3f} kl={m['approx_kl']:.5f} r={m['avg_reward']:.5f}", flush=True)
    jax.block_until_ready(train_state.params)
    dt = time.time() - t0
    steady = (steps - BATCH_SIZE * args.updates_per_jit) / dt if NUM_JIT_CALLS > 1 else 0
    print(f"STEADY_SPS={steady:,.0f}  total_steps={steps:,}  wall={dt:.1f}s  compile={t_compile:.1f}s", flush=True)
    if args.save_model:
        with open(args.save_path, "wb") as f:
            pickle.dump(train_state.params, f)


if __name__ == "__main__":
    main()
