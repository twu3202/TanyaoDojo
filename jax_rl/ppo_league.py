"""
League PPO:对手池版训练器(修"四座同策略自博弈不迁移"病根,fork 自 ppo_fast.py)。

与 ppo_fast 的差异(算法内核不变:PPO+GAE+magnet):
  1) 参数集 = [学习者] + K 个冻结对手(--league_pool 逗号分隔 pkl);
  2) 每 env 随机指定学习者座位,其余三座从池随机指派;每个 jit 块重采样;
  3) rollout 对 K+1 套参数并行前向((K+1)x 前向代价),按 [env,当前座位] 选取
     logits/value——学习者步取学习者头,GAE 的逐座位链自动只消费学习者的 V;
  4) 损失/优势归一化掩码收窄到"学习者座位的决策"(对手步完全不进梯度);
  5) 自快照:每 snapshot_every_blocks 把学习者参数写入池的轮转槽(PFSP-lite)。
指标:learner_reward = 学习者座位的终局回报均值(外部强度的在线代理,
纯自博弈中该值恒≈0,league 中>0 意味着压过池)。
"""
from __future__ import annotations

import sys
import time
import os
import pickle
from typing import Dict, Literal, NamedTuple, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import distrax
import optax
from jax import lax
from omegaconf import OmegaConf
from pydantic import BaseModel

import mahjax
from mahjax.wrappers.auto_reset_wrapper import auto_reset

NEG = -1e9
MAX_REWARD = 320.0  # 与 ppo_fast 一致:终局点数归一(千点位),critic 才在 O(1) 尺度


class Args(BaseModel):
    env_name: str = "red_mahjong"
    round_mode: str = "single"
    observe_type: Literal["dict", "2D", "lean"] = "lean"
    seed: int = 0
    num_envs: int = 8192
    num_steps: int = 32
    total_timesteps: float = 2e10
    updates_per_jit: int = 4
    update_epochs: int = 1
    minibatch_size: int = 4096
    gamma: float = 1.0
    gae_lambda: float = 0.95
    lr: float = 3e-4
    ent_coef: float = 0.01
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    mag_coef: float = 0.2
    mag_divergence_type: Literal["kl", "l2"] = "kl"
    pretrained_model_path: Optional[str] = None      # 学习者初始化 + magnet 锚
    league_pool: str = ""                             # 逗号分隔的冻结对手 pkl
    self_pool_slots: int = 2                          # 池尾追加的自快照轮转槽数
    snapshot_every_blocks: int = 32
    save_model: bool = True
    save_path: str = "ppo_league_params.pkl"
    ckpt_every_blocks: int = 16
    mem_fraction: float = 0.85
    channels: int = 128               # 与基座/池权重的网络规格一致
    blocks: int = 6


args = Args(**OmegaConf.to_object(OmegaConf.from_cli()))
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(args.mem_fraction))
print(args, file=sys.stderr)

assert args.observe_type == "lean", "league 版目前只走 lean 观测"
ENV = mahjax.make(args.env_name, round_mode=args.round_mode, observe_type="dict")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obs_lean import observe_lean
from net_lean import LeanACNet

OBSERVE_FN = observe_lean
STEP_FN = auto_reset(ENV.step, ENV.init)
NUM_PLAYERS = ENV.num_players
BATCH_SIZE = args.num_envs * args.num_steps
assert BATCH_SIZE % args.minibatch_size == 0
NUM_MINIBATCHES = BATCH_SIZE // args.minibatch_size
NUM_UPDATES = int(args.total_timesteps // BATCH_SIZE)
NUM_JIT_CALLS = max(1, NUM_UPDATES // args.updates_per_jit)


class Transition(NamedTuple):
    is_new_episode: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    observation: Dict[str, jnp.ndarray]
    action_mask: jnp.ndarray
    current_player: jnp.ndarray
    learner_seat: jnp.ndarray


def masked_mean(x, mask):
    m = mask.astype(jnp.float32)
    return (x * m).sum() / jnp.maximum(m.sum(), 1.0)


def make_train(network: nn.Module, magnet_params):
    use_magnet = args.mag_coef > 0.0

    def rollout(all_params, assign, learner_seat, env_state, key):
        """all_params: 叶前维 K+1(0=学习者);assign: (B,4)∈[0,K+1) 座位→参数集。"""

        def one_step(carry, _):
            state, rng = carry
            rng, k_act, k_env = jax.random.split(rng, 3)
            obs = jax.vmap(OBSERVE_FN)(state)
            mask = state.legal_action_mask.astype(jnp.bool_)
            cur = jnp.asarray(state.current_player, dtype=jnp.int32)
            done = jnp.asarray(state.terminated | state.truncated, dtype=jnp.bool_)
            all_logits, all_values = jax.vmap(lambda p: network.apply(p, obs))(all_params)
            sel = assign[jnp.arange(args.num_envs), cur]              # (B,)
            logits = all_logits[sel, jnp.arange(args.num_envs)]       # (B, A)
            value = all_values[sel, jnp.arange(args.num_envs)]        # (B,)
            logits = jnp.where(mask, logits, NEG)
            dist = distrax.Categorical(logits=logits)
            action, log_prob = dist.sample_and_log_prob(seed=k_act)
            next_state = jax.vmap(STEP_FN)(state, action, jax.random.split(k_env, args.num_envs))
            reward = jnp.asarray(next_state.rewards, dtype=jnp.float32) / MAX_REWARD
            t = Transition(done, action, value, reward, log_prob, obs, mask, cur, learner_seat)
            return (next_state, rng), t

        (env_state, _), traj = lax.scan(one_step, (env_state, key), None, length=args.num_steps)
        traj = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj)
        return env_state, traj

    # GAE 与 ppo_fast 逐字一致:逐座位链;对手链的输出靠掩码丢弃
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
        # league 掩码:只保留"学习者座位做的决策"
        seat_onehot = jax.nn.one_hot(traj.learner_seat, NUM_PLAYERS, dtype=bool)  # (B,T,4)
        vm = vm & seat_onehot
        flat = jax.tree.map(lambda x: x.reshape((BATCH_SIZE,) + x.shape[2:]), traj)
        adv = adv.reshape((BATCH_SIZE, NUM_PLAYERS))
        tgt = tgt.reshape((BATCH_SIZE, NUM_PLAYERS))
        vm = vm.reshape((BATCH_SIZE, NUM_PLAYERS))
        mu = masked_mean(adv, vm)
        std = jnp.sqrt(masked_mean((adv - mu) ** 2, vm))
        adv = (adv - mu) / (std + 1e-8)
        return flat, adv, tgt, vm

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

    @jax.jit
    def train_block(train_state, pool_params, assign, learner_seat, env_state, key):
        def one_update(carry, _):
            ts, es, rng = carry
            rng, k_roll, k_upd = jax.random.split(rng, 3)
            all_params = jax.tree.map(lambda l, p: jnp.concatenate([l[None], p], axis=0),
                                      ts.params, pool_params)
            es, traj = rollout(all_params, assign, learner_seat, es, k_roll)
            batch = process(traj)
            ts, metrics = update(ts, k_upd, batch)
            # 终局奖励落在 done 前一步:直接对学习者座位的全程奖励求和,按局数归一
            r_seat = jnp.take_along_axis(
                traj.reward, traj.learner_seat[..., None], axis=-1).squeeze(-1)  # (B,T)
            n_eps = jnp.maximum(traj.is_new_episode.sum(), 1)
            learner_reward = r_seat.sum() / n_eps
            extra = {"learner_reward": learner_reward,
                     "eps_rate": jnp.mean(traj.is_new_episode.astype(jnp.float32))}
            return (ts, es, rng), {**metrics, **extra}

        (train_state, env_state, key), metrics = lax.scan(
            one_update, (train_state, env_state, key), None, length=args.updates_per_jit)
        return train_state, env_state, key, jax.tree.map(jnp.mean, metrics)

    return train_block


def main():
    rng = jax.random.PRNGKey(args.seed)
    rng, k_net, k_reset = jax.random.split(rng, 3)
    network = LeanACNet(channels=args.channels, blocks=args.blocks)
    dummy_obs = jax.vmap(OBSERVE_FN)(jax.vmap(ENV.init)(jax.random.split(jax.random.PRNGKey(0), 2)))
    params = network.init(k_net, dummy_obs)
    magnet_params = params
    if args.pretrained_model_path:
        with open(args.pretrained_model_path, "rb") as f:
            params = pickle.load(f)
        magnet_params = params
        print(f"loaded pretrained/magnet from {args.pretrained_model_path}", file=sys.stderr)

    pool_list = []
    for p in [x for x in args.league_pool.split(",") if x.strip()]:
        with open(p.strip(), "rb") as f:
            pool_list.append(pickle.load(f))
        print(f"pool <- {p.strip()}", file=sys.stderr)
    for _ in range(args.self_pool_slots):
        pool_list.append(jax.tree.map(jnp.copy, params))  # 自快照槽,初始=学习者
    K = len(pool_list)
    assert K >= 1, "league_pool 至少一个对手或 self_pool_slots>0"
    pool_params = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *pool_list)
    print(f"pool size K={K}", file=sys.stderr)

    train_state = TrainState.create(apply_fn=network.apply, params=params,
                                    tx=optax.adamw(args.lr, eps=1e-5))
    env_state = jax.vmap(ENV.init)(jax.random.split(k_reset, args.num_envs))
    train_block = make_train(network, magnet_params)

    np_rng = jax.random.PRNGKey(args.seed + 7)

    def sample_assign(key):
        k1, k2 = jax.random.split(key)
        seats = jax.random.randint(k1, (args.num_envs,), 0, NUM_PLAYERS)
        opp = jax.random.randint(k2, (args.num_envs, NUM_PLAYERS), 1, K + 1)
        assign = opp.at[jnp.arange(args.num_envs), seats].set(0)
        return assign.astype(jnp.int32), seats.astype(jnp.int32)

    np_rng, k_a = jax.random.split(np_rng)
    assign, learner_seat = sample_assign(k_a)

    steps = 0
    snap_slot = 0
    t0c = time.time()
    train_state, env_state, rng, metrics = train_block(
        train_state, pool_params, assign, learner_seat, env_state, rng)
    jax.block_until_ready(metrics)
    t_compile = time.time() - t0c
    steps += BATCH_SIZE * args.updates_per_jit
    print(f"[compile+first block] {t_compile:.1f}s  metrics={jax.tree.map(float, metrics)}", flush=True)

    t0 = time.time()
    for i in range(1, NUM_JIT_CALLS):
        np_rng, k_a = jax.random.split(np_rng)
        assign, learner_seat = sample_assign(k_a)
        train_state, env_state, rng, metrics = train_block(
            train_state, pool_params, assign, learner_seat, env_state, rng)
        steps += BATCH_SIZE * args.updates_per_jit
        if args.self_pool_slots > 0 and i % args.snapshot_every_blocks == 0:
            slot = K - args.self_pool_slots + snap_slot % args.self_pool_slots
            pool_params = jax.tree.map(
                lambda pp, lp: pp.at[slot].set(lp), pool_params, train_state.params)
            snap_slot += 1
        if i % max(1, min(64, NUM_JIT_CALLS // 8)) == 0:
            m = jax.tree.map(float, metrics)
            el = time.time() - t0
            print(f"[block {i}/{NUM_JIT_CALLS-1}] steps={steps:,} "
                  f"sps={(steps - BATCH_SIZE*args.updates_per_jit)/el:,.0f} "
                  f"loss={m['loss']:.4f} ent={m['entropy']:.3f} kl={m['approx_kl']:.5f} "
                  f"mag={m['mag_kl']:.4f} lr_r={m['learner_reward']:+.5f}", flush=True)
        if args.ckpt_every_blocks > 0 and i % args.ckpt_every_blocks == 0:
            tmp = args.save_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(jax.device_get(train_state.params), f)
            os.replace(tmp, args.save_path)
    jax.block_until_ready(train_state.params)
    dt = time.time() - t0
    steady = (steps - BATCH_SIZE * args.updates_per_jit) / dt if NUM_JIT_CALLS > 1 else 0
    print(f"STEADY_SPS={steady:,.0f}  total_steps={steps:,}  wall={dt:.1f}s  compile={t_compile:.1f}s", flush=True)
    if args.save_model:
        with open(args.save_path, "wb") as f:
            pickle.dump(jax.device_get(train_state.params), f)


if __name__ == "__main__":
    main()
