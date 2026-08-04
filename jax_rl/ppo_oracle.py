"""
Oracle-critic League PPO:非对称 actor-critic(fork 自 ppo_league.py)。

动机(MAHJAX_MIGRATION.md 负结果三连):自家族 league 在 10-30 亿步无外部增益。
病根候选之一是 critic 方差——合法观测下终局点数几乎不可预测,优势信号被噪声淹没。
Suphx oracle guiding 的教训:训练期给 critic 喂全信息(四家手牌/牌墙/里宝),
值估计方差骤降;actor 仍只看合法观测,eval 时 critic 整套丢弃,不引入任何作弊。

与 ppo_league 的差异(其余逐字一致):
  1) 观测:rollout 只算/只存 observe_oracle 的 37 平面/29 标量;actor 与 magnet
     前向取切片 [:20]/[:26](= observe_lean,前缀逐位一致,冒烟已验),零重复计算;
  2) 独立 LeanCriticNet(默认 128x6,值函数在全信息下更好学,不必跟 actor 同宽)
     + 独立 optimizer;GAE 与值损失全部换用 oracle 值;actor 的 value 头弃用不训;
  3) critic 预热(critic_warmup_blocks):actor 损失乘 0,先让 critic 从随机初始
     拟合到位,再放开 actor——BC 基座不吃垃圾优势的梯度;
  4) 检查点:save_path 存 actor(格式与评测桥/league 池完全兼容),
     save_path+'.critic' 存 critic(仅续训用)。
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
MAX_REWARD = 320.0

LEAN_PLANES = 20
LEAN_SCALARS = 26


class Args(BaseModel):
    env_name: str = "red_mahjong"
    round_mode: str = "single"
    observe_type: Literal["lean"] = "lean"
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
    pretrained_model_path: Optional[str] = None      # 学习者 actor 初始化
    magnet_model_path: Optional[str] = None          # 单独指定锚(续训时锚仍指稳定基座)
    league_pool: str = ""                             # 逗号分隔的冻结对手 pkl
    self_pool_slots: int = 2
    snapshot_every_blocks: int = 32
    save_model: bool = True
    save_path: str = "ppo_oracle_params.pkl"
    ckpt_every_blocks: int = 16
    mem_fraction: float = 0.85
    channels: int = 128               # actor 规格(须与基座/池权重一致)
    blocks: int = 6
    critic_channels: int = 128        # oracle critic 规格(独立可调)
    critic_blocks: int = 6
    critic_lr: float = 3e-4
    critic_warmup_blocks: int = 16    # 前 N 个 jit 块只训 critic(actor 损失乘 0)
    critic_pretrained_path: Optional[str] = None     # 续训:save_path+'.critic'
    # oracle 信息退火(DCRL/NeurIPS24 与 Suphx dropout 的共识:静态全信息 critic 可能不稳,
    # 随进度把 oracle 附加通道线性缩到 0,critic 平滑过渡为普通 critic)。0=不退火。
    oracle_anneal_start_blocks: int = 0
    oracle_anneal_end_blocks: int = 0


args = Args(**OmegaConf.to_object(OmegaConf.from_cli()))
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(args.mem_fraction))
print(args, file=sys.stderr)

ENV = mahjax.make(args.env_name, round_mode=args.round_mode, observe_type="dict")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obs_oracle import observe_oracle
from net_lean import LeanACNet, LeanCriticNet

STEP_FN = auto_reset(ENV.step, ENV.init)
NUM_PLAYERS = ENV.num_players
BATCH_SIZE = args.num_envs * args.num_steps
assert BATCH_SIZE % args.minibatch_size == 0
NUM_MINIBATCHES = BATCH_SIZE // args.minibatch_size
NUM_UPDATES = int(args.total_timesteps // BATCH_SIZE)
NUM_JIT_CALLS = max(1, NUM_UPDATES // args.updates_per_jit)


def to_actor_obs(obs: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """oracle obs → 合法 obs 切片(前缀逐位 = observe_lean,永不含隐藏信息)。"""
    return {
        "planes": obs["planes"][..., :LEAN_PLANES],
        "scalars": obs["scalars"][..., :LEAN_SCALARS],
    }


class Transition(NamedTuple):
    is_new_episode: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray            # oracle critic 值
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    observation: Dict[str, jnp.ndarray]   # oracle 全量(actor 用时再切片)
    action_mask: jnp.ndarray
    current_player: jnp.ndarray
    learner_seat: jnp.ndarray


def masked_mean(x, mask):
    m = mask.astype(jnp.float32)
    return (x * m).sum() / jnp.maximum(m.sum(), 1.0)


def make_train(network: nn.Module, critic: nn.Module, magnet_params):
    use_magnet = args.mag_coef > 0.0

    def rollout(all_params, critic_params, assign, learner_seat, env_state, key, oracle_scale):
        def one_step(carry, _):
            state, rng = carry
            rng, k_act, k_env = jax.random.split(rng, 3)
            obs = jax.vmap(observe_oracle)(state)
            obs = {
                "planes": jnp.concatenate(
                    [obs["planes"][..., :LEAN_PLANES],
                     obs["planes"][..., LEAN_PLANES:] * oracle_scale], axis=-1),
                "scalars": jnp.concatenate(
                    [obs["scalars"][..., :LEAN_SCALARS],
                     obs["scalars"][..., LEAN_SCALARS:] * oracle_scale], axis=-1),
            }
            obs_a = to_actor_obs(obs)
            mask = state.legal_action_mask.astype(jnp.bool_)
            cur = jnp.asarray(state.current_player, dtype=jnp.int32)
            done = jnp.asarray(state.terminated | state.truncated, dtype=jnp.bool_)
            all_logits, _ = jax.vmap(lambda p: network.apply(p, obs_a))(all_params)
            sel = assign[jnp.arange(args.num_envs), cur]
            logits = all_logits[sel, jnp.arange(args.num_envs)]
            value = critic.apply(critic_params, obs)              # (B,) 1x 前向,非 K+1
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

    # GAE 与 ppo_league 逐字一致(值已换 oracle critic 的)
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
        seat_onehot = jax.nn.one_hot(traj.learner_seat, NUM_PLAYERS, dtype=bool)
        vm = vm & seat_onehot
        flat = jax.tree.map(lambda x: x.reshape((BATCH_SIZE,) + x.shape[2:]), traj)
        adv = adv.reshape((BATCH_SIZE, NUM_PLAYERS))
        tgt = tgt.reshape((BATCH_SIZE, NUM_PLAYERS))
        vm = vm.reshape((BATCH_SIZE, NUM_PLAYERS))
        mu = masked_mean(adv, vm)
        std = jnp.sqrt(masked_mean((adv - mu) ** 2, vm))
        adv = (adv - mu) / (std + 1e-8)
        return flat, adv, tgt, vm

    def update(ts_a, ts_c, key, batch, actor_scale):
        flat, adv, tgt, vm = batch

        def epoch(carry, _):
            tsa, tsc, rng = carry
            rng, pk = jax.random.split(rng)
            perm = jax.random.permutation(pk, BATCH_SIZE)
            sh = (jax.tree.map(lambda x: x[perm], flat), adv[perm], tgt[perm], vm[perm])
            mbs = jax.tree.map(lambda x: x.reshape((NUM_MINIBATCHES, args.minibatch_size) + x.shape[1:]), sh)

            def mb_step(carry_in, mb):
                tsa_in, tsc_in = carry_in
                tr, a_mb, t_mb, m_mb = mb
                obs_a = to_actor_obs(tr.observation)

                def actor_loss_fn(params):
                    logits, _ = network.apply(params, obs_a)
                    logits = jnp.where(tr.action_mask, logits, NEG)
                    dists = distrax.Categorical(logits=logits)
                    log_ratio = dists.log_prob(tr.action) - tr.log_prob
                    ratio = jnp.exp(log_ratio)[..., None]
                    ppo_loss = -masked_mean(jnp.minimum(ratio * a_mb,
                                jnp.clip(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * a_mb), m_mb)
                    mag_kl = 0.0
                    if use_magnet:
                        mlogits, _ = network.apply(magnet_params, obs_a)
                        mdists = distrax.Categorical(logits=jnp.where(tr.action_mask, mlogits, NEG))
                        if args.mag_divergence_type == "kl":
                            vals = dists.kl_divergence(mdists)
                        else:
                            vals = 0.5 * jnp.sum((dists.probs - mdists.probs) ** 2, axis=-1)
                        mag_kl = masked_mean(vals[..., None], m_mb)
                    entropy = masked_mean(dists.entropy()[..., None], m_mb)
                    loss_actor = ppo_loss - args.ent_coef * entropy + args.mag_coef * mag_kl
                    return loss_actor * actor_scale, {
                        "entropy": entropy, "mag_kl": mag_kl,
                        "approx_kl": masked_mean((ratio - 1.0) - log_ratio[..., None], m_mb),
                    }

                def critic_loss_fn(cparams):
                    v = critic.apply(cparams, tr.observation)[..., None]
                    v_old = tr.value[..., None]
                    v_clip = v_old + jnp.clip(v - v_old, -args.clip_eps, args.clip_eps)
                    return 0.5 * args.vf_coef * masked_mean(
                        jnp.maximum((v - t_mb) ** 2, (v_clip - t_mb) ** 2), m_mb)

                a_grads, a_metrics = jax.grad(actor_loss_fn, has_aux=True)(tsa_in.params)
                v_loss, c_grads = jax.value_and_grad(critic_loss_fn)(tsc_in.params)
                metrics = {**a_metrics, "v_loss": v_loss}
                return (tsa_in.apply_gradients(grads=a_grads),
                        tsc_in.apply_gradients(grads=c_grads)), metrics

            (tsa, tsc), metrics = lax.scan(mb_step, (tsa, tsc), mbs)
            return (tsa, tsc, rng), jax.tree.map(jnp.mean, metrics)

        (ts_a, ts_c, _), metrics = lax.scan(epoch, (ts_a, ts_c, key), None, length=args.update_epochs)
        return ts_a, ts_c, jax.tree.map(jnp.mean, metrics)

    @jax.jit
    def train_block(ts_a, ts_c, pool_params, assign, learner_seat, env_state, key,
                    actor_scale, oracle_scale):
        def one_update(carry, _):
            tsa, tsc, es, rng = carry
            rng, k_roll, k_upd = jax.random.split(rng, 3)
            all_params = jax.tree.map(lambda l, p: jnp.concatenate([l[None], p], axis=0),
                                      tsa.params, pool_params)
            es, traj = rollout(all_params, tsc.params, assign, learner_seat, es, k_roll,
                               oracle_scale)
            batch = process(traj)
            tsa, tsc, metrics = update(tsa, tsc, k_upd, batch, actor_scale)
            r_seat = jnp.take_along_axis(
                traj.reward, traj.learner_seat[..., None], axis=-1).squeeze(-1)
            n_eps = jnp.maximum(traj.is_new_episode.sum(), 1)
            learner_reward = r_seat.sum() / n_eps
            extra = {"learner_reward": learner_reward,
                     "eps_rate": jnp.mean(traj.is_new_episode.astype(jnp.float32))}
            return (tsa, tsc, es, rng), {**metrics, **extra}

        (ts_a, ts_c, env_state, key), metrics = lax.scan(
            one_update, (ts_a, ts_c, env_state, key), None, length=args.updates_per_jit)
        return ts_a, ts_c, env_state, key, jax.tree.map(jnp.mean, metrics)

    return train_block


def main():
    rng = jax.random.PRNGKey(args.seed)
    rng, k_net, k_cnet, k_reset = jax.random.split(rng, 4)
    network = LeanACNet(channels=args.channels, blocks=args.blocks)
    critic = LeanCriticNet(channels=args.critic_channels, blocks=args.critic_blocks)
    dummy_state = jax.vmap(ENV.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    dummy_obs = jax.vmap(observe_oracle)(dummy_state)
    params = network.init(k_net, to_actor_obs(dummy_obs))
    critic_params = critic.init(k_cnet, dummy_obs)
    magnet_params = params
    if args.pretrained_model_path:
        with open(args.pretrained_model_path, "rb") as f:
            params = pickle.load(f)
        magnet_params = params
        print(f"loaded pretrained from {args.pretrained_model_path}", file=sys.stderr)
    if args.magnet_model_path:
        with open(args.magnet_model_path, "rb") as f:
            magnet_params = pickle.load(f)
        print(f"magnet <- {args.magnet_model_path}", file=sys.stderr)
    if args.critic_pretrained_path:
        with open(args.critic_pretrained_path, "rb") as f:
            critic_params = pickle.load(f)
        print(f"critic <- {args.critic_pretrained_path}", file=sys.stderr)

    pool_list = []
    for p in [x for x in args.league_pool.split(",") if x.strip()]:
        with open(p.strip(), "rb") as f:
            pool_list.append(pickle.load(f))
        print(f"pool <- {p.strip()}", file=sys.stderr)
    for _ in range(args.self_pool_slots):
        pool_list.append(jax.tree.map(jnp.copy, params))
    K = len(pool_list)
    assert K >= 1, "league_pool 至少一个对手或 self_pool_slots>0"
    pool_params = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *pool_list)
    n_c = sum(x.size for x in jax.tree.leaves(critic_params))
    print(f"pool size K={K}  critic {args.critic_channels}x{args.critic_blocks} = {n_c/1e6:.2f}M", file=sys.stderr)

    ts_a = TrainState.create(apply_fn=network.apply, params=params,
                             tx=optax.adamw(args.lr, eps=1e-5))
    ts_c = TrainState.create(apply_fn=critic.apply, params=critic_params,
                             tx=optax.adamw(args.critic_lr, eps=1e-5))
    env_state = jax.vmap(ENV.init)(jax.random.split(k_reset, args.num_envs))
    train_block = make_train(network, critic, magnet_params)

    np_rng = jax.random.PRNGKey(args.seed + 7)

    def sample_assign(key):
        k1, k2 = jax.random.split(key)
        seats = jax.random.randint(k1, (args.num_envs,), 0, NUM_PLAYERS)
        opp = jax.random.randint(k2, (args.num_envs, NUM_PLAYERS), 1, K + 1)
        assign = opp.at[jnp.arange(args.num_envs), seats].set(0)
        return assign.astype(jnp.int32), seats.astype(jnp.int32)

    def scale_at(i):
        return jnp.float32(0.0 if i < args.critic_warmup_blocks else 1.0)

    def oracle_scale_at(i):
        s, e = args.oracle_anneal_start_blocks, args.oracle_anneal_end_blocks
        if e <= s or i <= s:
            return jnp.float32(1.0)
        return jnp.float32(max(0.0, 1.0 - (i - s) / (e - s)))

    def save_ckpt():
        for path, obj in ((args.save_path, ts_a.params), (args.save_path + ".critic", ts_c.params)):
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(jax.device_get(obj), f)
            os.replace(tmp, path)

    np_rng, k_a = jax.random.split(np_rng)
    assign, learner_seat = sample_assign(k_a)

    steps = 0
    snap_slot = 0
    t0c = time.time()
    ts_a, ts_c, env_state, rng, metrics = train_block(
        ts_a, ts_c, pool_params, assign, learner_seat, env_state, rng,
        scale_at(0), oracle_scale_at(0))
    jax.block_until_ready(metrics)
    t_compile = time.time() - t0c
    steps += BATCH_SIZE * args.updates_per_jit
    print(f"[compile+first block] {t_compile:.1f}s  metrics={jax.tree.map(float, metrics)}", flush=True)

    t0 = time.time()
    for i in range(1, NUM_JIT_CALLS):
        np_rng, k_a = jax.random.split(np_rng)
        assign, learner_seat = sample_assign(k_a)
        ts_a, ts_c, env_state, rng, metrics = train_block(
            ts_a, ts_c, pool_params, assign, learner_seat, env_state, rng,
            scale_at(i), oracle_scale_at(i))
        steps += BATCH_SIZE * args.updates_per_jit
        if args.self_pool_slots > 0 and i >= args.critic_warmup_blocks \
                and i % args.snapshot_every_blocks == 0:
            slot = K - args.self_pool_slots + snap_slot % args.self_pool_slots
            pool_params = jax.tree.map(
                lambda pp, lp: pp.at[slot].set(lp), pool_params, ts_a.params)
            snap_slot += 1
        if i % max(1, min(64, NUM_JIT_CALLS // 8)) == 0:
            m = jax.tree.map(float, metrics)
            el = time.time() - t0
            warm = " WARMUP" if i < args.critic_warmup_blocks else ""
            print(f"[block {i}/{NUM_JIT_CALLS-1}] steps={steps:,} "
                  f"sps={(steps - BATCH_SIZE*args.updates_per_jit)/el:,.0f} "
                  f"vloss={m['v_loss']:.5f} ent={m['entropy']:.3f} kl={m['approx_kl']:.5f} "
                  f"mag={m['mag_kl']:.4f} lr_r={m['learner_reward']:+.5f}{warm}", flush=True)
        if args.ckpt_every_blocks > 0 and i % args.ckpt_every_blocks == 0:
            save_ckpt()
    jax.block_until_ready(ts_a.params)
    dt = time.time() - t0
    steady = (steps - BATCH_SIZE * args.updates_per_jit) / dt if NUM_JIT_CALLS > 1 else 0
    print(f"STEADY_SPS={steady:,.0f}  total_steps={steps:,}  wall={dt:.1f}s  compile={t_compile:.1f}s", flush=True)
    if args.save_model:
        save_ckpt()


if __name__ == "__main__":
    main()
