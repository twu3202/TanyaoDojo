"""
R2 判据实验:同一批人类决策上,LeanACNet vs 上游 ACNet 的 BC 精度对照。

数据:make_bc_dataset.py 落盘的 npz shards(双观测同点采集)。
损失:合法掩码 CE(非法位 logits 置 -1e9)。
指标:val top-1 acc——总体 / 有选择(legal>1) / 打牌类(a<=71) / 鸣牌响应类(74-84)。
用法:PYTHONPATH=~/mahjax python bc_lean.py <data_dir> --net lean|upstream
      [--epochs 3] [--batch 512] [--lr 3e-4]
"""
from __future__ import annotations
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/r/mahjax/examples")
from net_lean import LeanACNet


def load(data_dir):
    """预分配+逐 shard 填充(concatenate 的双缓冲在 10M+ 样本时会顶爆内存)。"""
    files = sorted(glob.glob(str(Path(data_dir) / "shard_*.npz")))
    with np.load(files[0]) as z0:
        keys = list(z0.files)
        spec = {k: (z0[k].dtype, z0[k].shape[1:]) for k in keys}
    counts = []
    for fp in files:
        with np.load(fp) as z:
            counts.append(len(z["action"]))
    n = sum(counts)
    arrs = {k: np.empty((n, *shp), dtype=dt) for k, (dt, shp) in spec.items()}
    off = 0
    for fp, c in zip(files, counts):
        with np.load(fp) as z:
            for k in keys:
                arrs[k][off : off + c] = z[k]
        off += c
    return arrs


def lean_batch(d, ix):
    return {
        "planes": jnp.asarray(d["planes"][ix], jnp.float32) / 4.0,
        "scalars": jnp.asarray(d["scalars"][ix]),
    }


def dict_batch(d, ix):
    return {
        "hand": jnp.asarray(d["hand"][ix], jnp.int32),
        "last_draw": jnp.asarray(d["last_draw"][ix], jnp.int32),
        "action_history": jnp.asarray(d["action_history"][ix], jnp.int32),
        "shanten_count": jnp.asarray(d["shanten"][ix], jnp.int32),
        "furiten": jnp.asarray(d["furiten"][ix], jnp.int32),
        "scores": jnp.asarray(d["scores"][ix], jnp.int32),
        "round": jnp.asarray(d["round"][ix], jnp.int32),
        "honba": jnp.asarray(d["honba"][ix], jnp.int32),
        "kyotaku": jnp.asarray(d["kyotaku"][ix], jnp.int32),
        "prevalent_wind": jnp.asarray(d["prevalent"][ix], jnp.int32),
        "seat_wind": jnp.asarray(d["seat"][ix], jnp.int32),
        "dora_indicators": jnp.asarray(d["dora_indicators"][ix], jnp.int32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--net", choices=["lean", "upstream"], default="lean")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save", default=None, help="训练后 pickle 参数到此路径(ppo_fast 可直接加载)")
    args = ap.parse_args()

    d = load(args.data_dir)
    n = len(d["action"])
    val = d["game_id"] % 20 == 0
    tr_ix, va_ix = np.flatnonzero(~val), np.flatnonzero(val)
    print(f"samples={n} train={len(tr_ix)} val={len(va_ix)} net={args.net}")

    if args.net == "lean":
        net, make_batch = LeanACNet(), lean_batch
    else:
        from networks.red_network import ACNet
        net, make_batch = ACNet(), dict_batch

    sample = make_batch(d, tr_ix[:2])
    params = net.init(jax.random.PRNGKey(0), sample)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"params={n_params/1e6:.2f}M")
    tx = optax.adam(args.lr)
    opt_state = tx.init(params)

    def logits_of(p, obs):
        out = net.apply(p, obs)
        return out[0] if isinstance(out, tuple) else out

    @jax.jit
    def train_step(p, o, obs, labels, mask):
        def loss_fn(p_):
            logits = jnp.where(mask, logits_of(p_, obs), -1e9)
            return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, o = tx.update(grads, o, p)
        return optax.apply_updates(p, updates), o, loss

    @jax.jit
    def pred_step(p, obs, mask):
        logits = jnp.where(mask, logits_of(p, obs), -1e9)
        return jnp.argmax(logits, axis=-1)

    rng = np.random.default_rng(0)
    B = args.batch
    for ep in range(args.epochs):
        order = rng.permutation(tr_ix)
        t0, losses = time.time(), []
        for i in range(0, len(order) - B + 1, B):
            ix = order[i : i + B]
            params, opt_state, loss = train_step(
                params, opt_state, make_batch(d, ix),
                jnp.asarray(d["action"][ix], jnp.int32),
                jnp.asarray(d["legal_mask"][ix]),
            )
            losses.append(float(loss))
        # ---- val
        preds = []
        for i in range(0, len(va_ix), B):
            ix = va_ix[i : i + B]
            preds.append(np.asarray(pred_step(
                params, make_batch(d, ix), jnp.asarray(d["legal_mask"][ix]))))
        preds = np.concatenate(preds)
        labels = d["action"][va_ix]
        legal_n = d["legal_mask"][va_ix].sum(1)
        hit = preds == labels
        is_discard = labels <= 71
        is_react = (labels >= 74) & (labels <= 84)
        is_call = (labels >= 74) & (labels <= 83)  # 真实叫牌(不含 PASS)
        choice = legal_n > 1

        def acc(m):
            return hit[m].mean() if m.any() else float("nan")

        print(f"ep{ep} loss={np.mean(losses):.4f} "
              f"val_acc={hit.mean():.4f} choice={acc(choice):.4f} "
              f"discard={acc(is_discard & choice):.4f} react={acc(is_react & choice):.4f} "
              f"call={acc(is_call):.4f} ({time.time()-t0:.0f}s)", flush=True)

    if args.save:
        import pickle
        with open(args.save, "wb") as f:
            pickle.dump(jax.device_get(params), f)
        print(f"saved params -> {args.save}")


if __name__ == "__main__":
    main()
