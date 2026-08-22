"""
流式 BC 训练器:样本量超内存时按 shard 组轮转训练(全量数据 BC 用)。

与 bc_lean.py 同网络/同指标,差别仅在数据通路:
  - 每 epoch 以随机顺序遍历 shard 文件,一次载入 G 个(组内全乱序);
  - 划分:game_id % 50 == 0 的对局全局进 val(每 shard 载入时过滤,训练不见);
  - val 集取自前 --val-shards 个文件的 val 部分(封顶,避免巨型 val);
  - 优化器状态跨组持续;每 epoch 末落盘参数。
用法:python bc_stream.py <data_dir> [--epochs 2] [--batch 1024] [--group 16]
      [--save out.pkl] [--channels 128] [--blocks 6] [--snap-every-groups N]
      [--weight-mode none|rank|delta] [--weight-beta 8000]

advantage-filtered BC(--weight-mode):人类牌谱的决策质量参差,按"该局得失点"或
"终局名次"给样本加权,把梯度往赢家的决策上倾斜(需 v2adv 模式构建的数据集)。

选峰铁律(2026-08-04 实测:宽网欠训 -1.7pt、大网过训 -2.3pt,外部强度对训练量
单峰且 val ±0.2% 可对应外部 ±2pt)→ --snap-every-groups N 每 N 个 shard 组落一个
带序号快照({save}.gK.pkl)并打 val,供逐档 4k 跟评选峰,不能只留 epoch 末档。
"""
from __future__ import annotations
import argparse
import glob
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax

sys.path.insert(0, str(Path(__file__).resolve().parent))
from net_lean import LeanACNet

VAL_MOD = 50
RANK_W = np.array([1.0, 0.8, 0.5, 0.2], np.float32)   # 终局 1/2/3/4 位的克隆权重


def sample_weights(mode, d, m, beta):
    """advantage-filtered BC 权重:none=全 1;rank=按终局名次;delta=exp(该局得失/β)。"""
    n = int(m.sum())
    if mode == "none":
        return np.ones(n, np.float32)
    if mode == "rank":
        if "rank" not in d:
            raise SystemExit("数据集缺 rank 列,请用 make_bc_dataset 的 v2adv 模式重建")
        return RANK_W[np.clip(d["rank"][m], 0, 3)]
    if "delta" not in d:
        raise SystemExit("数据集缺 delta 列,请用 make_bc_dataset 的 v2adv 模式重建")
    return np.clip(np.exp(d["delta"][m].astype(np.float32) / beta), 0.1, 3.0)


def shard_files(data_dir):
    fs = sorted(glob.glob(str(Path(data_dir) / "**" / "shard_*.npz"), recursive=True))
    if not fs:
        raise SystemExit(f"no shards under {data_dir}")
    return fs


def load_shard(fp, extra=()):
    keys = ("planes", "scalars", "action", "legal_mask", "game_id") + tuple(extra)
    with np.load(fp) as z:
        return {k: z[k] for k in keys if k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--group", type=int, default=16, help="每组载入的 shard 数")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-shards", type=int, default=6)
    ap.add_argument("--save", default=None)
    ap.add_argument("--init", default=None, help="从已有参数 pickle 续训")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--weight-mode", default="none", choices=("none", "rank", "delta"),
                    help="advantage-filtered BC:按终局名次/该局得失点给样本加权")
    ap.add_argument("--weight-beta", type=float, default=8000.0,
                    help="delta 模式的温度(点数尺度);w = exp(delta/beta),再截到 [0.1, 3]")
    ap.add_argument("--snap-every-groups", type=int, default=0,
                    help="每 N 个 shard 组存带序号快照并打 val(0=关)")
    args = ap.parse_args()

    files = shard_files(args.data_dir)
    print(f"shards={len(files)}")

    # ---- val 集(固定)
    val = {k: [] for k in ("planes", "scalars", "action", "legal_mask")}
    for fp in files[: args.val_shards]:
        d = load_shard(fp)
        m = d["game_id"] % VAL_MOD == 0
        for k in val:
            val[k].append(d[k][m])
    val = {k: np.concatenate(v) for k, v in val.items()}
    # 量化尺度按通道数识别:lean 20ch=×4;v2 36ch=×24(构建侧同约定)
    plane_scale = 24.0 if val["planes"].shape[-1] >= 36 else 4.0
    print(f"val={len(val['action'])} planes_ch={val['planes'].shape[-1]} scale={plane_scale} "
          f"weight_mode={args.weight_mode}")

    net = LeanACNet(channels=args.channels, blocks=args.blocks)
    sample = {
        "planes": jnp.asarray(val["planes"][:2], jnp.float32) / plane_scale,
        "scalars": jnp.asarray(val["scalars"][:2]),
    }
    params = net.init(jax.random.PRNGKey(0), sample)
    if args.init:
        with open(args.init, "rb") as f:
            params = pickle.load(f)
        print(f"init from {args.init}")
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"params={n_params/1e6:.2f}M")
    tx = optax.adam(args.lr)
    opt_state = tx.init(params)

    def logits_of(p, obs):
        out = net.apply(p, obs)
        return out[0] if isinstance(out, tuple) else out

    @jax.jit
    def train_step(p, o, planes, scalars, labels, mask, w):
        obs = {"planes": planes.astype(jnp.float32) / plane_scale, "scalars": scalars}

        def loss_fn(p_):
            logits = jnp.where(mask, logits_of(p_, obs), -1e9)
            ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
            return (ce * w).sum() / jnp.maximum(w.sum(), 1e-6)

        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, o = tx.update(grads, o, p)
        return optax.apply_updates(p, updates), o, loss

    @jax.jit
    def pred_step(p, planes, scalars, mask):
        obs = {"planes": planes.astype(jnp.float32) / plane_scale, "scalars": scalars}
        return jnp.argmax(jnp.where(mask, logits_of(p, obs), -1e9), axis=-1)

    def evaluate():
        B = args.batch
        preds = []
        for i in range(0, len(val["action"]), B):
            preds.append(np.asarray(pred_step(
                params, jnp.asarray(val["planes"][i:i+B]),
                jnp.asarray(val["scalars"][i:i+B]),
                jnp.asarray(val["legal_mask"][i:i+B]))))
        preds = np.concatenate(preds)
        labels = val["action"]
        hit = preds == labels
        choice = val["legal_mask"].sum(1) > 1
        disc = (labels <= 71) & choice
        call = (labels >= 74) & (labels <= 83)
        def acc(m): return hit[m].mean() if m.any() else float("nan")
        return hit.mean(), acc(choice), acc(disc), acc(call)

    rng = np.random.default_rng(0)
    B = args.batch
    step = 0
    g_count = 0
    for ep in range(args.epochs):
        order = rng.permutation(len(files))
        t0, losses = time.time(), []
        for gi in range(0, len(order), args.group):
            grp = [files[j] for j in order[gi : gi + args.group]]
            extra = () if args.weight_mode == "none" else ("delta", "rank")
            ds = [load_shard(fp, extra) for fp in grp]
            tr_mask = [d["game_id"] % VAL_MOD != 0 for d in ds]
            weights = np.concatenate([sample_weights(args.weight_mode, d, m, args.weight_beta)
                                      for d, m in zip(ds, tr_mask)])
            planes = np.concatenate([d["planes"][m] for d, m in zip(ds, tr_mask)])
            scalars = np.concatenate([d["scalars"][m] for d, m in zip(ds, tr_mask)])
            action = np.concatenate([d["action"][m] for d, m in zip(ds, tr_mask)])
            lmask = np.concatenate([d["legal_mask"][m] for d, m in zip(ds, tr_mask)])
            del ds
            ix = rng.permutation(len(action))
            for i in range(0, len(ix) - B + 1, B):
                b = ix[i : i + B]
                params, opt_state, loss = train_step(
                    params, opt_state, jnp.asarray(planes[b]), jnp.asarray(scalars[b]),
                    jnp.asarray(action[b], jnp.int32), jnp.asarray(lmask[b]),
                    jnp.asarray(weights[b]))
                losses.append(float(loss))
                step += 1
            g_count += 1
            if args.snap_every_groups and args.save and g_count % args.snap_every_groups == 0:
                a_all, a_choice, a_disc, a_call = evaluate()
                snap = f"{args.save}.g{g_count}.pkl"
                with open(snap, "wb") as f:
                    pickle.dump(jax.device_get(params), f)
                print(f"[snap g{g_count}] steps={step} val_acc={a_all:.4f} "
                      f"choice={a_choice:.4f} discard={a_disc:.4f} call={a_call:.4f} "
                      f"-> {snap}", flush=True)
        a_all, a_choice, a_disc, a_call = evaluate()
        print(f"ep{ep} steps={step} loss={np.mean(losses):.4f} "
              f"val_acc={a_all:.4f} choice={a_choice:.4f} discard={a_disc:.4f} "
              f"call={a_call:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if args.save:
            with open(args.save, "wb") as f:
                pickle.dump(jax.device_get(params), f)
            print(f"saved -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
