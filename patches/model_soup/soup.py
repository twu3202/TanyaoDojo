"""Build a uniform weight average (model soup) of fine-tuned Mortal states that share one start, and report how the
training draws relate in weight space.

    python soup.py OUT_INFER STEPS START INGREDIENT [INGREDIENT ...]

- Every ingredient must have START's architecture (same keys and shapes in mortal / current_dqn / aux_net).
- Floating tensors are averaged; integer buffers (BatchNorm num_batches_tracked) are taken from the first ingredient.
  BatchNorm running statistics are averaged like the weights.
- The output is an infer file in the format xfer_eval.sh makes: mortal, current_dqn, aux_net, config (the first
  ingredient's), steps.
- Weight-space report: delta_i = ingredient_i - START over the trainable floating tensors (running statistics
  excluded). |delta_i| / |START|, pairwise cosine of the deltas, and |mean delta| / mean |delta_i|.
  Independent draws give cosines near 0 and a ratio near 1/sqrt(K); one shared direction gives cosines near 1.
"""
import sys

import torch

NETS = ("mortal", "current_dqn", "aux_net")


def load(path):
    return torch.load(path, weights_only=True, map_location="cpu")


def flat(st, ref=None):
    """Concatenate the trainable floating tensors (no BatchNorm running stats), optionally minus ref's."""
    parts = []
    for net in NETS:
        for k, v in st[net].items():
            if not v.is_floating_point() or "running_" in k:
                continue
            d = v.double().flatten()
            if ref is not None:
                d = d - ref[net][k].double().flatten()
            parts.append(d)
    return torch.cat(parts)


def main():
    out, steps, start_path, paths = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
    start = load(start_path)
    sts = [load(p) for p in paths]
    for p, st in zip(paths, sts):
        for net in NETS:
            a, b = start[net], st[net]
            assert a.keys() == b.keys(), f"{p} {net}: keys differ from the start"
            for k in a:
                assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype, f"{p} {net}.{k}: shape/dtype differs"
        assert st["config"]["resnet"] == start["config"]["resnet"], f"{p}: resnet config differs"

    soup = {}
    for net in NETS:
        sd = {}
        for k, v in sts[0][net].items():
            if v.is_floating_point():
                sd[k] = torch.stack([st[net][k].double() for st in sts]).mean(0).to(v.dtype)
            else:
                sd[k] = v.clone()
        soup[net] = sd
    soup["config"] = sts[0]["config"]
    soup["steps"] = steps
    torch.save(soup, out)

    base = flat(start)
    deltas = [flat(st, start) for st in sts]
    names = [p.rsplit("/", 1)[-1] for p in paths]
    print(f"start {start_path.rsplit('/', 1)[-1]}  |start| {base.norm():.2f}  ({base.numel():,} trainable values)")
    for n, d in zip(names, deltas):
        print(f"  {n:<40} |delta| / |start| {d.norm() / base.norm():.5f}")
    k = len(deltas)
    cos = [[float(torch.dot(deltas[i], deltas[j]) / (deltas[i].norm() * deltas[j].norm())) for j in range(k)] for i in range(k)]
    print("  pairwise cosine of the deltas:")
    for i in range(k):
        print("   " + " ".join(f"{cos[i][j]:+.3f}" for j in range(k)) + f"  {names[i]}")
    off = [cos[i][j] for i in range(k) for j in range(k) if i < j]
    mean_d = torch.stack(deltas).mean(0)
    ratio = float(mean_d.norm() / torch.stack([d.norm() for d in deltas]).mean())
    print(f"  mean pairwise cosine {sum(off) / len(off):+.3f}; |mean delta| / mean |delta_i| {ratio:.3f} "
          f"(independent draws: {1 / k ** 0.5:.3f}, one shared direction: 1.000)")
    print(f"soup written: {out} (steps {steps}, {k} ingredients)")


if __name__ == "__main__":
    main()
