"""Branch B, Stage 0: is the untaken side of our Q stale? (hypothesis H)

Online RL regresses Q only for the action the (99.5% greedy) worker took; an untaken option is held only by the
anchor and by shared features. H says that is why the online lineage is overconfident at push/fold and riichi:
the untaken option's Q sits at an old value, the margin is too wide in both directions, and the policy locks in.

Data: A3's rollout labels (riichi-eval `tools/policy_probe.py`, all four kinds, key 20260924 games of tpt 408k vs
3 x mortal_v4, never trained on). Each pair is a state where 408k took `ours`, an alternative `alt`, and
d_grp = value(alt) - value(ours) from a one-step rollout on the true wall (kyoku horizon, GRP pt, sacred units),
an unbiased sample of Q^pi(s, alt) - Q^pi(s, ours).

Both snapshots are forwarded at every labelled state. Mortal's Q target for every decision of kyoku r is
GRP(start of r+1) - GRP(start of r) (gamma = 1; the last kyoku uses the real placement), so the taken side's
absolute target T_ours is read off the log and T_alt = T_ours + d_grp. Q is converted to sacred pt by the
online pts (2.390 / 90 per pt).

Readouts, per kind (push2fold, fold2push, riichi2dama, dama2riichi), seed-clustered 95% CIs:
  (a) calibration: slope of d_grp on the model's predicted difference (Q_alt - Q_ours)/scale (1 = calibrated,
      << 1 = overconfident); reliability by decile of the prediction; in the most confident decile, the share
      of non-zero rollouts where alt was in fact better.
  (b) staleness:
      b-move  (primary, as pre-registered in the plan) — movement 408k -> 448k of the taken vs the untaken
              action, common mode (mean dQ over legal actions) removed: ratio mean|dQ_ours - c| / mean|dQ_alt - c|.
      b-side  where the error sits: residual T - Q/scale of the taken and of the untaken side (408k).
      b-track does the movement go towards the target: slope of the 408k residual on (dQ - c), per side.

Decision rule (written before the run), on push2fold + fold2push pooled (Stage 1's target):
  b-move ratio >= 1.5 with the CI lower bound > 1           -> H supported, go to Stage 1
  b-move ratio <= 1.2                                        -> H rejected, back to the plan's section 5
  in between -> b-side decides: |untaken residual| - |taken residual| > 0 with z >= 2 -> supported, else rejected.

Origin check (--extra NAME=PATH, repeatable): other lineage snapshots forwarded at the same states. Their Q is
in their own pts (e.g. [6,4,2,0]); the taken side's target is recomputed exactly in those pts, the untaken side's
as T_own + k * d_grp with k fitted on the taken side (d_grp is small next to the margins, so k's error hardly
matters). Also reported: how close 408k's taken / untaken Q sit to each extra model's raw Q — the online anchor
compares raw Q across whatever pts the two models were trained in.

    python stage0_stale.py --labels candidates.jsonl --log-dir DIR --grp grp.pth \
        --m408 tpt_tpt_s408000.pth --m448 tpt_tpt_s448000.pth --out stage0.npz [--max-seeds N] [--procs 4] \
        [--extra cprime=cprime_final.pth ...]
    python stage0_stale.py --report stage0.npz

Needs riichi-eval on the path (RIICHI_EVAL_SRC) plus MORTAL_DIR and RIICHI_EVAL_LIBRIICHI.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.environ.get("RIICHI_EVAL_SRC", os.path.expanduser("~/riichi-eval/src")))
from riichi_eval.logs import NAME_RE, read_game  # noqa: E402
from riichi_eval.protocol import placement_pt  # noqa: E402

PT_SCALE = 2.390 / 90
REF = "mortal-v4"
KINDS = ("push2fold", "fold2push", "riichi2dama", "dama2riichi")
_grp = None
_pts = None


def load_labels(path, keep_seed):
    """(seed, rot) -> {(r, pos): [(ours, alt, d_grp, kind, margin)]}, deduplicated per (state, alt)."""
    out = defaultdict(lambda: defaultdict(list))
    seen = set()
    with open(path) as f:
        for ln in f:
            x = json.loads(ln)
            if x.get("d_grp") is None or not keep_seed(x["seed"]):
                continue
            k = (x["seed"], x["rot"], x["r"], x["pos"], x["alt"])
            if k in seen:
                continue
            seen.add(k)
            out[(x["seed"], x["rot"])][(x["r"], x["pos"])].append(
                (x["ours"], x["alt"], x["d_grp"], KINDS.index(x["kind"]), x["margin"]))
    return out


def _init(grp_path, pts_list):
    global _grp, _pts
    import torch
    torch.set_num_threads(1)
    from riichi_eval.rollout.value import GrpValuer
    _grp = GrpValuer(grp_path)
    _pts = [torch.tensor(p, dtype=torch.float64) for p in pts_list]


def game_states(job):
    """Replays one game; returns the labelled states' obs/masks and the pairs with their kyoku target."""
    key, path, lab = job
    try:
        from riichi_eval.replay import iter_decisions
        g = read_game(path)
        seat = next(i for i, n in enumerate(g.names) if n != REF)
        torch = _grp.torch
        seqs = [[_grp.feature(q.index, q.honba, q.kyotaku, q.scores) for q in g.rounds[:r + 1]]
                for r in range(len(g.rounds))]
        fin = g.final_scores()
        with torch.inference_mode():
            mat = _grp.grp.calc_matrix(_grp.grp([torch.tensor(q, dtype=torch.float64) for q in seqs]))
        target = []                                          # per round: tuple over pts vectors (sacred first)
        tj = []
        for pv in [_grp.pts] + _pts:
            exp = (mat @ pv).numpy()[:, seat]
            real = placement_pt(fin, seat, tuple(float(v) for v in pv))
            tj.append([(exp[r + 1] if r + 1 < len(g.rounds) else real) - exp[r] for r in range(len(g.rounds))])
        target = list(zip(*tj))
        obs, masks, pairs = [], [], []
        for d in iter_decisions(g, [seat], select=lambda r, pos, s: (r, pos) in lab):
            pl = lab.get((d.r, d.pos))
            if not pl or d.bucket == "forced":
                continue
            ok = [p for p in pl if d.logged == p[0] and d.mask[p[1]]]
            if not ok:
                continue
            si = len(obs)
            obs.append(d.obs.astype(np.float16))
            masks.append(d.mask.astype(bool))
            for ours, alt, dg, kd, mg in ok:
                pairs.append((si, key[0], "abcd".index(key[1]), d.r, d.pos, kd, ours, alt, dg, mg, *target[d.r]))
        n_lab = sum(len(v) for v in lab.values())
        return (np.stack(obs) if obs else None, np.stack(masks) if masks else None, pairs, n_lab)
    except Exception as e:  # noqa: BLE001
        return path, repr(e)


def load_model(path, device, need_sacred=True):
    import torch
    from riichi_eval import env
    env.setup()
    from model import Brain, DQN  # Mortal/mortal/model.py
    st = torch.load(path, weights_only=True, map_location="cpu")
    cfg = st["config"]
    assert cfg["env"]["gamma"] == 1, cfg["env"]
    if need_sacred:
        assert abs(cfg["env"]["pts"][0] / 90 - PT_SCALE) < 1e-9, cfg["env"]
    b = Brain(version=cfg["control"].get("version", 1), conv_channels=cfg["resnet"]["conv_channels"],
              num_blocks=cfg["resnet"]["num_blocks"])
    q = DQN(version=cfg["control"].get("version", 1))
    b.load_state_dict(st["mortal"])
    q.load_state_dict(st["current_dqn"])
    return b.to(device).eval(), q.to(device).eval(), [float(v) for v in cfg["env"]["pts"]]


def forward(models, obs, masks, device, bs=1024):
    import torch
    outs = [[] for _ in models]
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for i in range(0, len(obs), bs):
            o = torch.from_numpy(obs[i:i + bs]).to(device).float()
            m = torch.from_numpy(masks[i:i + bs]).to(device)
            for k, (b, q, _) in enumerate(models):
                outs[k].append(torch.where(m, q(b(o), m).float(), torch.nan).cpu().numpy())
    return [np.concatenate(x) for x in outs]


COLS = ("seed", "rot", "r", "pos", "kind", "ours", "alt", "d_grp", "margin_log", "t_ours",
        "q408_o", "q408_a", "q448_o", "q448_a", "qbar408", "qbar448", "arg448", "n_legal")


def extract(args):
    import torch
    games = {}
    for d in args.log_dir:
        for p in glob.glob(os.path.join(d, "*.json.gz")):
            if "_20260711_" in os.path.basename(p):
                sys.exit(f"{p}: sacred-protocol eval game; Stage 0 runs on the A3 label games only")
            m = NAME_RE.search(os.path.basename(p))
            if m:
                games[(int(m.group(1)), m.group(3))] = p
    seeds = sorted({k[0] for k in games})
    if args.max_seeds:
        seeds = seeds[:: max(1, len(seeds) // args.max_seeds)][: args.max_seeds]
    extras = [e.split("=", 1) for e in args.extra]
    device = torch.device(args.device)
    if device.type == "cuda" and args.gpu_mem_frac:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac, device)
    models = [load_model(args.m408, device), load_model(args.m448, device)] + \
             [load_model(p, device, need_sacred=False) for _, p in extras]
    pts_list = [m[2] for m in models[2:]]
    cols = list(COLS) + [f"{n}_{x}" for n, _ in extras for x in ("q_o", "q_a", "qbar", "t_ours")]
    t0 = time.time()
    chunks, errors, n_lab, n_pairs = [], [], 0, 0
    # workers fork before any label is read, so they stay small
    with Pool(args.procs, initializer=_init, initargs=(args.grp, pts_list)) as pool:
        for s0 in range(0, len(seeds), args.shard):
            shard = set(seeds[s0:s0 + args.shard])
            labels = load_labels(args.labels, shard.__contains__)
            jobs = [(k, games[k], dict(labels[k])) for k in sorted(labels) if k in games]
            del labels
            buf_o, buf_m, buf_p = [], [], []

            def flush():
                if not buf_o:
                    return
                obs, masks = np.concatenate(buf_o), np.concatenate(buf_m)
                qs = forward(models, obs, masks, device)
                P = np.array([p[1:] for p in buf_p], dtype=np.float64)   # seed .. t_ours for every pts vector
                si = np.array([p[0] for p in buf_p])
                ours, alt = P[:, 5].astype(int), P[:, 6].astype(int)
                q408, q448 = qs[0][si], qs[1][si]
                n = len(si)
                ar = np.arange(n)
                cols_ = [P[:, :10], q408[ar, ours][:, None], q408[ar, alt][:, None], q448[ar, ours][:, None],
                         q448[ar, alt][:, None], np.nanmean(q408, 1)[:, None], np.nanmean(q448, 1)[:, None],
                         np.nanargmax(q448, 1)[:, None].astype(float), masks[si].sum(1)[:, None].astype(float)]
                for j in range(len(extras)):
                    qx = qs[2 + j][si]
                    cols_ += [qx[ar, ours][:, None], qx[ar, alt][:, None], np.nanmean(qx, 1)[:, None],
                              P[:, 10 + j][:, None]]
                chunks.append(np.concatenate(cols_, 1))
                buf_o.clear(), buf_m.clear(), buf_p.clear()

            base = 0
            # submit in blocks: imap's result queue is unbounded, and replay outruns the shared-GPU forward
            for res in (r for b0 in range(0, len(jobs), args.block)
                        for r in pool.imap_unordered(game_states, jobs[b0:b0 + args.block], chunksize=8)):
                if len(res) == 2:
                    errors.append(res)
                    continue
                obs, masks, pairs, nl = res
                n_lab += nl
                if obs is None:
                    continue
                buf_o.append(obs)
                buf_m.append(masks)
                buf_p.extend((p[0] + base, *p[1:]) for p in pairs)
                n_pairs += len(pairs)
                base += len(obs)
                if base >= 4096:
                    flush()
                    base = 0
            flush()
            del jobs
            print(f"  seeds {s0 + len(shard):,}/{len(seeds):,}: {n_pairs:,} pairs ({time.time() - t0:.0f}s)", flush=True)
    a = np.concatenate(chunks)
    np.savez_compressed(args.out, cols=np.array(cols), data=a,
                        extras=np.array([n for n, _ in extras]), extra_pts=np.array(pts_list or [[0, 0, 0, 0]]))
    print(f"-> {args.out}: {len(a):,} pairs matched of {n_lab:,} labelled, {len(errors)} errors, "
          f"{time.time() - t0:.0f}s")
    for p, e in errors[:5]:
        print("  error", p, e)


# ---------------------------------------------------------------- report

def _cl_mean(y, g):
    """Mean with a seed-clustered 95% CI half-width."""
    _, gi = np.unique(g, return_inverse=True)
    s = np.bincount(gi, weights=y)
    n = np.bincount(gi).astype(float)
    G = len(s)
    m = s.sum() / n.sum()
    return m, 1.96 * np.sqrt(G / (G - 1) * ((s - m * n) ** 2).sum()) / n.sum()


def _cl_slope(x, y, g, origin=False):
    """OLS slope of y on x (with intercept unless origin) and its seed-clustered 95% CI half-width."""
    xc = x if origin else x - x.mean()
    yc = y if origin else y - y.mean()
    sxx = (xc * xc).sum()
    b = (xc * yc).sum() / sxx
    e = yc - b * xc
    _, gi = np.unique(g, return_inverse=True)
    sc = np.bincount(gi, weights=xc * e)
    G = len(sc)
    return b, 1.96 * np.sqrt(G / (G - 1) * (sc ** 2).sum()) / sxx


def _cl_ratio(u, v, g, n_boot=400, seed=0):
    """mean(u)/mean(v) with a seed-bootstrap 95% interval."""
    _, gi = np.unique(g, return_inverse=True)
    su, sv = np.bincount(gi, weights=u), np.bincount(gi, weights=v)
    rng = np.random.default_rng(seed)
    G = len(su)
    bs = [su[i].sum() / sv[i].sum() for i in (rng.integers(0, G, G) for _ in range(n_boot))]
    return su.sum() / sv.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def report(paths):
    parts = [np.load(p, allow_pickle=False) for p in paths]
    cols = [str(c) for c in parts[0]["cols"]]
    A = np.concatenate([p["data"] for p in parts])
    c = {k: A[:, i] for i, k in enumerate(cols)}
    g = c["seed"]
    pred408 = (c["q408_a"] - c["q408_o"]) / PT_SCALE
    pred448 = (c["q448_a"] - c["q448_o"]) / PT_SCALE
    d = c["d_grp"]
    t_o = c["t_ours"]
    t_a = t_o + d
    res_o, res_a = t_o - c["q408_o"] / PT_SCALE, t_a - c["q408_a"] / PT_SCALE
    cm = (c["qbar448"] - c["qbar408"]) / PT_SCALE
    dq_o = (c["q448_o"] - c["q408_o"]) / PT_SCALE - cm
    dq_a = (c["q448_a"] - c["q408_a"]) / PT_SCALE - cm
    games = len(np.unique(c["seed"] * 4 + c["rot"]))
    ctrl = np.abs(c["margin_log"] - (c["q408_o"] - c["q408_a"]))
    print(f"{len(A):,} pairs from {games:,} games ({len(np.unique(g)):,} seeds); units: sacred pt\n"
          f"positive control, recomputed 408k margin vs label file: |diff| median {np.median(ctrl):.4f}, "
          f"99th pct {np.percentile(ctrl, 99):.4f} Q units (1 pt = {PT_SCALE:.4f})\n")
    sets = [(k, c["kind"] == i) for i, k in enumerate(KINDS)] + [("push/fold pooled", c["kind"] <= 1)]

    print("(a) calibration: d_grp regressed on the model's predicted (Q_alt - Q_ours)   [1 = calibrated]")
    print(f"   {'kind':<18}{'pairs':>9}{'/game':>7}{'pred mean':>11}{'d_grp mean':>13}"
          f"{'slope 408k':>18}{'slope 448k':>18}{'slope 408 thru 0':>20}")
    for name, mk in sets:
        s4, s4c = _cl_slope(pred408[mk], d[mk], g[mk])
        s8, s8c = _cl_slope(pred448[mk], d[mk], g[mk])
        s0, s0c = _cl_slope(pred408[mk], d[mk], g[mk], origin=True)
        dm, dmc = _cl_mean(d[mk], g[mk])
        print(f"   {name:<18}{mk.sum():>9,}{mk.sum() / games:>7.2f}{pred408[mk].mean():>+11.2f}"
              f"{dm:>+8.3f}±{dmc:<4.2f}{s4:>+11.4f}±{s4c:<6.4f}{s8:>+11.4f}±{s8c:<6.4f}{s0:>+13.4f}±{s0c:<6.4f}")

    print("\n   reliability by decile of the 408k prediction (pred mean -> realized d_grp mean ± CI)")
    for name, mk in sets[:4]:
        x, y, gg = pred408[mk], d[mk], g[mk]
        qs = np.percentile(x, np.linspace(0, 100, 11))
        cells = []
        for lo, hi in zip(qs[:-1], qs[1:]):
            b = (x >= lo) & (x <= hi)
            m, ci = _cl_mean(y[b], gg[b])
            cells.append(f"{x[b].mean():+6.1f}->{m:+5.2f}±{ci:.2f}")
        print(f"   {name:<12} " + "  ".join(cells))

    print("\n   most confident decile (largest 408k margin) vs least: share of non-zero rollouts where alt was better")
    for name, mk in sets[:4]:
        x, y = -pred408[mk], d[mk]
        hi, lo = x >= np.percentile(x, 90), x <= np.percentile(x, 10)
        f = lambda b: ((y[b] > 0).sum() / max(1, (y[b] != 0).sum()), (y[b] != 0).mean(), y[b].mean())
        (fh, nzh, mh), (fl, nzl, ml) = f(hi), f(lo)
        print(f"   {name:<12} top10%: margin {x[hi].mean():6.1f} pt, alt better {fh:5.1%} of {nzh:5.1%} non-zero,"
              f" mean d {mh:+6.2f} | bottom10%: margin {x[lo].mean():5.2f}, alt better {fl:5.1%} of {nzl:5.1%},"
              f" mean d {ml:+6.2f}")

    print("\n(b-move) movement 408k -> 448k, common mode (mean dQ over legal actions) removed, pt   [primary]")
    for name, mk in sets:
        r, lo, hi = _cl_ratio(np.abs(dq_o[mk]), np.abs(dq_a[mk]), g[mk])
        cmm = np.abs(cm[mk]).mean()
        print(f"   {name:<18} mean|dQ taken| {np.abs(dq_o[mk]).mean():6.3f}   mean|dQ untaken| "
              f"{np.abs(dq_a[mk]).mean():6.3f}   ratio {r:5.2f} [{lo:4.2f}, {hi:4.2f}]   (|common| {cmm:5.3f})"
              f"   margin change {np.mean(pred448[mk] - pred408[mk]):+6.2f}   448k flips to alt "
              f"{np.mean(c['arg448'][mk] == c['alt'][mk]):5.1%}")

    print("\n(b-side) where the error sits (408k): residual target - Q, pt   [+ = Q too low]")
    for name, mk in sets:
        mo, mco = _cl_mean(res_o[mk], g[mk])
        ma, mca = _cl_mean(res_a[mk], g[mk])
        gap = np.abs(res_a[mk]).mean() - np.abs(res_o[mk]).mean()
        # |mean| comparison with a clustered CI on the difference of the two signed means' magnitudes
        diff = np.sign(ma) * res_a[mk] - np.sign(mo) * res_o[mk]
        dd, ddc = _cl_mean(diff, g[mk])
        print(f"   {name:<18} taken {mo:+7.2f} ± {mco:<5.2f} untaken {ma:+7.2f} ± {mca:<5.2f}   "
              f"|untaken| - |taken| {dd:+6.2f} ± {ddc:<5.2f} (z {dd / (ddc / 1.96):+5.1f})   (mean abs gap {gap:+.2f})")

    print("\n(b-track) does 408k->448k movement go towards the target? slope of the 408k residual on dQ (+ = yes)")
    for name, mk in sets:
        so, soc = _cl_slope(dq_o[mk], res_o[mk], g[mk])
        sa, sac = _cl_slope(dq_a[mk], res_a[mk], g[mk])
        print(f"   {name:<18} taken {so:+7.2f} ± {soc:<5.2f}   untaken {sa:+7.2f} ± {sac:<5.2f}")

    mk = sets[-1][1]
    r, lo, hi = _cl_ratio(np.abs(dq_o[mk]), np.abs(dq_a[mk]), g[mk])
    diff = np.sign(res_a[mk].mean()) * res_a[mk] - np.sign(res_o[mk].mean()) * res_o[mk]
    dd, ddc = _cl_mean(diff, g[mk])
    if r >= 1.5 and lo > 1:
        verdict = "H SUPPORTED by b-move -> Stage 1"
    elif r <= 1.2:
        verdict = "H REJECTED by b-move -> back to section 5"
    else:
        verdict = ("b-move in between; b-side " + ("supports H -> Stage 1" if dd / (ddc / 1.96) >= 2
                                                  else "does not support H -> back to section 5"))
    print(f"\npre-registered rule (push/fold pooled): ratio {r:.2f} [{lo:.2f}, {hi:.2f}], "
          f"b-side z {dd / (ddc / 1.96):+.1f}  =>  {verdict}")

    extras = [str(x) for x in parts[0]["extras"]] if "extras" in parts[0] else []
    if not extras:
        return
    print("\n(origin) the same states through other lineage snapshots, in their own pts, shown as sacred-pt "
          "equivalents (/k)")
    print(f"   {'model':<10}{'pts':>18}{'k':>8} | {'set':<18}{'prefers ours':>13}{'margin':>9}{'slope':>16}"
          f"{'taken res':>16}{'untaken res':>17}")
    for n, pv in zip(extras, parts[0]["extra_pts"]):
        t_x, q_o, q_a = c[f"{n}_t_ours"], c[f"{n}_q_o"], c[f"{n}_q_a"]
        k = (t_x * t_o).sum() / (t_o * t_o).sum()
        rx_o, rx_a = (t_x - q_o) / k, (t_x + k * d - q_a) / k
        for name, mk in (sets[-1], sets[2], sets[3]):
            sl, slc = _cl_slope((q_a[mk] - q_o[mk]) / k, d[mk], g[mk])
            a_, ac = _cl_mean(rx_o[mk], g[mk])
            b_, bc = _cl_mean(rx_a[mk], g[mk])
            print(f"   {n:<10}{str([float(v) for v in pv]):>18}{k:>8.4f} | {name:<18}{np.mean(q_o[mk] > q_a[mk]):>13.1%}"
                  f"{np.mean(q_o[mk] - q_a[mk]) / k:>9.1f}{sl:>+10.4f}±{slc:<5.3f}{a_:>+10.2f}±{ac:<5.2f}"
                  f"{b_:>+11.2f}±{bc:<5.2f}")
    for name, ref in (("408k", ("q408_o", "q408_a", "qbar408")),):
        print(f"\n   raw-Q distance of {name} to each snapshot at the same states (push/fold pooled; raw units, "
              f"state mean removed) — the online anchor compares raw Q")
        mk = sets[-1][1]
        a_o = c[ref[0]][mk] - c[ref[2]][mk]
        a_a = c[ref[1]][mk] - c[ref[2]][mk]
        for n in extras:
            x_o = c[f"{n}_q_o"][mk] - c[f"{n}_qbar"][mk]
            x_a = c[f"{n}_q_a"][mk] - c[f"{n}_qbar"][mk]
            print(f"   {n:<10} taken |d| {np.abs(a_o - x_o).mean():6.3f}  untaken |d| {np.abs(a_a - x_a).mean():6.3f}"
                  f"   corr taken {np.corrcoef(a_o, x_o)[0, 1]:+.3f}  untaken {np.corrcoef(a_a, x_a)[0, 1]:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels")
    ap.add_argument("--log-dir", action="append")
    ap.add_argument("--grp")
    ap.add_argument("--m408")
    ap.add_argument("--m448")
    ap.add_argument("--out")
    ap.add_argument("--max-seeds", type=int, default=0)
    ap.add_argument("--shard", type=int, default=1500, help="seeds per shard (bounds memory)")
    ap.add_argument("--block", type=int, default=256, help="games in flight (bounds the result backlog)")
    ap.add_argument("--extra", action="append", default=[], help="NAME=PATH, another snapshot for the origin check")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpu-mem-frac", type=float, default=0.3)
    ap.add_argument("--report", nargs="+")
    args = ap.parse_args()
    if args.report:
        report(args.report)
    else:
        if not all((args.labels, args.log_dir, args.grp, args.m408, args.m448, args.out)):
            sys.exit("--labels --log-dir --grp --m408 --m448 --out are required for extraction")
        extract(args)


if __name__ == "__main__":
    main()
