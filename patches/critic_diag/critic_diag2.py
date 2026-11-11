"""Critic upper-bound diagnostic, second run (评论家上限诊断 v2).

The first run (critic_diag.py) trained its trunks at LR 1e-4 with an optimizer step every batch, about 80 times more
aggressive per sample than the probe's own online value training (1e-5, a step every 8 batches). The three trunk critics
ended below their own starting point on the held-out games, so they measured a failed fit, not a ceiling. This run:

  - trunk LR 1e-5 (the probe's), head LR 1e-4;
  - a fixed-capacity shuffle buffer (6,000 decisions per loader worker, about 45 games), so a batch no longer holds
    ~13 decisions of the same kyoku;
  - step 0 is a validation look, so early stopping can always fall back to the starting point (= head0);
  - two side critics that cannot damage the public features: P160k's frozen features plus a fresh small network on
    the public and hidden input, fused into the head through an output layer that starts at zero.

  head0        the probe's own value head on P160k's features: no training (the reference)
  sep          a copy of P160k's trunk plus the head, trained on value (public information)
  hands        sep with Mortal's oracle input, wall rows zeroed (opponents' hands, shanten, waits, furiten)
  oracle       sep with the full oracle input (also the wall order, dora, ura)
  hands_side   frozen P160k features + a fresh small net on public + hands rows
  oracle_side  frozen P160k features + a fresh small net on public + all oracle rows

Pre-registration and departures: patches/critic_diag/README.md, section "Second run".

Run in Mortal/mortal with MORTAL_CFG pointing at a config that has [grp] and [env] (the probe's own):
  python critic_diag.py split  OUT          # dedupe, split by seed -> OUT/split.json
  python critic_diag.py smoke  OUT          # 30 batches + a small evaluation + the report code; prints no EV
  python critic_diag.py bench  OUT N        # N training batches: timing and memory only, no explained variance
  python critic_diag.py train  OUT          # the run -> OUT/val_curve.json, OUT/test_preds.npz, OUT/best_*.pth
  python critic_diag.py report OUT          # tables with bootstrap CIs (CPU)
"""
import copy
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from glob import glob
from os import path

import numpy as np

ROOT = '/home/r/Projects/better_mortal'
P_STATE = f'{ROOT}/runs/xfer_p1/archive/xfer_p1_s160000.pth'
P_ARM = f'{ROOT}/runs/xfer_p1'
# training-only data: the other stage-2 arms from A80k (Q line, T 0.1, eps 0.005)
EXTRA_ARMS = [f'{ROOT}/runs/xfer_v11c2', f'{ROOT}/runs/xfer_v11r', f'{ROOT}/runs/xfer_v11w']
SALT = 'critic_diag_20261109'
TEST_FRAC = 0.16
VAL_FRAC = 0.04
HANDS_ROWS = 51          # oracle rows 0..50: the three opponents' hand blocks (17 rows each); 51..216: wall, dora, ura

BATCH = 512
TOTAL = int(os.environ.get('CD_TOTAL', '12000'))
VAL_EVERY = int(os.environ.get('CD_VAL_EVERY', '1000'))
LR_TRUNK = 1e-5          # the probe's own online LR
LR_HEAD = 1e-4
LR_SIDE = 1e-4
WARMUP = 200
LR_FLOOR = 0.1           # cosine to 10% of peak at TOTAL
MAX_GRAD_NORM = 1.0
KL_COEF = 10.0           # shared: weight of KL(pi_P160k || pi) at T 0.1
TEMP = 0.1
NUM_WORKERS = int(os.environ.get('CD_WORKERS', '4'))
EVAL_WORKERS = 2        # WSL has 15 GB; the training loader's workers stay alive during validation
FILE_BATCH = 8            # files decoded at a time (float32 transients ~0.35 GB per worker)
SHUF_CAP = 6000          # decisions kept in each loader worker's shuffle buffer (float16, ~0.5 GB; error <= 2.4e-4,
                         # below the bfloat16 rounding the model applies to its input)
CRITICS = ['head0', 'sep', 'hands', 'oracle', 'hands_side', 'oracle_side']
TRAINED = CRITICS[1:]
FAMILIES = {'public': ['sep'], 'hands': ['hands', 'hands_side'], 'oracle': ['oracle', 'oracle_side']}


def ident(p):
    parts = path.basename(p)[:-len('.json.gz')].split('_')
    return parts[-3], parts[-2], parts[-1]          # game index, worker key, side a/b


def which_split(idx, key):
    u = int(hashlib.md5(f'{SALT}:{idx}_{key}'.encode()).hexdigest()[:8], 16) / 2 ** 32
    return 'test' if u < TEST_FRAC else 'val' if u < TEST_FRAC + VAL_FRAC else 'train'


def arm_files(arm):
    seen = {}
    for sub in ['train_play_w1', 'train_play_w2', 'train_play_w3', 'train_play_w4', 'drain', 'buffer']:
        for p in sorted(glob(f'{arm}/{sub}/*.json.gz')):
            seen.setdefault(ident(p), p)
    return seen


def cmd_split(out):
    os.makedirs(out, exist_ok=True)
    p = arm_files(P_ARM)
    split = {'train': [], 'val': [], 'test': []}
    for (idx, key, side), f in sorted(p.items()):
        split[which_split(idx, key)].append(f)
    held = {(i, k) for (i, k, s), f in p.items() if which_split(i, k) != 'train'}
    p_train = len(split['train'])
    extra = {}
    for arm in EXTRA_ARMS:
        files = arm_files(arm)
        keep = [f for (i, k, s), f in sorted(files.items()) if (i, k) not in held]
        extra[path.basename(arm)] = len(keep)
        split['train'] += keep
    # duplicates across buffer / drain / train_play must be byte-identical: check a sample
    dup_ok = dup_n = 0
    groups = {}
    for sub in ['train_play_w1', 'drain', 'buffer']:
        for f in glob(f'{P_ARM}/{sub}/*.json.gz'):
            groups.setdefault(ident(f), []).append(f)
    multi = [v for v in groups.values() if len(v) > 1]
    random.Random(0).shuffle(multi)
    for v in multi[:60]:
        h = {hashlib.md5(open(f, 'rb').read()).hexdigest() for f in v}
        dup_n += 1
        dup_ok += len(h) == 1
    info = {'p_unique': len(p), 'p_train': p_train, 'val': len(split['val']), 'test': len(split['test']),
            'extra_train': extra, 'train_total': len(split['train']), 'dup_checked': dup_n, 'dup_identical': dup_ok,
            'seed_pairs_val': len({ident(f)[:2] for f in split['val']}),
            'seed_pairs_test': len({ident(f)[:2] for f in split['test']})}
    split['info'] = info
    with open(f'{out}/split.json', 'w') as fh:
        json.dump(split, fh)
    print(json.dumps(info, indent=1))
    assert dup_ok == dup_n, 'duplicate game files differ'


# ---------------------------------------------------------------- data
def make_iter_class():
    import torch
    from torch.utils.data import IterableDataset
    from libriichi.dataset import GameplayLoader
    from model import GRP
    from reward_calculator import RewardCalculator
    from config import config

    class DiagIter(IterableDataset):
        """Trainee decisions with obs, oracle obs, the return and (for evaluation) per-sample metadata."""

        def __init__(self, files, *, shuffle, meta, loop=False, seed=0):
            super().__init__()
            self.items = list(enumerate(files))
            self.shuffle, self.meta, self.loop, self.seed = shuffle, meta, loop, seed

        def decode(self, chunk, loader, calc):
            try:
                data = loader.load_gz_log_files([f for _, f in chunk])
            except RuntimeError as e:
                logging.warning(f'skipping unreadable batch {chunk[0][1]}: {str(e).splitlines()[0]}')
                return []
            out = []
            for (gid, _), file in zip(chunk, data):
                for game in file:
                    obs = np.asarray(game.take_obs(), dtype=np.float16)
                    inv = np.asarray(game.take_invisible_obs(), dtype=np.float16)
                    actions = game.take_actions()
                    masks = game.take_masks()
                    at_kyoku = game.take_at_kyoku()
                    turns = game.take_at_turns() if self.meta else None
                    shantens = game.take_shantens() if self.meta else None
                    grp = game.take_grp()
                    player_id = game.take_player_id()
                    rewards = calc.calc_delta_pt(player_id, grp.take_feature(), grp.take_rank_by_player())
                    pos = np.zeros(len(obs), dtype=np.int64)
                    for i in range(1, len(obs)):
                        pos[i] = pos[i - 1] + 1 if at_kyoku[i] == at_kyoku[i - 1] else 0
                    for i in range(len(obs)):
                        # copies, not views: a view keeps its whole game's array alive until the game's last
                        # decision leaves the shuffle buffer (about 4x the memory)
                        e = [obs[i].copy(), inv[i].copy(), actions[i], np.array(masks[i]),
                             np.float32(rewards[at_kyoku[i]])]
                        if self.meta:
                            e += [gid, int(at_kyoku[i]), int(pos[i]), int(turns[i]), int(shantens[i])]
                        out.append(e)
            return out

        def __iter__(self):
            info = torch.utils.data.get_worker_info()
            items = self.items
            wid = 0
            if info is not None:
                per = math.ceil(len(items) / info.num_workers)
                items = items[info.id * per:(info.id + 1) * per]
                wid = info.id
            assert config['env']['gamma'] == 1, 'the return here is the kyoku reward; gamma must be 1'
            g = GRP(**config['grp']['network'])
            g.load_state_dict(torch.load(config['grp']['state_file'], weights_only=True, map_location='cpu')['model'])
            calc = RewardCalculator(g, config['env']['pts'])
            loader = GameplayLoader(version=4, oracle=True, player_names=['trainee'], augmented=False)
            rng = random.Random(self.seed * 1000 + wid)
            buf = []
            while True:
                items = list(items)
                if self.shuffle:
                    rng.shuffle(items)
                fb = FILE_BATCH if self.shuffle else 4
                for s in range(0, len(items), fb):
                    new = self.decode(items[s:s + fb], loader, calc)
                    if not self.shuffle:
                        yield from new
                        continue
                    buf += new
                    if len(buf) > SHUF_CAP:
                        rng.shuffle(buf)
                        yield from buf[SHUF_CAP:]
                        del buf[SHUF_CAP:]
                if not self.loop:
                    rng.shuffle(buf)
                    yield from buf
                    return

    return DiagIter


# ---------------------------------------------------------------- models
def build_models(device):
    import torch
    from torch import nn
    from model import Brain, DQN

    s = torch.load(P_STATE, weights_only=True, map_location='cpu')
    assert s['steps'] == 160000 and s['config']['resnet'] == {'conv_channels': 192, 'num_blocks': 40}
    rn = s['config']['resnet']

    def public():
        b = Brain(version=4, **rn)
        b.load_state_dict(s['mortal'])
        return b

    def oracle():
        b = Brain(version=4, is_oracle=True, **rn)
        sd = dict(s['mortal'])
        w = sd['encoder.net.0.weight']
        new = torch.zeros(w.shape[0], w.shape[1] + 217, w.shape[2], dtype=w.dtype)
        new[:, :w.shape[1]] = w
        sd['encoder.net.0.weight'] = new
        b.load_state_dict(sd)
        return b

    def head():
        h = nn.Sequential(nn.Linear(1024, 256), nn.Mish(inplace=True), nn.Linear(256, 1))
        h.load_state_dict(s['value_head'])
        return h

    def dqn():
        d = DQN(version=4)
        d.load_state_dict(s['current_dqn'])
        return d

    class Side(nn.Module):
        """A fresh small network on the public + hidden input, fused with P160k's frozen features. The output layer
        starts at zero, so the critic starts equal to head0."""

        def __init__(self, c_in):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(c_in, 128, 3, padding=1), nn.Mish(inplace=True),
                nn.Conv1d(128, 128, 3, padding=1), nn.Mish(inplace=True), nn.Flatten(),
                nn.Linear(128 * 34, 256), nn.Mish(inplace=True))
            self.fuse = nn.Sequential(nn.Linear(1024 + 256, 256), nn.Mish(inplace=True), nn.Linear(256, 1))
            nn.init.zeros_(self.fuse[-1].weight)
            nn.init.zeros_(self.fuse[-1].bias)

        def forward(self, x, phi):
            return self.fuse(torch.cat([phi, self.conv(x)], -1))

    ref_trunk, ref_dqn = public().to(device).eval(), dqn().to(device).eval()
    for p in list(ref_trunk.parameters()) + list(ref_dqn.parameters()):
        p.requires_grad_(False)
    m = {
        'head0': {'head': head().to(device).eval()},
        'sep': {'trunk': public().to(device), 'head': head().to(device)},
        'hands': {'trunk': oracle().to(device), 'head': head().to(device)},
        'oracle': {'trunk': oracle().to(device), 'head': head().to(device)},
        'hands_side': {'head': head().to(device), 'side': Side(1012 + HANDS_ROWS).to(device)},
        'oracle_side': {'head': head().to(device), 'side': Side(1012 + 217).to(device)},
    }
    for k in m['head0']['head'].parameters():
        k.requires_grad_(False)
    for name in TRAINED:
        if 'trunk' in m[name]:
            m[name]['trunk'].freeze_bn(True)     # BN statistics frozen, as in every online stage
    return ref_trunk, ref_dqn, m


def make_optims(m):
    import torch
    opts = {}
    for name in TRAINED:
        groups = [{'params': list(m[name]['head'].parameters()), 'lr': LR_HEAD, 'peak': LR_HEAD}]
        trunk_params = []
        for part in ('trunk', 'dqn'):
            if part in m[name]:
                trunk_params += list(m[name][part].parameters())
        if trunk_params:
            groups.append({'params': trunk_params, 'lr': LR_TRUNK, 'peak': LR_TRUNK})
        if 'side' in m[name]:
            groups.append({'params': list(m[name]['side'].parameters()), 'lr': LR_SIDE, 'peak': LR_SIDE})
        opts[name] = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    return opts


def lr_factor(step):
    if step < WARMUP:
        return (step + 1) / WARMUP
    t = min(1.0, (step - WARMUP) / max(1, TOTAL - WARMUP))
    return LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * t))


def forward_all(models, ref_trunk, ref_dqn, obs, inv, masks, train, opts=None, g=None, step=0):
    """Returns {critic: V} (float32, detached). In training mode also steps each trained critic in turn, so only one
    network's activations are alive at a time."""
    import torch
    from torch.nn import functional as F
    from torch.nn.utils import clip_grad_norm_

    ac = torch.autocast('cuda', dtype=torch.bfloat16)
    out, extra = {}, {}
    with torch.no_grad(), ac:
        phi_ref = ref_trunk(obs)
        q_ref = ref_dqn(phi_ref, masks)
        out['head0'] = models['head0']['head'](phi_ref).squeeze(-1).float()
    lp_ref = (q_ref.float().masked_fill(~masks, -1e9) / TEMP).log_softmax(-1)
    inv_hands = inv.clone()
    inv_hands[:, HANDS_ROWS:] = 0
    phi_ref = phi_ref.detach()

    def run(name):
        mm = models[name]
        if name in ('hands_side', 'oracle_side'):
            with ac:
                x = torch.cat([obs, inv[:, :HANDS_ROWS] if name == 'hands_side' else inv], 1)
                v = mm['head'](phi_ref) + mm['side'](x, phi_ref)
            return v.squeeze(-1).float(), None, None
        with ac:
            if name in ('oracle', 'hands'):
                phi = mm['trunk'](obs, inv if name == 'oracle' else inv_hands)
            else:
                phi = mm['trunk'](obs)
            v = mm['head'](phi).squeeze(-1).float()
            kl = lp = None
            if name == 'shared':
                lp = (mm['dqn'](phi, masks).float().masked_fill(~masks, -1e9) / TEMP).log_softmax(-1)
                kl = (lp_ref.exp() * (lp_ref - lp)).sum(-1)
        return v, kl, lp

    for name in TRAINED:
        mm = models[name]
        for part in mm.values():
            part.train(train)
        if 'trunk' in mm:
            mm['trunk'].freeze_bn(True)
        if train:
            opt = opts[name]
            for grp in opt.param_groups:
                grp['lr'] = grp['peak'] * lr_factor(step)
            v, kl, _ = run(name)
            loss = F.mse_loss(v, g)
            if kl is not None:
                loss = loss + KL_COEF * kl.mean()
                extra['shared_kl'] = float(kl.detach().mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            params = [p for grp in opt.param_groups for p in grp['params']]
            clip_grad_norm_(params, MAX_GRAD_NORM)
            opt.step()
            out[name] = v.detach()
        else:
            with torch.no_grad():
                v, kl, lp = run(name)
            out[name] = v
            if kl is not None:
                extra['shared_kl_vec'] = kl
                extra['shared_agree'] = lp.argmax(-1) == lp_ref.argmax(-1)
    return out, extra, lp_ref


def loader_for(files, *, shuffle, meta, loop, seed, workers):
    import torch
    from torch.utils.data import DataLoader
    DiagIter = make_iter_class()
    return DataLoader(DiagIter(files, shuffle=shuffle, meta=meta, loop=loop, seed=seed), batch_size=BATCH,
                      num_workers=workers, pin_memory=True, persistent_workers=False, prefetch_factor=2)


def to_dev(batch, device):
    import torch
    obs, inv, actions, masks, g = batch[:5]
    return (obs.to(device, non_blocking=True).float(), inv.to(device, non_blocking=True).float(), actions.to(device),
            masks.to(device=device, dtype=torch.bool, non_blocking=True), g.to(device=device, dtype=torch.float32))


def evaluate(files, models, ref_trunk, ref_dqn, device, *, meta):
    """Sum statistics (and per-sample predictions when meta) over a fixed list of games."""
    import torch
    sums = {k: [0.0, 0.0] for k in CRITICS}     # SSE, and for g: n, sum, sumsq
    gs = [0, 0.0, 0.0]
    rows = {k: [] for k in CRITICS + ['g', 'meta', 'action', 'n_legal', 'kl', 'agree']}
    for batch in loader_for(files, shuffle=False, meta=meta, loop=False, seed=0, workers=EVAL_WORKERS):
        obs, inv, actions, masks, g = to_dev(batch, device)
        with torch.no_grad():
            out, extra, lp_ref = forward_all(models, ref_trunk, ref_dqn, obs, inv, masks, train=False)
        g64 = g.double()
        gs[0] += len(g)
        gs[1] += float(g64.sum())
        gs[2] += float((g64 ** 2).sum())
        for k in CRITICS:
            sums[k][0] += float(((g64 - out[k].double()) ** 2).sum())
        if meta:
            rows['g'].append(g.cpu().numpy())
            for k in CRITICS:
                rows[k].append(out[k].cpu().numpy())
            rows['meta'].append(torch.stack([t.to(torch.int64) for t in batch[5:]], -1).numpy())
            rows['action'].append(actions.cpu().numpy())
            rows['n_legal'].append(masks.sum(-1).cpu().numpy())
            if 'shared_kl_vec' in extra:
                rows['kl'].append(extra['shared_kl_vec'].cpu().numpy())
                rows['agree'].append(extra['shared_agree'].cpu().numpy())
    n, s1, s2 = gs
    sst = s2 - s1 * s1 / n
    ev = {k: 1 - sums[k][0] / sst for k in CRITICS}
    res = {'n': n, 'mean_g': s1 / n, 'var_g': sst / n, 'ev': ev}
    if meta:
        res['rows'] = {k: np.concatenate(v) for k, v in rows.items() if v}
    return res


def snapshot(models, names=TRAINED):
    return {name: {part: {k: v.detach().cpu().clone() for k, v in mod.state_dict().items()}
                   for part, mod in models[name].items()} for name in names}


def restore(models, snap, names):
    for name in names:
        for part, sd in snap[name].items():
            models[name][part].load_state_dict(sd)


def cmd_train(out, bench_n=0):
    import torch
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    split = json.load(open(f'{out}/split.json'))
    ref_trunk, ref_dqn, models = build_models(device)
    opts = make_optims(models)
    total = bench_n or TOTAL
    logging.info(f'train files {len(split["train"]):,}, val {len(split["val"]):,}, test {len(split["test"]):,}; '
                 f'{total:,} batches of {BATCH}, validation every {VAL_EVERY:,}')
    curve, best, best_snap = [], {k: -1e9 for k in TRAINED}, snapshot(models)
    train_iter = iter(loader_for(split['train'], shuffle=True, meta=False, loop=True, seed=1, workers=NUM_WORKERS))
    win = {k: [0.0, 0.0, 0.0, 0.0] for k in TRAINED}    # n, sum g, sum g^2, SSE (training window, before the step)
    t0 = time.time()
    t_wait = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(total):
        if not bench_n and step % VAL_EVERY == 0:      # step 0 included: early stopping can keep the start
            r = evaluate(split['val'], models, ref_trunk, ref_dqn, device, meta=False)
            curve.append({'step': step, **r['ev']})
            for k in TRAINED:
                if r['ev'][k] > best[k]:
                    best[k] = r['ev'][k]
                    best_snap[k] = snapshot(models, [k])[k]
            logging.info(f'VAL at {step:,}: ' + ' '.join(f'{k} {r["ev"][k]:.4f}' for k in CRITICS))
            json.dump(curve, open(f'{out}/val_curve.json', 'w'), indent=1)
        tw = time.time()
        batch = next(train_iter)
        t_wait += time.time() - tw
        obs, inv, actions, masks, g = to_dev(batch, device)
        out_v, extra, _ = forward_all(models, ref_trunk, ref_dqn, obs, inv, masks, train=True, opts=opts, g=g,
                                      step=step)
        if not bench_n:
            g64 = g.double()
            for k in TRAINED:
                w = win[k]
                w[0] += len(g)
                w[1] += float(g64.sum())
                w[2] += float((g64 ** 2).sum())
                w[3] += float(((g64 - out_v[k].double()) ** 2).sum())
        if (step + 1) % 200 == 0:
            el = time.time() - t0
            msg = (f'{step + 1:,} batches, {(step + 1) / el:.2f} batch/s, waiting for data {t_wait / el:.0%}, '
                   f'peak GPU {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB')
            if not bench_n:
                evs = {k: 1 - w[3] / max(w[2] - w[1] ** 2 / w[0], 1e-9) for k, w in win.items()}
                msg += ' | train-window EV ' + ' '.join(f'{k} {v:.3f}' for k, v in evs.items())
                msg += f' | shared KL {extra.get("shared_kl", 0):.4f}'
                win = {k: [0.0, 0.0, 0.0, 0.0] for k in TRAINED}
            logging.info(msg)
    if bench_n:
        logging.info('BENCH DONE')
        return
    r = evaluate(split['val'], models, ref_trunk, ref_dqn, device, meta=False)
    curve.append({'step': total, **r['ev']})
    for k in TRAINED:
        if r['ev'][k] > best[k]:
            best[k] = r['ev'][k]
            best_snap[k] = snapshot(models, [k])[k]
    json.dump(curve, open(f'{out}/val_curve.json', 'w'), indent=1)
    logging.info(f'VAL at {total:,}: ' + ' '.join(f'{k} {r["ev"][k]:.4f}' for k in CRITICS))
    restore(models, best_snap, TRAINED)
    torch.save(best_snap, f'{out}/best_critics.pth')
    logging.info('best validation step per critic: ' + ' '.join(
        f'{k} {max(curve, key=lambda c: c[k])["step"]:,}' for k in TRAINED))
    r = evaluate(split['test'], models, ref_trunk, ref_dqn, device, meta=True)
    rows = r.pop('rows')
    np.savez_compressed(f'{out}/test_preds.npz', **rows)
    json.dump(r, open(f'{out}/test_summary.json', 'w'), indent=1)
    logging.info('TEST ' + ' '.join(f'{k} {r["ev"][k]:.4f}' for k in CRITICS))
    logging.info('CD_TRAIN_DONE')


def cmd_smoke(out):
    """30 training batches on 40 training games, an evaluation with metadata on 8 test games, then the report code.
    Prints no explained variance (the report's output goes to a file nobody reads before the run)."""
    import contextlib
    import torch
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    device = torch.device('cuda')
    split = json.load(open(f'{out}/split.json'))
    ref_trunk, ref_dqn, models = build_models(device)
    opts = make_optims(models)
    it = iter(loader_for(split['train'][:40], shuffle=True, meta=False, loop=True, seed=1, workers=2))
    from libriichi.dataset import GameplayLoader
    raw = GameplayLoader(version=4, oracle=True, player_names=['trainee'], augmented=False).load_gz_log_files(
        split['train'][:2])
    for file in raw:
        for game in file:
            for a in (game.take_obs(), game.take_invisible_obs()):
                a = np.asarray(a, dtype=np.float32)
                err = np.abs(a.astype(np.float16).astype(np.float32) - a).max()
                logging.info(f'float16 storage: max abs error {err:.3g}, max |x| {np.abs(a).max():.3g}')
                # the model casts its input to bfloat16 under autocast (step 2^-8 near 1): float16 storage (2^-11)
                # must stay well below that rounding
                assert err < 1e-3, 'float16 storage error is not below the bfloat16 input rounding'
    obs, inv, actions, masks, g = to_dev(next(it), device)
    with torch.no_grad():
        out_v, _, _ = forward_all(models, ref_trunk, ref_dqn, obs, inv, masks, train=False)
    d0 = {k: float((out_v[k] - out_v['head0']).abs().max()) for k in TRAINED}
    logging.info('start vs head0, max |dV|: ' + ' '.join(f'{k} {v:.2e}' for k, v in d0.items()))
    assert max(d0.values()) < 0.02, d0
    t0 = time.time()
    for step in range(30):
        obs, inv, actions, masks, g = to_dev(next(it), device)
        assert obs.shape[1:] == (1012, 34) and inv.shape[1:] == (217, 34), (obs.shape, inv.shape)
        out_v, extra, _ = forward_all(models, ref_trunk, ref_dqn, obs, inv, masks, train=True, opts=opts, g=g,
                                      step=step)
        assert all(torch.isfinite(v).all() for v in out_v.values())
    logging.info(f'smoke: 30 batches in {time.time() - t0:.0f} s, peak GPU '
                 f'{torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB, keys {sorted(out_v)}')
    r = evaluate(split['test'][:8], models, ref_trunk, ref_dqn, device, meta=True)
    rows = r.pop('rows')
    sd = f'{out}/smoke'
    os.makedirs(sd, exist_ok=True)
    np.savez_compressed(f'{sd}/test_preds.npz', **rows)
    logging.info('smoke eval: ' + ', '.join(f'{k} {v.shape}' for k, v in rows.items()))
    with open(f'{sd}/report.txt', 'w') as fh, contextlib.redirect_stdout(fh):
        cmd_report(sd, B=50)
    logging.info('SMOKE OK')


# ---------------------------------------------------------------- report
def cmd_report(out, B=2000):
    d = np.load(f'{out}/test_preds.npz')
    g = d['g'].astype(np.float64)
    meta = d['meta']                         # gid, kyoku, position in kyoku, turn, shanten
    gid = meta[:, 0]
    games, inv_idx = np.unique(gid, return_inverse=True)
    G = len(games)
    rng = np.random.default_rng(20261109)
    W = rng.multinomial(G, np.full(G, 1 / G), size=B).astype(np.float64)    # bootstrap weights per game

    def per_game(x):
        return np.bincount(inv_idx, weights=x, minlength=G)

    n_g, s_g, q_g = per_game(np.ones_like(g)), per_game(g), per_game(g * g)
    sse = {k: per_game((g - d[k].astype(np.float64)) ** 2) for k in CRITICS}

    def ev_of(w, k):
        n, s, q = w @ n_g, w @ s_g, w @ q_g
        return 1 - (w @ sse[k]) / (q - s * s / n)

    one = np.ones(G)
    print(f'test: {len(g):,} decisions in {G:,} games; mean return {g.mean():+.4f}, SD {g.std():.4f}')
    print('\nExplained variance on the held-out games (95% bootstrap over games):')
    print(f'{"critic":8s} {"EV":>8s} {"95% CI":>18s}  {"residual var x":>14s}')
    base = ev_of(one, 'head0')
    boot = {k: ev_of(W, k) for k in CRITICS}
    for k in CRITICS:
        e = ev_of(one, k)
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        print(f'{k:8s} {e:8.4f} [{lo:7.4f}, {hi:7.4f}]  {(1 - base) / (1 - e):14.2f}')
    print('\nPaired differences:')
    for a, b in [('sep', 'head0'), ('hands', 'sep'), ('oracle', 'sep'), ('oracle', 'hands'),
                 ('hands_side', 'head0'), ('oracle_side', 'head0'), ('oracle_side', 'hands_side')]:
        dd = boot[a] - boot[b]
        e = ev_of(one, a) - ev_of(one, b)
        lo, hi = np.percentile(dd, [2.5, 97.5])
        print(f'  {a:7s} - {b:7s} {e:+.4f} [{lo:+.4f}, {hi:+.4f}]')

    def sub(mask, title):
        gg = g[mask]
        sst = ((gg - gg.mean()) ** 2).sum()
        cells = ' '.join(f'{k} {1 - ((gg - d[k][mask]) ** 2).sum() / sst:6.3f}' for k in CRITICS)
        print(f'  {title:28s} n {mask.sum():>8,}  {cells}')

    act = d['action']
    print('\nBy decision (EV within each subset):')
    sub(act <= 36, 'discard')
    sub(act == 37, 'riichi')
    sub((act >= 38) & (act <= 42), 'chi / pon / kan')
    sub(act == 43, 'agari')
    sub(act == 45, 'pass')
    turn = meta[:, 3]
    print('\nBy turn (junme):')
    for lo, hi in [(0, 5), (6, 10), (11, 14), (15, 99)]:
        sub((turn >= lo) & (turn <= hi), f'turn {lo}-{hi}')
    vc = f'{out}/val_curve.json'
    if path.exists(vc):
        curve = json.load(open(vc))
        print('\nValidation curves (step 0 is the start, equal to head0):')
        print('  step   ' + ' '.join(f'{k:>11s}' for k in CRITICS))
        for c in curve:
            print(f'  {c["step"]:6,} ' + ' '.join(f'{c[k]:11.4f}' for k in CRITICS))
        best = {}
        for k in TRAINED:
            i = max(range(len(curve)), key=lambda j: curve[j][k])
            best[k] = (curve[i]['step'], curve[i][k])
            last3 = [c[k] for c in curve[-3:]]
            rising = i == len(curve) - 1 and last3[0] < last3[1] < last3[2]
            print(f'  {k:11s} best val {curve[i][k]:.4f} at step {curve[i]["step"]:,}'
                  + ('  << still rising at the last look: its EV is a lower bound' if rising else '')
                  + ('  (the start: training never beat it)' if curve[i]['step'] == 0 else ''))
        print('\nVerdict critics (in each family the variant is chosen on validation, then read on test):')
        for fam, ks in FAMILIES.items():
            k = max(ks, key=lambda x: best[x][1])
            print(f'  {fam:7s} {k:11s} test EV {ev_of(one, k):.4f}')


if __name__ == '__main__':
    cmd, out = sys.argv[1], sys.argv[2]
    if cmd == 'split':
        cmd_split(out)
    elif cmd == 'bench':
        cmd_train(out, bench_n=int(sys.argv[3]))
    elif cmd == 'train':
        cmd_train(out)
    elif cmd == 'smoke':
        cmd_smoke(out)
    elif cmd == 'report':
        cmd_report(out)
    else:
        sys.exit(__doc__)
