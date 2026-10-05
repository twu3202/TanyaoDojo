"""Unit check for symaug (run from the Mortal tree that holds symaug.py and libriichi.so).

For N random houou logs and each of the 18 symmetries, the relabelled log is parsed by libriichi and compared with
the plain load (load_gz_log_files, the path training uses without augmentation):
  - identity through load_log == load_gz_log_files (bit-identical observations);
  - relabel is invertible, and the log has no tile strings the regex misses;
  - same players, same number of samples, same kyoku index / dones / gamma flags, same GRP features and final scores
    (rewards are unchanged);
  - actions and masks are the original ones under the action permutation;
  - per observation row: is it the original row with the tile columns permuted? (rows that are not, e.g. red-five
    rows, are listed; they are encoded by libriichi from the relabelled game, so they are correct either way.)
Also times the plain and the augmented load.

    python symaug_check.py --data ~/Projects/better_mortal/data/houou --files 200
"""
import argparse
import collections
import glob
import gzip
import itertools
import json
import os
import random
import time

import numpy as np
from libriichi.dataset import GameplayLoader

import symaug


def perm_maps(g):
    sp = list(itertools.permutations(symaug.SUITS))[g // 3]
    k = g % 3
    smap = dict(zip(symaug.SUITS, sp))
    col = np.arange(34)
    for si, s in enumerate(symaug.SUITS):
        col[si * 9:(si + 1) * 9] = symaug.SUITS.index(smap[s]) * 9 + np.arange(9)
    for i in range(3):
        col[31 + i] = 31 + (i + k) % 3
    act = np.arange(46)
    act[:34] = col
    for si, s in enumerate(symaug.SUITS):
        act[34 + si] = 34 + symaug.SUITS.index(smap[s])
    return col, act


def take(gp):
    grp = gp.take_grp()
    return dict(
        obs=np.stack(gp.take_obs()), actions=np.array(gp.take_actions()), masks=np.stack(gp.take_masks()),
        at_kyoku=np.array(gp.take_at_kyoku()), dones=np.array(gp.take_dones()),
        apply_gamma=np.array(gp.take_apply_gamma()), player_id=gp.take_player_id(),
        grp_feature=grp.take_feature(), rank=np.array(grp.take_rank_by_player()), final=np.array(grp.take_final_scores()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--files', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20261005)
    args = ap.parse_args()
    files = []
    for y in range(2019, 2027):
        files += glob.glob(os.path.join(args.data, str(y), '*.mjson*'))
    files.sort()
    random.Random(args.seed).shuffle(files)
    files = files[:args.files]

    inv = {}
    for g in range(symaug.N_ELEMENTS):
        col, _ = perm_maps(g)
        for h in range(symaug.N_ELEMENTS):
            if (perm_maps(h)[0][col] == np.arange(34)).all():
                inv[g] = h
    assert len(inv) == symaug.N_ELEMENTS

    # every quoted short string in the logs: anything tile-like the regex does not cover would be left unpermuted
    short = collections.Counter()
    loader = GameplayLoader(version=4, oracle=False, player_names=[], excludes=[])
    bad_rows = collections.Counter()
    n_dec = 0
    t_plain = t_aug = 0.0
    for fi, f in enumerate(files):
        t0 = time.time()
        ref = [take(gp) for gp in loader.load_gz_log_files([f])[0]]
        t_plain += time.time() - t0
        with gzip.open(f, 'rt', encoding='utf-8') as fh:
            raw = fh.read()
        for line in raw.splitlines():
            ev = json.loads(line)
            stack = [v for k, v in ev.items() if k not in ('type', 'names')]
            while stack:
                v = stack.pop()
                if isinstance(v, list):
                    stack.extend(v)
                elif isinstance(v, str):
                    short[v] += 1
        ident = [take(gp) for gp in loader.load_log(raw)]
        assert len(ident) == len(ref)
        for a, b in zip(ref, ident):
            for key in a:
                assert np.array_equal(a[key], b[key]), (f, 'identity', key)
        n_dec += sum(len(r['actions']) for r in ref)
        for g in range(1, symaug.N_ELEMENTS):
            col, act = perm_maps(g)
            t0 = time.time()
            rl = symaug.relabel(raw, g)
            out = [take(gp) for gp in loader.load_log(rl)]
            t_aug += (time.time() - t0) / (symaug.N_ELEMENTS - 1)
            assert symaug.relabel(rl, inv[g]) == raw, (f, g, 'inverse')
            assert len(out) == len(ref), (f, g)
            for a, b in zip(ref, out):
                assert a['player_id'] == b['player_id']
                assert len(a['actions']) == len(b['actions']), (f, g, 'length')
                for key in ('at_kyoku', 'dones', 'apply_gamma', 'grp_feature', 'rank', 'final'):
                    assert np.array_equal(a[key], b[key]), (f, g, key)
                assert np.array_equal(act[a['actions']], b['actions']), (f, g, 'actions')
                assert np.array_equal(b['masks'][:, act], a['masks']), (f, g, 'masks')
                same = (b['obs'][:, :, col] == a['obs']).all(axis=(0, 2))
                for r in np.flatnonzero(~same):
                    bad_rows[int(r)] += 1
        if fi % 50 == 0:
            print(f'{fi + 1}/{len(files)} files ok', flush=True)

    tiles = {f'{n}{s}' for n in '123456789' for s in 'mps'} | {f'5{s}r' for s in 'mps'} | set('ESWNPFC') | {'?'}
    others = collections.Counter({k: v for k, v in short.items() if k not in tiles})
    print(f'\n{len(files)} logs, {n_dec:,} decisions x 17 non-trivial symmetries: rewards / lengths / actions / masks all match')
    print(f'tile strings seen: {sum(v for k, v in short.items() if k in tiles):,}; other string values (should be none '
          f'besides winds already counted as tiles): {dict(others.most_common(30))}')
    print(f'observation rows that are not a pure column permutation (row: count of (file, g, player) cases): '
          f'{len(bad_rows)} rows -> {sorted(bad_rows)[:60]}{" ..." if len(bad_rows) > 60 else ""}')
    print(f'load time per log: plain {t_plain / len(files) * 1e3:.1f} ms, augmented (relabel + parse) '
          f'{t_aug / len(files) * 1e3:.1f} ms')


if __name__ == '__main__':
    main()
