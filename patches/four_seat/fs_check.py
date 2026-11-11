"""Four-seat self-play: the checks before any training (patches/four_seat/DESIGN.md, "Open before pre-registration").

Run in Mortal/mortal with the patched player.py, MORTAL_CFG = a copy of the control arm's config whose train_play
log_dir points at OUT and whose train_play games is GAMES:
    FOUR_SEAT=1 python fs_check.py OUT STATE

1. One worker round of four-seat self-play from STATE (A80k), exactly as client.py plays it.
2. All four seats load: the trainer's reader (player_names = ['trainee']) returns four trajectories per file.
3. Samples per file, against the 1v3 control's 157.4.
4. The champion engine follows the trainee's parameters: it wraps the same module objects.
5. How far the four rotations of a seed diverge: they are four rollouts of one deal by one policy, different only by
   sampling (T 0.1, eps 0.005).
"""
import gzip
import json
import os
import sys
import time
from collections import defaultdict
from os import path

import numpy as np
import torch

OUT, STATE = sys.argv[1], sys.argv[2]
assert os.environ.get('FOUR_SEAT') == '1'

from config import config                     # noqa: E402
from model import Brain, DQN                  # noqa: E402
from player import TrainPlayer                # noqa: E402
from libriichi.dataset import GameplayLoader  # noqa: E402

device = torch.device('cuda')
s = torch.load(STATE, weights_only=True, map_location='cpu')
rn = s['config']['resnet']
mortal = Brain(version=4, **rn).to(device).eval()
dqn = DQN(version=4).to(device)
mortal.load_state_dict(s['mortal'])
dqn.load_state_dict(s['current_dqn'])

tp = TrainPlayer()
assert tp.four_seat
t0 = time.time()
rankings, files = tp.train_play(mortal, dqn, device)
print(f'round: {len(files)} files in {time.time() - t0:.0f} s; challenger rankings {list(rankings)}')

# 2 and 3: the trainer's reader
loader = GameplayLoader(version=4, oracle=False, player_names=['trainee'], excludes=[])
per_file, seats = [], []
for f in files:
    games = loader.load_gz_log_files([f])[0]
    seats.append(len(games))
    per_file.append(sum(len(g.take_actions()) for g in games))
print(f'trajectories per file: {sorted(set(seats))} (expect [4]); trainee samples per file {np.mean(per_file):.1f} '
      f'(+- {1.96 * np.std(per_file, ddof=1) / np.sqrt(len(per_file)):.1f}); the 1v3 control has 157.4')
assert set(seats) == {4}, 'not all four seats load'

# 4: the champion engine wraps the trainee's own modules (checked on a fresh pair of engines built the same way)
from engine import MortalEngine               # noqa: E402
champ = MortalEngine(mortal, dqn, is_oracle=False, version=4, boltzmann_epsilon=tp.boltzmann_epsilon,
                     boltzmann_temp=tp.boltzmann_temp, top_p=tp.top_p, device=device, enable_amp=True, name='trainee')
assert champ.brain is mortal and champ.dqn is dqn, 'the champion engine holds copies, not the trainee modules'
print('champion engine wraps the trainee modules: parameter updates reach all four seats')

# 5: divergence of the four rotations of each seed
def events(f):
    with gzip.open(f, 'rt') as fh:
        return [json.loads(x) for x in fh]

groups = defaultdict(dict)
for f in files:
    idx, key, side = path.basename(f)[:-len('.json.gz')].split('_')[-3:]
    groups[(idx, key)][side] = events(f)
first_div, same_game, kyoku_same, kyoku_n = [], 0, 0, 0
for g in groups.values():
    rot = [g[k] for k in sorted(g)]
    a = rot[0]
    for b in rot[1:]:
        n = min(len(a), len(b))
        d = next((i for i in range(n) if a[i] != b[i]), n)
        first_div.append(d / len(a))
        same_game += a == b
        # kyoku by kyoku: the same result (deltas of hora / ryukyoku) in the same kyoku index
        ka = [e.get('deltas') for e in a if e['type'] in ('hora', 'ryukyoku')]
        kb = [e.get('deltas') for e in b if e['type'] in ('hora', 'ryukyoku')]
        for x, y in zip(ka, kb):
            kyoku_n += 1
            kyoku_same += x == y
first_div = np.array(first_div)
print(f'rotation pairs (vs rotation a): {len(first_div)} in {len(groups)} seeds; identical games {same_game}')
print(f'first divergence, as a share of the game: median {np.median(first_div):.3f}, '
      f'quartiles {np.percentile(first_div, 25):.3f} / {np.percentile(first_div, 75):.3f}, '
      f'within the first 10% {np.mean(first_div < 0.1):.0%}')
print(f'kyoku results equal in the same kyoku index: {kyoku_same / max(kyoku_n, 1):.1%} of {kyoku_n}')
print('FS_CHECK_DONE')
