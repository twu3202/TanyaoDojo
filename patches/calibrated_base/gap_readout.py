"""Q-gap readout for offline bases (patches/calibrated_base/README.md).

    python gap_readout.py --files 'GLOB' --n-files 60 --ref NAME=PATH [NAME=PATH ...]

Loads the last N files of a sorted glob of Houou logs (all four seats, as offline training does), runs every model on
the same decisions and reports, per model:
- the median gap between the best and the second-best legal action's Q (decisions with at least two legal actions),
  overall and by decision type: riichi available, a call available (chi/pon/kan), discard-only;
- agreement of the greedy action with the logged human action;
- the share of decisions whose greedy action differs from the first model's (the reference).
Gaps are in the model's own Q units, so compare models trained on the same pts only.
Runs on CPU by default (the card may be evaluating); set DEVICE=cuda:0 to use the GPU.
"""
import argparse
import os
from glob import glob

import numpy as np
import torch

from libriichi.dataset import GameplayLoader
from model import Brain, DQN

p = argparse.ArgumentParser()
p.add_argument("--files", required=True)
p.add_argument("--n-files", type=int, default=60)
p.add_argument("models", nargs="+")
a = p.parse_args()
dev = torch.device(os.environ.get("DEVICE", "cpu"))
torch.set_num_threads(int(os.environ.get("THREADS", "4")))

files = sorted(glob(a.files))[-a.n_files:]
loader = GameplayLoader(version=4, oracle=False, player_names=[], augmented=False)
obs, masks, acts = [], [], []
for f in files:
    for game in loader.load_gz_log_files([f])[0]:
        obs.append(np.asarray(game.take_obs(), dtype=np.float32))
        masks.append(np.asarray(game.take_masks(), dtype=bool))
        acts.append(np.asarray(game.take_actions(), dtype=np.int64))
obs = np.concatenate(obs); masks = np.concatenate(masks); acts = np.concatenate(acts)
multi = masks.sum(1) >= 2
riichi = masks[:, 37]
call = masks[:, 38:43].any(1)
discard_only = ~riichi & ~call & ~masks[:, 43:].any(1)
print(f"{len(files)} files, {len(obs):,} decisions, {multi.sum():,} with >= 2 legal actions "
      f"(riichi available {(multi & riichi).sum():,}, call available {(multi & call).sum():,}, "
      f"discard-only {(multi & discard_only).sum():,})")

ref_greedy = None
for spec in a.models:
    name, path = spec.split("=", 1)
    st = torch.load(path, weights_only=True, map_location="cpu")
    res = st["config"]["resnet"]
    mortal = Brain(version=4, conv_channels=res["conv_channels"], num_blocks=res["num_blocks"]).to(dev).eval()
    dqn = DQN(version=4).to(dev).eval()
    mortal.load_state_dict(st["mortal"]); dqn.load_state_dict(st["current_dqn"])
    qs = []
    with torch.inference_mode():
        for i in range(0, len(obs), 1024):
            o = torch.from_numpy(obs[i:i + 1024]).to(dev); m = torch.from_numpy(masks[i:i + 1024]).to(dev)
            qs.append(dqn(mortal(o), m).float().cpu().numpy())
    q = np.concatenate(qs)
    greedy = q.argmax(1)
    top2 = np.sort(np.where(masks, q, -np.inf), axis=1)[:, -2:]
    gap = top2[:, 1] - top2[:, 0]
    med = lambda sel: float(np.median(gap[multi & sel])) if (multi & sel).any() else float("nan")
    allsel = np.ones(len(q), dtype=bool)
    agree = float((greedy == acts).mean())
    flip = float((greedy != ref_greedy).mean()) if ref_greedy is not None else 0.0
    if ref_greedy is None:
        ref_greedy = greedy
    print(f"  {name:<14} steps {st.get('steps')}  median top-2 gap {med(allsel):.3f} (riichi {med(riichi):.3f}, "
          f"call {med(call):.3f}, discard-only {med(discard_only):.3f}); p90 {np.percentile(gap[multi], 90):.3f}; "
          f"agrees with the human {100 * agree:.2f}%; greedy differs from the reference {100 * flip:.2f}%")
