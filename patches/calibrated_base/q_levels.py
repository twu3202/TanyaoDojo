"""Q levels on logged Houou decisions (patches/calibrated_base/README.md, "Online result").

    python q_levels.py --files 'GLOB' --n-files N NAME=PATH [NAME=PATH ...]

Per model, over decisions with at least two legal actions: mean Q of the greedy action, of the logged human action, of
the best untaken action (the second-best) and of all other legal actions; and the same split by calls available.
Tests the collapse explanation: online training pulls the greedy action down onto untaken actions that did not move.
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
p.add_argument("--n-files", type=int, default=10)
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
call = masks[:, 38:43].any(1)
print(f"{len(files)} files, {multi.sum():,} decisions with >= 2 legal actions ({(multi & call).sum():,} with a call available)")
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
    qm = np.where(masks, q, np.nan)
    srt = np.sort(np.where(masks, q, -np.inf), axis=1)
    greedy, second = srt[:, -1], srt[:, -2]
    human = q[np.arange(len(q)), acts]
    others = (np.nansum(qm, 1) - greedy) / (masks.sum(1) - 1)
    def row(sel):
        return (f"greedy {greedy[sel].mean():+.3f}  human {human[sel].mean():+.3f}  second {second[sel].mean():+.3f}  "
                f"mean of the rest {others[sel].mean():+.3f}")
    print(f"  {name:<8} all:  {row(multi)}")
    print(f"  {'':<8} call: {row(multi & call)}")
