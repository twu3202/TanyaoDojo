"""Stage 1 smoke: play a few games with the frontier-exploration trainee, then check the loader alignment.

    MORTAL_CFG=... python test_explore_run.py OUT_DIR SEED_COUNT TRAINEE.pth BASE.pth [extra log dirs to align ...]
"""
import glob
import gzip
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/twu/Projects/better_mortal/Mortal/mortal")
from model import Brain, DQN  # noqa: E402
from engine import MortalEngine  # noqa: E402
from explore_engine import FrontierExploreEngine, KINDS  # noqa: E402
from libriichi.arena import OneVsThree  # noqa: E402
from libriichi.dataset import GameplayLoader  # noqa: E402
from dataloader import explore_keep_mask  # noqa: E402

PT = 2.390 / 90


def load(path, dev):
    st = torch.load(path, weights_only=True, map_location="cpu")
    c = st["config"]
    b = Brain(version=4, conv_channels=c["resnet"]["conv_channels"], num_blocks=c["resnet"]["num_blocks"]).to(dev).eval()
    q = DQN(version=4).to(dev).eval()
    b.load_state_dict(st["mortal"])
    q.load_state_dict(st["current_dqn"])
    return b, q


def align(log_dir, name):
    loader = GameplayLoader(version=4, oracle=False, player_names=[name])
    n_games = mis = n_ent = n_drop = n_ng = 0
    for f in sorted(glob.glob(log_dir + "/*.json.gz")):
        lines = gzip.open(f, "rt").read().splitlines()
        for game in loader.load_gz_log_files([f])[0]:
            pid = game.take_player_id()
            acts, atk = game.take_actions(), game.take_at_kyoku()
            keep = explore_keep_mask(lines, pid, acts, atk)
            n_games += 1
            n_ent += len(acts)
            n_ng += sum(1 for ln in lines if '"is_greedy":false' in ln.replace(" ", "")
                        and json.loads(ln).get("actor") == pid)
            if keep is None:
                mis += 1
            else:
                n_drop += int((~keep).sum())
    print(f"  {log_dir} [{name}]: {n_games} games, {mis} misaligned, non-greedy events {n_ng}, "
          f"dropped {n_drop:,}/{n_ent:,} entries ({n_drop / max(1, n_ent):.2%})")
    return mis


def main():
    out, seed_count, trainee, base = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    dev = torch.device("cuda:0")
    tb, tq = load(trainee, dev)
    bb, bq = load(base, dev)
    chal = FrontierExploreEngine(tb, tq, is_oracle=False, version=4, boltzmann_epsilon=0.005, boltzmann_temp=0.1,
                                 top_p=1.0, device=dev, enable_amp=True, name="trainee",
                                 explore_margin=10 * PT, explore_prob=0.3)
    champ = MortalEngine(bb, bq, is_oracle=False, version=4, device=dev, enable_amp=True,
                         enable_rule_based_agari_guard=True, name="baseline")
    if os.path.isdir(out):
        shutil.rmtree(out)
    env = OneVsThree(disable_progress_bar=True, log_dir=out)
    rk = env.py_vs_py(challenger=chal, champion=champ, seed_start=(10000, 0x5EED1), seed_count=seed_count)
    rk = np.array(rk)
    n = rk.sum()
    el, ex = chal.stats
    print(f"{n} games, trainee rankings {rk.tolist()} ({rk @ np.array([90, 45, 0, -135]) / n:+.2f} pt)")
    print("  eligible/explored per game: " + ", ".join(f"{k} {e / n:.2f}/{x / n:.2f}" for k, e, x in zip(KINDS, el, ex)))
    bad = align(out, "trainee")
    for d in sys.argv[5:]:
        for nm in ("trainee", "mortal"):
            if glob.glob(d + "/*.json.gz"):
                bad += align(d, nm)
    print("ALIGNMENT OK" if bad == 0 else f"ALIGNMENT FAILURES: {bad}")


if __name__ == "__main__":
    main()
