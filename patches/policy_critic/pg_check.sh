#!/bin/bash
# CPU checks for train_pg.py (DESIGN.md, "Before any GPU time"), in check mode: no server, a fixed set of drained
# games, batch 32, nice 19 so the running arm is not slowed.
#   warm: policy_start_step out of reach, 64 batches -> the policy (mortal, current_dqn) must equal the base bit for
#         bit and only the value head may move
#   pg:   64 warm-up batches then 136 policy-gradient batches -> the policy moves, every diagnostic is finite
#   load: the pg state loads in run_eval.load_champion and answers a react_batch
# Usage: bash pg_check.sh   (train_pg.py must already be in Mortal/mortal); output in ~/pg_check
set -u
B=$HOME/Projects/better_mortal; M=$B/Mortal/mortal; PY=$B/.venv/bin/python; C=$HOME/pg_check
BASE=$B/runs/xfer_v11/archive/xfer_v11_s80000.pth; FILES="$B/runs/xfer_v11r/drain/1*.json.gz"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 RAYON_NUM_THREADS=2
rm -rf $C; mkdir -p $C; cp $BASE $C/base.pth
$PY - $B $C <<'PY'
import sys, toml
B, C = sys.argv[1:]
c = toml.load(f"{B}/configs/online_selfplay_xfer_v11s2.toml")
c["control"].update(device="cpu", enable_amp=False, enable_cudnn_benchmark=False, batch_size=32, save_every=1000,
                    test_every=2000, submit_every=40, state_file=f"{C}/state.pth", best_state_file=f"{C}/best.pth",
                    tensorboard_dir=f"{C}/tb")
c["dataset"].update(num_workers=2, file_index=f"{C}/file_index.pth")
for p in c["train_play"].values():
    p.update(boltzmann_epsilon=1.0, boltzmann_temp=0.1, top_p=1.0, log_dir=f"{C}/train_play")
c["anchor"].update(weight=0.0, state_file=f"{C}/base.pth")
c["pg"] = dict(temp=0.1, clip=0.2, clip_c=3.0, kl_coef=1.0, kl_target=0.02, kl_horizon=2000, ent_coef=0.0,
               value_coef=1.0, value_hidden=256, policy_start_step=10**9, lr_warmup=64)
toml.dump(c, open(f"{C}/check_warm.toml", "w"))
c["pg"]["policy_start_step"] = 80064
toml.dump(c, open(f"{C}/check_pg.toml", "w"))
PY
cd $M
run() {  # run <name> <config> <steps>
  cp $C/base.pth $C/state.pth
  MORTAL_CFG=$C/$2 PG_CHECK_FILES="$FILES" PG_CHECK_STEPS=$3 nice -n 19 $PY train_pg.py > $C/$1.log 2>&1
  echo "$1: exit $?; $(grep -a -oE 'pg (warmup )?at [0-9,]+: .*' $C/$1.log | tail -1)"
  cp $C/state.pth $C/$1_state.pth
}
run warm check_warm.toml 64
run pg check_pg.toml 200
$PY - $C <<'PY'
import sys, torch
C = sys.argv[1]
base = torch.load(f"{C}/base.pth", weights_only=True, map_location="cpu")
for name in ("warm", "pg"):
    s = torch.load(f"{C}/{name}_state.pth", weights_only=True, map_location="cpu")
    diff = {k: sum(float((s[k][t].float() - base[k][t].float()).abs().max()) for t in base[k]) for k in ("mortal", "current_dqn")}
    print(f"{name}: steps {s['steps']:,}; max |param - base| summed per module: "
          + ", ".join(f"{k} {v:.3g}" for k, v in diff.items()) + f"; value head saved: {'value_head' in s}; pg meta {s.get('pg')}")
PY
cd /mnt/d/Better_mortal/jax_rl/mjai_bot
MORTAL_DIR=$M $PY - $C <<'PY'
import sys, numpy as np, run_eval
from libriichi.dataset import GameplayLoader
import glob
C = sys.argv[1]
e = run_eval.load_champion(f"{C}/pg_state.pth", "cpu", "off")
f = sorted(glob.glob("/home/r/Projects/better_mortal/runs/xfer_v11r/drain/1*.json.gz"))[:1]
g = GameplayLoader(version=4, oracle=False, player_names=["trainee"]).load_gz_log_files(f)[0][0]
obs, masks = g.take_obs()[:8], g.take_masks()[:8]
a, q, m, greedy = e.react_batch(list(obs), list(masks), None)
print(f"load: run_eval.load_champion ok; react_batch on 8 states -> actions {a}")
PY
