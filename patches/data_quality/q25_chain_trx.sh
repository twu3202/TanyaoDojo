#!/bin/bash
# q25 evaluation chain on trx (runs without the Mac), after symaug_run.sh (NAME=q25) writes SYMAUG_TRAIN_DONE:
#   1. weights-only copy of q25_final.pth
#   2. 12k vs v4 on the evaluation station (s10000 4k + s14000 8k, with diag_gap / place / bust), paired with v11
#      on the same seeds -> stop gate: q25 - v11 < 0
#   3. unless stopped: s11000 12k + s16000-34999 76k -> 100k pairing with v11
# q25 = the data-quality arm (README v6/v7): the best 25% of houou games by the players' own regret, v11 recipe.
# Pre-registered gates as B1: 12k < 0 stops; 100k >= +0.5 with z >= 2 says selecting by player quality helps.
# Same chain as q25_chain.sh (5060 Ti), on the evaluation station; only one of the two may run.
# Log: runs/q25/chain.log
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/q25/q25_chain_trx.sh > /dev/null 2>&1 &)
B=/home/twu/Projects/better_mortal; R=$B/runs/q25; E=/home/twu/evalsta; PY=$B/.venv/bin/python
LOG=$R/chain.log; TAG=q25
say() { echo "$(date +%m-%d_%H:%M:%S) $*" >> $LOG; }
pair() { PAIR_DIR=$E/dump SYMTAG=${TAG}_final $PY $R/symaug_pair.py "$@"; }
echo $$ > $R/chain.pid
say "eval chain waiting for training to finish"
# the marker is written by symaug_run.sh as the last word of its line; never write it from here
until grep -q " SYMAUG_TRAIN_DONE$" $R/run.log 2>/dev/null; do sleep 300; done
ck=$R/${TAG}_final_infer.pth
$PY - >> $LOG 2>&1 <<PY
import torch
st = torch.load("$R/${TAG}_final.pth", weights_only=True, map_location="cpu")
torch.save({k: st[k] for k in ("mortal", "current_dqn", "aux_net", "config", "steps")}, "$ck")
print("infer copy, steps", st["steps"])
PY
say "eval start, md5 $(md5sum $ck | cut -c1-8)"
bash $E/srv_evalone.sh final $ck $TAG 10000 10
bash $E/srv_evalone.sh final $ck $TAG 14000 20
pair 10000 10999 14000 15999 > $R/${TAG}_12k.txt 2>&1
say "12k: $(tr '\n' ' ' < $R/${TAG}_12k.txt)"
D12=$($PY - <<PY
import glob, numpy as np
def load(pat):
    m = {}
    for f in sorted(glob.glob("$E/dump/" + pat)):
        z = np.load(f)
        for s, v in zip(z["seed"], z["mean"]):
            m.setdefault(int(s), float(v))
    return m
a, b = load("${TAG}_final_s*_gpufp32.npz"), load("v11_manual_s*.npz")
c = [s for s in a if s in b and (10000 <= s <= 10999 or 14000 <= s <= 15999)]
print(f"{np.mean([a[s] - b[s] for s in c]):.4f}" if len(c) == 3000 else "")
PY
)
if [ -z "$D12" ]; then
  say "12k gate: could not compute ${TAG} - v11 on 3,000 groups; stopping for a manual look"; say Q25_CHAIN_STOPPED; exit 1
fi
if awk -v d="$D12" 'BEGIN{exit !(d < 0)}'; then
  say "12k gate: ${TAG} - v11 = $D12 < 0 -> futility stop (pre-registered), not extending to 100k"
  say Q25_CHAIN_DONE; exit 0
fi
say "12k gate passed (${TAG} - v11 = $D12 >= 0); extending to 100k"
bash $E/srv_evalone.sh final $ck $TAG 11000 30
bash $E/srv_evalone.sh final $ck $TAG 16000 190
pair 10000 34999 > $R/${TAG}_100k.txt 2>&1
say "100k: $(tr '\n' ' ' < $R/${TAG}_100k.txt)"
say Q25_CHAIN_DONE
