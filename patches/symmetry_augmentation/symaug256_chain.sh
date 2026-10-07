#!/bin/bash
# symaug256 evaluation chain on trx (runs without the Mac), after symaug_run.sh (NAME=symaug256) writes SYMAUG_TRAIN_DONE:
#   1. weights-only copy of symaug256_final.pth
#   2. 12k vs v4 on the evaluation station (s10000 4k + s14000 8k, with diag_gap / place / bust), paired with B1 (its
#      unaugmented twin) and v11 on the same seeds -> stop gate: symaug256 - B1 < 0
#   3. unless stopped: s11000 12k + s16000-34999 76k -> 100k pairing (the seeds of the v11 and B1 100k)
# Pre-registered: patches/symmetry_augmentation/README.md. Log: runs/symaug256/chain.log
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/symaug256/symaug256_chain.sh > /dev/null 2>&1 &)
B=/home/twu/Projects/better_mortal; R=$B/runs/symaug256; E=/home/twu/evalsta; PY=$B/.venv/bin/python
LOG=$R/chain.log; TAG=symaug256
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
a, b = load("${TAG}_final_s*_gpufp32.npz"), load("b1_b1final_s*_gpufp32.npz")
c = [s for s in a if s in b and (10000 <= s <= 10999 or 14000 <= s <= 15999)]
print(f"{np.mean([a[s] - b[s] for s in c]):.4f}" if len(c) == 3000 else "")
PY
)
if [ -z "$D12" ]; then
  say "12k gate: could not compute ${TAG} - B1 on 3,000 groups; stopping for a manual look"; say SYMAUG256_CHAIN_STOPPED; exit 1
fi
if awk -v d="$D12" 'BEGIN{exit !(d < 0)}'; then
  say "12k gate: ${TAG} - B1 = $D12 < 0 -> futility stop (pre-registered), not extending to 100k"
  say SYMAUG256_CHAIN_DONE; exit 0
fi
say "12k gate passed (${TAG} - B1 = $D12 >= 0); extending to 100k"
bash $E/srv_evalone.sh final $ck $TAG 11000 30
bash $E/srv_evalone.sh final $ck $TAG 16000 190
pair 10000 34999 > $R/${TAG}_100k.txt 2>&1
say "100k: $(tr '\n' ' ' < $R/${TAG}_100k.txt)"
say SYMAUG256_CHAIN_DONE
