#!/bin/bash
# symaug evaluation chain on the 5060 Ti, after symaug_run.sh writes SYMAUG_TRAIN_DONE (1.22M steps):
#   1. weights-only copy of symaug_final.pth
#   2. sacred eval s14000 8k + s10000 4k, each with diag_gap / diag_place / diag_bust, paired with v11 and B1
#      on the same seeds (symaug_pair.py) -> 12k stop gate
#   3. D1 symmetry readout on the s14000 games (rotation-only dragon group, the one trained on): symaug vs v11
#   4. unless stopped: s11000 12k + s16000-34999 76k -> 100k pairing
# Pre-registered (patches/symmetry_augmentation/README.md): 12k symaug - v11 < 0 stops; 100k >= +0.5 with z >= 2 passes.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/symaug/symaug_chain.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; SA=$R/symaug; P=$SA/pair; D=$R/evaldump
PY=$B/.venv/bin/python; LOG=$SA/chain.log
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
echo $$ > $SA/chain.pid
say "eval chain waiting for training to finish"
# the marker is written by symaug_run.sh as the last word of its line; never write it from here
until grep -q " SYMAUG_TRAIN_DONE$" $SA/run.log 2>/dev/null; do sleep 300; done
mkdir -p $R/done_symaug $R/gapdump $R/placedump $R/bustdump $D
ck=$SA/symaug_final_infer.pth; tag=symaug_final
$PY - >> $LOG 2>&1 <<PY
import torch
st = torch.load("$SA/symaug_final.pth", weights_only=True, map_location="cpu")
torch.save({k: st[k] for k in ("mortal", "current_dqn", "aux_net", "config", "steps")}, "$ck")
print("infer copy, steps", st["steps"])
PY
say "eval start, md5 $(md5sum $ck | cut -c1-8)"
cd /mnt/d/Better_mortal/jax_rl/mjai_bot
seg() {
  local s=$1 it=$2; local hi=$((s + it*100 - 1))
  [ -f $R/done_symaug/s$s ] && return 0
  local L=$R/eval_${tag}_s${s}_gpufp32; rm -rf "$L"; mkdir -p "$L"
  $PY run_eval.py "$ck" --challenger-type mortal --challenger-name "$tag" --device cuda:0 --amp off \
      --games 400 --iters $it --seed-start $s --log-dir "$L" < /dev/null > "$L.out" 2>&1
  $PY group_ci.py --log-dir "$L" --seed-lo $s --seed-hi $hi --dump "$D/${tag}_s${s}_gpufp32.npz" < /dev/null > "$L.ci" 2>&1
  for t in gap place bust; do
    $PY diag_$t.py --log-dir "$L" --seed-lo $s --seed-hi $hi --expect-games $((it*400)) \
        --dump "$R/${t}dump/${tag}_s${s}.json" < /dev/null > "$L.$t" 2>&1
  done
  if grep -a -q "正确口径" "$L.ci"; then cp "$D/${tag}_s${s}_gpufp32.npz" $P/; touch $R/done_symaug/s$s; say "s$s-$hi done"
  else say "s$s group_ci FAILED, see $L.ci"; fi
}
seg 14000 20
seg 10000 10
$PY $P/symaug_pair.py 10000 10999 14000 15999 > $SA/symaug_12k.txt 2>&1
say "12k: $(tr '\n' ' ' < $SA/symaug_12k.txt)"
(cd /home/r/a3 && MORTAL_DIR=$B/Mortal/mortal RIICHI_EVAL_LIBRIICHI=/home/r/riichi-eval-build/pylib_1002 \
  RIICHI_EVAL_SRC=/home/r/riichi-eval/src $PY -W ignore symmetry_check.py --log-dir $R/eval_${tag}_s14000_gpufp32 \
  --games 300 --chunk 10 --dragon-perms rot --model symaug=$ck --model v11=$SA/v11_best.pth < /dev/null > $SA/symaug_d1.txt 2>&1)
say "D1 symmetry readout -> $SA/symaug_d1.txt"
D12=$($PY - <<PY
import glob, numpy as np
def load(pat):
    m = {}
    for f in sorted(glob.glob("$P/" + pat)):
        z = np.load(f)
        for s, v in zip(z["seed"], z["mean"]):
            m.setdefault(int(s), float(v))
    return m
a, b = load("${tag}_s*_gpufp32.npz"), load("v11_manual_s*.npz")
c = [s for s in a if s in b and (10000 <= s <= 10999 or 14000 <= s <= 15999)]
print(f"{np.mean([a[s] - b[s] for s in c]):.4f}" if len(c) == 3000 else "")
PY
)
if [ -z "$D12" ]; then
  say "12k gate: could not compute symaug - v11 on 3,000 groups; stopping for a manual look"; say SYMAUG_CHAIN_STOPPED; exit 1
fi
if awk -v d="$D12" 'BEGIN{exit !(d < 0)}'; then
  say "12k gate: symaug - v11 = $D12 < 0 -> futility stop (pre-registered), not extending to 100k"
  say SYMAUG_CHAIN_DONE; exit 0
fi
say "12k gate passed (symaug - v11 = $D12 >= 0); extending to 100k"
for sg in "11000 30" "16000 40" "20000 50" "25000 50" "30000 50"; do seg $sg; done
$PY $P/symaug_pair.py 10000 34999 > $SA/symaug_100k.txt 2>&1
say "100k: $(tr '\n' ' ' < $SA/symaug_100k.txt)"
say SYMAUG_CHAIN_DONE
