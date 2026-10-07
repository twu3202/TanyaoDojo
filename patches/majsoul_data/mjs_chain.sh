#!/bin/bash
# New-data arm (mjs: Tenhou + Mahjong Soul throne/jade, v11 recipe) evaluation chain on the 5060 Ti, after
# symaug_run.sh (NAME=mjs) writes SYMAUG_TRAIN_DONE (2,255,200 steps):
#   1. weights-only copy of mjs_final.pth
#   2. sacred eval s14000 8k + s10000 4k, each with diag_gap / diag_place / diag_bust, paired with v11 and B1
#      on the same seeds (symaug_pair.py with SYMTAG=mjs_final) -> 12k stop gate
#   3. unless stopped: s11000 12k + s16000-34999 76k -> 100k pairing
# Pre-registered (patches/majsoul_data/README.md): 12k mjs - v11 < 0 stops; 100k >= +0.5 with z >= 2 passes.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/mjs/mjs_chain.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; SA=$R/mjs; P=$R/symaug/pair; D=$R/evaldump
PY=$B/.venv/bin/python; LOG=$SA/chain.log
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
echo $$ > $SA/chain.pid
say "eval chain waiting for training to finish"
# the marker is written by symaug_run.sh as the last word of its line; never write it from here
until grep -q " SYMAUG_TRAIN_DONE$" $SA/run.log 2>/dev/null; do sleep 300; done
mkdir -p $R/done_mjs $R/gapdump $R/placedump $R/bustdump $D
ck=$SA/mjs_final_infer.pth; tag=mjs_final
$PY - >> $LOG 2>&1 <<PY
import torch
st = torch.load("$SA/mjs_final.pth", weights_only=True, map_location="cpu")
torch.save({k: st[k] for k in ("mortal", "current_dqn", "aux_net", "config", "steps")}, "$ck")
print("infer copy, steps", st["steps"])
PY
say "eval start, md5 $(md5sum $ck | cut -c1-8)"
cd /mnt/d/Better_mortal/jax_rl/mjai_bot
seg() {
  local s=$1 it=$2; local hi=$((s + it*100 - 1))
  [ -f $R/done_mjs/s$s ] && return 0
  local L=$R/eval_${tag}_s${s}_gpufp32; rm -rf "$L"; mkdir -p "$L"
  $PY run_eval.py "$ck" --challenger-type mortal --challenger-name "$tag" --device cuda:0 --amp off \
      --games 400 --iters $it --seed-start $s --log-dir "$L" < /dev/null > "$L.out" 2>&1
  $PY group_ci.py --log-dir "$L" --seed-lo $s --seed-hi $hi --dump "$D/${tag}_s${s}_gpufp32.npz" < /dev/null > "$L.ci" 2>&1
  for t in gap place bust; do
    $PY diag_$t.py --log-dir "$L" --seed-lo $s --seed-hi $hi --expect-games $((it*400)) \
        --dump "$R/${t}dump/${tag}_s${s}.json" < /dev/null > "$L.$t" 2>&1
  done
  if grep -a -q "正确口径" "$L.ci"; then cp "$D/${tag}_s${s}_gpufp32.npz" $P/; touch $R/done_mjs/s$s; say "s$s-$hi done"
  else say "s$s group_ci FAILED, see $L.ci"; fi
}
seg 14000 20
seg 10000 10
SYMTAG=$tag $PY $P/symaug_pair.py 10000 10999 14000 15999 > $SA/mjs_12k.txt 2>&1
say "12k: $(tr '\n' ' ' < $SA/mjs_12k.txt)"
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
  say "12k gate: could not compute mjs - v11 on 3,000 groups; stopping for a manual look"; say MJS_CHAIN_STOPPED; exit 1
fi
if awk -v d="$D12" 'BEGIN{exit !(d < 0)}'; then
  say "12k gate: mjs - v11 = $D12 < 0 -> futility stop (pre-registered), not extending to 100k"
  say MJS_CHAIN_DONE; exit 0
fi
say "12k gate passed (mjs - v11 = $D12 >= 0); extending to 100k"
for sg in "11000 30" "16000 40" "20000 50" "25000 50" "30000 50"; do seg $sg; done
SYMTAG=$tag $PY $P/symaug_pair.py 10000 34999 > $SA/mjs_100k.txt 2>&1
say "100k: $(tr '\n' ' ' < $SA/mjs_100k.txt)"
say MJS_CHAIN_DONE
