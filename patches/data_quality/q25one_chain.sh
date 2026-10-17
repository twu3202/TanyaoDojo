#!/bin/bash
# q25one (q25 with one pass instead of about 1.5; README v9) evaluation chain on the 5060 Ti, after symaug_run.sh
# (NAME=q25one) writes SYMAUG_TRAIN_DONE:
#   1. weights-only copy of q25one_final.pth
#   2. sacred eval s14000 8k + s10000 4k, then s11000 12k + s16000-34999 76k = 100k, each with diag_gap /
#      diag_place / diag_bust; no 12k futility stop, because the arm exists for the 100k pairing with q25
#   3. 100k pairings on the same seeds: q25one - v11 (and B1) with symaug_pair.py, q25one - q25 below
# Pre-registered (README v9): q25one - v11 >= +0.5 with z >= 2 passes (the bar of every base arm); q25one - q25 is
# the repetition effect, reported with its CI.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/q25one/q25one_chain.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; SA=$R/q25one; P=$R/symaug/pair; D=$R/evaldump
PY=$B/.venv/bin/python; LOG=$SA/chain.log
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
echo $$ > $SA/chain.pid
say "eval chain waiting for training to finish"
# the marker is written by symaug_run.sh as the last word of its line; never write it from here
until grep -q " SYMAUG_TRAIN_DONE$" $SA/run.log 2>/dev/null; do sleep 300; done
mkdir -p $R/done_q25one $R/gapdump $R/placedump $R/bustdump $D
ck=$SA/q25one_final_infer.pth; tag=q25one_final
$PY - >> $LOG 2>&1 <<PY
import torch
st = torch.load("$SA/q25one_final.pth", weights_only=True, map_location="cpu")
torch.save({k: st[k] for k in ("mortal", "current_dqn", "aux_net", "config", "steps")}, "$ck")
print("infer copy, steps", st["steps"])
PY
say "eval start, md5 $(md5sum $ck | cut -c1-8)"
cd /mnt/d/Better_mortal/jax_rl/mjai_bot
seg() {
  local s=$1 it=$2; local hi=$((s + it*100 - 1))
  [ -f $R/done_q25one/s$s ] && return 0
  local L=$R/eval_${tag}_s${s}_gpufp32; rm -rf "$L"; mkdir -p "$L"
  $PY run_eval.py "$ck" --challenger-type mortal --challenger-name "$tag" --device cuda:0 --amp off \
      --games 400 --iters $it --seed-start $s --log-dir "$L" < /dev/null > "$L.out" 2>&1
  $PY group_ci.py --log-dir "$L" --seed-lo $s --seed-hi $hi --dump "$D/${tag}_s${s}_gpufp32.npz" < /dev/null > "$L.ci" 2>&1
  for t in gap place bust; do
    $PY diag_$t.py --log-dir "$L" --seed-lo $s --seed-hi $hi --expect-games $((it*400)) \
        --dump "$R/${t}dump/${tag}_s${s}.json" < /dev/null > "$L.$t" 2>&1
  done
  if grep -a -q "正确口径" "$L.ci"; then cp "$D/${tag}_s${s}_gpufp32.npz" $P/; touch $R/done_q25one/s$s; say "s$s-$hi done"
  else say "s$s group_ci FAILED, see $L.ci"; fi
}
seg 14000 20
seg 10000 10
SYMTAG=$tag $PY $P/symaug_pair.py 10000 10999 14000 15999 > $SA/q25one_12k.txt 2>&1
say "12k: $(tr '\n' ' ' < $SA/q25one_12k.txt)"
say "12k reported only (no futility stop for this arm); extending to 100k"
for sg in "11000 30" "16000 40" "20000 50" "25000 50" "30000 50"; do seg $sg; done
SYMTAG=$tag $PY $P/symaug_pair.py 10000 34999 > $SA/q25one_100k.txt 2>&1
$PY - >> $SA/q25one_100k.txt 2>&1 <<PY
import glob, numpy as np
def load(pat):
    m = {}
    for f in sorted(glob.glob("$P/" + pat)):
        z = np.load(f)
        for s, v in zip(z["seed"], z["mean"]):
            m.setdefault(int(s), float(v))
    return m
a, b = load("q25one_final_s*_gpufp32.npz"), load("q25_final_s*_gpufp32.npz")
c = sorted(s for s in a if s in b and 10000 <= s <= 34999)
d = np.array([a[s] - b[s] for s in c]); se = d.std(ddof=1) / np.sqrt(len(d))
print(f"  q25one - q25 (one pass - two passes)   {d.mean():+.3f} +- {1.96 * se:.3f}  z {d.mean() / se:+.2f}   [{len(c):,} groups]")
PY
say "100k: $(tr '\n' ' ' < $SA/q25one_100k.txt)"
say Q25ONE_CHAIN_DONE
