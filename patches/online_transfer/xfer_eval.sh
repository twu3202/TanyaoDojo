#!/bin/bash
# Online transfer test: sacred evaluation of one snapshot, segment by segment (duplicate 1v3 vs mortal_v4,
# seed_key 20260711, GPU fp32 / AMP off, the engine of every 100k so far), each with diag_gap / diag_place / diag_bust.
#   bash xfer_eval.sh ARM STEP 12k|rest
# 12k = s10000 (1,000 groups) + s14000 (2,000); rest = s11000 + s16000-34999, together 25,000 groups = 100k games.
# group_ci runs --strict: a segment counts only with every seed's four rotations present and ended.
set -u
ARM=$1; STEP=$2; WHICH=$3
B=$HOME/Projects/better_mortal; R=$B/runs; SP=$R/xfer_$ARM; T=$R/xfer_tools; D=$R/evaldump
PY=$B/.venv/bin/python; LOG=$SP/arm.log; E=/mnt/d/Better_mortal/jax_rl/mjai_bot
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
tag=xfer_${ARM}_s${STEP}; ck=$SP/infer/${tag}_infer.pth
mkdir -p $SP/infer $R/done_xfer $R/gapdump $R/placedump $R/bustdump $D
if [ ! -f $ck ]; then
  $PY - $SP/archive/${tag}.pth $ck >> $LOG 2>&1 <<'PY'
import sys, torch
st = torch.load(sys.argv[1], weights_only=True, map_location="cpu")
torch.save({k: st[k] for k in ("mortal", "current_dqn", "aux_net", "config", "steps")}, sys.argv[2])
print("infer copy", sys.argv[2], "steps", st["steps"])
PY
fi
say "eval $tag $WHICH, md5 $(md5sum $ck | cut -c1-8)"
case $WHICH in
  12k)  SEGS="14000 20|10000 10";;
  rest) SEGS="11000 30|16000 40|20000 50|25000 50|30000 50";;
esac
fail=0
IFS='|' read -ra SEG <<< "$SEGS"
cd $E
for seg in "${SEG[@]}"; do
  set -- $seg; s=$1; it=$2; hi=$((s + it*100 - 1))
  [ -f $R/done_xfer/${tag}_s$s ] && continue
  L=$R/eval_${tag}_s${s}_gpufp32; rm -rf "$L"; mkdir -p "$L"
  $PY run_eval.py "$ck" --challenger-type mortal --challenger-name "$tag" --device cuda:0 --amp off \
      --games 400 --iters $it --seed-start $s --log-dir "$L" < /dev/null > "$L.out" 2>&1
  $PY $T/group_ci.py --strict --log-dir "$L" --seed-lo $s --seed-hi $hi --dump "$D/${tag}_s${s}_gpufp32.npz" < /dev/null > "$L.ci" 2>&1
  rc=$?
  for t in gap place bust; do
    $PY diag_$t.py --log-dir "$L" --seed-lo $s --seed-hi $hi --expect-games $((it*400)) \
        --dump "$R/${t}dump/${tag}_s${s}.json" < /dev/null > "$L.$t" 2>&1
  done
  if [ $rc = 0 ] && grep -a -q "正确口径" "$L.ci"; then touch $R/done_xfer/${tag}_s$s; say "$tag s$s-$hi done"
  else say "$tag s$s group_ci FAILED (exit $rc), see $L.ci"; fail=1; fi
done
exit $fail
