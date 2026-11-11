#!/bin/bash
# Snapshot ensemble (README.md): sacred evaluation of a mean-Q ensemble, one segment of ITERS x 100 seed groups from
# START. Same engine, GPU fp32, 4-process layout, thread caps and --strict acceptance as
# patches/online_transfer/xfer_eval.sh; the challenger is the comma-separated MEMBERS, played by
# run_eval.py --challenger-type ensemble. A segment with a done marker is skipped.
#   bash ens_eval.sh NAME MEMBERS START ITERS
set -u
NAME=$1; MEMBERS=$2; s=$3; it=$4
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; D=$R/evaldump
PY=$B/.venv/bin/python; LOG=${LOG:-$T/ens_chain.log}; E=/mnt/d/Better_mortal/jax_rl/mjai_bot
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
mkdir -p $R/done_xfer $R/gapdump $R/placedump $R/bustdump $D
hi=$((s + it*100 - 1))
[ -f $R/done_xfer/${NAME}_s$s ] && exit 0
L=$R/eval_${NAME}_s${s}_gpufp32; rm -rf "$L"; mkdir -p "$L"
cd $E
NP=${NPROC:-4}; off=0; pids=()
for p in $(seq 0 $((NP - 1))); do
  k=$(( it / NP + (p < it % NP ? 1 : 0) )); [ $k -eq 0 ] && continue
  RAYON_NUM_THREADS=5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY run_eval.py "$MEMBERS" --challenger-type ensemble --challenger-name "$NAME" --device cuda:0 --amp off \
      --games 400 --iters $k --seed-start $((s + off * 100)) --log-dir "$L" < /dev/null > "$L.out.$p" 2>&1 &
  pids+=($!); off=$((off + k))
done
for pid in "${pids[@]}"; do wait $pid; done
dis=$(for f in "$L".out.*; do grep -a "differs from member 0" $f | tail -1 | grep -oE "[0-9]+/[0-9]+"; done \
      | awk -F/ '{a += $1; b += $2} END {if (b) printf "%d/%d (%.2f%%)", a, b, 100 * a / b}')
cat "$L".out.* > "$L.out"; rm -f "$L".out.*
$PY $T/group_ci.py --strict --log-dir "$L" --seed-lo $s --seed-hi $hi --dump "$D/${NAME}_s${s}_gpufp32.npz" < /dev/null > "$L.ci" 2>&1
rc=$?
for t in gap place bust; do
  $PY diag_$t.py --log-dir "$L" --seed-lo $s --seed-hi $hi --expect-games $((it*400)) \
      --dump "$R/${t}dump/${NAME}_s${s}.json" < /dev/null > "$L.$t" 2>&1
done
if [ $rc = 0 ] && grep -a -q "正确口径" "$L.ci"; then
  touch $R/done_xfer/${NAME}_s$s; say "$NAME s$s-$hi done; differs from member 0 on $dis engine decisions"
else say "$NAME s$s group_ci FAILED (exit $rc), see $L.ci"; exit 1; fi
