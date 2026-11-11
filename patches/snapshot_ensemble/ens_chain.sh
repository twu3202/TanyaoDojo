#!/bin/bash
# Snapshot ensemble (README.md): the mean-Q ensemble of the four best own models at 100k, screened against tpt 408k.
#   1. engine check: the ensemble path with 408k alone must play seeds 10000-10099 exactly as the single-model path
#   2. SPRT E - 408k (0 vs +1, alpha 0.025, beta 0.2), looks per 1,000 groups from seed 10000, cap 25,000 groups;
#      a promote plays on to the full 100k
#   3. paired readouts: E vs v4, E - 408k, E - each member
#   4. the default follow-on: mainline_next.sh
# Log runs/xfer_tools/ens_chain.log; markers ENS_CHAIN_DONE / ENS_CHAIN_STOPPED. Re-runnable: the engine check and
# finished segments are not repeated. Every segment needs DMIN (10) GB free on D:, which holds the WSL disk.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/xfer_tools/ens_chain.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/ens_chain.log
D=$R/evaldump; E=/mnt/d/Better_mortal/jax_rl/mjai_bot; NAME=ens4; DMIN=${DMIN:-10}
M0=$R/tpt_tpt_s408000.pth
MEMBERS=$M0,$R/stage1_infer/stage1_s448000_infer.pth,$R/v4p_v4p_s448000.pth,$R/lam01_state_s368000.pth
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
dfree() { df -BG --output=avail /mnt/d | tail -1 | tr -dc 0-9; }
echo $$ > $T/ens_chain.pid
idle=0
while [ $idle -lt 3 ]; do
  if pgrep -f "^$PY .*(train|run_eval|client|server|ev_map)\.py" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
  sleep 60
done
for m in ${MEMBERS//,/ }; do [ -f $m ] || { say "no $m: ENS_CHAIN_STOPPED"; exit 1; }; done
say "GPU idle; members $(for m in ${MEMBERS//,/ }; do echo -n "$(basename $m) $(md5sum $m | cut -c1-8), "; done)D: $(dfree) GB free"
if [ ! -f $T/ens_check.ok ]; then
  C=$R/ens_check; rm -rf $C; mkdir -p $C/single $C/ens; cd $E
  for v in single ens; do
    ct=mortal; [ $v = ens ] && ct=ensemble
    RAYON_NUM_THREADS=5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY run_eval.py $M0 --challenger-type $ct --challenger-name chk_$v \
        --device cuda:0 --amp off --games 400 --iters 1 --seed-start 10000 --log-dir $C/$v < /dev/null > $C/$v.out 2>&1
  done
  # the logs carry each decision's wall-clock eval_time_ns, which no two runs share; everything else must match
  h() { (cd $C/$1 && for f in $(ls); do zcat $f | sed -E "s/chk_$1/X/g; s/\"eval_time_ns\":[0-9]+,?//g"; done | md5sum | cut -c1-12); }
  a=$(h single); b=$(h ens); n=$(ls $C/ens | wc -l); dis=$(grep -a -oE "on [0-9]+/[0-9]+" $C/ens.out | tail -1)
  if [ "$a" = "$b" ] && [ $n = 400 ]; then touch $T/ens_check.ok; say "engine check: $n games identical ($a), differs $dis"
  else say "engine check FAILED: $n games, $a vs $b: ENS_CHAIN_STOPPED"; exit 1; fi
fi
look() {
  $PY $T/seqtest.py "$D/${NAME}_s*_gpufp32.npz" "$D/tpt_tpt_s408000_s*_gpufp32.npz" \
      --delta0 0 --delta1 1 --alpha 0.025 --beta 0.2 --look 1000 --nmax 25000 > $T/ens_seq.txt 2>&1
  echo $?
}
verdict=3; hi=9999
for s in $(seq 10000 1000 34000); do
  [ "$verdict" = 3 ] || [ "$verdict" = 0 ] || break      # drop or cap: stop playing; promote: play on to 100k
  [ "$(dfree)" -ge $DMIN ] || { say "D: below $DMIN GB free before s$s: ENS_CHAIN_STOPPED"; exit 1; }
  bash $T/ens_eval.sh $NAME "$MEMBERS" $s 10 || { say "segment s$s FAILED: ENS_CHAIN_STOPPED"; exit 1; }
  hi=$((s + 999))
  if [ "$verdict" = 3 ]; then
    verdict=$(look); say "look through s$hi: exit $verdict, $(tail -2 $T/ens_seq.txt | tr -s ' \n' ' ')"
  fi
done
case $verdict in
  0) say "SPRT: promote (played on to s$hi)";;
  1) say "SPRT: drop at $hi";;
  4) say "SPRT: undecided at the cap; read the fixed 100k";;
  *) say "SPRT: exit $verdict";;
esac
$PY - "$D" > $T/ens_final.txt 2>&1 <<'PY'
import glob, sys
import numpy as np
D = sys.argv[1]
pats = {"E": "ens4", "408k": "tpt_tpt_s408000", "Stage 1 448k": "stage1_s448000", "v4 pool 448k": "v4p_v4p_s448000",
        "s368000": "lam01_state_s368000"}
m = {}
for k, p in pats.items():
    d = {}
    for f in glob.glob(f"{D}/{p}_s*_gpufp32.npz"):
        z = np.load(f)
        d.update({int(s): float(v) for s, v in zip(z["seed"], z["mean"])})
    m[k] = d
def show(label, a, b=None):
    seeds = sorted(set(m[a]) & (set(m[b]) if b else set(m[a])))
    if len(seeds) < 2:
        return
    y = np.array([m[a][s] - (m[b][s] if b else 0.0) for s in seeds])
    h = 1.96 * y.std(ddof=1) / np.sqrt(len(y))
    print(f"  {label:<24} {len(seeds):>6,} groups  {y.mean():+.3f} +- {h:.3f}  z {y.mean() / (h / 1.96):+.2f}")
show("E vs v4", "E")
for k in ("408k", "Stage 1 448k", "v4 pool 448k", "s368000"):
    show(f"E - {k}", "E", k)
PY
say "pairings: $(tr '\n' ' ' < $T/ens_final.txt)"
say "ENS_CHAIN_DONE"
say "default follow-on: mainline_next.sh from $(cat $T/mainline.txt)"
exec bash $T/mainline_next.sh
