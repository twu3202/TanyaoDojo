#!/bin/bash
# Anchor-schedule test (README.md): the v11 line's stage 2 with the anchor schedule of the historical C' stage, λ 0.5
# and then 0.2, against the control already played (runs/xfer_v11s2, A160k: straight λ 0.2).
#   0. waits for an idle GPU (no main-line loop or reward chain, no trainer / worker / eval)
#   1. runs/xfer_v11w from runs/xfer_v11/archive/xfer_v11_s80000.pth, set up as xfer_v11s2 except the anchor weight
#      0.5; the generated config must differ from v11s2's only in the weight line (paths and port aside)
#   2. phase 1: 80,000 -> 160,000 batches at λ 0.5. phase 2: the weight line set back to v11s2's (λ 0.2; the config
#      then equals v11s2's), state.pth = the 160k archive, 160,000 -> 200,000 batches
#   3. 12k of W160k and W200k; SPRT W200k - A160k (0 vs +1, alpha 0.025, beta 0.2), looks per 1,000 groups from seed
#      10000, capped at the 16,000 groups A160k has
#   4. paired readouts: W160k - A80k (the λ 0.5 phase), W200k - W160k (the relaxation), W200k - A160k,
#      W200k - tpt 408k, both snapshots vs v4
#   5. promote: the line continues by mainline_next.sh from W200k; otherwise the chain ends
# Log runs/xfer_tools/anc_chain.log; markers ANC_CHAIN_DONE / ANC_CHAIN_STOPPED. Re-runnable: a set-up arm is not set
# up again, a finished phase is not trained again (SP/phase2 marks that state.pth was reset to the 160k archive), and
# xfer_eval.sh skips finished segments. Every phase and segment needs DMIN (10) GB free on D:.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/xfer_tools/anc_chain.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/anc_chain.log
D=$R/evaldump; ARM=v11w; SP=$R/xfer_v11w; SRC=$R/xfer_v11/archive/xfer_v11_s80000.pth; DMIN=${DMIN:-10}
C=$B/configs; CFG=$C/online_selfplay_xfer_$ARM.toml
P1=$SP/archive/xfer_${ARM}_s160000.pth; P2=$SP/archive/xfer_${ARM}_s200000.pth
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
dfree() { df -BG --output=avail /mnt/d | tail -1 | tr -dc 0-9; }
norm() { sed -E -e "s/xfer_$1/ARM/g" -e 's/55[0-9]{2}/PORT/g' $C/online_selfplay_xfer_$1.toml; }
cfgdiff() {
  [ -s $C/online_selfplay_xfer_v11s2.toml ] && [ -s $CFG ] || { echo "MISSING CONFIG"; return; }
  diff <(norm v11s2) <(norm $ARM) | grep "^[<>]"
}
echo $$ > $T/anc_chain.pid
say "waiting for an idle GPU"
idle=0
while [ $idle -lt 3 ]; do
  if pgrep -f "mainline_next\.sh|rew_chain\.sh" > /dev/null || pgrep -f "^$PY .*(train|run_eval|client|server|ev_map)\.py" > /dev/null
  then idle=0; else idle=$((idle + 1)); fi
  sleep 60
done
if [ -e $SP/state.pth ]; then
  say "arm $ARM already set up ($(head -1 $SP/manifest.txt)), D: $(dfree) GB free"
else
  [ "$(dfree)" -ge $DMIN ] || { say "D: below $DMIN GB free: ANC_CHAIN_STOPPED"; exit 1; }
  say "GPU idle; arm $ARM from $SRC with anchor weight 0.5, D: $(dfree) GB free"
  ANCHOR=0.5 bash $T/xfer_setup.sh $ARM $SRC 5589 >> $LOG 2>&1 || { say "setup FAILED: ANC_CHAIN_STOPPED"; exit 1; }
fi
if [ ! -f $P1 ]; then
  d=$(cfgdiff)
  if [ "$(echo "$d" | grep -c .)" != 2 ] || [ "$(echo "$d" | grep -c -E '^< weight = 0\.2 ')" != 1 ] \
     || [ "$(echo "$d" | grep -c -E '^> weight = 0\.5 ')" != 1 ]; then
    say "phase 1 config differs from v11s2 beyond the anchor weight: $(echo "$d" | tr '\n' ' '): ANC_CHAIN_STOPPED"; exit 1
  fi
  say "phase 1 config vs v11s2: $(echo "$d" | tr '\n' ' ')"
  STOP_AT=160000 bash $T/xfer_arm.sh $ARM
  [ -f $P1 ] || { say "phase 1 did not reach 160k: ANC_CHAIN_STOPPED"; exit 1; }
fi
if [ ! -f $P2 ]; then
  [ "$(dfree)" -ge $DMIN ] || { say "D: below $DMIN GB free before phase 2: ANC_CHAIN_STOPPED"; exit 1; }
  L=$(grep '^weight = ' $C/online_selfplay_xfer_v11s2.toml)
  awk -v L="$L" '/^weight = /{print L; next} {print}' $CFG > $CFG.new && mv $CFG.new $CFG
  d=$(cfgdiff)
  [ -z "$d" ] || { say "phase 2 config differs from v11s2: $(echo "$d" | tr '\n' ' '): ANC_CHAIN_STOPPED"; exit 1; }
  if [ ! -f $SP/phase2 ]; then
    cp $P1 $SP/state.pth && echo "state.pth = $(basename $P1) $(md5sum < $P1 | cut -c1-12), $(date '+%F %T')" > $SP/phase2
  fi
  say "phase 2: config = v11s2's (λ 0.2), $(cat $SP/phase2), D: $(dfree) GB free"
  START_AFTER=160000 STOP_AT=200000 bash $T/xfer_arm.sh $ARM
  [ -f $P2 ] || { say "phase 2 did not reach 200k: ANC_CHAIN_STOPPED"; exit 1; }
fi
say "arm trained; $(grep -a '^overall' $SP/arm.log | tr '\n' ' ')"
pair() {
  $PY - "$D" <<'PY'
import glob, sys
import numpy as np
D = sys.argv[1]
pats = {"W200k": "xfer_v11w_s200000", "W160k": "xfer_v11w_s160000", "A160k": "xfer_v11s2_s160000",
        "A80k": "xfer_v11_s80000", "408k": "tpt_tpt_s408000"}
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
    print(f"  {label:<26} {len(seeds):>6,} groups  {y.mean():+.3f} +- {h:.3f}  z {y.mean() / (h / 1.96):+.2f}")
for k in ("W160k", "W200k"):
    show(f"{k} vs v4", k)
show("W160k - A80k (λ 0.5)", "W160k", "A80k")
show("W200k - W160k (relaxed)", "W200k", "W160k")
show("W200k - A160k", "W200k", "A160k")
show("A160k - A80k (same seeds)", "A160k", "A80k")
show("W200k - tpt 408k", "W200k", "408k")
PY
}
for st in 160000 200000; do bash $T/xfer_eval.sh $ARM $st 12k; done
pair > $T/anc_12k.txt 2>&1; say "12k: $(tr '\n' ' ' < $T/anc_12k.txt)"
seq_look() {
  $PY $T/seqtest.py "$D/xfer_${ARM}_s200000_s*_gpufp32.npz" "$D/xfer_v11s2_s160000_s*_gpufp32.npz" \
      --delta0 0 --delta1 1 --alpha 0.025 --beta 0.2 --look 1000 --nmax 16000 > $T/anc_seq.txt 2>&1
  echo $?
}
verdict=3; hi=10999
rc=$(seq_look); say "look at 1,000 groups: exit $rc, $(tail -2 $T/anc_seq.txt | tr -s " \n" " ")"; [ "$rc" = 3 ] || verdict=$rc
for s in 11000 12000 13000 $(seq 16000 1000 25000); do
  [ "$verdict" = 3 ] || [ "$verdict" = 0 ] || break      # drop or cap: stop playing; promote: play to the cap
  [ "$(dfree)" -ge $DMIN ] || { say "D: below $DMIN GB free before s$s: ANC_CHAIN_STOPPED"; exit 1; }
  bash $T/xfer_eval.sh $ARM 200000 seg $s 10 || { say "segment s$s FAILED: ANC_CHAIN_STOPPED"; exit 1; }
  if [ $s = 13000 ]; then hi=15999; else hi=$((s + 999)); fi
  if [ "$verdict" = 3 ]; then
    rc=$(seq_look); say "look through s$hi: exit $rc, $(tail -2 $T/anc_seq.txt | tr -s " \n" " ")"; [ "$rc" = 3 ] || verdict=$rc
  fi
done
case $verdict in
  0) say "SPRT: promote";;
  1) say "SPRT: drop at $hi";;
  4) say "SPRT: undecided at the 16,000-group cap; read the fixed 16k";;
  *) say "SPRT: exit $verdict";;
esac
pair > $T/anc_final.txt 2>&1; say "pairings: $(tr '\n' ' ' < $T/anc_final.txt)"
$PY $T/xfer_reuse.py $SP | tail -1 >> $LOG 2>&1
say "ANC_CHAIN_DONE"
if [ "$verdict" = 0 ]; then
  rm -f $T/MAINLINE_STOP; echo "v11w 200000 1000000000" > $T/mainline.txt
  say "promoted: the warmed line continues by mainline_next.sh from $(cat $T/mainline.txt)"
  exec bash $T/mainline_next.sh
fi
say "not promoted: chain ends, GPU free"
