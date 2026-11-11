#!/bin/bash
# Probe of the structural operator (a), separate policy and critic (README.md, DESIGN.md): arm P = runs/xfer_p1 from
# A80k with train_pg.py, against the controls at 160k (A160k = runs/xfer_v11s2).
#   0. waits for the anchor-schedule chain to end not promoted (if it promoted, the main line holds the GPU and this
#      stops for the user), then for the X1 timing chain, then for 3 idle GPU minutes
#   1. installs the TRAINER switch (xfer_helpers.sh.pg prepared beside the live copy; run_selfplay_m1.sh by sed), each by
#      an atomic mv while nothing runs; both default to train.py
#   2. sets up runs/xfer_p1 (port 5590) like xfer_v11s2, then: workers sample pi exactly (boltzmann_epsilon 1.0 at
#      temp 0.1, top_p 1.0), the Q-space anchor off (weight 0), and a [pg] section; every other line must equal v11s2's
#   3. trains 80,000 -> 160,000 batches with TRAINER=train_pg.py: critic warm-up to 88,000, policy learning-rate ramp to
#      96,000, adaptive KL to the base at 0.02 nats. A watcher stops the run if KL to the base exceeds 0.05 nats at two
#      saves in a row
#   4. 12k of P120k and P160k; SPRT P160k - A160k (0 vs +1, alpha 0.025, beta 0.2), looks per 1,000 groups from seed
#      10000, capped at 16,000 groups; pairings and the last PG diagnostics
# Log runs/xfer_tools/pg_chain.log; markers PG_CHAIN_DONE / PG_CHAIN_STOPPED. Re-runnable: a set-up arm is not set up
# again, a trained arm is not trained again, xfer_eval.sh skips finished segments. Each step needs DMIN (10) GB on D:.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/xfer_tools/pg_chain.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/pg_chain.log
D=$R/evaldump; ARM=p1; SP=$R/xfer_p1; SRC=$R/xfer_v11/archive/xfer_v11_s80000.pth; DMIN=${DMIN:-10}
C=$B/configs; CFG=$C/online_selfplay_xfer_$ARM.toml; KL_CAP=0.05
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
dfree() { df -BG --output=avail /mnt/d | tail -1 | tr -dc 0-9; }
echo $$ > $T/pg_chain.pid
say "waiting for the anchor-schedule chain, the X1 timing chain and an idle GPU"
until grep -a -q -E "ANC_CHAIN_DONE|ANC_CHAIN_STOPPED" $T/anc_chain.log 2>/dev/null; do sleep 300; done
if grep -a -q -E "_[0-9:]+ promoted: " $T/anc_chain.log; then   # not "not promoted: "
  say "the anchor schedule was promoted and the main line holds the GPU: (a) waits for the user: PG_CHAIN_STOPPED"; exit 1
fi
while pgrep -f "x1_5060\.sh" > /dev/null && ! grep -a -q X1_5060_DONE $R/bench_eval/x1.log 2>/dev/null; do sleep 120; done
idle=0
while [ $idle -lt 3 ]; do
  if pgrep -f "mainline_next\.sh|anc_chain\.sh|rew_chain\.sh|x1_5060\.sh" > /dev/null \
     || pgrep -f "^$PY .*(train|train_pg|run_eval|client|server|ev_map|x1_bench|x1_server)\.py" > /dev/null; then idle=0
  else idle=$((idle + 1)); fi
  sleep 60
done
[ "$(dfree)" -ge $DMIN ] || { say "D: below $DMIN GB free: PG_CHAIN_STOPPED"; exit 1; }

# 1. the TRAINER switch
if ! grep -q 'TRAINER:-train.py' $T/xfer_helpers.sh; then
  [ -f $T/xfer_helpers.sh.pg ] && grep -q 'TRAINER:-train.py' $T/xfer_helpers.sh.pg \
    || { say "no prepared xfer_helpers.sh.pg with the TRAINER switch: PG_CHAIN_STOPPED"; exit 1; }
  cp $T/xfer_helpers.sh $T/xfer_helpers.sh.bak_pg && mv $T/xfer_helpers.sh.pg $T/xfer_helpers.sh
fi
RS=$B/scripts/run_selfplay_m1.sh
if ! grep -q 'TRAINER:-train.py' $RS; then
  cp $RS $RS.bak_pg
  sed 's/\$PY train\.py > \$LOGD\/trainer\.log/$PY ${TRAINER:-train.py} > $LOGD\/trainer.log/' $RS > $RS.new
  [ "$(diff $RS $RS.new | grep -c '^[<>]')" = 2 ] && bash -n $RS.new && mv $RS.new $RS \
    || { say "could not install the TRAINER switch in run_selfplay_m1.sh: PG_CHAIN_STOPPED"; exit 1; }
fi
say "TRAINER switch: $(grep -c 'TRAINER:-train.py' $T/xfer_helpers.sh) in xfer_helpers.sh, $(grep -c 'TRAINER:-train.py' $RS) in run_selfplay_m1.sh"

# 2. the arm and its config
if [ -e $SP/state.pth ]; then
  say "arm $ARM already set up ($(head -1 $SP/manifest.txt))"
else
  bash $T/xfer_setup.sh $ARM $SRC 5590 >> $LOG 2>&1 || { say "setup FAILED: PG_CHAIN_STOPPED"; exit 1; }
  sed -i -e 's/^boltzmann_epsilon = .*/boltzmann_epsilon = 1.0   # pg: workers sample pi = softmax(q \/ T) exactly/' \
         -e 's/^boltzmann_temp = [0-9.]*/boltzmann_temp = 0.1/' -e 's/^top_p = [0-9.]*/top_p = 1.0/' \
         -e 's/^weight = .*/weight = 0.0   # pg: no Q-space anchor; train_pg.py uses state_file as the reference policy/' $CFG
  cat >> $CFG <<'TOML'

# structural operator (a), patches/policy_critic: read by train_pg.py only
[pg]
temp = 0.1
clip = 0.2
clip_c = 3.0
kl_coef = 1.0
kl_target = 0.02
kl_horizon = 2000
ent_coef = 0.0
value_coef = 1.0
value_hidden = 256
policy_start_step = 88000
lr_warmup = 8000
TOML
fi
norm() { sed -E -e "s/xfer_$1/ARM/g" -e 's/55[0-9]{2}/PORT/g' $C/online_selfplay_xfer_$1.toml; }
d=$(diff <(norm v11s2) <(norm $ARM) | grep "^[<>]")
bad=$(echo "$d" | grep -v -E '^[<>] (boltzmann_epsilon|weight) = |^> \[pg\]|^> # structural operator|^> $|^> (temp|clip|clip_c|kl_coef|kl_target|kl_horizon|ent_coef|value_coef|value_hidden|policy_start_step|lr_warmup) = ')
if [ -n "$bad" ] || [ "$(echo "$d" | grep -c -E '^[<>] (boltzmann_epsilon|weight) = ')" != 4 ]; then
  say "config differs from v11s2 beyond the pg lines: $(echo "$bad" | tr '\n' ' '): PG_CHAIN_STOPPED"; exit 1
fi
say "config vs v11s2: $(echo "$d" | grep -E '^[<>] (boltzmann_epsilon|weight) = ' | tr '\n' ' ') + [pg]"

# 3. training, with the KL watcher
if ! grep -a -q " XFER_ARM_DONE$" $SP/arm.log 2>/dev/null; then
  [ "$(dfree)" -ge $DMIN ] || { say "D: below $DMIN GB free: PG_CHAIN_STOPPED"; exit 1; }
  (
    over=0
    while sleep 300; do
      [ -f $SP/pids ] || continue
      k=$(tr -d '\r' < $SP/logs/trainer.log 2>/dev/null | grep -a -oE "pg at [0-9,]+: .* kl_ref [0-9.e+-]+" | tail -1 \
          | grep -oE "kl_ref [0-9.e+-]+" | cut -d' ' -f2)
      [ -z "$k" ] && continue
      line=$(tr -d '\r' < $SP/logs/trainer.log | grep -a -c "pg at ")
      [ "$line" = "${last:-}" ] && continue
      last=$line
      if awk -v k="$k" -v c=$KL_CAP 'BEGIN {exit !(k > c)}'; then over=$((over + 1)); else over=0; fi
      if [ $over -ge 2 ]; then
        say "KL to the base above $KL_CAP at two saves in a row (last $k): stopping the arm"
        bash $T/xfer_helpers.sh stop $SP >> $LOG 2>&1; exit 0
      fi
    done
  ) & watcher=$!
  TRAINER=train_pg.py STOP_AT=160000 bash $T/xfer_arm.sh $ARM
  kill $watcher 2>/dev/null
  grep -a -q " XFER_ARM_DONE$" $SP/arm.log || { say "arm did not reach 160k: PG_CHAIN_STOPPED"; exit 1; }
fi
say "arm trained; $(grep -a '^overall' $SP/arm.log | tail -1)"
say "last PG diagnostics: $(tr -d '\r' < $SP/logs/trainer.log | grep -a -oE 'pg at [0-9,]+: .*' | tail -1)"

# 4. evaluation
pair() {
  $PY - "$D" <<'PY'
import glob, sys
import numpy as np
D = sys.argv[1]
pats = {"P160k": "xfer_p1_s160000", "P120k": "xfer_p1_s120000", "A160k": "xfer_v11s2_s160000",
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
for k in ("P120k", "P160k"):
    show(f"{k} vs v4", k)
show("P160k - A160k", "P160k", "A160k")
show("P160k - A80k", "P160k", "A80k")
show("A160k - A80k (same seeds)", "A160k", "A80k")
show("P160k - tpt 408k", "P160k", "408k")
PY
}
for st in 120000 160000; do bash $T/xfer_eval.sh $ARM $st 12k; done
pair > $T/pg_12k.txt 2>&1; say "12k: $(tr '\n' ' ' < $T/pg_12k.txt)"
seq_look() {
  $PY $T/seqtest.py "$D/xfer_${ARM}_s160000_s*_gpufp32.npz" "$D/xfer_v11s2_s160000_s*_gpufp32.npz" \
      --delta0 0 --delta1 1 --alpha 0.025 --beta 0.2 --look 1000 --nmax 16000 > $T/pg_seq.txt 2>&1
  echo $?
}
verdict=3; hi=10999
rc=$(seq_look); say "look at 1,000 groups: exit $rc, $(tail -2 $T/pg_seq.txt | tr -s " \n" " ")"; [ "$rc" = 3 ] || verdict=$rc
for s in 11000 12000 13000 $(seq 16000 1000 25000); do
  [ "$verdict" = 3 ] || [ "$verdict" = 0 ] || break      # drop or cap: stop playing; promote: play to the cap
  [ "$(dfree)" -ge $DMIN ] || { say "D: below $DMIN GB free before s$s: PG_CHAIN_STOPPED"; exit 1; }
  bash $T/xfer_eval.sh $ARM 160000 seg $s 10 || { say "segment s$s FAILED: PG_CHAIN_STOPPED"; exit 1; }
  if [ $s = 13000 ]; then hi=15999; else hi=$((s + 999)); fi
  if [ "$verdict" = 3 ]; then
    rc=$(seq_look); say "look through s$hi: exit $rc, $(tail -2 $T/pg_seq.txt | tr -s " \n" " ")"; [ "$rc" = 3 ] || verdict=$rc
  fi
done
case $verdict in
  0) say "SPRT: promote";;
  1) say "SPRT: drop at $hi";;
  4) say "SPRT: undecided at the 16,000-group cap; read the fixed 16k";;
  *) say "SPRT: exit $verdict";;
esac
pair > $T/pg_final.txt 2>&1; say "pairings: $(tr '\n' ' ' < $T/pg_final.txt)"
$PY $T/xfer_reuse.py $SP | tail -1 >> $LOG 2>&1
say "PG_CHAIN_DONE"
