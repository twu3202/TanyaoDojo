#!/bin/bash
# Default follow-on for the 5060 Ti (user's standing choice, 2026-10-28): when a chain ends and nothing else is
# queued, the current main line trains one more stage. It is re-anchored to its own latest snapshot, which also
# becomes the 'base' opponents (v4 stays in the pool), for 80,000 batches. The stage is screened against tpt 408k by
# the SPRT (0 vs +1, alpha 0.025, beta 0.2, looks per 1,000 groups, cap 100k). The pointer then moves to the new
# snapshot, and the next stage follows.
#   runs/xfer_tools/mainline.txt = "ARM STEP EPOCHS [PTS]": the snapshot runs/xfer_ARM/archive/xfer_ARM_sSTEP.pth to
#   start from, num_epochs for the new stage (1 = each fresh sample once; 1000000000 = the default reuse), and
#   optionally the training reward's placement points (e.g. [6, 4, 2, 0]; empty = the template's eval-proportional)
#   touch runs/xfer_tools/MAINLINE_STOP to stop after the current stage; kill $(cat runs/xfer_tools/mainline.pid) to
#   stop it before a stage starts. Log runs/xfer_tools/mainline.log.
# The WSL disk lives on D:; a stage and every segment start only with DMIN (10) GB free there (xfer_arm.sh checks the
# same during training), otherwise the loop stops cleanly.
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/mainline.log; D=$R/evaldump
DMIN=${DMIN:-10}
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
dfree() { df -BG --output=avail /mnt/d | tail -1 | tr -dc 0-9; }
echo $$ > $T/mainline.pid
while [ ! -f $T/MAINLINE_STOP ]; do
  read -r ARM STEP EPOCHS PTS < $T/mainline.txt
  SRC=$R/xfer_$ARM/archive/xfer_${ARM}_s${STEP}.pth
  [ -f $SRC ] || { say "no $SRC: MAINLINE_STOPPED"; exit 1; }
  n=$(ls -d $R/xfer_ml* 2>/dev/null | wc -l); NEW=ml$((n + 1)); STOP=$((STEP + 80000)); MID=$((STEP + 40000))
  idle=0
  while [ $idle -lt 3 ]; do
    if pgrep -f "^$PY .*(train|run_eval|client|server|ev_map)\.py" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
    sleep 60
  done
  [ "$(dfree)" -ge $DMIN ] || { say "D: below $DMIN GB free: MAINLINE_STOPPED"; exit 1; }
  say "stage $NEW from $SRC (num_epochs $EPOCHS${PTS:+, pts $PTS}), to $STOP, D: $(dfree) GB free"
  PTS=$PTS EPOCHS=$EPOCHS bash $T/xfer_setup.sh $NEW $SRC $((5590 + n % 5)) >> $LOG 2>&1 || { say "setup FAILED: MAINLINE_STOPPED"; exit 1; }
  STOP_AT=$STOP bash $T/xfer_arm.sh $NEW
  grep -a -q " XFER_ARM_DONE$" $R/xfer_$NEW/arm.log || { say "$NEW did not reach $STOP: MAINLINE_STOPPED"; exit 1; }
  say "$NEW trained; $(grep -a '^overall' $R/xfer_$NEW/arm.log | tail -1)"
  for st in $MID $STOP; do bash $T/xfer_eval.sh $NEW $st 12k; done
  look() {
    $PY $T/seqtest.py "$D/xfer_${NEW}_s${STOP}_s*_gpufp32.npz" "$D/tpt_tpt_s408000_s*_gpufp32.npz" \
        --delta0 0 --delta1 1 --alpha 0.025 --beta 0.2 --look 1000 > $T/mainline_seq.txt 2>&1
    echo $?
  }
  verdict=$(look); say "$NEW look 1: exit $verdict, $(tail -2 $T/mainline_seq.txt | tr -s ' \n' ' ')"
  for s in 11000 12000 13000 $(seq 16000 1000 34000); do
    [ "$verdict" = 3 ] || [ "$verdict" = 0 ] || break
    [ "$(dfree)" -ge $DMIN ] || { say "D: below $DMIN GB free before s$s: MAINLINE_STOPPED"; exit 1; }
    bash $T/xfer_eval.sh $NEW $STOP seg $s 10 || { say "segment s$s FAILED: MAINLINE_STOPPED"; exit 1; }
    if [ "$verdict" = 3 ]; then verdict=$(look); say "$NEW look through s$s: exit $verdict, $(tail -2 $T/mainline_seq.txt | tr -s ' \n' ' ')"; fi
  done
  $PY - "$D" "xfer_${NEW}_s${STOP}" "xfer_${NEW}_s${MID}" "xfer_${ARM}_s${STEP}" > $T/mainline_${NEW}.txt 2>&1 <<'PY'
import glob, sys
import numpy as np
D, new, mid, old = sys.argv[1:5]
def load(p):
    d = {}
    for f in glob.glob(f"{D}/{p}_s*_gpufp32.npz"):
        z = np.load(f)
        d.update({int(s): float(v) for s, v in zip(z["seed"], z["mean"])})
    return d
m = {"new": load(new), "mid": load(mid), "old": load(old), "408k": load("tpt_tpt_s408000")}
def show(label, a, b=None):
    seeds = sorted(set(m[a]) & (set(m[b]) if b else set(m[a])))
    if len(seeds) < 2:
        return
    y = np.array([m[a][s] - (m[b][s] if b else 0.0) for s in seeds])
    h = 1.96 * y.std(ddof=1) / np.sqrt(len(y))
    print(f"  {label:<30} {len(seeds):>6,} groups  {y.mean():+.3f} +- {h:.3f}  z {y.mean() / (h / 1.96):+.2f}")
show(f"{mid} vs v4", "mid"); show(f"{new} vs v4", "new")
show("new - previous snapshot", "new", "old"); show("new - tpt 408k", "new", "408k")
PY
  say "$NEW: SPRT exit $verdict; $(tr '\n' ' ' < $T/mainline_${NEW}.txt)"
  echo "$NEW $STOP $EPOCHS${PTS:+ $PTS}" > $T/mainline.txt
done
say "MAINLINE_STOP found: stopping"
