#!/bin/bash
# Model-soup probe (patches/model_soup/README.md): evaluation only, no training.
#   SV = uniform weight average of the five 160k arms trained from A80k (A160k, C160k, F160k, R160k, FS160k),
#        16,000 groups (the pooled control's seeds 10000-25999).
#   ST = uniform weight average of the four 448k continuations of tpt 408k (historical 448k, Stage 1, M1 c0.3,
#        M1-sel), 25,000 groups = the 100k protocol (seeds 10000-34999).
# The soups are built beforehand by soup.py (runs/xfer_soup{v,t}/infer/*_infer.pth, md5 in the README).
# Each soup's first call is one 1,000-group segment; a soup under -8 against v4 there is broken and is not evaluated
# further. Then paired readouts (soup_pair.py), mechanism readouts, SOUP_CHAIN_DONE, and the idle card is lent for
# a day (ends as soon as other mortal GPU work starts).
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/xfer_tools/soup_chain.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/soup_chain.log
D=$R/evaldump; W=/mnt/c/Users/asus; ST=$W/GPU_STATUS_mortal.txt; DMIN=${DMIN:-10}
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
dfree() { df -BG --output=avail /mnt/d | tail -1 | tr -dc 0-9; }
status() {
  printf '%s\r\n' "mortal GPU status (TanyaoDojo session), $(date '+%F %T %Z')" "$@" \
    "To book: create C:\\Users\\asus\\GPU_BOOK_<name>.txt (one line: who and what; add a line not_before=YYYY-MM-DD HH:MM to defer it)." \
    > $ST.new && mv $ST.new $ST
}
idle() {
  status "State: FREE (idle) since $(date +%T): ${1:-the model-soup evaluation has ended and no mortal GPU work is queued.}" \
    "An active booking (GPU_BOOK_<name>.txt, once past its not_before) gets the card within a minute; this file then shows the hold." \
    "This file is rewritten before any new mortal GPU work starts."
}
stopped() { say "$1: SOUP_CHAIN_STOPPED"; idle "the model-soup chain stopped ($1)."; exit 1; }
mortal_busy() { pgrep -f "^$PY .*(train|train_pg|run_eval|client|server|critic_diag2?|fs_check)\.py" > /dev/null; }
vs_v4() {  # mean vs v4 over one segment's dump
  $PY -c "import numpy as np,sys; z=np.load(sys.argv[1]); print(f'{z[\"mean\"].mean():.3f}')" "$1" 2>/dev/null
}
echo $$ > $T/soup_chain.pid
say "waiting for an idle card"
idle=0; n=0
while [ $idle -lt 3 ]; do
  if mortal_busy || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] \
     || compgen -G "$W/GPU_BUSY_*.txt" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
  n=$((n + 1)); [ $n -gt 720 ] && stopped "card not idle for 12 hours"
  sleep 60
done
for f in $R/xfer_soupv/infer/xfer_soupv_s160000_infer.pth $R/xfer_soupt/infer/xfer_soupt_s448000_infer.pth; do
  [ -f $f ] || stopped "missing $f"; say "soup $(basename $f) md5 $(md5sum $f | cut -c1-8)"
done
status "State: BUSY - evaluation only (two model soups, 164,000 games, about 8 hours from $(date +%H:%M)). Exclusive: do not start GPU work." \
  "Lending: a booking fires at the next evaluation call (calls come at most ~60 min apart)." \
  "This file is rewritten when the chain ends."

run_soup() {  # ARM STEP then segments START:ITERS (s10000 x 10 is always the first call)
  local arm=$1 step=$2; shift 2
  local tag=xfer_${arm}_s${step}
  [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free before $tag"
  bash $T/xfer_eval.sh $arm $step seg 10000 10 || stopped "$tag s10000 FAILED"
  local v=$(vs_v4 $D/${tag}_s10000_gpufp32.npz)
  say "$tag first 1,000 groups vs v4: $v"
  [ -n "$v" ] || stopped "$tag: no dump for s10000"
  if $PY -c "import sys; sys.exit(0 if float(sys.argv[1]) < -8 else 1)" "$v"; then
    say "$tag under -8 against v4 on its first segment: broken soup, not evaluated further"; return 0
  fi
  local sp
  for sp in "$@"; do
    [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free before $tag s${sp%:*}"
    bash $T/xfer_eval.sh $arm $step seg ${sp%:*} ${sp#*:} || stopped "$tag segment s${sp%:*} FAILED"
  done
  say "$tag evaluated"
}
# SV: 16,000 groups = s10000 + s11000-13999 + s14000-15999 + s16000-25999 (the pooled control's seeds)
run_soup soupv 160000 14000:20 11000:10 12000:10 13000:10 $(for s in $(seq 16000 1000 25000); do echo -n "$s:10 "; done)
# ST: 25,000 groups = the 100k protocol's seeds 10000-34999 (12k, then xfer_eval.sh's "rest" segments)
run_soup soupt 448000 14000:20 11000:30 16000:40 20000:50 25000:50 30000:50

$PY $T/soup_pair.py > $T/soup_final.txt 2>&1; say "pairings: $(tr -s ' ' < $T/soup_final.txt | tr '\n' '|')"
mech() {
  cd /mnt/d/Better_mortal/jax_rl/mjai_bot || return
  local tag segs
  for tag in xfer_soupv_s160000 xfer_soupt_s448000; do
    segs=$(ls $R/gapdump | grep -oE "^${tag}_s[0-9]+\.json$" | sed 's/\.json$//' | tr '\n' ' ')
    [ -n "$segs" ] && { $PY merge_dumps.py ${tag}_fall $segs > /dev/null || echo "merge $tag failed"; }
  done
  $PY - <<'PY'
import json, os
R = os.path.expanduser("~/Projects/better_mortal/runs")
for tag in ("xfer_v11_s80000_f3k", "xfer_v11s2_s160000_f16k", "xfer_v11c2_s160000_f16k", "xfer_v11f_s160000_f6k",
            "xfer_v11r_s160000_f6k", "xfer_v11fs_s160000_fall", "xfer_soupv_s160000_fall",
            "tpt_tpt_s408000_100k", "tpt_tpt_s448000_100k", "stage1_s448000_100k", "anc30_s448000_100k",
            "m1sel_s448000_100k", "xfer_soupt_s448000_fall"):
    try:
        g = json.load(open(f"{R}/gapdump/{tag}.json")); p = json.load(open(f"{R}/placedump/{tag}.json")); b = json.load(open(f"{R}/bustdump/{tag}.json"))
    except OSError:
        print(f"  {tag}: missing"); continue
    o, t = g["ours"], g["theirs"]
    r = lambda d, k: d[k] / d["kyoku"]
    att = r(o, "win_pts") - r(t, "win_pts"); dfn = -(r(o, "deal_in_pts") - r(t, "deal_in_pts"))
    n = p["ours"]["n"]; f4 = p["ours"]["rank"][3] / n; f4v = (n - p["ours"]["rank"][3]) / (3 * n)
    print(f"  {tag:<26} n={g['n_games']} call {100*r(o,'naki'):.2f} ({100*r(t,'naki'):.2f}) win {100*(r(o,'win')-r(t,'win')):+.2f} "
          f"dealin {100*(r(o,'deal_in')-r(t,'deal_in')):+.2f} riichi {100*(r(o,'riichi')-r(t,'riichi')):+.2f} "
          f"net {att+dfn:+.0f} ({att:+.0f} / {dfn:+.0f}) bust {100*b['ours']['bust']/b['ours']['n']:.2f} / {100*b['v4']['bust']/b['v4']['n']:.2f} "
          f"4th {100*(f4-f4v):+.2f}")
PY
}
mech > $T/soup_mech.txt 2>&1; say "mechanism: $(tr -s ' ' < $T/soup_mech.txt | tr '\n' '|')"
idle; say "SOUP_CHAIN_DONE"
# lend the idle card for a day; ends as soon as other mortal GPU work starts
END=$(( $(date +%s) + 86400 ))
while [ $(date +%s) -lt $END ]; do
  mortal_busy && { say "mortal GPU work started: lending loop ends"; exit 0; }
  n0=$(ls $W/GPU_BOOK_*.txt 2>/dev/null | wc -l)
  if [ $n0 -gt 0 ]; then
    bash $T/gpu_share.sh >> $LOG 2>&1
    [ $(ls $W/GPU_BOOK_*.txt 2>/dev/null | wc -l) -lt $n0 ] && { idle; say "hold over; status: FREE (idle)"; }
  fi
  sleep 60
done
say "lending loop ends after a day"
