#!/bin/bash
# P0, the C' re-check (README.md): evaluation only. rl1_best, C392k and C' final, the three snapshots of the only big jump
# in the project's history, played on the same walls under the current harness (GPU fp32, 4 processes, the 5060 Ti).
#   0. waits for an idle GPU and rewrites GPU_STATUS_mortal.txt
#   1. runs/xfer_hist/archive/xfer_hist_s{60000,215800,278600}.pth = copies of the three checkpoints (md5 in the log)
#   2. the 13,000 new groups first (seeds 11000-13999, 16000-25999), segment by segment for all three models, then the
#      old 12k of 09-12 (10000-10999, 14000-15999), report only
#   3. verdict on C' final - C392k over the new groups: >= +0.7 with z >= 2 holds, <= +0.3 fails, otherwise
#      seeds 26000-34999 are added once and the same lines read once more (still between: undecided)
#   4. pairings, a loading check against the 09-12 dumps, mechanism merges; the card is marked FREE and lent to bookings
#      for a day. Nothing trains afterwards: the faithful C' replay waits for the user.
# Log runs/xfer_tools/p0_chain.log; markers P0_CHAIN_DONE / P0_CHAIN_STOPPED. Re-runnable: xfer_eval.sh skips finished
# segments. Stop: kill $(cat runs/xfer_tools/p0_chain.pid) (a running segment finishes or is lost; rerun resumes).
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/xfer_tools/p0_chain.sh > /dev/null 2>&1 &)
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/p0_chain.log
D=$R/evaldump; SP=$R/xfer_hist; DMIN=${DMIN:-10}; W=/mnt/c/Users/asus; ST=$W/GPU_STATUS_mortal.txt
STEPS="60000 215800 278600"
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
dfree() { df -BG --output=avail /mnt/d | tail -1 | tr -dc 0-9; }
status() {
  printf '%s\r\n' "mortal GPU status (TanyaoDojo session), $(date '+%F %T %Z')" "$@" \
    "To book: create C:\\Users\\asus\\GPU_BOOK_<name>.txt (one line: who and what; add a line not_before=YYYY-MM-DD HH:MM to defer it)." \
    > $ST.new && mv $ST.new $ST
}
idle() {
  status "State: FREE (idle) since $(date +%T): the C' re-check (P0) has ended and no mortal GPU work is queued." \
    "An active booking (GPU_BOOK_<name>.txt, once past its not_before) gets the card within a minute; this file then shows the hold." \
    "This file is rewritten before any new mortal GPU work starts."
}
stopped() { say "$1: P0_CHAIN_STOPPED"; idle; exit 1; }
echo $$ > $T/p0_chain.pid
say "waiting for an idle GPU"
idle_n=0
while [ $idle_n -lt 3 ]; do
  # a guest holding the card (gpu_share.sh waiting, or a GPU_BUSY_ file) counts as busy too
  if pgrep -f "(mainline_next|anc_chain|pg_chain)\.sh" > /dev/null || pgrep -f "^$PY .*(train|train_pg|run_eval|client|server|ev_map)\.py" > /dev/null \
     || pgrep -f "gpu_share\.sh" > /dev/null || ls $W/GPU_BUSY_*.txt > /dev/null 2>&1
  then idle_n=0; else idle_n=$((idle_n + 1)); fi
  sleep 60
done
status "State: BUSY - evaluation only: the C' re-check (P0), three historical snapshots on 13,000-22,000 groups, about 9-14 hours from $(date +%H:%M). Exclusive: do not start GPU work." \
  "Lending: a booking fires at the next evaluation call (calls come about 12 minutes apart)." \
  "This file is rewritten when the chain ends."
mkdir -p $SP/archive $SP/infer $SP/logs
cp_ck() { cp "$1" $SP/archive/xfer_hist_s$2.pth.new && mv -f $SP/archive/xfer_hist_s$2.pth.new $SP/archive/xfer_hist_s$2.pth
          say "xfer_hist_s$2 = $1 $(md5sum < $SP/archive/xfer_hist_s$2.pth | cut -c1-8)"; }
[ -f $SP/archive/xfer_hist_s60000.pth ] || cp_ck /mnt/d/Better_mortal/weights_backup/rl1_best_155k.pth 60000
[ -f $SP/archive/xfer_hist_s215800.pth ] || cp_ck $R/cprime_C392k_0723_1921.pth 215800
[ -f $SP/archive/xfer_hist_s278600.pth ] || cp_ck $R/cprime_final_0724_0939.pth 278600
for st in $STEPS; do [ -s $SP/archive/xfer_hist_s$st.pth ] || stopped "checkpoint s$st missing"; done
ev() {   # one segment of 1,000 groups for all three snapshots
  for st in $STEPS; do
    [ "$(dfree)" -ge $DMIN ] || stopped "D: below $DMIN GB free before s$1"
    bash $T/xfer_eval.sh hist $st seg $1 10 || stopped "xfer_hist_s$st segment s$1 FAILED"
  done
}
pair() {
  $PY - "$D" "$1" <<'PY'
import glob, sys
import numpy as np
D, which = sys.argv[1], sys.argv[2]
pats = {"rl1": "xfer_hist_s60000", "C392k": "xfer_hist_s215800", "Cfinal": "xfer_hist_s278600",
        "A80k": "xfer_v11_s80000", "408k": "tpt_tpt_s408000", "old_Cfinal": "cprime_final", "old_C392k": "cprime_C392k"}
m = {}
for k, p in pats.items():
    d = {}
    for f in glob.glob(f"{D}/{p}_s*_gpufp32.npz"):
        z = np.load(f)
        d.update({int(s): float(v) for s, v in zip(z["seed"], z["mean"])})
    m[k] = d
new = set(range(11000, 14000)) | set(range(16000, 26000)) | set(range(26000, 35000))
old = set(range(10000, 11000)) | set(range(14000, 16000))
def show(label, a, b=None, only=None):
    seeds = set(m[a]) & (set(m[b]) if b else set(m[a]))
    if only is not None:
        seeds &= only
    seeds = sorted(seeds)
    if len(seeds) < 2:
        return None
    y = np.array([m[a][s] - (m[b][s] if b else 0.0) for s in seeds])
    h = 1.96 * y.std(ddof=1) / np.sqrt(len(y)); z = y.mean() / (h / 1.96)
    print(f"  {label:<36} {len(seeds):>6,} groups  {y.mean():+.3f} +- {h:.3f}  z {z:+.2f}")
    return y.mean(), z, len(seeds)
if which == "verdict":
    r = show("C' final - C392k (new groups)", "Cfinal", "C392k", new)
    print(f"VERDICT {r[0]:.4f} {r[1]:.4f} {r[2]}" if r else "VERDICT nan nan 0")
    sys.exit(0)
for lab, only in (("new groups", new), ("old 12k of 09-12", old), ("all", None)):
    print(f" [{lab}]")
    show("C' final - C392k (relaxation)", "Cfinal", "C392k", only)
    show("C392k - rl1_best (lambda 0.5 phase)", "C392k", "rl1", only)
    show("C' final - rl1_best", "Cfinal", "rl1", only)
    for k in ("rl1", "C392k", "Cfinal"):
        show(f"{k} vs v4", k, None, only)
print(" [other lines, all common groups]")
show("rl1_best - A80k (the two lines' starts)", "rl1", "A80k")
show("C' final - tpt 408k", "Cfinal", "408k")
print(" [loading check: same model, 09-12 dumps vs now]")
show("C' final now - 09-12", "Cfinal", "old_Cfinal")
show("C392k now - 09-12", "C392k", "old_C392k")
PY
}
mech() {
  cd /mnt/d/Better_mortal/jax_rl/mjai_bot || return
  for st in $STEPS; do
    p=xfer_hist_s$st; segs=""
    for f in $R/gapdump/${p}_s*.json; do segs="$segs $(basename $f .json)"; done
    $PY merge_dumps.py ${p}_all $segs > /dev/null || echo "merge ${p}_all failed"
  done
  $PY - <<'PY'
import json, os
R = os.path.expanduser("~/Projects/better_mortal/runs")
for tag in ("xfer_hist_s60000_all", "xfer_hist_s215800_all", "xfer_hist_s278600_all"):
    g = json.load(open(f"{R}/gapdump/{tag}.json")); p = json.load(open(f"{R}/placedump/{tag}.json")); b = json.load(open(f"{R}/bustdump/{tag}.json"))
    o, t = g["ours"], g["theirs"]
    r = lambda d, k: d[k] / d["kyoku"]
    att = r(o, "win_pts") - r(t, "win_pts"); dfn = -(r(o, "deal_in_pts") - r(t, "deal_in_pts"))
    n = p["ours"]["n"]; f4 = p["ours"]["rank"][3] / n; f4v = (n - p["ours"]["rank"][3]) / (3 * n)
    print(f"  {tag:<22} n={g['n_games']} call {100*r(o,'naki'):.2f} ({100*r(t,'naki'):.2f}) win {100*(r(o,'win')-r(t,'win')):+.2f} "
          f"dealin {100*(r(o,'deal_in')-r(t,'deal_in')):+.2f} riichi {100*(r(o,'riichi')-r(t,'riichi')):+.2f} "
          f"net {att+dfn:+.0f} ({att:+.0f} / {dfn:+.0f}) bust {100*b['ours']['bust']/b['ours']['n']:.2f} / {100*b['v4']['bust']/b['v4']['n']:.2f} "
          f"4th {100*(f4-f4v):+.2f}")
PY
}
say "new groups: 11000-13999, 16000-25999"
for s in 11000 12000 13000 $(seq 16000 1000 25000); do ev $s; done
for st in $STEPS; do bash $T/xfer_eval.sh hist $st 12k || stopped "xfer_hist_s$st old 12k FAILED"; done
read -r mean z n <<< "$(pair verdict | grep VERDICT | cut -d' ' -f2-)"
say "verdict read on $n new groups: C' final - C392k = $mean (z $z)"
decide() { $PY -c "import sys; m, z = float(sys.argv[1]), float(sys.argv[2]); print('HOLDS' if m >= 0.7 and z >= 2 else ('FAILS' if m <= 0.3 else 'BETWEEN'))" "$1" "$2"; }
v=$(decide $mean $z)
if [ "$v" = BETWEEN ]; then
  say "between +0.3 and +0.7 (or z < 2): adding seeds 26000-34999 once"
  for s in $(seq 26000 1000 34000); do ev $s; done
  read -r mean z n <<< "$(pair verdict | grep VERDICT | cut -d' ' -f2-)"
  say "verdict read on $n new groups: C' final - C392k = $mean (z $z)"
  v=$(decide $mean $z); [ "$v" = BETWEEN ] && v=UNDECIDED
fi
say "P0 verdict: $v"
pair all > $T/p0_final.txt 2>&1; say "pairings: $(tr '\n' ' ' < $T/p0_final.txt)"
mech > $T/p0_mech.txt 2>&1; say "mechanism: $(tr '\n' ' ' < $T/p0_mech.txt)"
idle; say "P0_CHAIN_DONE"
# nothing is queued after this chain and gpu_share.sh runs only at a boundary: lend the idle card for a day
END=$(( $(date +%s) + 86400 ))
while [ $(date +%s) -lt $END ]; do
  pgrep -f "^$PY .*(train|train_pg|run_eval|client|server)\.py" > /dev/null && { say "mortal GPU work started: lending loop ends"; exit 0; }
  n0=$(ls $W/GPU_BOOK_*.txt 2>/dev/null | wc -l)
  if [ $n0 -gt 0 ]; then
    bash $T/gpu_share.sh >> $LOG 2>&1
    [ $(ls $W/GPU_BOOK_*.txt 2>/dev/null | wc -l) -lt $n0 ] && { idle; say "hold over; status: FREE (idle)"; }
  fi
  sleep 60
done
say "lending loop ends after a day"
