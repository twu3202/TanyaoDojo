#!/bin/bash
# Exploratory (README, "Online result"): is the Cal-QL base better than v11 offline? 2,000 more groups each
# (seeds 14000-15999), so the pair has 3,000 groups on the 12k seeds, same protocol (GPU fp32).
set -u
B=$HOME/Projects/better_mortal; R=$B/runs; T=$R/xfer_tools; PY=$B/.venv/bin/python; LOG=$T/cq_chain.log
W=/mnt/c/Users/asus; ST=$W/GPU_STATUS_mortal.txt; D=$R/evaldump
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
status() {
  printf '%s\r\n' "mortal GPU status (TanyaoDojo session), $(date '+%F %T %Z')" "$@" \
    "To book: create C:\\Users\\asus\\GPU_BOOK_<name>.txt (one line: who and what; add a line not_before=YYYY-MM-DD HH:MM to defer it)." \
    > $ST.new && mv $ST.new $ST
}
echo $$ > $T/cq_diag2.pid
status "State: BUSY - exploratory evaluation of the Cal-QL offline base against v11 (2 x 2,000 groups, about 50 minutes from $(date +%H:%M)). Exclusive: do not start GPU work." \
  "Lending: a booking fires at each evaluation call (about 25 minutes apart)." "This file is rewritten when it ends."
say "exploratory: CQ base and v11 on seeds 14000-15999"
for a in "v11cq 0" "v11base 0"; do set -- $a; bash $T/xfer_eval.sh $1 $2 seg 14000 20 || say "diag2 eval $1 FAILED"; done
$PY - "$D" > $T/cq_diag2.txt 2>&1 <<'PY'
import sys
import numpy as np
D = sys.argv[1]
def load(t):
    m = {}
    for s in (10000, 14000):
        z = np.load(f"{D}/{t}_s{s}_gpufp32.npz"); m.update(zip(z["seed"].tolist(), z["mean"].tolist()))
    return m
a, b = load("xfer_v11cq_s0"), load("xfer_v11base_s0")
s = sorted(set(a) & set(b))
for name, y in (("CQbase vs v4", np.array([a[i] for i in s])), ("v11 vs v4", np.array([b[i] for i in s])),
                ("CQbase - v11", np.array([a[i] - b[i] for i in s]))):
    h = 1.96 * y.std(ddof=1) / np.sqrt(len(y))
    print(f"  {name:<14} {len(y):>5,} groups  {y.mean():+.3f} +- {h:.3f}  z {y.mean() / (h / 1.96):+.2f}")
PY
say "diag2: $(tr -s ' ' < $T/cq_diag2.txt | tr '\n' '|')"
status "State: FREE (idle) since $(date +%T): the Cal-QL exploratory evaluation has ended and no mortal GPU work is queued." \
  "An active booking (GPU_BOOK_<name>.txt, once past its not_before) gets the card within a minute; this file then shows the hold." \
  "This file is rewritten before any new mortal GPU work starts."
say "CQ_DIAG2_DONE"
