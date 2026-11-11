#!/bin/bash
# Early-end gap, Step 2b (README.md): game-horizon EV maps of tpt 408k on the fresh key-20261021 games, restricted
# to the slices Step 1 flagged. (a) East kyoku that start with someone below 8k, about 4 hours; (b) every East kyoku,
# about a day and a half. Each priced decision also rolls out its successor, so its delta is exact (riichi-eval
# evaluate(select=...), checked 234/234 identical to the full map on 16 games at CPU fp32). --grp adds the
# kyoku-horizon GRP valuation of the same rollouts. Waits for EEG_GEN_DONE.
# Start: (setsid nohup bash ~/Projects/better_mortal/runs/eeg/tools/eeg_evmap.sh > /dev/null 2>&1 &)
# Log runs/eeg/evmap.log, marker EEG_EVMAP_DONE.
set -u
export MORTAL_DIR=/home/r/Projects/better_mortal/Mortal/mortal RIICHI_EVAL_LIBRIICHI=/home/r/riichi-eval-build/pylib_commit
B=$HOME/Projects/better_mortal; R=$B/runs; O=$R/eeg; PY=$B/.venv/bin/python; RE=$HOME/riichi-eval
LOG=$O/evmap.log
say() { echo "$(date +%m-%d_%T) $*" >> $LOG; }
echo $$ > $O/evmap.pid
say "waiting for EEG_GEN_DONE"
until grep -a -q -E "EEG_GEN_(DONE|INCOMPLETE)" $O/gen.log 2>/dev/null; do sleep 60; done
grep -a -q "EEG_GEN_DONE" $O/gen.log || { say "generation incomplete; not starting"; exit 1; }
DIRS=""; for s in 10000 15000 20000 25000 30000; do DIRS="$DIRS --log-dir $O/logs_s$s"; done
cd $RE
for v in "east_low|--only-phase East --only-below 8000" "east|--only-phase East"; do
  IFS='|' read -r name filt <<< "$v"
  out=runs/evmap_eeg_$name
  say "start $name ($filt)"
  $PY -W ignore tools/ev_map.py --resume $DIRS --ours $R/tpt_tpt_s408000.pth \
      --ref $B/baseline/mortal_v4.pth --grp $R/grp_full/grp.pth --horizon game --chunk 200 $filt \
      --out $out > $out.log 2>&1
  rc=$?
  for m in d_game_pt d_kyoku_grp; do
    $PY -W ignore tools/ev_context.py --dir $out --workers 8 --metric $m --dump $out.ctx_$m.json > $out.ctx_$m.txt 2>&1
  done
  say "$name done (rc $rc): $(grep -m1 -E '^all: all' $out.ctx_d_game_pt.txt)"
done
say "EEG_EVMAP_DONE"
