#!/bin/bash
# Online transfer test (README.md): prepares runs/xfer_<arm> for one arm.
#   bash xfer_setup.sh ARM BASE_STATE PORT
# state.pth and BASE.pth = copies of the offline base (full training state: train.py's warm start from an offline
# state resets steps / best_perf and builds a fresh Adam), configs/online_selfplay_xfer_<arm>.toml from the template,
# the param primer next to the run (run_selfplay_m1.sh looks for it there), and manifest.txt with the md5 of every
# input. Refuses to touch an existing run.
set -eu
ARM=$1; BASE=$2; PORT=$3
B=$HOME/Projects/better_mortal; R=$B/runs; SP=$R/xfer_$ARM; T=$R/xfer_tools
CFG=$B/configs/online_selfplay_xfer_$ARM.toml
[ -e $SP/state.pth ] && { echo "!! $SP/state.pth exists, not overwriting"; exit 1; }
[ -f $BASE ] || { echo "!! no $BASE"; exit 1; }
mkdir -p $SP/logs $SP/buffer $SP/drain $SP/archive $SP/infer
cp $BASE $SP/state.pth
cp $BASE $SP/BASE.pth
sed -e "s#@SP@#$SP#g" -e "s#@PORT@#$PORT#g" $T/online_selfplay_xfer.toml > $CFG
cp $T/m1_prime_param.py $SP/
m() { md5sum "$1" | cut -c1-32; }
MM=$B/Mortal/mortal
{
  echo "arm $ARM, prepared $(date '+%F %T')"
  echo "base $BASE $(m $BASE)"
  echo "state.pth $(m $SP/state.pth)  BASE.pth $(m $SP/BASE.pth)"
  echo "config $CFG $(m $CFG)"
  echo "train.py $(m $MM/train.py)  server.py $(m $MM/server.py)  client.py $(m $MM/client.py)"
  echo "dataloader.py $(m $MM/dataloader.py)  model.py $(m $MM/model.py)"
  echo "mortal_v4 $(m $B/baseline/mortal_v4.pth)  grp $(m $R/grp_full/grp.pth)"
  $B/.venv/bin/python -c "import torch,sys; s=torch.load(sys.argv[1],weights_only=True,map_location='cpu'); c=s['config']; print('base steps', s['steps'], 'online', c['control']['online'], 'resnet', c['resnet'], 'version', c['control']['version'])" $SP/state.pth
} > $SP/manifest.txt
cat $SP/manifest.txt
