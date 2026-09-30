#!/bin/bash
# Branch B Stage 1 on trx: frontier exploration, otherwise identical to the tpt run 408k -> 448k.
set -eu
ROOT=$HOME/Projects/better_mortal
SP=$ROOT/runs/stage1_explore
OLD=$ROOT/runs/selfplay
mkdir -p $SP/logs $SP/buffer $SP/drain $SP/archive
[ -f $SP/state.pth ] && { echo "state.pth exists, not overwriting"; exit 1; }
cp $OLD/archive_resume/tpt_s408000.pth $SP/state.pth
cp $OLD/BASE.pth $SP/BASE.pth
cp $OLD/cprime_final_0724_0939.pth $SP/cprime_final_0724_0939.pth
md5sum $SP/state.pth $SP/BASE.pth $SP/cprime_final_0724_0939.pth | cut -c1-12,34-

# config: the tpt config with every run path moved to $SP, its own port, plus [explore]
CFG=$ROOT/configs/online_selfplay_stage1.toml
sed -e "s#runs/selfplay/#runs/stage1_explore/#g" -e "s#^port = 5577#port = 5578#" \
    $ROOT/configs/online_selfplay_tenhoupt.toml > $CFG
cat >> $CFG <<'TOML'

# B 分支 Stage 1(2026-09-30):决策边界定向探索(Mortal/mortal/explore_engine.py + player.py/dataloader.py 本地补丁)。
# 押/弃与立直/默听这两类决策上,贪心动作与对应另一侧的 Q 差 < margin_pt 时,以 prob 概率改走另一侧;
# 训练时丢弃同一局里最后一次非贪心动作之前的样本(整局奖励目标被探索改变)。其余与 tpt 408k→448k 完全相同。
[explore]
enable = true
margin_pt = 10.0
prob = 0.3
drop_pre_explore = true
TOML
grep -n "stage1_explore\|5578\|^\[explore\]" $CFG | head -20

# launcher / stopper: copies of the tpt-era scripts with SP, CFG and the train_play log-dir pattern changed
sed -e "s#^SP=\$ROOT/runs/selfplay#SP=\$ROOT/runs/stage1_explore#" \
    -e "s#^CFG=\$ROOT/configs/online_selfplay.toml#CFG=\$ROOT/configs/online_selfplay_stage1.toml#" \
    -e "s#selfplay/train_play\.\*#stage1_explore/train_play.*#" \
    -e "s#bash \$ROOT/scripts/stop_selfplay.sh#bash \$ROOT/scripts/stop_selfplay_stage1.sh#" \
    $ROOT/scripts/run_selfplay.sh > $ROOT/scripts/run_selfplay_stage1.sh
sed -e "s#^SP=\$ROOT/runs/selfplay#SP=\$ROOT/runs/stage1_explore#" -e "s#run_selfplay.sh --resume#run_selfplay_stage1.sh --resume ...#" \
    $ROOT/scripts/stop_selfplay.sh > $ROOT/scripts/stop_selfplay_stage1.sh
grep -n "^SP=\|^CFG=\|LOGDIR_SED=\|stop_selfplay" $ROOT/scripts/run_selfplay_stage1.sh
grep -n "^SP=" $ROOT/scripts/stop_selfplay_stage1.sh

# step archiver (every 8,000 steps) and collapse guard, pointed at $SP
sed -e "s#^S=\$HOME/Projects/better_mortal/runs/selfplay#S=\$HOME/Projects/better_mortal/runs/stage1_explore#" \
    -e "s#^A=\$S/archive_resume#A=\$S/archive#" $OLD/archiver_steps.sh > $SP/archiver_steps.sh
sed -e "s#runs/selfplay/tb#runs/stage1_explore/tb#" -e "s#runs/selfplay/collapse_guard.log#runs/stage1_explore/collapse_guard.log#" \
    -e "s#scripts/stop_selfplay.sh#scripts/stop_selfplay_stage1.sh#" \
    -e "s#^START = .*#START = datetime.datetime(2026, 9, 30, 0, 0).timestamp()#" $OLD/collapse_guard.py > $SP/collapse_guard.py
grep -n "^S=\|^A=" $SP/archiver_steps.sh; grep -n "^TB\|^LOG\|^STOP\|^START" $SP/collapse_guard.py
