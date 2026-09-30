# B 分支 Stage 1:决策边界定向探索(2026-09-30)

Stage 0(`jax_rl/mjai_bot/stage0_stale.py`)发现:押/弃与立直决策上,被采取动作的 Q 已校准,
未被采取的另一侧被低估约 49 pt(离线 CQL 遗留,rl1 之后冻结),在线 RL 从不给它目标。
这里让 worker 在决策边界上探索另一侧,并让训练丢掉被探索污染的样本。

- `explore_engine.py`:放进 `Mortal/mortal/`。worker 端引擎,向 libriichi 要 v5 观测(v4 + 10 行防守特征),
  网络只吃前 1012 行;贪心动作与其押/弃或立直/默听对侧的 Q 差 < `margin_pt` 时以 `prob` 改走对侧。
  `is_greedy` 改记为「动作 == argmax Q」。
- `player_dataloader.patch`:`Mortal/mortal/player.py`(`[explore] enable` 时 TrainPlayer 用上面的引擎,
  每轮日志记可探索/已探索次数)与 `dataloader.py`(`drop_pre_explore` 时丢弃同一局里最后一次非贪心动作
  之前的样本;样本与牌谱事件逐条对齐,对不上的整局保留并计数)。配置缺省时两处补丁完全惰性。
  应用:`cd ~/Mortal && patch -p1 < player_dataloader.patch`。
- `test_frontier.py`:引擎向量化分类 vs PlayerState 重放的纯 Python 参考,逐决策比对(trx:40,114 决策 0 不一致;
  选杠标志行 494/494)。
- `test_explore_run.py`:小批真实对局 + 加载器对齐(trx:200 局探索局 + 800 局旧局 0 对不上;
  每局探索约 1.07 次;丢弃 8.1% 样本)。
- `stage1_setup.sh`:在 trx 上建 `runs/stage1_explore`(起点 tpt 408k,md5 6fb1d3e1…)、配置、启动/停止脚本、
  每 8000 步归档与崩溃保险。除 `[explore]` 外与 tpt 408k→448k 历史续跑完全相同
  (7 worker,对手池 base×4 / v4×2 / C392k×1,锚 cprime_final λ=0.1,ε 0.005 温度 0.1)。

启动:`bash scripts/run_selfplay_stage1.sh --resume --workers 7 --pool base,base,v4,<C392k>,base,base,v4`;
停止:`bash scripts/stop_selfplay_stage1.sh`(按 PID 进程组,含归档与保险)。
