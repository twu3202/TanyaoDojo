# Better_mortal

训一个**稳定强于 Mortal v4**(256ch×54blk,23.8M 参数,离线+在线 RL 完全体)的立直麻将 AI。

- **数据**:天凤凤凰卓 18 年牌谱(251 万局 mjai 格式;受天凤条款约束不入库、不再分发)
- **评测协议**:10 万局复式 1v3(challenger 轮换 4 座打同一批牌山,seed_key=20260711,
  jun_pt=[90,45,0,-135],champion=mortal_v4)。avg_pt 95% CI 整体 >0 = 真超 v4。
  训练内 1000 局 test_play 噪声 ±5pt,只作趋势参考,从不作结论。

## 现状(2026-07-18)

离线阶段已收官,**离线天花板 ≈ -2.5pt**;已转入 Track C 自对弈 RL(在线值回归)补最后差距。

| 模型 | 配方 | avg_pt vs v4(95% CI) | 结论 |
|---|---|---|---|
| v1_best | 192宽·恒定LR·近8年 | -3.99 ± 0.53 | Phase 1 基线 |
| **v11** | 192宽·LR余弦衰减·近8年 | **-2.50 ± 0.53** | 纯 LR 修复 +1.5pt;**Track C 基座** |
| v5 | 256宽·LR衰减·+10防守通道·近8年 | -2.67 ± 0.53 | 防守特征+加宽净贡献 ≈ 0 |
| v18 | v5配方·全18年(251万局) | -3.02 ± 0.53 | 数据翻倍无增益 |

三个离线方向(特征/宽度/数据)边际收益均 ≈ 0——详细判读见 [RESULTS.md](RESULTS.md)。

## 文档地图

| 文档 | 内容 |
|---|---|
| [HANDOFF.md](HANDOFF.md) | **从这里开始**:当前状态、机器布局、常用命令、坑清单 |
| [ROADMAP.md](ROADMAP.md) | 总纲:做什么、不做什么 |
| [RESULTS.md](RESULTS.md) | 全部评测数字与判读 |
| [TRACK_C_SELFPLAY.md](TRACK_C_SELFPLAY.md) | 自对弈 RL 执行定稿(当前主线) |
| [RL_SIM_PLAN.md](RL_SIM_PLAN.md) | 算力账 + Mahjax/PPO 后手方案 |
| [SETUP_NOTES.md](SETUP_NOTES.md) | 环境搭建(libriichi 编译、数据处理的坑) |
| PHASE2_PLAN / EVAL_PLAN / ARCH_* | 背景与设计文档 |
| CHINA_MJ_*.md | 中国麻将调研(已归档,不做) |

## 布局速览

- `Mortal/` 上游 clone + 本地补丁(v5 obs 防守特征、train_grp 修复;改动清单见 ARCH_TRACKB_DEFENSE.md)
- `configs/` 全部训练/评测/自对弈 toml(`MORTAL_CFG` 指向)
- `scripts/` 训练 track、自对弈启停、评测聚合、数据下载
- `data/` `runs/` `weights_backup/` gitignore(数据不可分发;权重超 GitHub 单文件限)

机器:训练服务器(RTX 6000 Ada 49GB/48核)+ 本地 Windows 台式机(RTX 5060 Ti 8GB,
WSL2,与服务器同 Ubuntu 22.04 + Python 3.10 + torch 2.11+cu128,权重/编译产物可直接互搬)。
