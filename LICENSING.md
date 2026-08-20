# 许可说明(分目录双许可)

| 路径 | 许可 | 原因 |
|---|---|---|
| 仓库根下除下列目录外的全部代码<br/>(`jax_rl/` 的环境/观测/网络/训练/数据桥、`scripts/`、`configs/`、文档) | **MIT** | 本项目原创,无 copyleft 依赖 |
| `jax_rl/mjai_bot/` | **AGPL-3.0** | 运行时链接上游 [Mortal](https://github.com/Equim-chan/Mortal) 的 `libriichi`(AGPL-3.0) |

## 说明

- 评测桥(`jax_rl/mjai_bot/`)通过 `import libriichi` 使用上游的对局引擎与合法性校验,
  按 AGPL-3.0 的链接条款,该目录以 AGPL-3.0 发布。
- 上游 Mortal 源码**不随本仓库分发**,请按 [SETUP.md](SETUP.md) 自行克隆构建;
  其许可与版权声明归上游所有。
- 训练/推理主干(Mahjax 环境适配、`obs_lean`/`obs_v2` 观测、`net_lean` 网络、
  `bc_stream`/`ppo_*` 训练器、`data_bridge` 牌谱重放)不依赖 libriichi,以 MIT 提供。
- 本项目不分发天凤牌谱及其派生数据集(见 SETUP.md §4)。

*本说明不构成法律意见。*
