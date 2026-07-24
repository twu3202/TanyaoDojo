# 立直线迁移 Mahjax 栈(2026-07-24 用户决策)

**决定**:原 Mortal 值网络在线管线(CPU 引擎 actor-learner,~1.1 万局/h)**停止发展**,立直线
切换到 Mahjax(JAX GPU 向量化)+ policy 线。依据:本地实测单卡 29-34 万步/秒(见
SICHUAN_RL_PLAN.md),policy 线 5000 万局需求在旧管线不可行(~190 天)、新栈单卡 1-2 周可行;
且与川麻线共栈,一次投资两线复用。

## 旧值线资产处置

- **rl1_best(-1.85~-1.96±0.53)= 值线最终交付**,双机备份(weights_backup/rl1_best_155k.pth)。
  比离线天花板 +0.55pt,rank CI 与基座分离——值线的历史使命完成。
- C2(松锚 λ=0.2)中止,未做终审;state 归档于服务器 runs/selfplay/(可考古,不再续)。
- 教训入档(TRACK_C_SELFPLAY.md 复盘节):值回归 churn/锚定权衡/峰值提取运营法,
  对 policy 线的价值 = ppo_with_reg 的正则强度直觉。

## 新管线(全部基于 Mahjax examples 现成组件)

- **R0 冒烟**:examples 依赖装齐,collect_offline_data → bc.py → ppo_with_reg.py 玩具规模跑通。
- **R1 数据桥(关键工程)**:251 万局天凤 mjson 回放进 Mahjax env(red_mahjong,规则对齐天凤),
  在 Mahjax 自己的 obs 编码下产出 (obs, action) 监督数据。要点:mjai↔Mahjax 动作空间映射、
  牌山重构回放、抽样差分校验。
- **R2 BC**:全量人类数据行为克隆(examples/bc.py,CNN 或 transformer 网络)→ 得 Mahjax 原生基座。
- **R2.5 训练循环优化(2026-07-24 代码已交付,定量待服务器窗口)**:
  - 背景实测:官方示例 as-is 稳态 ~1,260 步/秒(Ada,两点法)≈ 6.5k 局/h,慢于旧栈。
    勘误:示例三段各自有 jit,"漏 jit"不成立;真实慢源 = ①尺寸哲学(1024×128:env 步延迟对
    batch 平坦,窄而深的 rollout 浪费墙钟)②逐 update 的 Python 分发 + 9 个 float() 强制同步
    ③dict+transformer 默认(最贵组合)。
  - **交付:jax_rl/ppo_fast.py**——算法与上游严格一致(含 magnet),工程改动:加宽减深默认、
    K-update 大 jit(lax.scan 内联,设备端累积指标)、obs 类型可配。经验:donate_argnums 与
    vmap(init) 的 XLA 输出别名冲突,弃用;dict obs 单样本数百 KB,8GB 卡仅容微型批。
  - 本机(5060 Ti)验证:**功能全绿**(收敛/熵/KL 正常,大 jit 机制工作);吞吐 A/B 在 8GB 上
    **测不出差异**(minibatch 512 的 transformer 更新时间 ~10.8s/update 垄断一切,两组同
    ~760 步/秒)——**定量收益判定需在服务器 48GB 上做 10 分钟对照**(mb 4096、num_envs 8k+),
    待 GPU 空闲窗口执行;或直接并入 R3 首跑对照。目标不变:2-8 万步/秒。
- **R3 PPO**:ppo_with_reg 配方(BC 起步 + magnet 正则,≈ Mortal-Policy 配方)跑在 R2.5 优化后
  的 harness 上;5000 万局 = 优化后单卡 1-2 周 / 8×A100 短租 2-4 天(待 R2.5 后重估)。
- **R4 评测桥(神圣不可变)**:mjai 协议适配器把 Mahjax 模型包成 mjai bot,接入既有 libriichi
  one_vs_three(同一批牌山 seed_key=20260711)对 v4 打 100k——与 -2.50/-1.96 系列直接可比。
  里程碑判据不变:avg_pt 95% CI 整体 >0 = 真超 v4。

## 风险与对策

- Mahjax "obs 未定稿/API provisional" → 锁定 commit,升级需过回归测试。
- red_mahjong 与天凤规则的细节差异 → R1 回放差分即是最强的规则校验(251 万局逐决策比对,
  任何规则不一致会在回放合法性检查中暴露)。
- BC 天花板未知 → R2 后先过评测桥定坐标,再投 PPO。
- 旧管线随时可考古复跑(全套脚本/配置/权重在 repo 与服务器)。

## 与川麻线的协同

同一 JAX 栈、同一批运维件(哨兵/里程碑/落袋)、同一套正确性方法论(参考实现+差分)。
排期上 R1(立直数据桥)与川麻 P0(规则冻结)可并行——前者纯工程,后者纯规则。
