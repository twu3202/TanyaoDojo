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
  - **重放回路已达标(2026-07-24,data_bridge/replay_env.py)**:300 场真实凤凰牌谱
    **3,213/3,213 局 100% 合法重放 + 终局类型全对**(判据达成)。要点:牌山布局
    deck[84+13p..]=配牌 / deck[83]↓=自摸 / deck[10+k]=岭上 / deck[9-2k]、[8-2k]=宝、里;
    init_from_deck 镜像 env._init 注入牌山与局面;动作映射利用"本回合 id(0-73,85)与
    响应 id(74-84)不相交"做无歧义驱动(下一事件属当前玩家且合法→执行,否则 PASS);
    九种九牌在 mjai 无显式动作,verify_step 的 KYUUSHU 分支会自动开新局,需就地终局。
    剩余:BC 数据集落盘格式(等 R2 网络选型定 obs 编码后一并做)。
  - 备选捷径(社区 Trisolar-ian/mahjax-mortal-merge 的思路):不重放人类谱,直接
    Mahjax 自产对局 + Mortal v4 当老师打标签(蒸馏)。状态分布对齐 PPO 实际访问域,
    可作 R2 的补充数据源;其 action_spec.py 的 87 动作解码器可参考。
- **R2 BC**:全量人类数据行为克隆 → 得 Mahjax 原生基座。
  - **轻量网络选型定案(2026-07-25,jax_rl/obs_lean.py + net_lean.py + bench_lean.py)**:
    上游 dict obs 把牌局信息压在 (3,200) 动作流水账里逼网络用注意力重建状态,且
    policy/critic 双抽取器纯 2×浪费。改 Mortal 式结构化观测:(34,20) 特征平面
    (手牌/牌河/副露/宝牌/可见计数)+ 26 标量,直接从 State 读取;1D-CNN 残差主干
    (128ch×6blk,1.75M 参数)共享干双头。**5060 Ti 实测 mb512 fwd+bwd:
    14.4µs/样本 vs 上游 295.6µs = 20.5×**;obs 体积亦从数百 KB 降至 ~6KB/样本
    (mb16k 从 OOM 变可行)。按矩阵分解外推 Ada 全训吞吐 ~2-4 万步/秒(达 R2.5
    目标带),待 R3 首跑实测定数。质量未验:BC 精度对照 = R2 的判据(平面通道
    不够再加宽,Mortal 938 平面是上限参照)。
  - **BC 对照实验通过(2026-07-25,data_bridge/make_bc_dataset.py + bc_lean.py)**:
    200 场 2,164 局 → 133,050 决策样本(双观测同点采集,零 reject,129s);
    两网同数据同预算(5 epochs,mb512,adam 3e-4)在 5060 Ti 对照:

    | 网络 | 参数 | val 总精度 | 打牌类(有选择) | 响应类 | epoch 耗时 |
    |---|---|---|---|---|---|
    | LeanACNet | 1.75M | **67.9%** | **60.3%** | 86.2% | 2-3s |
    | 上游 ACNet | 1.91M | 59.6% | 49.2% | 85.3% | 23s |

    轻网络不只快 20.5×,**同预算下学得还更好**(+8.3pt/+11.1pt)——结构化特征的
    归纳偏置优势;transformer 需要多得多的数据/轮次先学会"解析流水账"。注意项:
    真实叫牌子类(样本少、噪声大)上游略优,提示后续可补"近期舍牌时序"平面。
    判据达成:**R2 网络选型定版 LeanACNet**,BC/PPO 管线全部换装。
  - **规模化 BC(2026-07-25)**:2,000 场 21,082 局 → **1,298,153 决策样本**(21k 局仅
    1 reject,18.5 分钟单进程;全量 251 万局可按年分片并行,数十小时级)。10 epochs:
    **val 72.7% / 打牌 65.5% / 响应 90.3% / 真实叫牌 65.8%**——叫牌类从 133k 样本时的
    37% 翻倍,证实此前弱项是数据饥饿非架构缺陷。基座权重 runs/bc_lean_2k.pkl(WSL)。
    ⚠️ 上述系半盲观测成绩(见 obs_lean 死数组勘误,commit c390764)。
  - **全信息 20k BC(2026-07-26,obs 修复后)**:20,000 场 211,297 局 →
    **12,986,451 样本**(4 rejects,3.8h)。6 epochs:**val 77.9% / 打牌 72.3% /
    响应 91.9% / 叫牌 ~75%**——观测修复+10×数据合计 +5.2pt,打牌 +6.8pt,已近
    houou 大模型参照区(76-77%)。数据还有 ×125 余量(全量 251 万局)。
    基座权重 runs/bc_lean_20k.pkl(WSL+服务器)。
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
    ~760 步/秒)。
  - **服务器定量矩阵(2026-07-24 夜完结,Ada 49GB,两点法稳态)**:

    | 臂 | 尺寸/配置 | 稳态步/秒 | 编译 |
    |---|---|---|---|
    | A | 1024×128, ep4, mb4096, jit1(=上游尺寸) | 1,401(两次复现) | 138s |
    | B | 8192×32, ep4, mb4096, jit8 | 1,354 | 1,561s |
    | C1 | 8192×32, **ep2**, mb4096, jit4 | 2,477 | 485s |
    | C2 | 8192×32, **ep1**, mb4096, jit4 | **4,369** | 292s |
    | C3/C4 | mb16384 系 | **OOM**(重物化后仍需 98GiB) | — |

    结论:①**吞吐由 update 阶段的 epochs 垄断**——ep4→ep1 得 3.2×,反解 update 占比
    ep4 时 ~92%、ep1 时 ~74%;加宽减深与大 jit 在 compute-bound 区**无增益**(B≤A),
    大 jit 反而放大编译成本(jit8×ep4 编译 26 分钟)。②巨型 minibatch 物理不通:200-token
    注意力激活随 mb 线性爆炸,49GB 装不下 mb16k。③上游默认 ep4 对海量向量化样本是浪费,
    **定版配方 ep1-2 + mb4096**(大批量 PPO 惯例,OpenAI Five 亦 ~ep1);重网络下 4.4k 步/秒
    为当前天花板,**继续提速的唯一大杠杆 = 换轻网络(R2 选型)与 BF16**。
- **R3 PPO**:ppo_with_reg 配方(BC 起步 + magnet 正则,≈ Mortal-Policy 配方)跑在 R2.5 优化后
  的 harness 上,**采用矩阵定版配方 ep1-2 + mb4096**。
  - **端到端彩排通过(2026-07-25,本机 5060 Ti)**:`ppo_fast env_name=red_mahjong
    observe_type=lean mag_coef=0.2 pretrained_model_path=<bc_lean_2k.pkl>`——BC 权重
    加载✓、magnet 生效(mag_kl 0.23)✓、熵 0.90 起步(锐利 BC 策略)✓;
    **256 环境即 19.7k 步/秒**(magnet 代价 -8%),无 magnet 21.4k。重网络 Ada 天花板
    4.4k 作废,轻网络在 Ada + 8k 环境的正式吞吐待服务器空闲窗实测(预期数万级)。
  - **Ada 正式吞吐矩阵(2026-07-25,全臂 BC 起步+magnet 0.2)**:
    L1 8192env/ep1/mb4k=**57,801**;L2 ep2=35,756;L3 16k env/mb8k=53,635;
    L4 mb16k=58,032;L5 32k env/mb16k=55,374。结论:lean 网络下 update 不再是瓶颈,
    吞吐在 ~55-58k 步/秒进入 rollout/env 平台区,加大 env 数与 minibatch 均无增益
    (mb16k 也不再 OOM,lean obs 内存红利兑现)。**R3 定版配置 = 8192×32, ep1, mb4096,
    mag 0.2 → 57.8k 步/秒 ≈ 上游默认的 46×**;编译 57s。1 亿步 ≈ 半天(单 Ada)。
  - ⚠️ **shanten 标量 bug(2026-07-25 发现并修复)**:verify_step 不更新
    shanten_current_player → BC 数据集 obs 标量[8] 恒为 0,而 PPO/评测侧是真值
    (env.step 会更新)——特征分布错位。已在 make_bc_dataset 采集点重算真实向听
    (Shanten.number(当前手牌))。此前 2k 数据/权重带此瑕疵(对照结论仍立,两网同底),
    20k 重建已带修复,后续 BC 一律用修复版数据。
  - **过夜动力学彩排(07-25→26,旧基座旧观测,自洽可弃)**:10 亿步+无一崩,
    54.5k 步/秒恒定,熵 0.76→0.59 平台企稳,KL 健康,滚动检查点全程在岗——
    长跑稳定性与基础设施双验证(价值线的崩塌模式未出现)。
  - **R3 run1 正式启动(2026-07-26 00:19,服务器)**:全信息 obs + bc_lean_20k 基座 +
    定版配置(8192×32/ep1/mb4096/mag0.2),目标 200 亿步(~4 天),滚动检查点
    ~/jax_rl_lean/r3_run1.pkl,日志 /tmp/r3_run1.log。首块:熵 0.65、mag_kl 0.02
    (起步即贴锚)。评分板待 R4 评测桥上线后回填。
  - 配方模板:BC 基座 → lean PPO(ep1, mb4096, mag 0.2)→ R4 评测桥对 v4 打 100k。
- **R4 评测桥(神圣不可变)**:mjai 协议适配器把 Mahjax 模型包成 mjai bot,接入既有 libriichi
  one_vs_three(同一批牌山 seed_key=20260711)对 v4 打 100k——与 -2.50/-1.96 系列直接可比。
  里程碑判据不变:avg_pt 95% CI 整体 >0 = 真超 v4。
  - **贯通(2026-07-27 凌晨,mjai_bot/)**:tracker(events→obs,差分 100 场全等)+
    jax_engine(mjai-log 引擎,cans+tracker 双重掩码)+ run_eval(神圣协议)。
    两坑志:①validate_reaction_json 未导出 Python,宽 except 把 AttributeError 当非法
    动作→全程 none→手牌爆容量(假安全网);②`[x]*0` 仍先求值 x(FIVES[tt] KeyError)。
    400 局 ×2 模型共 **128,788 决策 fallback=0**——掩码/发射零瑕疵。评测速度
    ~2.5 局/秒(本机,v4 冠军 GPU + LeanJax CPU),100k ≈ 11h。
  - **首块评分板(400 局,±8.5)**:BC 基座(20k 场)= **-14.29pt**(排名分布
    [76,99,99,126],avg_rank 2.69);r3_run1@~50 亿步 = **-19.46pt**(2.76)。
  - **BC 对外强度缩放(400 局系列)**:20k 场=-14.29 → **353k 场(2.30 亿样本,
    2023+24 全量,val 79.4%/打牌 73.7%)= -10.80±8.4**(avg_rank 2.65)。
    数据 ×18 ≈ +3.5pt;对数外推 BC 独走的平台在 -6~-8pt 区间 → 收尾靠 league RL
    (与 Suphx/Mortal 的 SL→RL 分工一致)。
  - **4k 局定标(2026-07-29)**:bc_full_2y_ep3 = **-12.52±2.73**(avg_rank 2.653,
    64 万决策 fallback=0)。
  - **BC v3(2026-07-29,743k 场/4.87 亿样本,2019-20+2023-24,本机过夜)**:
    val 79.9%/打牌 74.5%;4k 局 = **-10.17±2.70(新最优)**。缩放律:353k→743k
    (×2.1)≈ +2.35pt,即**每翻倍 ≈ +2.3pt**;外推全量 251 万场 BC 平台 ≈ -6pt,
    与此前估计吻合——超 v4 的最后一段仍须 league RL。v3.5(六年 6.9 亿样本池)
    续训中;数据版图 2019-2024 共 7,219 shards/32GB(本机)。
  - **⚠️ BC 强度天花板实测(2026-07-30,三点收敛)**:窄网 v3(743k 场)=-10.17,
    窄网 v3.5(105 万场)=-10.46,**宽网 v4(192×8,5.3M 参数,val 80.6/打牌 75.3
    全面新高)=-10.53**——三者统计等同。结论:**模仿学习对 v4 的兑换在 ≈-10.3
    处触顶**,更高的人类预测精度不再换来对超人对手的胜率(Suphx/Mortal 的 SL→RL
    分工在我们的数据上重演)。战略定调:①BC 线冻结(-10.3 基座已够,后续年份
    数据/更大网络暂停);②全部强度预算押 league RL;③宽网仍有价值——作为 RL
    载具(容量余量留给超越模仿的策略),league run3 用 bc_w192 当基座与锚。
  - **宽网破台(2026-08-02,bc_w192 = 192ch×8blk/5.3M 参数,六年池)**:首测 -10.53
    系欠训假象(仅 1 epoch);补第 2 epoch 后 val 80.7%/打牌 75.4%(首超窄网台),
    4k 局 = **-8.87±2.70(新最优,平台击穿)**。结论:①容量路线有效且未见顶,
    但**欠训敏感**(少 1 epoch 损失 ~1.7pt)——宽网必须足训;②三个 BC 杠杆全部
    在效:更大网(256×10 候选)、全量数据(2009-16 待建,宽网未喂饱)、obs v2 加富
    (与容量疑似叠加而非互斥);③league run3 基座候选切换为宽网(ppo_league 已带
    channels/blocks 参数)。当前记分:窄网台 -10.2 → 宽网 -8.87 → 外推组合杠杆
    BC 可达 -5~-7,league 收尾。
  - **league run1 首验(2.7 亿步即被让机中断,4k 局)**:**-14.81±2.70**——与基座
    统计边缘持平略降,未现 r3 式崩退(同期 r3 已 -19)。判读:①lr_r 显示刚追平
    对手池,3% 训练量未到兑现期,RL 微调的"先小跌后爬"曲线属常态;②approx_kl
    0.005-6/update 偏快,续跑(league run2)拟 lr 3e-4→1e-4 + mag 0.2→0.3,
    护住强基座再攒 league 信号。检查点 league_run1_ckpt_banked.pkl。方法论分层定版:400 局=冒烟(±8.5,只辨大差),
    4k 局=选型(±2.7),100k=里程碑(±0.55)。此前 -10.8/-13.2 均为同一水位的
    噪声抽样;ep2/ep3 在 4k 分辨率下同级。数据缩放对外部 pt 的兑换率低于
    val 精度增速——BC 平台估计下修保守化,league RL 的担子更重。基座 bc_full_2y.pkl(ep1,ep2 因让机
    停在 2/3,补完收益 <1pt 可选)。构建吞吐注:服务器 48 核 8 工人 7.5h 产出
    2.3 亿样本(0.003% reject),全量 2009-2024 约需再 ×3 时长。
    两结论:①未超 v4(参照:价值线 rl1_best=-1.96,离线上限=-2.50);②**纯自博弈
    PPO 未把强度迁移到外部对手**(四座同策略均衡漂移 + avg_reward≈0 无外部信号,
    方向与 OpenAI Five 的 league 教训一致)。下一杠杆排序:全量数据 BC(数据 ×125,
    确定性收益)≫ PPO 重设计(对手池 league / 更强锚)。r3_run1 暂继续跑
    (零机会成本),全量 BC 基座就绪后以 league 版重启替换。
  - **架构定案(2026-07-25 侦察完毕,实现待做)**:用 libriichi 的 `'mjai-log'` 引擎类型
    (见 Mortal/mortal/engine.py 的 ExampleMjaiLogEngine)——OneVsThree 每个决策回调给
    `game_state.events_json`(开局以来全部 mjai 事件)+ `game_state.state.last_cans`
    (ActionCandidate:can_discard/chi_low·mid·high/pon/daiminkan/kakan/ankan/riichi/
    tsumo_agari/ron_agari/ryukyoku + target_actor,rust 端 validate_reaction 把关)。
    因此:**合法性引擎白嫖 libriichi,obs 无状态重建**——写 events→obs_lean 的纯 numpy
    构建器(每回调从事件流重建,消灭状态同步 bug),差分判据 = 对 houou 牌谱逐决策点与
    replay_env 的 obs_lean 完全一致;掩码由 cans 映射到 87 动作(粒度差处用 PlayerState
    getter 补:暗/加杠候选、食替禁打)。响应转 mjai JSON(打牌需回赤五面与 tsumogiri 旗)。
    待办:①mjai_obs.py 构建器+差分验证 ②cans→mask+动作→JSON ③LeanJaxEngine 批推理
    ④100 场 1v3 冒烟 → 100k 正式。

## 风险与对策

- Mahjax "obs 未定稿/API provisional" → 锁定 commit,升级需过回归测试。
- red_mahjong 与天凤规则的细节差异 → R1 回放差分即是最强的规则校验(251 万局逐决策比对,
  任何规则不一致会在回放合法性检查中暴露)。
- BC 天花板未知 → R2 后先过评测桥定坐标,再投 PPO。
- 旧管线随时可考古复跑(全套脚本/配置/权重在 repo 与服务器)。

## 与川麻线的协同

同一 JAX 栈、同一批运维件(哨兵/里程碑/落袋)、同一套正确性方法论(参考实现+差分)。
排期上 R1(立直数据桥)与川麻 P0(规则冻结)可并行——前者纯工程,后者纯规则。
